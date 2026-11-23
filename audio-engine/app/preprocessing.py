"""Prétraitement audio.

Copie FIDÈLE du notebook (fonctions preprocess_file + mel_spectrogram_db).
Toute différence ici = training-serving skew (le modèle se trompe sans erreur).
"""
import json
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt

from .errors import AudioTooShort, ModelContractError

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "models"))
META_PATH = MODEL_DIR / "cardio_model_meta.json"
MAX_WINDOWS = 6  # comme MAX_SEG_PER_FILE dans le notebook


def load_meta(path: Path = META_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bandpass(y: np.ndarray, fs: int, low: float, high: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [low / nyq, min(high, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, y)


def clean_signal(y: np.ndarray, meta: dict) -> np.ndarray:
    y = y - np.mean(y)                                   # enlève la composante continue
    y = bandpass(y, meta["sample_rate"], *meta["bandpass_hz"])
    return y / (np.max(np.abs(y)) + 1e-8)                # amplitude dans [-1, 1]


def split_windows(y: np.ndarray, n_samples: int, max_windows: int = MAX_WINDOWS) -> list[np.ndarray]:
    segs = []
    for start in range(0, len(y), n_samples):
        seg = y[start:start + n_samples]
        if len(seg) < n_samples * 0.5:                   # morceau final trop court -> ignoré
            continue
        if len(seg) < n_samples:                         # sinon zero-padding
            seg = np.pad(seg, (0, n_samples - len(seg)))
        segs.append(seg.astype(np.float32))
    if len(segs) > max_windows:                          # fenêtres réparties sur tout le fichier
        idx = np.linspace(0, len(segs) - 1, max_windows).astype(int)
        segs = [segs[i] for i in idx]
    return segs


def to_log_mel(seg: np.ndarray, meta: dict) -> np.ndarray:
    # fmin/fmax/top_db ne sont pas (encore) dans meta.json -> valeurs du notebook par défaut
    S = librosa.feature.melspectrogram(
        y=seg, sr=meta["sample_rate"], n_fft=meta["n_fft"], hop_length=meta["hop_length"],
        n_mels=meta["n_mels"], fmin=meta.get("fmin", 20), fmax=meta.get("fmax", 900),
    )
    return librosa.power_to_db(S, ref=1.0, top_db=meta.get("top_db", 80)).astype(np.float32)


def preprocess(source, meta: dict) -> np.ndarray:
    """.wav -> tenseur (N_fenêtres, 1, 40, 157) float32 prêt pour ONNX.

    PAS de normalisation (mel_mean/mel_std) ici: elle est DÉJÀ dans le graphe ONNX.
    """
    sr = meta["sample_rate"]
    n_samples = int(sr * meta["duration_s"])

    y, _ = librosa.load(source, sr=sr, mono=True)
    if len(y) < sr:
        raise AudioTooShort("Audio trop court (< 1 s)")

    segs = split_windows(clean_signal(y, meta), n_samples)
    if not segs:
        raise AudioTooShort("Audio trop court pour former une fenêtre exploitable")

    mels = np.stack([to_log_mel(s, meta) for s in segs])         # (N, 40, 157)
    expected = tuple(meta["input_shape"][2:])
    if mels.shape[1:] != expected:
        raise ModelContractError(f"Shape {mels.shape[1:]} != attendu {expected}")
    return mels[:, None, :, :].astype(np.float32)