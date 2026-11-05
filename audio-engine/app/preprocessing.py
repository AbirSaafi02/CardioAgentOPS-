import tempfile
from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 2000
TARGET_LENGTH = 10000
APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"


class InvalidAudioError(ValueError):
    """Raised when the uploaded bytes are not a readable audio file."""


def process_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes to a Mel spectrogram with shape (1, 128, 20, 1)."""
    if not audio_bytes:
        raise InvalidAudioError("Le fichier audio est vide.")

    if audio_bytes.startswith(APPLEDOUBLE_MAGIC):
        raise InvalidAudioError(
            "Le fichier envoyé est un fichier de métadonnées macOS (AppleDouble), "
            "pas l'audio WAV. Envoyez le fichier original, sans le préfixe '._'."
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "input.wav"
            audio_path.write_bytes(audio_bytes)
            # Passing a path lets librosa fall back to ffmpeg/audioread for WAV
            # codecs that libsndfile cannot decode.
            signal, _ = librosa.load(str(audio_path), sr=TARGET_SR)
    except Exception as exc:
        raise InvalidAudioError(
            "Impossible de décoder ce fichier audio. Vérifiez qu'il s'agit "
            "d'un vrai fichier WAV pris en charge."
        ) from exc

    if len(signal) < TARGET_LENGTH:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)), mode="constant")
    else:
        signal = signal[:TARGET_LENGTH]

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=TARGET_SR,
        n_mels=128,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    spec_expanded = np.expand_dims(mel_spec_db, axis=-1)
    return np.expand_dims(spec_expanded, axis=0).astype(np.float32)
