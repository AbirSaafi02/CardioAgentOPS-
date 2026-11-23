import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .errors import ModelContractError
from .preprocessing import MODEL_DIR

MODEL_PATH = MODEL_DIR / "cardio_model.onnx"
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))


class CardioModel:
    def __init__(self, meta: dict, model_path: Path = MODEL_PATH):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if inp.name != meta["input_name"]:
            raise ModelContractError(f"Entrée ONNX '{inp.name}' != meta '{meta['input_name']}'")
        self.input_name = inp.name
        self.classes = meta["classes"]
        self.patho_idx = self.classes.index("pathologique")

    def predict_windows(self, tensor: np.ndarray) -> np.ndarray:
        """(N,1,40,157) -> (N,2) probabilités (softmax déjà dans le modèle)."""
        return self.session.run(None, {self.input_name: tensor})[0]


def aggregate(window_probs: np.ndarray, classes: list[str], patho_idx: int,
              threshold: float = THRESHOLD) -> dict:
    """Moyenne des probabilités sur les fenêtres, puis seuil (comme predict_wav du notebook)."""
    mean = window_probs.mean(axis=0)
    p_patho = float(mean[patho_idx])
    label = "pathologique" if p_patho >= threshold else "normal"
    return {
        "label": label,
        "confidence": p_patho if label == "pathologique" else 1.0 - p_patho,
        "probabilities": {c: float(p) for c, p in zip(classes, mean)},
        "n_windows": int(len(window_probs)),
        "window_probabilities": [round(float(p), 4) for p in window_probs[:, patho_idx]],
    }