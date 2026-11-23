import hashlib
import io
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .errors import AudioTooShort, ModelContractError
from .inference import CardioModel, aggregate
from .mlflow_utils import log_inference
from .preprocessing import load_meta, preprocess, to_log_mel

logger = logging.getLogger("audio-engine")
logging.basicConfig(level=logging.INFO)

meta = load_meta()
model = CardioModel(meta)  # chargé UNE fois au démarrage


class PredictResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    n_windows: int
    window_probabilities: list[float]
    latency_ms: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: la 1re exécution de librosa (numba) est lente -> on la fait avant la 1re vraie requête
    to_log_mel(np.zeros(int(meta["sample_rate"] * meta["duration_s"]), dtype=np.float32), meta)
    model.predict_windows(np.zeros(tuple([1] + meta["input_shape"][1:]), dtype=np.float32))
    logger.info("audio-engine prêt (modèle: %s)", meta["model"])
    yield


app = FastAPI(title="audio-engine", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model"]}


# `def` (et pas `async def`): le travail est CPU-bound (filtrage, mel, ONNX).
# FastAPI exécute alors la fonction dans un thread pool -> le serveur reste réactif.
@app.post("/predict", response_model=PredictResponse)
def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    raw = file.file.read()
    t0 = time.perf_counter()
    try:
        tensor = preprocess(io.BytesIO(raw), meta)
    except AudioTooShort as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelContractError as e:
        logger.error("Contrat modèle violé: %s", e)
        raise HTTPException(status_code=500, detail="Erreur interne de prétraitement")
    except Exception as e:  # fichier illisible / pas un vrai WAV
        raise HTTPException(status_code=422, detail=f"Audio illisible: {type(e).__name__}")

    result = aggregate(model.predict_windows(tensor), model.classes, model.patho_idx)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Log MLflow APRÈS l'envoi de la réponse (n'allonge pas la latence du client)
    audio_hash = hashlib.sha256(raw).hexdigest()[:12]
    background_tasks.add_task(log_inference, result, latency_ms, audio_hash, meta["model"])

    return {**result, "latency_ms": latency_ms, "model": meta["model"]}