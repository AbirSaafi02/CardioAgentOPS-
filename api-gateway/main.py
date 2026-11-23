import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import ValidationError

from schemas import AnalyzeResponse, AudioPrediction

logger = logging.getLogger("api-gateway")
logging.basicConfig(level=logging.INFO)

# Configuration via variables d'environnement (valeurs par défaut = noms DNS Docker)
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8001")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Un seul client HTTP partagé: réutilise les connexions (plus rapide)
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    yield
    await app.state.client.aclose()


app = FastAPI(title="CardioAgent-Ops API Gateway", lifespan=lifespan)


def is_wav(data: bytes) -> bool:
    # Un vrai WAV commence par "RIFF" + 4 octets de taille + "WAVE".
    # On ne fait pas confiance au nom du fichier, seulement au contenu.
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


async def call_audio_engine(client: httpx.AsyncClient, filename: str, content: bytes) -> dict:
    try:
        r = await client.post(
            f"{AUDIO_ENGINE_URL}/predict",
            files={"file": (filename, content, "audio/wav")},
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="audio-engine: délai dépassé")
    except httpx.HTTPStatusError as e:
        if 400 <= e.response.status_code < 500:  # la faute vient de l'audio -> on transmet
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        logger.error("audio-engine a répondu %s", e.response.status_code)
        raise HTTPException(status_code=502, detail="audio-engine a échoué")
    except httpx.RequestError as e:  # connexion refusée, DNS, etc.
        logger.error("audio-engine injoignable: %s", e)
        raise HTTPException(status_code=503, detail="audio-engine injoignable")
    except ValueError:  # r.json() a échoué: réponse pas en JSON
        raise HTTPException(status_code=502, detail="audio-engine: réponse invalide")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_pcg", response_model=AnalyzeResponse)
async def analyze_pcg(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "audio.wav"

    # --- 1. Validation (avant d'embêter les autres services) ---
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=415, detail="Seuls les fichiers .wav sont acceptés")

    content = await file.read(MAX_UPLOAD_BYTES + 1)  # on lit au max limite+1 octet
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux")
    if not is_wav(content):
        raise HTTPException(status_code=415, detail="Contenu non reconnu comme WAV")

    # --- 2. Appel audio-engine ---
    audio_json = await call_audio_engine(request.app.state.client, filename, content)

    # --- 3. Guardrail: on valide la réponse avant de la renvoyer ---
    try:
        prediction = AudioPrediction(**audio_json)
    except ValidationError as e:
        logger.error("Réponse audio-engine invalide: %s", e)
        raise HTTPException(status_code=502, detail="audio-engine: réponse non conforme")

    # --- 4. (étape suivante) appel agent-orchestrator ici ---

    return AnalyzeResponse(audio=prediction)