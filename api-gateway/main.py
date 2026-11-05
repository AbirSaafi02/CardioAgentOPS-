import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="CardioAgent-Ops API Gateway",
    version="1.0.0",
    description="Porte d'entrée unique de la plateforme de cardio-acoustique",
)

AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8001")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://agent-orchestrator:8002")
PREDICTION_TIMEOUT_SECONDS = float(
    os.getenv("PREDICTION_TIMEOUT_SECONDS", "60")
)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Affiche l'interface utilisateur médicale."""
    dashboard_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


@app.get("/api/v1/health")
async def health_check():
    """Vérifie l'état de la gateway et des micro-services."""
    services_status = {
        "api-gateway": "online",
        "audio-engine": "pending",
        "agent-orchestrator": "pending",
        "vector-db": "pending",
        "llm-server": "pending",
        "mlflow-server": "pending",
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{AUDIO_ENGINE_URL}/health")
        if response.status_code != 200:
            services_status["audio-engine"] = "offline"
        else:
            audio_health = response.json()
            services_status["audio-engine"] = (
                "online"
                if isinstance(audio_health, dict)
                and audio_health.get("status") == "online"
                and audio_health.get("model_loaded") is True
                else "degraded"
            )
    except httpx.RequestError:
        services_status["audio-engine"] = "offline"
    except ValueError:
        services_status["audio-engine"] = "offline"

    overall_status = (
        "ok" if services_status["audio-engine"] == "online" else "degraded"
    )
    return JSONResponse(
        content={"status": overall_status, "services": services_status}
    )


@app.post("/api/v1/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Transmet le fichier audio au micro-service audio-engine."""
    try:
        if not file.filename or not file.filename.lower().endswith(".wav"):
            raise HTTPException(
                status_code=400,
                detail="Seuls les fichiers .wav sont acceptés.",
            )

        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Le fichier audio est vide.",
            )

        files = {
            "file": (
                file.filename,
                contents,
                file.content_type or "audio/wav",
            )
        }

        async with httpx.AsyncClient(
            timeout=PREDICTION_TIMEOUT_SECONDS
        ) as client:
            response = await client.post(f"{AUDIO_ENGINE_URL}/predict", files=files)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Impossible de joindre le micro-service audio-engine. "
                "Vérifiez qu'il est démarré et que les services partagent "
                "le réseau Docker."
            ),
        ) from exc
    finally:
        await file.close()

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = {"detail": response.text or "Erreur audio-engine."}
        if isinstance(error_detail, dict):
            error_detail = error_detail.get("detail", error_detail)
        else:
            error_detail = "Réponse d'erreur invalide du micro-service audio-engine."
        raise HTTPException(
            status_code=response.status_code,
            detail=error_detail,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Le micro-service audio-engine a renvoyé une réponse invalide.",
        ) from exc
