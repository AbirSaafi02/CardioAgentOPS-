import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.preprocessing import InvalidAudioError, process_audio_bytes

app = FastAPI(
    title="CardioAgent-Ops Audio Engine",
    version="1.0.0",
    description="Micro-service d'inférence audio ONNX pour le diagnostic cardio-acoustique",
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "cardio_model.onnx"),
    )
)

try:
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    print(f"Modèle ONNX chargé avec succès depuis '{MODEL_PATH}'")
except Exception as exc:
    print(f"Erreur de chargement du modèle ONNX : {exc}")
    session = None
    input_name = None


@app.get("/health")
async def health_check():
    """Vérifie l'état du micro-service."""
    return {
        "status": "online" if session is not None else "degraded",
        "service": "audio-engine",
        "model_loaded": session is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Reçoit un fichier WAV et renvoie le résultat de l'inférence."""
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers .wav sont acceptés.",
        )

    if session is None or input_name is None:
        raise HTTPException(status_code=503, detail="Modèle ONNX non chargé.")

    start_time = time.time()

    try:
        contents = await file.read()
        input_data = process_audio_bytes(contents)
        outputs = session.run(None, {input_name: input_data})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Le modèle a renvoyé une probabilité invalide.")

        label_code = 1 if probability >= 0.35 else 0
        prediction_label = "Abnormal" if label_code == 1 else "Normal"
        elapsed_time = round((time.time() - start_time) * 1000, 2)

        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "prediction": prediction_label,
                "label_code": label_code,
                "confidence": round(
                    probability if label_code == 1 else 1.0 - probability,
                    4,
                ),
                "raw_probability": round(probability, 4),
                "processing_time_ms": elapsed_time,
            }
        )
    except HTTPException:
        raise
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'inférence : {exc}",
        ) from exc
    finally:
        await file.close()
