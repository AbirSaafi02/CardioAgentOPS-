import logging
import os
import time

# Sans ces limites, MLflow réessaie ~7 fois avec backoff si le serveur est down.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

logger = logging.getLogger("audio-engine")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
EXPERIMENT = "audio-inference"
_experiment_id: str | None = None  # mis en cache après le 1er appel


def log_inference(result: dict, latency_ms: float, audio_hash: str, model_name: str) -> None:
    """Trace une inférence dans MLflow. Best-effort: ne doit JAMAIS casser le service."""
    global _experiment_id
    try:
        from mlflow.entities import Metric, Param
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=TRACKING_URI)
        if _experiment_id is None:
            exp = client.get_experiment_by_name(EXPERIMENT)
            _experiment_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)

        run = client.create_run(_experiment_id, run_name="inference")
        ts = int(time.time() * 1000)
        client.log_batch(
            run.info.run_id,
            params=[
                Param("label", result["label"]),
                Param("audio_sha256", audio_hash),      # hash, PAS le nom du fichier (donnée patient)
                Param("model", model_name),
            ],
            metrics=[
                Metric("confidence", result["confidence"], ts, 0),
                Metric("p_pathologique", result["probabilities"].get("pathologique", 0.0), ts, 0),
                Metric("latency_ms", latency_ms, ts, 0),
                Metric("n_windows", float(result["n_windows"]), ts, 0),
            ],
        )
        client.set_terminated(run.info.run_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow logging échoué (ignoré): %s", e)