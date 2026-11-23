from pydantic import BaseModel, Field


class AudioPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float | None = None
    model: str | None = None
    n_windows: int | None = None
    window_probabilities: list[float] | None = None


class AnalyzeResponse(BaseModel):
    audio: AudioPrediction
    report: dict | None = None  # rempli plus tard par agent-orchestrator