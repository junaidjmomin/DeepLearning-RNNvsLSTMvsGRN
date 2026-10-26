"""FastAPI service for next-word model inference."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import NextWordPredictor, PredictionError

predictor = NextWordPredictor()
app = FastAPI(title="ContextNext API", version="1.0.0")


class PredictRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="lstm", pattern="^(gru|rnn|lstm)$")
    top_k: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health() -> dict:
    status = predictor.status()
    status["ready"] = status["vocabulary_available"] and bool(predictor.available_models())
    return status


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return predictor.predict(request.sentence, request.model, request.top_k)
    except PredictionError as exc:
        message = str(exc)
        status_code = 503 if "checkpoint" in message.lower() or "tensorflow" in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
