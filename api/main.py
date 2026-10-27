"""FastAPI service for next-word model inference."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.data.examples import DatasetExampleStore
from src.data.tokenizer import tokenize
from src.inference.predictor import NextWordPredictor, PredictionError

predictor = NextWordPredictor()
examples = DatasetExampleStore(Path("data/processed"))
app = FastAPI(title="ContextNext API", version="1.0.0")


class PredictRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="lstm", pattern="^(gru|rnn|lstm)$")
    top_k: int = Field(default=5, ge=1, le=50)
    example_id: str | None = None


@app.get("/health")
def health() -> dict:
    status = predictor.status()
    status["ready"] = status["vocabulary_available"] and bool(predictor.available_models())
    return status


@app.get("/examples")
def list_examples(
    split: str = Query(default="train"),
    q: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Search or page through real labelled examples from processed splits."""
    try:
        return examples.list_examples(split=split, query=q, limit=limit, offset=offset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _unavailable_ground_truth(reason: str, candidates: list[dict] | None = None) -> dict:
    result = {"available": False, "reason": reason}
    if candidates:
        result["candidates"] = candidates
    return result


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        model_context_length = predictor.model_sequence_lengths.get(request.model)
        context = " ".join(tokenize(request.sentence)[-3:])
        source = None
        target = None

        if request.example_id:
            example = examples.get(request.example_id)
            if example is None:
                raise HTTPException(status_code=404, detail=f"Unknown dataset example: {request.example_id}")
            if context != example["input"]:
                raise HTTPException(
                    status_code=400,
                    detail="The selected dataset example does not match the final three input words.",
                )
            source = {"id": example["id"], "split": example["split"], "index": example["index"]}
            target = example["target"]
            if model_context_length != 3:
                target = None
                ground_truth = _unavailable_ground_truth(
                    f"This labelled example has a 3-word context, but the loaded checkpoint expects {model_context_length} words."
                )
        elif model_context_length != 3:
            ground_truth = _unavailable_ground_truth(
                f"The processed labelled examples use 3-word contexts, but this checkpoint expects {model_context_length} words."
            )
        else:
            matches = examples.find(context)
            distinct_targets = sorted({match["target"] for match in matches})
            if not matches:
                ground_truth = _unavailable_ground_truth(
                    "No exact labelled dataset example uses this 3-word context."
                )
            elif len(distinct_targets) > 1:
                candidates = []
                for candidate_target in distinct_targets:
                    target_matches = [match for match in matches if match["target"] == candidate_target]
                    candidates.append(
                        {
                            "target": candidate_target,
                            "examples": [
                                {"id": match["id"], "split": match["split"]}
                                for match in target_matches[:5]
                            ],
                        }
                    )
                ground_truth = _unavailable_ground_truth(
                    "This context has multiple labelled targets. Choose a specific dataset example to score one record.",
                    candidates,
                )
            else:
                target = distinct_targets[0]
                source = {
                    "matches": len(matches),
                    "splits": sorted({match["split"] for match in matches}),
                    "example_ids": [match["id"] for match in matches[:10]],
                }

        result = predictor.predict(request.sentence, request.model, request.top_k, target_word=target)
        if target is not None:
            result["ground_truth"]["source"] = source
        else:
            result["ground_truth"] = ground_truth
        return result
    except PredictionError as exc:
        message = str(exc)
        status_code = 503 if "checkpoint" in message.lower() or "tensorflow" in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
