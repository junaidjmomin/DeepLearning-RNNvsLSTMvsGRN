"""FastAPI service for next-word model inference."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.data.examples import DatasetExampleStore
from src.data.tokenizer import tokenize
from src.inference.predictor import NextWordPredictor, PredictionError

predictor = NextWordPredictor()
other_predictors: dict[int, NextWordPredictor] = {}
examples = DatasetExampleStore(Path("data/processed"))
other_examples: dict[int, DatasetExampleStore] = {}
app = FastAPI(title="ContextNext API", version="1.0.0")


class PredictRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="lstm", pattern="^(gru|rnn|lstm)$")
    top_k: int = Field(default=5, ge=1, le=50)
    example_id: str | None = None
    include_trace: bool = False
    context_size: int = Field(default=3, ge=1)


def predictor_for_context(context_size: int) -> NextWordPredictor:
    if context_size == 3:
        return predictor
    if context_size not in other_predictors:
        other_predictors[context_size] = NextWordPredictor(sequence_length=context_size)
    return other_predictors[context_size]


@app.get("/health")
def health(context_size: int = Query(default=3, ge=1)) -> dict:
    selected_predictor = predictor_for_context(context_size)
    status = selected_predictor.status()
    status["ready"] = status["vocabulary_available"] and bool(selected_predictor.available_models())
    status["available_context_lengths"] = [3, 5, 10]
    return status


@app.get("/examples")
def list_examples(
    split: str = Query(default="train"),
    q: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    context_size: int = Query(default=3, ge=1),
) -> dict:
    """Search or page through real labelled examples from processed splits."""
    try:
        store = example_store(context_size)
        return store.list_examples(split=split, query=q, limit=limit, offset=offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No prepared examples for context length {context_size}. Run python -m src.data.preprocess --context-size {context_size}.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _unavailable_ground_truth(reason: str, candidates: list[dict] | None = None) -> dict:
    result = {"available": False, "reason": reason}
    if candidates:
        result["candidates"] = candidates
    return result


def example_store(context_size: int) -> DatasetExampleStore:
    if context_size == 3:
        return examples
    if context_size not in other_examples:
        other_examples[context_size] = DatasetExampleStore(Path(f"data/processed_ctx{context_size}"))
    return other_examples[context_size]


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        selected_predictor = predictor_for_context(request.context_size)
        model_context_length = selected_predictor.model_sequence_lengths.get(request.model)
        context_size = model_context_length or request.context_size
        context = " ".join(tokenize(request.sentence)[-context_size:])
        store = example_store(context_size)
        source = None
        target = None

        if request.example_id:
            try:
                example = store.get(request.example_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"No prepared examples for context length {context_size}.") from exc
            if example is None:
                raise HTTPException(status_code=404, detail=f"Unknown dataset example: {request.example_id}")
            if context != example["input"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"The selected dataset example does not match the final {context_size} input words.",
                )
            source = {"id": example["id"], "split": example["split"], "index": example["index"]}
            target = example["target"]
        else:
            try:
                matches = store.find(context)
            except FileNotFoundError:
                matches = []
            distinct_targets = sorted({match["target"] for match in matches})
            if not matches:
                ground_truth = _unavailable_ground_truth(
                    f"No exact labelled dataset example uses this {context_size}-word context (or its split files are not prepared)."
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

        result = selected_predictor.predict(request.sentence, request.model, request.top_k, target_word=target, include_trace=request.include_trace)
        if target is not None:
            result["ground_truth"]["source"] = source
        else:
            result["ground_truth"] = ground_truth
        return result
    except PredictionError as exc:
        message = str(exc)
        status_code = 503 if "checkpoint" in message.lower() or "tensorflow" in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
