"""Shared, lazily loaded next-word inference for available model checkpoints."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from src.training.data import OOV_ID, PAD_ID, Vocabulary


DEFAULT_CHECKPOINTS = {
    "gru": Path("artifacts/models/gru_best.keras"),
    "rnn": Path("artifacts/models/rnn_best.keras"),
    "lstm": Path("artifacts/models/lstm_best.keras"),
}


@dataclass(frozen=True)
class Prediction:
    word: str
    probability: float


class PredictionError(ValueError):
    """An invalid prediction request or unavailable model resource."""


class NextWordPredictor:
    """Load model artifacts only when selected and reuse them across requests."""

    def __init__(
        self,
        vocab_path: Path = Path("data/processed/vocab.json"),
        model_paths: dict[str, Path] | None = None,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.model_paths = dict(model_paths or DEFAULT_CHECKPOINTS)
        for name in DEFAULT_CHECKPOINTS:
            env_path = os.getenv(f"{name.upper()}_MODEL_PATH")
            if env_path:
                self.model_paths[name] = Path(env_path)
        self._vocabulary: Vocabulary | None = None
        self._id_to_token: dict[int, str] | None = None
        self._models: dict[str, Any] = {}
        self._lock = RLock()

    def available_models(self) -> list[str]:
        """Return model names with checkpoint files currently present."""

        return sorted(name for name, path in self.model_paths.items() if path.is_file())

    def status(self) -> dict[str, Any]:
        return {
            "vocabulary_available": self.vocab_path.is_file(),
            "models": {name: {"available": path.is_file(), "path": str(path)} for name, path in self.model_paths.items()},
        }

    def predict(self, sentence: str, model_name: str = "lstm", top_k: int = 5) -> dict[str, Any]:
        if not isinstance(sentence, str) or not sentence.strip():
            raise PredictionError("sentence must contain at least one word.")
        if not isinstance(model_name, str) or model_name.lower() not in self.model_paths:
            choices = ", ".join(sorted(self.model_paths))
            raise PredictionError(f"model must be one of: {choices}.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise PredictionError("top_k must be an integer from 1 to 50.")

        model_name = model_name.lower()
        tokens = re.sub(r"\s+", " ", sentence.lower().strip()).split()
        model = self._load_model(model_name)
        vocabulary = self._load_vocabulary()
        sequence_length = self._sequence_length(model)
        context_tokens = tokens[-sequence_length:]
        context_ids = [vocabulary.encode(token) for token in context_tokens]
        input_ids = np.full((1, sequence_length), PAD_ID, dtype=np.int32)
        input_ids[0, sequence_length - len(context_ids) :] = context_ids

        probabilities = np.asarray(model(input_ids, training=False)).reshape(-1)
        if probabilities.size != vocabulary.size:
            raise PredictionError(
                f"{model_name} output has {probabilities.size} classes but vocabulary has {vocabulary.size} IDs."
            )
        if not np.all(np.isfinite(probabilities)):
            raise PredictionError(f"{model_name} returned non-finite prediction probabilities.")
        count = min(top_k, len(probabilities))
        top_ids = np.argpartition(probabilities, -count)[-count:]
        top_ids = top_ids[np.argsort(probabilities[top_ids])[::-1]]
        id_to_token = self._load_id_to_token()
        predictions = [
            Prediction(id_to_token.get(int(token_id), "<unk>"), float(probabilities[token_id]))
            for token_id in top_ids
        ]
        return {
            "model": model_name,
            "context": context_tokens,
            "predictions": [item.__dict__ for item in predictions],
        }

    def _load_model(self, model_name: str):
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            model_path = self.model_paths[model_name]
            if not model_path.is_file():
                raise PredictionError(f"No checkpoint is available for {model_name}: {model_path}")
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PredictionError("TensorFlow is required to load prediction models.") from exc
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as exc:
                raise PredictionError(f"Could not load {model_name} checkpoint at {model_path}: {exc}") from exc
            self._models[model_name] = model
            return model

    def _load_vocabulary(self) -> Vocabulary:
        if not self.vocab_path.is_file():
            raise PredictionError(f"Vocabulary file was not found: {self.vocab_path}")
        if self._vocabulary is None:
            try:
                self._vocabulary = Vocabulary.from_json(self.vocab_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise PredictionError(f"Could not load vocabulary at {self.vocab_path}: {exc}") from exc
        return self._vocabulary

    def _load_id_to_token(self) -> dict[int, str]:
        if self._id_to_token is None:
            vocab = self._load_vocabulary()
            self._id_to_token = {token_id: token for token, token_id in vocab.token_to_id.items()}
            self._id_to_token[PAD_ID] = "<pad>"
            self._id_to_token[OOV_ID] = "<unk>"
        return self._id_to_token

    @staticmethod
    def _sequence_length(model: Any) -> int:
        shape = model.input_shape
        if isinstance(shape, list):
            if len(shape) != 1:
                raise PredictionError("Model must have exactly one input tensor.")
            shape = shape[0]
        if len(shape) != 2 or shape[1] is None or int(shape[1]) < 1:
            raise PredictionError("Model must have a fixed positive context length.")
        return int(shape[1])
