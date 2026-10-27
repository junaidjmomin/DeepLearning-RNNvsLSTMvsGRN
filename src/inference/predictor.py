"""Inference for the shared GRU, vanilla RNN, and LSTM vocabulary models."""

from __future__ import annotations

import os
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from src.data.tokenizer import tokenize
from src.training.data import OOV_ID, OOV_TOKEN, PAD_ID, PAD_TOKEN, Vocabulary


MODEL_NAMES = ("gru", "rnn", "lstm")
MODEL_ENV_VARS = {"gru": "GRU_MODEL_PATH", "rnn": "RNN_MODEL_PATH", "lstm": "LSTM_MODEL_PATH"}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class PredictionError(Exception):
    """An expected error while preparing or running model inference."""


class NextWordPredictor:
    def __init__(
        self,
        vocab_path: Path = Path("data/processed/vocab.json"),
        models_dir: Path = Path("artifacts/models"),
        sequence_length: int = 3,
        models: dict[str, Any] | None = None,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.models_dir = Path(models_dir)
        self.sequence_length = sequence_length
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.vocab: Vocabulary | None = None
        self._id_to_token: dict[int, str] = {}
        self.models: dict[str, Any] = {}
        self.checkpoints: dict[str, Path] = {}
        self.model_sequence_lengths: dict[str, int] = {}
        self.load_errors: dict[str, str] = {}
        self.tensorflow_error: str | None = None
        self._trace_layers: dict[str, Any] = {}
        self._load_vocab()
        if models is not None:
            for name, model in models.items():
                self._register_model(name, model, Path("<injected>"))
        else:
            self._load_models()

    def _load_vocab(self) -> None:
        try:
            self.vocab = Vocabulary.from_json(self.vocab_path)
            self._id_to_token = self.vocab.id_to_token
        except Exception as exc:
            raise PredictionError(f"Failed to load vocabulary: {exc}") from exc

    def _candidate_paths(self, model_name: str) -> list[Path]:
        override = os.getenv(self._override_name(model_name))
        if override:
            return [Path(override).expanduser()]
        suffix = "" if self.sequence_length == 3 else f"_ctx{self.sequence_length}"
        return [
            self.models_dir / f"{model_name}{suffix}_best.keras",
            self.models_dir / f"{model_name}{suffix}_final.keras",
            self.models_dir / f"{model_name}{suffix}.keras",
        ]

    def _override_name(self, model_name: str) -> str:
        return MODEL_ENV_VARS[model_name] if self.sequence_length == 3 else f"{model_name.upper()}_CTX{self.sequence_length}_MODEL_PATH"

    def _register_model(self, model_name: str, model: Any, checkpoint: Path) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model name: {model_name}")
        if self.vocab is None:
            raise PredictionError("Vocabulary not loaded")

        input_shape = getattr(model, "input_shape", None)
        output_shape = getattr(model, "output_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        model_sequence_length = (
            int(input_shape[-1])
            if input_shape is not None and input_shape[-1] is not None
            else self.sequence_length
        )
        if model_sequence_length != self.sequence_length:
            raise ValueError(f"checkpoint expects {model_sequence_length} context words, selected {self.sequence_length}")
        output_size = int(output_shape[-1]) if output_shape is not None and output_shape[-1] is not None else None
        if output_size is not None and output_size != self.vocab.size:
            raise ValueError(
                f"checkpoint output size {output_size} does not match vocabulary size {self.vocab.size}"
            )

        self.models[model_name] = model
        self.checkpoints[model_name] = checkpoint
        self.model_sequence_lengths[model_name] = model_sequence_length

    def _load_models(self) -> None:
        try:
            import tensorflow as tf
        except Exception as exc:
            self.tensorflow_error = str(exc)
            return

        for model_name in MODEL_NAMES:
            candidates = self._candidate_paths(model_name)
            existing = [path for path in candidates if path.is_file()]
            if not existing:
                override = os.getenv(self._override_name(model_name))
                if override:
                    self.load_errors[model_name] = f"checkpoint does not exist: {override}"
                continue

            last_error: Exception | None = None
            for checkpoint in existing:
                try:
                    model = self._load_checkpoint(tf, checkpoint)
                    self._register_model(model_name, model, checkpoint)
                    break
                except Exception as exc:
                    last_error = exc
            if model_name not in self.models and last_error is not None:
                self.load_errors[model_name] = str(last_error)

    def available_models(self) -> list[str]:
        return [name for name in MODEL_NAMES if name in self.models]

    @staticmethod
    def _load_checkpoint(tf: Any, checkpoint: Path) -> Any:
        try:
            return tf.keras.models.load_model(checkpoint, compile=False)
        except Exception as exc:
            if "quantization_config" not in str(exc):
                raise
        # Older Keras releases cannot deserialize a newer archive's null
        # quantization_config. Remove only that no-op field in a temporary copy.
        def strip_null_config(value):
            if isinstance(value, dict):
                return {key: strip_null_config(item) for key, item in value.items()
                        if not (key == "quantization_config" and item is None)}
            if isinstance(value, list):
                return [strip_null_config(item) for item in value]
            return value

        with tempfile.TemporaryDirectory() as directory:
            compatible = Path(directory) / checkpoint.name
            with zipfile.ZipFile(checkpoint) as source, zipfile.ZipFile(compatible, "w") as destination:
                for entry in source.infolist():
                    content = source.read(entry.filename)
                    if entry.filename == "config.json":
                        content = json.dumps(strip_null_config(json.loads(content))).encode("utf-8")
                    destination.writestr(entry, content)
            return tf.keras.models.load_model(compatible, compile=False)

    def status(self) -> dict:
        return {
            "vocabulary_available": self.vocab is not None,
            "models": {
                name: {
                    "available": name in self.models,
                    "checkpoint": str(self.checkpoints[name]) if name in self.checkpoints else None,
                    "sequence_length": self.model_sequence_lengths.get(name),
                    "load_error": self.load_errors.get(name),
                }
                for name in MODEL_NAMES
            },
            "tensorflow_error": self.tensorflow_error,
            "context_length": self.sequence_length,
            "vocab_size": self.vocab.size if self.vocab else 0,
        }

    def predict(self, sentence: str, model_name: str = "lstm", top_k: int = 5, target_word: str | None = None, include_trace: bool = False) -> dict:
        if not isinstance(sentence, str) or not sentence.strip():
            raise PredictionError("Sentence cannot be empty")
        if model_name not in MODEL_NAMES:
            raise PredictionError("Invalid model name. Choose from: gru, rnn, lstm")
        if top_k < 1 or top_k > 50:
            raise PredictionError("top_k must be between 1 and 50")
        if model_name not in self.models:
            detail = self.load_errors.get(model_name)
            message = f"Model {model_name} is not loaded. Check its checkpoint."
            if detail:
                message = f"{message} {detail}"
            if self.tensorflow_error:
                message = f"{message} TensorFlow unavailable: {self.tensorflow_error}"
            raise PredictionError(message)
        if self.vocab is None:
            raise PredictionError("Vocabulary not loaded")

        full_tokens = tokenize(sentence)
        sequence_length = self.model_sequence_lengths[model_name]
        context_tokens = full_tokens[-sequence_length:]
        padding_count = sequence_length - len(context_tokens)
        model_tokens = [PAD_TOKEN] * padding_count + context_tokens

        # Vocabulary.encode owns the reserved-ID offset. Do not shift these IDs
        # again here: training's load_examples uses this same method.
        input_ids = [PAD_ID] * padding_count + [self.vocab.encode(token) for token in context_tokens]
        input_tensor = np.asarray([input_ids], dtype=np.int32)
        model = self.models[model_name]
        try:
            probabilities = np.asarray(model(input_tensor, training=False))[0]
        except Exception as exc:
            raise PredictionError(f"Model inference failed: {exc}") from exc

        if probabilities.ndim != 1 or probabilities.size != self.vocab.size:
            raise PredictionError(
                f"Model output has {probabilities.size} classes; expected {self.vocab.size} vocabulary IDs"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PredictionError("Model returned invalid probabilities")

        ranked_ids = np.argsort(-probabilities, kind="stable")
        predictions = [
            {
                "rank": rank,
                "token_id": int(token_id),
                "word": self._id_to_token.get(int(token_id), OOV_TOKEN),
                "probability": float(probabilities[token_id]),
            }
            for rank, token_id in enumerate(ranked_ids[:top_k], start=1)
        ]

        result = {
            "sentence": sentence,
            "model": model_name,
            "checkpoint": str(self.checkpoints[model_name]),
            "sequence_length": sequence_length,
            "vocabulary_size": self.vocab.size,
            "context": context_tokens,
            "tokenization": {
                "tokens": full_tokens,
                "context_tokens": model_tokens,
                "token_ids": input_ids,
                "decoded_tokens": [self._id_to_token.get(token_id, OOV_TOKEN) for token_id in input_ids],
                "padding_count": padding_count,
            },
            "predictions": predictions,
        }
        if target_word is not None:
            result["ground_truth"] = self._score_probabilities(probabilities, target_word)
        if include_trace:
            result["trace"] = self._trace_model(model_name, input_tensor, model_tokens, probabilities, predictions)
        return result

    def _trace_model(self, model_name: str, input_tensor: np.ndarray, tokens: list[str], probabilities: np.ndarray, predictions: list[dict]) -> dict:
        """Inspect the saved model's actual embeddings, recurrent states, and output layer."""
        model = self.models[model_name]
        try:
            embedding_layer = model.get_layer("token_embedding")
            recurrent_layer = model.get_layer("vanilla_rnn" if model_name == "rnn" else model_name)
            output_layer = model.get_layer("next_token")
            embeddings = np.asarray(embedding_layer(input_tensor))[0]
            if model_name not in self._trace_layers:
                config = recurrent_layer.get_config()
                config["return_sequences"] = True
                config["return_state"] = True
                probe = type(recurrent_layer).from_config(config)
                probe(embeddings[None, :, :], mask=input_tensor != PAD_ID, training=False)
                probe.set_weights(recurrent_layer.get_weights())
                self._trace_layers[model_name] = probe
            probe = self._trace_layers[model_name]
            states = probe(embeddings[None, :, :], mask=input_tensor != PAD_ID, training=False)
            sequence = np.asarray(states[0])[0]
            hidden = np.asarray(states[1])[0]
            gate_values = self._gate_trace(model_name, recurrent_layer, embeddings, sequence, input_tensor[0])
            kernel, bias = output_layer.get_weights()
            logits = hidden @ kernel + bias
            shifted = logits - np.max(logits)
            reconstructed = np.exp(shifted) / np.exp(shifted).sum()
            if not np.allclose(reconstructed, probabilities, atol=1e-5, rtol=1e-5):
                raise ValueError("Trace output does not reproduce checkpoint probabilities")
            trace = {
                "available": True,
                "embedding_dimension": int(embeddings.shape[1]),
                "hidden_units": int(hidden.size),
                "steps": [
                    {
                        "position": index + 1,
                        "token": token,
                        "token_id": int(input_tensor[0, index]),
                        "active": bool(input_tensor[0, index] != PAD_ID),
                        "embedding_norm": float(np.linalg.norm(embeddings[index])),
                        "embedding_preview": [float(value) for value in embeddings[index, :6]],
                        "hidden_norm": float(np.linalg.norm(sequence[index])),
                        "hidden_preview": [float(value) for value in sequence[index, :6]],
                        "gates": gate_values[index],
                    }
                    for index, token in enumerate(tokens)
                ],
                "final_hidden_preview": [float(value) for value in hidden[:8]],
                "output": {
                    "softmax_sum": float(np.sum(probabilities)),
                    "top_logits": [
                        {"word": item["word"], "token_id": item["token_id"], "logit": float(logits[item["token_id"]])}
                        for item in predictions
                    ],
                },
            }
            if model_name == "lstm":
                trace["final_cell_preview"] = [float(value) for value in np.asarray(states[2])[0][:8]]
            return trace
        except Exception as exc:
            return {"available": False, "reason": f"Could not inspect this checkpoint: {exc}"}

    @staticmethod
    def _gate_trace(model_name: str, layer: Any, embeddings: np.ndarray, sequence: np.ndarray, input_ids: np.ndarray) -> list[dict[str, float]]:
        """Reconstruct gate activations and verify each saved recurrent state."""
        gates: list[dict[str, float]] = []
        if model_name == "rnn":
            return [{} for _ in input_ids]
        kernel, recurrent_kernel, bias = layer.get_weights()
        units = sequence.shape[1]
        previous_hidden = np.zeros(units, dtype=np.float32)
        previous_cell = np.zeros(units, dtype=np.float32)
        for index, token_id in enumerate(input_ids):
            if token_id == PAD_ID:
                gates.append({})
                continue
            x = embeddings[index] @ kernel
            recurrent = previous_hidden @ recurrent_kernel
            if model_name == "gru":
                if layer.reset_after:
                    x = x + bias[0]
                    recurrent = recurrent + bias[1]
                else:
                    x = x + bias
                update = _sigmoid(x[:units] + recurrent[:units])
                reset = _sigmoid(x[units:2 * units] + recurrent[units:2 * units])
                recurrent_candidate = recurrent[2 * units:] if layer.reset_after else (reset * previous_hidden) @ recurrent_kernel[:, 2 * units:]
                candidate = np.tanh(x[2 * units:] + reset * recurrent_candidate if layer.reset_after else x[2 * units:] + recurrent_candidate)
                new_hidden = update * previous_hidden + (1 - update) * candidate
                gate = {"update": float(update.mean()), "reset": float(reset.mean())}
            else:
                inputs = x + recurrent + bias
                input_gate = _sigmoid(inputs[:units])
                forget_gate = _sigmoid(inputs[units:2 * units])
                candidate = np.tanh(inputs[2 * units:3 * units])
                output_gate = _sigmoid(inputs[3 * units:])
                previous_cell = forget_gate * previous_cell + input_gate * candidate
                new_hidden = output_gate * np.tanh(previous_cell)
                gate = {
                    "forget": float(forget_gate.mean()),
                    "input": float(input_gate.mean()),
                    "output": float(output_gate.mean()),
                }
            if not np.allclose(new_hidden, sequence[index], atol=1e-4, rtol=1e-4):
                raise ValueError(f"Gate reconstruction differs from the saved {model_name} state at step {index + 1}")
            previous_hidden = new_hidden
            gates.append(gate)
        return gates

    def _score_probabilities(self, probabilities: np.ndarray, target: str) -> dict:
        if self.vocab is None:
            raise PredictionError("Vocabulary not loaded")
        target_id = self.vocab.encode(target)
        representable = target_id != OOV_ID
        model_token = self._id_to_token.get(target_id, OOV_TOKEN)
        target_probability = float(probabilities[target_id])
        ranked_ids = np.argsort(-probabilities, kind="stable")
        rank = int(np.flatnonzero(ranked_ids == target_id)[0]) + 1
        return {
            "available": True,
            "target": target,
            "target_id": target_id,
            "representable": representable,
            "model_token": model_token,
            "probability": target_probability,
            "rank": rank,
            "correct": bool(rank == 1) if representable else None,
            "cross_entropy": float(-np.log(max(target_probability, 1e-12))),
        }

    def score_target(self, result: dict, target: str) -> dict:
        """Calculate a target score using a repeat inference pass."""
        probabilities = np.asarray(
            self.models[result["model"]](np.asarray([result["tokenization"]["token_ids"]], dtype=np.int32), training=False)
        )[0]
        return self._score_probabilities(probabilities, target)


def predict_next_words(sentence: str, model=None, top_k: int = 5):
    """Backward-compatible helper around the shared predictor."""
    predictor = NextWordPredictor()
    if model is not None:
        predictor._register_model("lstm", model, Path("<provided model>"))
    return predictor.predict(sentence, model_name="lstm", top_k=top_k)
