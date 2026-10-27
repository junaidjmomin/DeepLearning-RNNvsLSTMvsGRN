"""Memory-bounded evaluation and metrics reporting for next-word models."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.training.data import Vocabulary, ensure_non_empty, load_examples


def evaluate_model(model: Any, contexts: np.ndarray, targets: np.ndarray, batch_size: int = 256) -> dict[str, float | int]:
    """Evaluate a model without materialising all vocabulary probabilities at once.

    MSE, MAE, and RMSE compare the softmax probabilities with the one-hot target.
    Cross-entropy, Top-1/Top-3/Top-5 accuracy, and perplexity are the main
    next-word metrics.
    """

    ensure_non_empty(contexts, targets)
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have matching lengths.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    sample_count = len(targets)
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    cross_entropy_sum = 0.0
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    vocabulary_size: int | None = None

    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        probabilities = np.asarray(model(contexts[start:stop], training=False))
        if probabilities.ndim != 2 or probabilities.shape[0] != stop - start:
            raise ValueError("Model output must be shaped (batch_size, vocabulary_size).")
        if vocabulary_size is None:
            vocabulary_size = probabilities.shape[1]
        elif probabilities.shape[1] != vocabulary_size:
            raise ValueError("Model vocabulary size changed between batches.")

        batch_targets = targets[start:stop]
        if np.any(batch_targets < 0) or np.any(batch_targets >= vocabulary_size):
            raise ValueError("A target ID is outside the model vocabulary.")
        target_probabilities = probabilities[np.arange(len(batch_targets)), batch_targets]
        clipped_target_probabilities = np.clip(target_probabilities, 1e-12, 1.0)

        # CHOICE: calculate MSE/MAE from probabilities, not from word IDs.
        # For a probability vector p and one-hot target y:
        # sum((p-y)^2) = sum(p^2) - 2*p[target] + 1
        # sum(abs(p-y)) = 2 * (1 - p[target])
        squared_error_sum += float(np.sum(np.square(probabilities)) - 2 * np.sum(target_probabilities) + len(batch_targets))
        absolute_error_sum += float(2 * np.sum(1 - target_probabilities))
        cross_entropy_sum += float(-np.sum(np.log(clipped_target_probabilities)))

        predictions = np.argmax(probabilities, axis=1)
        top_1_correct += int(np.sum(predictions == batch_targets))
        top_k = np.argpartition(probabilities, -min(3, vocabulary_size), axis=1)[:, -min(3, vocabulary_size) :]
        top_3_correct += int(np.sum(np.any(top_k == batch_targets[:, None], axis=1)))
        top_k = np.argpartition(probabilities, -min(5, vocabulary_size), axis=1)[:, -min(5, vocabulary_size) :]
        top_5_correct += int(np.sum(np.any(top_k == batch_targets[:, None], axis=1)))

    assert vocabulary_size is not None
    mean_squared_error = squared_error_sum / (sample_count * vocabulary_size)
    cross_entropy = cross_entropy_sum / sample_count
    return {
        "samples": sample_count,
        "vocabulary_size": vocabulary_size,
        "mse": mean_squared_error,
        "mae": absolute_error_sum / (sample_count * vocabulary_size),
        "rmse": math.sqrt(mean_squared_error),
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 100.0)),
        "top_1_accuracy": top_1_correct / sample_count,
        "top_3_accuracy": top_3_correct / sample_count,
        "top_5_accuracy": top_5_correct / sample_count,
        "parameter_count": int(model.count_params()),
        "inference_latency_ms": _mean_single_example_latency_ms(model, contexts),
    }


def _mean_single_example_latency_ms(model: Any, contexts: np.ndarray, samples: int = 32) -> float:
    """Measure warmed, single-example inference latency over a small fixed sample."""

    measured = contexts[: min(samples, len(contexts))]
    # CHOICE: warm up once so TensorFlow setup time is not treated as model latency.
    model(measured[:1], training=False)
    started = time.perf_counter()
    for context in measured:
        model(context[None, :], training=False)
    elapsed_seconds = time.perf_counter() - started
    return elapsed_seconds * 1000 / len(measured)


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained vanilla RNN next-word model.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/rnn_best.keras"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/rnn_test_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    vocabulary = Vocabulary.from_json(args.vocab)
    contexts, targets = load_examples(args.test_data, vocabulary, args.sequence_length, args.max_test_samples)
    model = tf.keras.models.load_model(args.model)
    metrics = evaluate_model(model, contexts, targets, args.batch_size)
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
