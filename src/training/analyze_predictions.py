"""Export reproducible correct and incorrect next-word examples for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.training.data import Vocabulary, ensure_non_empty, load_examples


def analyze_predictions(
    model: Any,
    contexts: np.ndarray,
    targets: np.ndarray,
    vocabulary: Vocabulary,
    sample_count: int = 10,
    seed: int = 42,
    batch_size: int = 256,
) -> dict[str, list[dict[str, Any]]]:
    """Collect shuffled, deterministic samples of correct, wrong, and Top-3 cases."""

    ensure_non_empty(contexts, targets)
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have matching lengths.")
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive.")

    id_to_token = {token_id: token for token, token_id in vocabulary.token_to_id.items()}
    categories: dict[str, list[dict[str, Any]]] = {"top1_correct": [], "top1_incorrect": [], "top3_hit": []}
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(targets))

    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size]
        probabilities = np.asarray(model(contexts[batch_indexes], training=False))
        if probabilities.ndim != 2 or probabilities.shape[1] != vocabulary.size:
            raise ValueError("Model output must be (batch, vocabulary_size) and match the vocabulary.")
        top_count = min(3, probabilities.shape[1])
        top_ids = np.argpartition(probabilities, -top_count, axis=1)[:, -top_count:]
        top_ids = np.take_along_axis(
            top_ids,
            np.argsort(np.take_along_axis(probabilities, top_ids, axis=1), axis=1)[:, ::-1],
            axis=1,
        )

        for row, example_index in enumerate(batch_indexes):
            target_id = int(targets[example_index])
            ranked_ids = [int(token_id) for token_id in top_ids[row]]
            is_top1 = ranked_ids[0] == target_id
            is_top3 = target_id in ranked_ids
            category_names = []
            if is_top1:
                category_names.append("top1_correct")
            else:
                category_names.append("top1_incorrect")
            if is_top3 and not is_top1:
                category_names.append("top3_hit")

            record = {
                "input": " ".join(id_to_token.get(int(token_id), "<unk>") for token_id in contexts[example_index] if token_id != 0),
                "target": id_to_token.get(target_id, "<unk>"),
                "top1_correct": is_top1,
                "top3_correct": is_top3,
                "predictions": [
                    {
                        "word": id_to_token.get(token_id, "<unk>"),
                        "probability": float(probabilities[row, token_id]),
                    }
                    for token_id in ranked_ids
                ],
            }
            for category in category_names:
                if len(categories[category]) < sample_count:
                    categories[category].append(record)
        if all(len(records) >= sample_count for records in categories.values()):
            break
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Export example predictions for error analysis.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/lstm_best.keras"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.json"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/lstm_prediction_examples.json"))
    args = parser.parse_args()

    import tensorflow as tf

    vocabulary = Vocabulary.from_json(args.vocab)
    contexts, targets = load_examples(args.test_data, vocabulary, args.sequence_length)
    model = tf.keras.models.load_model(args.model, compile=False)
    result = analyze_predictions(model, contexts, targets, vocabulary, args.sample_count, args.seed, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({category: len(records) for category, records in result.items()}, indent=2))


if __name__ == "__main__":
    main()
