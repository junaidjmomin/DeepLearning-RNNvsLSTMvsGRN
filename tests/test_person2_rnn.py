"""Unit tests for Person 2's data encoding and metric calculations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.data import OOV_ID, PAD_ID, Vocabulary, load_examples
from src.training.evaluate import evaluate_model


class _FakeModel:
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = probabilities
        self.position = 0

    def __call__(self, contexts, training=False):
        count = len(contexts)
        selected = self.probabilities[self.position : self.position + count]
        self.position += count
        return selected

    def count_params(self):
        return 17


class Person2RNNTests(unittest.TestCase):
    def test_loader_reserves_padding_and_oov_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vocab.json").write_text(json.dumps({"hello": 0, "world": 1}), encoding="utf-8")
            (root / "data.json").write_text(
                json.dumps([{"input": "hello unseen world", "target": "unseen"}]), encoding="utf-8"
            )
            vocabulary = Vocabulary.from_json(root / "vocab.json")
            contexts, targets = load_examples(root / "data.json", vocabulary, sequence_length=4)

        self.assertEqual(contexts.tolist(), [[PAD_ID, 2, OOV_ID, 3]])
        self.assertEqual(targets.tolist(), [OOV_ID])

    def test_evaluator_reports_expected_classification_metrics(self):
        probabilities = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=np.float64)
        model = _FakeModel(probabilities)
        metrics = evaluate_model(model, np.array([[1], [2]], dtype=np.int32), np.array([1, 2]), batch_size=2)

        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["top_1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["top_3_accuracy"], 1.0)
        self.assertEqual(metrics["parameter_count"], 17)


if __name__ == "__main__":
    unittest.main()
