"""End-to-end API checks for labelled and ambiguous dynamic input."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api import main as api
from src.data.examples import DatasetExampleStore
from src.inference.predictor import NextWordPredictor
from src.training.data import Vocabulary


class FixedModel:
    input_shape = (None, 3)

    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities], dtype=np.float32)
        self.output_shape = (None, len(probabilities))

    def __call__(self, inputs, training=False):
        return self.probabilities


class TestPredictionFlow(unittest.TestCase):
    def test_exact_training_row_shows_wrong_prediction_and_correct_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "c": 2, "right": 3, "wrong": 4, "other": 5}), encoding="utf-8")
            (root / "train.json").write_text(json.dumps([{"input": "a b c", "target": "right"}]), encoding="utf-8")
            (root / "validation.json").write_text(json.dumps([{"input": "a b c", "target": "other"}]), encoding="utf-8")
            (root / "test.json").write_text("[]", encoding="utf-8")
            vocab = Vocabulary.from_json(root / "vocab.json")
            probabilities = [0.001] * vocab.size
            probabilities[vocab.encode("wrong")] = 0.8
            probabilities[vocab.encode("right")] = 0.1
            probabilities[vocab.encode("other")] = 0.094
            old_predictor, old_examples = api.predictor, api.examples
            api.predictor = NextWordPredictor(vocab_path=root / "vocab.json", models={"rnn": FixedModel(probabilities)})
            api.examples = DatasetExampleStore(root)
            try:
                client = TestClient(api.app)
                response = client.post("/predict", json={"sentence": "A B C", "model": "rnn", "example_id": "train:0", "include_trace": True})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["predictions"][0]["word"], "wrong")
                self.assertEqual(result["ground_truth"]["target"], "right")
                self.assertFalse(result["ground_truth"]["correct"])
                self.assertEqual(result["ground_truth"]["source"]["id"], "train:0")
                self.assertFalse(result["trace"]["available"])

                ambiguous = client.post("/predict", json={"sentence": "a b c", "model": "rnn"}).json()
                self.assertFalse(ambiguous["ground_truth"]["available"])
                self.assertEqual({item["target"] for item in ambiguous["ground_truth"]["candidates"]}, {"right", "other"})
                self.assertEqual(client.get("/examples", params={"split": "train", "q": "a b c"}).json()["examples"][0]["id"], "train:0")
            finally:
                api.predictor, api.examples = old_predictor, old_examples

    def test_five_word_context_uses_matching_checkpoint_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text(json.dumps({word: index for index, word in enumerate("a b c d e target wrong".split())}), encoding="utf-8")
            (root / "train.json").write_text(json.dumps([{"input": "a b c d e", "target": "target"}]), encoding="utf-8")
            (root / "validation.json").write_text("[]", encoding="utf-8")
            (root / "test.json").write_text("[]", encoding="utf-8")
            vocab = Vocabulary.from_json(root / "vocab.json")
            probabilities = [0.01] * vocab.size
            probabilities[vocab.encode("wrong")] = 0.8
            model = FixedModel(probabilities)
            model.input_shape = (None, 5)
            old_predictor = api.other_predictors.get(5)
            old_examples = api.other_examples.get(5)
            api.other_predictors[5] = NextWordPredictor(vocab_path=root / "vocab.json", sequence_length=5, models={"gru": model})
            api.other_examples[5] = DatasetExampleStore(root)
            try:
                client = TestClient(api.app)
                self.assertTrue(client.get("/health", params={"context_size": 5}).json()["models"]["gru"]["available"])
                self.assertEqual(client.get("/examples", params={"context_size": 5}).json()["examples"][0]["input"], "a b c d e")
                response = client.post("/predict", json={"sentence": "a b c d e", "model": "gru", "context_size": 5, "example_id": "train:0"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["sequence_length"], 5)
                self.assertEqual(response.json()["ground_truth"]["target"], "target")
            finally:
                if old_predictor is None:
                    api.other_predictors.pop(5, None)
                else:
                    api.other_predictors[5] = old_predictor
                if old_examples is None:
                    api.other_examples.pop(5, None)
                else:
                    api.other_examples[5] = old_examples


if __name__ == "__main__":
    unittest.main()
