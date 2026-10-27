from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.examples import DatasetExampleStore
from src.inference.predictor import NextWordPredictor
from src.training.data import Vocabulary


class _FakePredictorModel:
    input_shape = (None, 3)
    output_shape = (None, 7)

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray([probabilities], dtype=np.float32)
        self.last_input = None

    def __call__(self, inputs, training=False):
        self.last_input = np.asarray(inputs).copy()
        return self.probabilities


class TestPredictorVocabularyAlignment(unittest.TestCase):
    def test_input_ids_and_decoded_predictions_match_training_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.json"
            vocab_path.write_text(json.dumps({"a": 0, "b": 1, "c": 2, "target": 3, "wrong": 4}), encoding="utf-8")
            vocabulary = Vocabulary.from_json(vocab_path)
            probabilities = [0.01] * vocabulary.size
            probabilities[vocabulary.encode("wrong")] = 0.10
            probabilities[vocabulary.encode("target")] = 0.80
            model = _FakePredictorModel(probabilities)
            predictor = NextWordPredictor(vocab_path=vocab_path, models={"gru": model})

            result = predictor.predict("A   B c", "gru", top_k=2, target_word="target")

            expected_ids = [vocabulary.encode(token) for token in ("a", "b", "c")]
            self.assertEqual(model.last_input.tolist(), [expected_ids])
            self.assertEqual(result["tokenization"]["token_ids"], expected_ids)
            self.assertEqual(result["tokenization"]["decoded_tokens"], ["a", "b", "c"])
            self.assertEqual(result["predictions"][0]["word"], "target")
            self.assertEqual(result["predictions"][0]["token_id"], vocabulary.encode("target"))
            self.assertEqual(result["ground_truth"]["rank"], 1)
            self.assertAlmostEqual(result["ground_truth"]["probability"], 0.8, places=6)

    def test_unknown_words_use_training_oov_id_and_padding_uses_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.json"
            vocab_path.write_text(json.dumps({"known": 0}), encoding="utf-8")
            model = _FakePredictorModel([0.1, 0.2, 0.7])
            model.output_shape = (None, 3)
            predictor = NextWordPredictor(vocab_path=vocab_path, models={"gru": model})

            result = predictor.predict("mystery known", "gru")

            self.assertEqual(result["tokenization"]["token_ids"], [0, 1, 2])
            self.assertEqual(result["tokenization"]["decoded_tokens"], ["<pad>", "<unk>", "known"])


class TestDatasetExampleStore(unittest.TestCase):
    def test_exact_examples_preserve_conflicting_labels_and_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.json").write_text(json.dumps([
                {"input": "vegetarian supreme and", "target": "one"},
                {"input": "and one double", "target": "pepperoni."},
            ]), encoding="utf-8")
            (root / "validation.json").write_text(json.dumps([
                {"input": "and one double", "target": "room"},
            ]), encoding="utf-8")
            (root / "test.json").write_text("[]", encoding="utf-8")
            store = DatasetExampleStore(root)

            found = store.find(" AND   ONE DOUBLE ")
            page = store.list_examples(split="all", query="and one double")

            self.assertEqual([record["target"] for record in found], ["pepperoni.", "room"])
            self.assertEqual([record["id"] for record in found], ["train:1", "validation:0"])
            self.assertEqual(page["total"], 2)
            self.assertEqual(store.get("train:0")["target"], "one")


if __name__ == "__main__":
    unittest.main()
