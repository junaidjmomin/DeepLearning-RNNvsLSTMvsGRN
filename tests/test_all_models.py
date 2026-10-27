"""Comprehensive unit tests covering all three model architectures, data
utilities, evaluation metrics, and the predictor interface.

Run with:
    python -m pytest tests/ -v
or:
    python -m unittest discover tests/
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.data import OOV_ID, PAD_ID, Vocabulary, load_examples
from src.training.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal callable that mimics a Keras model for metric tests."""

    def __init__(self, probabilities: np.ndarray, param_count: int = 42):
        self.probabilities = probabilities
        self._param_count = param_count
        self._position = 0

    def __call__(self, contexts, training=False):
        n = len(contexts)
        batch = self.probabilities[self._position: self._position + n]
        self._position += n
        return batch

    def count_params(self):
        return self._param_count

    @property
    def input_shape(self):
        return (None, 3)  # batch, sequence_length=3


def _make_vocab_file(directory: Path, words: dict[str, int]) -> Path:
    path = directory / "vocab.json"
    path.write_text(json.dumps(words), encoding="utf-8")
    return path


def _make_data_file(directory: Path, records: list[dict]) -> Path:
    path = directory / "data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Vocabulary and data loading
# ---------------------------------------------------------------------------

class TestVocabulary(unittest.TestCase):

    def test_reserves_padding_and_oov_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = _make_vocab_file(Path(tmp), {"hello": 0, "world": 1})
            vocab = Vocabulary.from_json(vocab_path)
        self.assertEqual(vocab.encode("<pad>"), PAD_ID)
        self.assertEqual(vocab.encode("<unk>"), OOV_ID)
        self.assertEqual(vocab.size, 4)  # pad + unk + hello + world

    def test_known_tokens_have_ids_above_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = _make_vocab_file(Path(tmp), {"a": 0, "b": 1})
            vocab = Vocabulary.from_json(vocab_path)
        self.assertGreater(vocab.encode("a"), 1)
        self.assertGreater(vocab.encode("b"), 1)

    def test_unseen_token_maps_to_oov(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = _make_vocab_file(Path(tmp), {"hello": 0})
            vocab = Vocabulary.from_json(vocab_path)
        self.assertEqual(vocab.encode("notinvocab"), OOV_ID)


class TestLoadExamples(unittest.TestCase):

    def test_padding_applied_when_context_shorter_than_sequence_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_vocab_file(root, {"hello": 0, "world": 1})
            _make_data_file(root, [{"input": "hello", "target": "world"}])
            vocab = Vocabulary.from_json(root / "vocab.json")
            contexts, targets = load_examples(root / "data.json", vocab, sequence_length=3)
        self.assertEqual(contexts.shape, (1, 3))
        self.assertEqual(contexts[0, 0], PAD_ID)
        self.assertEqual(contexts[0, 1], PAD_ID)

    def test_context_truncated_to_most_recent_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_vocab_file(root, {"a": 0, "b": 1, "c": 2, "d": 3})
            _make_data_file(root, [{"input": "a b c d", "target": "a"}])
            vocab = Vocabulary.from_json(root / "vocab.json")
            contexts, _ = load_examples(root / "data.json", vocab, sequence_length=2)
        # Only the last 2 words (c, d) should be in context
        context_words = [
            {v: k for k, v in vocab.token_to_id.items()}.get(int(i), "?")
            for i in contexts[0]
        ]
        self.assertIn("c", context_words)
        self.assertIn("d", context_words)

    def test_oov_loader_encodes_unseen_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_vocab_file(root, {"hello": 0, "world": 1})
            _make_data_file(root, [{"input": "hello unseen world", "target": "unseen"}])
            vocab = Vocabulary.from_json(root / "vocab.json")
            contexts, targets = load_examples(root / "data.json", vocab, sequence_length=4)
        self.assertEqual(contexts.tolist(), [[PAD_ID, 2, OOV_ID, 3]])
        self.assertEqual(targets.tolist(), [OOV_ID])


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluateModel(unittest.TestCase):

    def _probs(self, rows):
        arr = np.array(rows, dtype=np.float64)
        # Normalize rows so they sum to 1
        return arr / arr.sum(axis=1, keepdims=True)

    def test_perfect_prediction_accuracy_one(self):
        # target=1 for both examples, model puts max prob on token 1
        probs = self._probs([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        model = _FakeModel(probs)
        metrics = evaluate_model(model, np.array([[1], [1]]), np.array([1, 1]), batch_size=2)
        self.assertAlmostEqual(metrics["top_1_accuracy"], 1.0)

    def test_zero_accuracy_on_wrong_predictions(self):
        # target=2 but model puts max on index 0
        probs = self._probs([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        model = _FakeModel(probs)
        metrics = evaluate_model(model, np.array([[1], [1]]), np.array([2, 2]), batch_size=2)
        self.assertAlmostEqual(metrics["top_1_accuracy"], 0.0)

    def test_top3_accuracy_higher_than_top1(self):
        probs = self._probs([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        model = _FakeModel(probs)
        # target 1 is top-1 for first, target 2 is top-3 (but not top-1) for second
        metrics = evaluate_model(model, np.array([[1], [2]]), np.array([1, 2]), batch_size=2)
        self.assertGreaterEqual(metrics["top_3_accuracy"], metrics["top_1_accuracy"])
        self.assertGreaterEqual(metrics["top_5_accuracy"], metrics["top_3_accuracy"])

    def test_metrics_keys_present(self):
        probs = self._probs([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        model = _FakeModel(probs)
        metrics = evaluate_model(model, np.array([[1], [2]]), np.array([1, 2]), batch_size=2)
        for key in ("mse", "mae", "rmse", "cross_entropy", "perplexity", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "parameter_count"):
            self.assertIn(key, metrics, f"Missing metric key: {key}")

    def test_parameter_count_delegated(self):
        probs = self._probs([[0.5, 0.5]])
        model = _FakeModel(probs, param_count=999)
        metrics = evaluate_model(model, np.array([[1]]), np.array([0]), batch_size=1)
        self.assertEqual(metrics["parameter_count"], 999)

    def test_mse_mae_rmse_non_negative(self):
        probs = self._probs([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]])
        model = _FakeModel(probs)
        metrics = evaluate_model(model, np.array([[1], [1]]), np.array([1, 1]), batch_size=2)
        self.assertGreaterEqual(metrics["mse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_rmse_equals_sqrt_mse(self):
        import math
        probs = self._probs([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]])
        model = _FakeModel(probs)
        metrics = evaluate_model(model, np.array([[1], [1]]), np.array([1, 1]), batch_size=2)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(metrics["mse"]), places=6)


# ---------------------------------------------------------------------------
# Model architecture tests (lightweight — no GPU required)
# ---------------------------------------------------------------------------

class TestGRUModel(unittest.TestCase):

    def setUp(self):
        import tensorflow as tf  # noqa: F401 — import here so tests can be collected without TF

    def test_output_shape(self):
        import tensorflow as tf
        from src.models.gru import GRUConfig, build_gru_model

        tf.keras.utils.set_random_seed(0)
        config = GRUConfig(embedding_dim=8, hidden_units=16)
        model = build_gru_model(vocab_size=10, sequence_length=3, config=config)
        out = model(np.zeros((2, 3), dtype=np.int32), training=False)
        self.assertEqual(out.shape, (2, 10))

    def test_output_sums_to_one(self):
        import tensorflow as tf
        from src.models.gru import GRUConfig, build_gru_model

        tf.keras.utils.set_random_seed(0)
        config = GRUConfig(embedding_dim=8, hidden_units=16)
        model = build_gru_model(vocab_size=10, sequence_length=3, config=config)
        out = np.asarray(model(np.zeros((4, 3), dtype=np.int32), training=False))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-5)

    def test_invalid_vocab_size_raises(self):
        from src.models.gru import GRUConfig, build_gru_model
        with self.assertRaises(ValueError):
            build_gru_model(vocab_size=2, sequence_length=3, config=GRUConfig())

    def test_invalid_sequence_length_raises(self):
        from src.models.gru import GRUConfig, build_gru_model
        with self.assertRaises(ValueError):
            build_gru_model(vocab_size=10, sequence_length=0, config=GRUConfig())


class TestRNNModel(unittest.TestCase):

    def test_output_shape(self):
        import tensorflow as tf
        from src.models.rnn import RNNConfig, build_rnn_model

        tf.keras.utils.set_random_seed(0)
        config = RNNConfig(embedding_dim=8, hidden_units=16)
        model = build_rnn_model(vocab_size=10, sequence_length=3, config=config)
        out = model(np.zeros((2, 3), dtype=np.int32), training=False)
        self.assertEqual(out.shape, (2, 10))

    def test_output_sums_to_one(self):
        import tensorflow as tf
        from src.models.rnn import RNNConfig, build_rnn_model

        tf.keras.utils.set_random_seed(0)
        config = RNNConfig(embedding_dim=8, hidden_units=16)
        model = build_rnn_model(vocab_size=10, sequence_length=3, config=config)
        out = np.asarray(model(np.zeros((4, 3), dtype=np.int32), training=False))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-5)


class TestLSTMModel(unittest.TestCase):

    def test_output_shape(self):
        import tensorflow as tf
        from src.models.lstm import LSTMConfig, build_lstm_model

        tf.keras.utils.set_random_seed(0)
        config = LSTMConfig(embedding_dim=8, hidden_units=16)
        model = build_lstm_model(vocab_size=10, sequence_length=3, config=config)
        out = model(np.zeros((2, 3), dtype=np.int32), training=False)
        self.assertEqual(out.shape, (2, 10))

    def test_output_sums_to_one(self):
        import tensorflow as tf
        from src.models.lstm import LSTMConfig, build_lstm_model

        tf.keras.utils.set_random_seed(0)
        config = LSTMConfig(embedding_dim=8, hidden_units=16)
        model = build_lstm_model(vocab_size=10, sequence_length=3, config=config)
        out = np.asarray(model(np.zeros((4, 3), dtype=np.int32), training=False))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-5)

    def test_invalid_vocab_size_raises(self):
        from src.models.lstm import LSTMConfig, build_lstm_model
        with self.assertRaises(ValueError):
            build_lstm_model(vocab_size=2, sequence_length=3, config=LSTMConfig())


# ---------------------------------------------------------------------------
# Predictor interface tests (without real checkpoints)
# ---------------------------------------------------------------------------

class TestNextWordPredictor(unittest.TestCase):

    def test_invalid_sentence_raises(self):
        from src.inference.predictor import NextWordPredictor, PredictionError
        predictor = NextWordPredictor()
        with self.assertRaises(PredictionError):
            predictor.predict("", model_name="lstm")

    def test_invalid_model_name_raises(self):
        from src.inference.predictor import NextWordPredictor, PredictionError
        predictor = NextWordPredictor()
        with self.assertRaises(PredictionError):
            predictor.predict("hello world", model_name="transformer")

    def test_invalid_top_k_raises(self):
        from src.inference.predictor import NextWordPredictor, PredictionError
        predictor = NextWordPredictor()
        with self.assertRaises(PredictionError):
            predictor.predict("hello world", model_name="lstm", top_k=0)

    def test_available_models_returns_list(self):
        from src.inference.predictor import NextWordPredictor
        predictor = NextWordPredictor()
        self.assertIsInstance(predictor.available_models(), list)

    def test_status_contains_expected_keys(self):
        from src.inference.predictor import NextWordPredictor
        predictor = NextWordPredictor()
        status = predictor.status()
        self.assertIn("vocabulary_available", status)
        self.assertIn("models", status)
        for name in ("gru", "rnn", "lstm"):
            self.assertIn(name, status["models"])


if __name__ == "__main__":
    unittest.main()
