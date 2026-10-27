"""Small memorization check for the shared encoding and training path.

This synthetic fixture is diagnostic only; it does not replace the DailyDialog
training configuration or contribute results to the model comparison.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.training.data import Vocabulary, load_examples
from src.training.evaluate import evaluate_model
from src.training.train_unified import _build_and_compile


@unittest.skipUnless(os.getenv("CONTEXTNEXT_RUN_TRAINING_TESTS") == "1", "Training test is opt-in")
class TestSharedTrainingPipelineSanity(unittest.TestCase):
    def test_each_architecture_learns_three_controlled_examples(self):
        import tensorflow as tf

        records = [
            {"input": "vegetarian supreme and", "target": "one"},
            {"input": "supreme and one", "target": "double"},
            {"input": "and one double", "target": "pepperoni."},
        ]
        tokens = sorted(
            {
                token
                for record in records
                for text in (record["input"], record["target"])
                for token in text.split()
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text(
                json.dumps({token: index for index, token in enumerate(tokens)}), encoding="utf-8"
            )
            (root / "train.json").write_text(json.dumps(records * 32), encoding="utf-8")
            (root / "eval.json").write_text(json.dumps(records), encoding="utf-8")

            vocabulary = Vocabulary.from_json(root / "vocab.json")
            train_x, train_y = load_examples(root / "train.json", vocabulary, sequence_length=3)
            eval_x, eval_y = load_examples(root / "eval.json", vocabulary, sequence_length=3)
            args = SimpleNamespace(
                embedding_dim=8,
                hidden_units=16,
                dropout=0.0,
                recurrent_dropout=0.0,
                learning_rate=0.02,
                batch_size=16,
                seed=42,
            )

            for model_name in ("rnn", "gru", "lstm"):
                with self.subTest(model=model_name):
                    tf.keras.backend.clear_session()
                    tf.keras.utils.set_random_seed(42)
                    model, _ = _build_and_compile(model_name, vocabulary.size, 3, args)
                    history = model.fit(
                        train_x,
                        train_y,
                        validation_data=(eval_x, eval_y),
                        batch_size=args.batch_size,
                        epochs=40,
                        verbose=0,
                        shuffle=True,
                    )
                    metrics = evaluate_model(model, eval_x, eval_y, batch_size=3)
                    prediction_ids = np.argmax(np.asarray(model(eval_x, training=False)), axis=1)

                    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
                    self.assertEqual(metrics["top_1_accuracy"], 1.0)
                    self.assertEqual(
                        [vocabulary.decode(token_id) for token_id in prediction_ids],
                        [record["target"] for record in records],
                    )


if __name__ == "__main__":
    unittest.main()
