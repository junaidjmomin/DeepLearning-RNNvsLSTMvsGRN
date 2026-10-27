"""Context generation and training data contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_vocab import build_vocabulary
from src.data.preprocess import create_samples, prepare_data
from src.training.data import Vocabulary, load_examples


class TestDataPipeline(unittest.TestCase):
    def test_long_context_uses_actual_preceding_words(self):
        dialogues = [{"turns": [{"utterance": "A B C D E F G"}]}]
        self.assertEqual(create_samples(dialogues, 5), [
            {"input": "a b c d e", "target": "f"},
            {"input": "b c d e f", "target": "g"},
        ])

    def test_preparation_does_not_modify_input_and_training_rejects_wrong_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialogues = [{"turns": [{"utterance": f"one two three four five six {index}"}]} for index in range(10)]
            source = root / "dialogues.json"
            source.write_text(json.dumps(dialogues), encoding="utf-8")
            counts = prepare_data(source, root / "processed", context_size=5)
            self.assertEqual(sum(counts.values()), 20)
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), dialogues)
            train_path = root / "processed" / "train.json"
            build_vocabulary(train_path, root / "vocab.json")
            vocab = Vocabulary.from_json(root / "vocab.json")
            contexts, targets = load_examples(train_path, vocab, sequence_length=5, require_exact_context=True)
            self.assertEqual(contexts.shape[1], 5)
            self.assertEqual(len(contexts), len(targets))
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                load_examples(train_path, vocab, sequence_length=3, require_exact_context=True)


if __name__ == "__main__":
    unittest.main()
