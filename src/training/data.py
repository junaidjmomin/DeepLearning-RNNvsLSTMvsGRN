"""Data loading and deterministic token encoding for Person 2 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PAD_TOKEN = "<pad>"
OOV_TOKEN = "<unk>"
PAD_ID = 0
OOV_ID = 1


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        """Load the existing training vocabulary while reserving padding/OOV IDs."""

        with path.open("r", encoding="utf-8") as handle:
            raw_vocab = json.load(handle)
        if not isinstance(raw_vocab, dict) or not all(isinstance(token, str) for token in raw_vocab):
            raise ValueError(f"Vocabulary at {path} must be a JSON object of token IDs.")

        # CHOICE: reserve ID 0 for padding and ID 1 for a word we have not seen.
        # The project vocabulary starts at zero, so real words are shifted by two.
        tokens = sorted(raw_vocab, key=lambda token: (raw_vocab[token], token))
        token_to_id = {PAD_TOKEN: PAD_ID, OOV_TOKEN: OOV_ID}
        token_to_id.update({token: index + 2 for index, token in enumerate(tokens)})
        return cls(token_to_id=token_to_id)

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, OOV_ID)

    @property
    def id_to_token(self) -> dict[int, str]:
        """Return the exact inverse of the IDs used by training."""
        return {token_id: token for token, token_id in self.token_to_id.items()}

    def decode(self, token_id: int) -> str:
        """Decode a model output ID, including the reserved special IDs."""
        return self.id_to_token.get(int(token_id), OOV_TOKEN)


def load_examples(
    path: Path,
    vocabulary: Vocabulary,
    sequence_length: int,
    limit: int | None = None,
    require_exact_context: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded context IDs and next-token IDs from one processed split."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Dataset at {path} must be a JSON list.")

    if limit is not None:
        records = records[:limit]
    contexts = np.full((len(records), sequence_length), PAD_ID, dtype=np.int32)
    targets = np.empty(len(records), dtype=np.int32)

    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("input"), str) or not isinstance(record.get("target"), str):
            raise ValueError(f"Invalid record at index {index} in {path}; expected string input and target.")
        # CHOICE: when a context is too long, use its most recent words because
        # they are usually the most useful for next-word prediction.
        tokens = record["input"].split()[-sequence_length:]
        if require_exact_context and len(record["input"].split()) != sequence_length:
            raise ValueError(
                f"Record {index} in {path} has {len(record['input'].split())} context words; "
                f"expected {sequence_length}. Regenerate that split with "
                f"python -m src.data.preprocess --context-size {sequence_length}."
            )
        encoded = [vocabulary.encode(token) for token in tokens]
        contexts[index, sequence_length - len(encoded) :] = encoded
        targets[index] = vocabulary.encode(record["target"])

    return contexts, targets


def ensure_non_empty(*datasets: Iterable[object]) -> None:
    """Fail early with a helpful message when a selected split contains no samples."""

    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("At least one selected dataset split is empty.")
