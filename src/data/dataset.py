"""PyTorch-style Dataset wrapper around the project's pre-split JSON files.

Each JSON file is a list of ``{"input": "<context words>", "target": "<word>"}``
records produced by ``src/data/preprocess.py``.

This module exposes a lightweight ``NextWordDataset`` so training loops can
iterate over (context, target) pairs without loading everything into memory at
once. It complements ``src/training/data.py`` which provides the NumPy-based
batch-loading used by the Keras training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.training.data import PAD_ID, Vocabulary


class NextWordDataset:
    """Iterable dataset of (context_ids, target_id) integer tuples.

    Parameters
    ----------
    path:
        Path to a processed JSON split (train / validation / test).
    vocabulary:
        Shared project vocabulary; unseen tokens map to ``OOV_ID``.
    sequence_length:
        Fixed context window size. Shorter contexts are left-padded with
        ``PAD_ID``; longer contexts are right-truncated to the most recent
        ``sequence_length`` words.
    limit:
        If provided, yield only the first *limit* records.
    """

    def __init__(
        self,
        path: Path,
        vocabulary: Vocabulary,
        sequence_length: int,
        limit: int | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        self._path = Path(path)
        self._vocabulary = vocabulary
        self._sequence_length = sequence_length
        self._limit = limit
        self._records: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if self._records is None:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {self._path}.")
            self._records = data[: self._limit] if self._limit is not None else data
        return self._records

    def _encode(self, record: dict) -> tuple[list[int], int]:
        tokens = record["input"].split()[-self._sequence_length:]
        encoded = [self._vocabulary.encode(t) for t in tokens]
        # Left-pad with PAD_ID
        padded = [PAD_ID] * (self._sequence_length - len(encoded)) + encoded
        target = self._vocabulary.encode(record["target"])
        return padded, target

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[tuple[list[int], int]]:
        for record in self._load():
            yield self._encode(record)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self._encode(self._load()[index])
