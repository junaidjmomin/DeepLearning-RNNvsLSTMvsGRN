"""Lookup helpers for labelled examples in the project's processed splits."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.tokenizer import tokenize


SPLITS = ("train", "validation", "test")


class DatasetExampleStore:
    """Lazily index processed examples so the UI can select real labels.

    IDs use ``<split>:<zero-based-row>`` and refer directly to the source JSON
    row. Repeated contexts are kept as separate records because their targets
    can differ across conversations or splits.
    """

    def __init__(self, data_dir: Path = Path("data/processed")) -> None:
        self.data_dir = Path(data_dir)
        self._rows: dict[str, list[tuple[str, str]]] | None = None
        self._by_input: dict[str, list[tuple[str, int, str]]] | None = None

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(tokenize(text))

    def _load(self) -> None:
        if self._rows is not None:
            return

        rows: dict[str, list[tuple[str, str]]] = {}
        by_input: dict[str, list[tuple[str, int, str]]] = {}
        for split in SPLITS:
            path = self.data_dir / f"{split}.json"
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON list in {path}.")

            split_rows: list[tuple[str, str]] = []
            for index, record in enumerate(records):
                if not isinstance(record, dict) or not isinstance(record.get("input"), str) or not isinstance(record.get("target"), str):
                    raise ValueError(f"Invalid labelled example at row {index} in {path}.")
                context = self.normalize(record["input"])
                target = self.normalize(record["target"])
                split_rows.append((context, target))
                by_input.setdefault(context, []).append((split, index, target))
            rows[split] = split_rows

        self._rows = rows
        self._by_input = by_input

    @staticmethod
    def _record(split: str, index: int, context: str, target: str) -> dict:
        return {
            "id": f"{split}:{index}",
            "split": split,
            "index": index,
            "input": context,
            "target": target,
        }

    def get(self, example_id: str) -> dict | None:
        self._load()
        try:
            split, raw_index = example_id.rsplit(":", 1)
            index = int(raw_index)
        except (AttributeError, ValueError):
            return None
        if split not in self._rows or index < 0 or index >= len(self._rows[split]):
            return None
        context, target = self._rows[split][index]
        return self._record(split, index, context, target)

    def find(self, context: str) -> list[dict]:
        """Return every labelled record with this normalized context."""
        self._load()
        normalized = self.normalize(context)
        return [
            self._record(split, index, normalized, target)
            for split, index, target in self._by_input.get(normalized, [])
        ]

    def list_examples(
        self,
        split: str = "train",
        query: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return a bounded page of examples, optionally searching contexts."""
        self._load()
        if split != "all" and split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join((*SPLITS, 'all'))}.")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1-100 and offset cannot be negative.")

        query = self.normalize(query)
        splits = SPLITS if split == "all" else (split,)
        page: list[dict] = []
        total = 0

        if not query:
            total = sum(len(self._rows[row_split]) for row_split in splits)
            remaining = offset
            for row_split in splits:
                split_rows = self._rows[row_split]
                if remaining >= len(split_rows):
                    remaining -= len(split_rows)
                    continue
                start = remaining
                remaining = 0
                for index in range(start, min(len(split_rows), start + limit - len(page))):
                    context, target = split_rows[index]
                    page.append(self._record(row_split, index, context, target))
                if len(page) == limit:
                    break
        else:
            exact_matches = self._by_input.get(query)
            if exact_matches is not None:
                matches = (
                    (row_split, index, query, target)
                    for row_split, index, target in exact_matches
                    if split == "all" or row_split == split
                )
            else:
                matches = (
                    (row_split, index, context, target)
                    for row_split in splits
                    for index, (context, target) in enumerate(self._rows[row_split])
                    if query in context
                )

            for row_split, index, context, target in matches:
                if offset <= total < offset + limit:
                    page.append(self._record(row_split, index, context, target))
                total += 1

        return {
            "examples": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total,
        }
