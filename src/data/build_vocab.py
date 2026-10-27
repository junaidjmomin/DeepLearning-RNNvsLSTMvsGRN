"""Build a deterministic vocabulary from the training split only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_vocabulary(train_path: Path, output_path: Path) -> int:
    with Path(train_path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Training split must be a JSON list")
    words = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("input"), str) or not isinstance(record.get("target"), str):
            raise ValueError("Each training record needs string input and target")
        words.update(record["input"].split())
        words.update(record["target"].split())
    vocabulary = {word: index for index, word in enumerate(sorted(words))}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, ensure_ascii=False, indent=2)
    return len(vocabulary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from a training split")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/vocab.json"))
    args = parser.parse_args()
    print(f"Vocabulary size: {build_vocabulary(args.train_data, args.output)}")


if __name__ == "__main__":
    main()
