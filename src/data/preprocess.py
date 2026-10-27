"""Create dialogue-disjoint next-word examples for a chosen context length."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.tokenizer import tokenize


def create_samples(dialogues: list[dict], context_size: int) -> list[dict[str, str]]:
    if context_size < 1:
        raise ValueError("context_size must be positive")
    samples = []
    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            words = tokenize(turn["utterance"])
            for index in range(context_size, len(words)):
                samples.append({
                    "input": " ".join(words[index - context_size:index]),
                    "target": words[index],
                })
    return samples


def prepare_data(source: Path, output_dir: Path, context_size: int = 3, seed: int = 42) -> dict[str, int]:
    if context_size < 1:
        raise ValueError("context_size must be positive")
    with Path(source).open(encoding="utf-8") as handle:
        dialogues = json.load(handle)
    if not isinstance(dialogues, list):
        raise ValueError("Source dialogues must be a JSON list")
    shuffled = list(dialogues)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    validation_end = int(len(shuffled) * 0.9)
    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in splits.items():
        samples = create_samples(split, context_size)
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False)
        counts[name] = len(samples)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"context_size": context_size, "seed": seed, "dialogue_counts": {name: len(split) for name, split in splits.items()}}, handle, indent=2)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DailyDialog next-word examples")
    parser.add_argument("--source", type=Path, default=Path("data/raw/dailydialog/data/dialogues.json"))
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.context_size < 1:
        parser.error("--context-size must be positive")
    output_dir = args.output_dir or Path(f"data/processed_ctx{args.context_size}" if args.context_size != 3 else "data/processed")
    print(prepare_data(args.source, output_dir, args.context_size, args.seed))


if __name__ == "__main__":
    main()
