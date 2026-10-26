"""Tokenizer for next-word prediction — minimal and stopword-preserving.

Tokenization rules:
  1. Lowercase the input text.
  2. Strip leading/trailing whitespace.
  3. Collapse multiple spaces into one.
  4. Split on whitespace — words are already separated in DailyDialog.

Stopwords are intentionally kept because removing them would destroy the
natural word-sequence structure needed for next-word prediction
(e.g. "I am going to the market" → "going market" breaks the sequence).
"""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Return a list of lowercase tokens from *text*.

    Parameters
    ----------
    text:
        Raw utterance string.

    Returns
    -------
    list[str]
        Tokens in order. Empty list for blank input.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}.")
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    return cleaned.split() if cleaned else []
