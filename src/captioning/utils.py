"""Utility helpers for captioning workflows."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def clean_caption(text: str) -> str:
    """Normalize whitespace in generated captions."""
    return " ".join(text.split())


def caption_statistics(captions: list[str], tokenizer: Callable[[str], list[int]]) -> dict[str, float]:
    """Compute basic caption and token-length statistics."""
    if not captions:
        return {"count": 0.0, "avg_words": 0.0, "avg_tokens": 0.0, "max_tokens": 0.0}

    word_counts = [len(c.split()) for c in captions]
    token_counts = [len(tokenizer(c)) for c in captions]
    return {
        "count": float(len(captions)),
        "avg_words": float(sum(word_counts) / len(word_counts)),
        "avg_tokens": float(sum(token_counts) / len(token_counts)),
        "max_tokens": float(max(token_counts)),
    }


def add_caption_columns(df: pd.DataFrame, caption_column: str = "caption") -> pd.DataFrame:
    """Add helper analysis columns to caption dataframe."""
    result = df.copy()
    result[caption_column] = result[caption_column].fillna("").map(clean_caption)
    result["word_count"] = result[caption_column].map(lambda text: len(text.split()))
    return result
