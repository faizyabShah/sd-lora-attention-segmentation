"""Dataset filtering utilities."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def add_token_length_column(df: pd.DataFrame, tokenizer: Callable[[str], int]) -> pd.DataFrame:
    """Add CLIP token lengths for each caption."""
    result = df.copy()
    result["token_length"] = result["caption"].fillna("").map(tokenizer)
    return result


def filter_by_token_length(
    df: pd.DataFrame,
    tokenizer: Callable[[str], int],
    max_tokens: int = 77,
) -> pd.DataFrame:
    """Keep rows where caption token length is <= max_tokens."""
    with_lengths = add_token_length_column(df, tokenizer)
    return with_lengths[with_lengths["token_length"] <= max_tokens].reset_index(drop=True)


def filter_dataset(records: pd.DataFrame) -> pd.DataFrame:
    """Default filter pass-through to keep backward compatibility."""
    return records
