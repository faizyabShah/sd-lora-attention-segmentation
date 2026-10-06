"""Streaming cross-attention aggregation for segmentation ablations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .captured_dataset import CapturedSample


def find_token_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, ...]:
    """Return the first exact occurrence of token IDs in the prompt sequence."""
    if not subsequence:
        raise ValueError("Target token sequence is empty")
    width = len(subsequence)
    for start in range(len(sequence) - width + 1):
        if tuple(sequence[start : start + width]) == tuple(subsequence):
            return tuple(range(start, start + width))
    raise ValueError(f"Token IDs {list(subsequence)} are absent from the prompt token IDs")


def load_step_resolution_sums(
    sample: CapturedSample,
    step_index: int,
    token_positions: Sequence[int],
    resolutions: Iterable[int],
    branch: int = 1,
) -> dict[int, tuple[torch.Tensor, int]]:
    """Load one step and sum head-averaged target-token maps per resolution.

    Only cross-attention probability tensors are read from the large step file.
    Q/K/V and self-attention tensors are never loaded.
    """
    requested = set(resolutions)
    output: dict[int, tuple[torch.Tensor, int]] = {}
    path = sample.attention_path(step_index)
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for resolution in sorted(requested):
            module_sum: torch.Tensor | None = None
            count = 0
            for module_name in sample.modules_by_resolution.get(resolution, ()):
                probability = tensors.get_tensor(f"{module_name}.probabilities")
                # [branch, head, spatial query, text token]
                selected = probability[branch, :, :, list(token_positions)].float()
                attention_map = selected.mean(dim=(0, 2)).reshape(resolution, resolution)
                module_sum = attention_map if module_sum is None else module_sum + attention_map
                count += 1
                del probability, selected, attention_map
            if module_sum is not None:
                output[resolution] = (module_sum, count)
    return output


def resize_map(attention_map: torch.Tensor, size: int) -> torch.Tensor:
    if attention_map.shape == (size, size):
        return attention_map
    return F.interpolate(
        attention_map[None, None], size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]


def combine_resolution_sums(
    resolution_sums: dict[int, torch.Tensor],
    resolution_counts: dict[int, int],
    resolutions: Sequence[int],
    working_size: int = 64,
) -> torch.Tensor:
    """Combine raw module sums while weighting every module equally."""
    combined: torch.Tensor | None = None
    count = 0
    for resolution in resolutions:
        if resolution not in resolution_sums or resolution_counts.get(resolution, 0) == 0:
            continue
        value = resize_map(resolution_sums[resolution], working_size)
        combined = value if combined is None else combined + value
        count += resolution_counts[resolution]
    if combined is None or count == 0:
        raise ValueError(f"No attention maps available for resolutions {tuple(resolutions)}")
    return combined / count


def normalize_map(attention_map: torch.Tensor, method: str = "minmax") -> torch.Tensor:
    if method == "none":
        return attention_map
    if method == "max":
        maximum = attention_map.max()
        return attention_map / maximum if maximum > 0 else torch.zeros_like(attention_map)
    if method == "minmax":
        minimum, maximum = attention_map.min(), attention_map.max()
        scale = maximum - minimum
        return (attention_map - minimum) / scale if scale > 0 else torch.zeros_like(attention_map)
    raise ValueError(f"Unknown normalization method: {method}")


def attention_to_mask(
    attention_map: torch.Tensor,
    output_shape: tuple[int, int],
    threshold: float,
    normalization: str,
) -> torch.Tensor:
    normalized = normalize_map(attention_map, normalization)
    resized = F.interpolate(
        normalized[None, None], size=output_shape, mode="bilinear", align_corners=False
    )[0, 0]
    return resized >= threshold
