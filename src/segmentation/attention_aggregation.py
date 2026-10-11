"""Streaming cross-attention aggregation for segmentation ablations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import torch
import torch.nn.functional as F
from .captured_dataset import AttentionSample


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
    sample: AttentionSample,
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
    with sample.open_attention(step_index) as tensors:
        for resolution in sorted(requested):
            module_sum: torch.Tensor | None = None
            count = 0
            for module_name in sample.modules_by_resolution.get(resolution, ()):
                compact_key = f"{module_name}.{sample.cross_map_key}"
                if compact_key in tensors.keys():
                    selected = tensors.get_tensor(compact_key).float()
                    attention_map = selected.mean(dim=0).reshape(resolution, resolution)
                    probability = None
                else:
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


def load_step_module_maps(
    sample: AttentionSample,
    step_index: int,
    token_positions: Sequence[int],
    module_names: Iterable[str],
    branch: int = 1,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Load target-token cross-attention maps for selected individual modules."""
    requested = set(module_names)
    module_resolutions = {
        name: resolution
        for resolution, names in sample.modules_by_resolution.items()
        for name in names
        if name in requested
    }
    output: dict[str, tuple[torch.Tensor, int]] = {}
    with sample.open_attention(step_index) as tensors:
        for module_name in sorted(module_resolutions):
            resolution = module_resolutions[module_name]
            compact_key = f"{module_name}.{sample.cross_map_key}"
            if compact_key in tensors.keys():
                selected = tensors.get_tensor(compact_key).float()
                probability = None
                attention_map = selected.mean(dim=0).reshape(resolution, resolution)
            else:
                probability = tensors.get_tensor(f"{module_name}.probabilities")
                selected = probability[branch, :, :, list(token_positions)].float()
                attention_map = selected.mean(dim=(0, 2)).reshape(resolution, resolution)
            output[module_name] = (attention_map, resolution)
            del probability, selected
    return output


def load_step_module_head_maps(
    sample: AttentionSample,
    step_index: int,
    token_positions: Sequence[int],
    module_names: Iterable[str],
    branch: int = 1,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Load per-head target-token cross-attention maps for selected modules."""
    requested = set(module_names)
    module_resolutions = {
        name: resolution
        for resolution, names in sample.modules_by_resolution.items()
        for name in names
        if name in requested
    }
    output: dict[str, tuple[torch.Tensor, int]] = {}
    with sample.open_attention(step_index) as tensors:
        for module_name in sorted(module_resolutions):
            resolution = module_resolutions[module_name]
            compact_key = f"{module_name}.{sample.cross_map_key}"
            if compact_key in tensors.keys():
                selected = tensors.get_tensor(compact_key).float()
                probability = None
                maps = selected.reshape(selected.shape[0], resolution, resolution)
            else:
                probability = tensors.get_tensor(f"{module_name}.probabilities")
                selected = probability[branch, :, :, list(token_positions)].float()
                maps = selected.mean(dim=-1).reshape(selected.shape[0], resolution, resolution)
            output[module_name] = (maps, resolution)
            del probability, selected
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
    if method == "percentile_01_99":
        lower, upper = torch.quantile(
            attention_map.float(),
            torch.tensor([0.01, 0.99], device=attention_map.device),
        )
        scale = upper - lower
        if scale <= 0:
            return torch.zeros_like(attention_map)
        return ((attention_map - lower) / scale).clamp_(0, 1)
    raise ValueError(f"Unknown normalization method: {method}")


def attention_to_mask(
    attention_map: torch.Tensor,
    output_shape: tuple[int, int],
    threshold: float,
    normalization: str,
) -> torch.Tensor:
    if normalization == "ovam_max_after_resize":
        resized = F.interpolate(
            attention_map[None, None], size=output_shape, mode="bilinear", align_corners=False
        )[0, 0]
        maximum = resized.max()
        normalized = resized / maximum if maximum > 0 else torch.zeros_like(resized)
        return normalized > threshold
    normalized = normalize_map(attention_map, normalization)
    resized = F.interpolate(
        normalized[None, None], size=output_shape, mode="bilinear", align_corners=False
    )[0, 0]
    return resized >= threshold
