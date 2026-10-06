"""Research-grade streaming runners for attention segmentation ablations."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from .attention_aggregation import (
    attention_to_mask,
    combine_resolution_sums,
    find_token_subsequence,
    load_step_resolution_sums,
    resize_map,
)
from .captured_dataset import CapturedAttentionDataset, CapturedSample
from .metrics import BinarySegmentationMetrics


@dataclass(frozen=True)
class SegmentationSettings:
    threshold: float = 0.4
    normalization: str = "minmax"
    branch: int = 1
    working_size: int = 64


def target_token_positions(sample: CapturedSample, tokenizer: CLIPTokenizer) -> tuple[int, ...]:
    target_ids = tokenizer(sample.classname, add_special_tokens=False)["input_ids"]
    return find_token_subsequence(sample.token_ids, target_ids)


def run_time_window_ablation(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    windows: dict[str, range],
    resolutions: Sequence[int],
    settings: SegmentationSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict]:
    """Evaluate all timestep windows in one streaming pass over every sample."""
    metrics = {name: BinarySegmentationMetrics() for name in windows}
    for sample_index, sample in enumerate(tqdm(dataset, desc="Time-window ablation"), start=1):
        positions = target_token_positions(sample, tokenizer)
        sums = {name: torch.zeros((settings.working_size, settings.working_size)) for name in windows}
        counts = {name: 0 for name in windows}
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_resolution_sums(
                sample, step_index, positions, resolutions, branch=settings.branch
            )
            step_sum: torch.Tensor | None = None
            step_count = 0
            for resolution, (value, count) in step_maps.items():
                resized = resize_map(value, settings.working_size)
                step_sum = resized if step_sum is None else step_sum + resized
                step_count += count
            if step_sum is None:
                continue
            for name, indices in windows.items():
                if step_index in indices:
                    sums[name] += step_sum
                    counts[name] += step_count
        target = sample.load_target_mask()
        for name in windows:
            if counts[name] == 0:
                raise ValueError(f"Window {name} selected no attention for {sample.sample_id}")
            prediction = attention_to_mask(
                sums[name] / counts[name], target.shape, settings.threshold, settings.normalization
            )
            metrics[name].update(sample.classname, prediction, target)
        if progress is not None:
            progress("time_window", sample_index, len(dataset))
        del sums, target
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def run_resolution_ablation(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    configurations: dict[str, tuple[int, ...]],
    settings: SegmentationSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict]:
    """Evaluate resolution combinations in one streaming pass over every sample."""
    metrics = {name: BinarySegmentationMetrics() for name in configurations}
    needed_resolutions = sorted({r for values in configurations.values() for r in values})
    for sample_index, sample in enumerate(tqdm(dataset, desc="Resolution ablation"), start=1):
        positions = target_token_positions(sample, tokenizer)
        sums = {resolution: torch.zeros((resolution, resolution)) for resolution in needed_resolutions}
        counts = {resolution: 0 for resolution in needed_resolutions}
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_resolution_sums(
                sample, step_index, positions, needed_resolutions, branch=settings.branch
            )
            for resolution, (value, count) in step_maps.items():
                sums[resolution] += value
                counts[resolution] += count
        target = sample.load_target_mask()
        for name, resolutions in configurations.items():
            attention_map = combine_resolution_sums(
                sums, counts, resolutions, working_size=settings.working_size
            )
            prediction = attention_to_mask(
                attention_map, target.shape, settings.threshold, settings.normalization
            )
            metrics[name].update(sample.classname, prediction, target)
        if progress is not None:
            progress("resolution", sample_index, len(dataset))
        del sums, target
    gc.collect()
    return {name: accumulator.result() for name, accumulator in metrics.items()}
