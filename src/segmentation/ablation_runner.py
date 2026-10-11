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
    load_step_module_maps,
    load_step_resolution_sums,
    normalize_map,
    resize_map,
)
from .captured_dataset import AttentionSample, CapturedAttentionDataset
from .metrics import BinarySegmentationMetrics


@dataclass(frozen=True)
class SegmentationSettings:
    threshold: float = 0.4
    normalization: str = "minmax"
    branch: int = 1
    working_size: int = 64


def target_token_positions(sample: AttentionSample, tokenizer: CLIPTokenizer) -> tuple[int, ...]:
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


def run_time_resolution_ablation(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    windows: dict[str, range],
    resolution_configurations: dict[str, tuple[int, ...]],
    settings: SegmentationSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict]:
    """Evaluate the Cartesian product of time windows and resolution sets.

    Each attention tensor is read once per sample. Only small accumulated maps and
    streaming confusion counts are retained; predictions are never written to disk.
    """
    pairs = {
        f"{window_name}__{resolution_name}": (window_name, resolutions)
        for window_name in windows
        for resolution_name, resolutions in resolution_configurations.items()
    }
    metrics = {name: BinarySegmentationMetrics() for name in pairs}
    needed_resolutions = sorted(
        {resolution for values in resolution_configurations.values() for resolution in values}
    )

    for sample_index, sample in enumerate(
        tqdm(dataset, desc="Joint time-resolution ablation"), start=1
    ):
        positions = target_token_positions(sample, tokenizer)
        sums = {
            window_name: {
                resolution: torch.zeros((resolution, resolution))
                for resolution in needed_resolutions
            }
            for window_name in windows
        }
        counts = {
            window_name: {resolution: 0 for resolution in needed_resolutions}
            for window_name in windows
        }

        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_resolution_sums(
                sample, step_index, positions, needed_resolutions, branch=settings.branch
            )
            selected_windows = [
                name for name, indices in windows.items() if step_index in indices
            ]
            for window_name in selected_windows:
                for resolution, (value, count) in step_maps.items():
                    sums[window_name][resolution] += value
                    counts[window_name][resolution] += count

        target = sample.load_target_mask()
        for configuration_name, (window_name, resolutions) in pairs.items():
            attention_map = combine_resolution_sums(
                sums[window_name],
                counts[window_name],
                resolutions,
                working_size=settings.working_size,
            )
            prediction = attention_to_mask(
                attention_map, target.shape, settings.threshold, settings.normalization
            )
            metrics[configuration_name].update(sample.classname, prediction, target)

        if progress is not None:
            progress("time_resolution", sample_index, len(dataset))
        del sums, counts, target

    gc.collect()
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def run_cross_attention_layer_ablation(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    configurations: dict[str, tuple[str, ...]],
    settings: SegmentationSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict]:
    """Evaluate individual cross-attention modules and module groups."""
    metrics = {name: BinarySegmentationMetrics() for name in configurations}
    needed_modules = sorted({module for values in configurations.values() for module in values})

    for sample_index, sample in enumerate(
        tqdm(dataset, desc="Cross-attention layer ablation"), start=1
    ):
        positions = target_token_positions(sample, tokenizer)
        sums: dict[str, torch.Tensor] = {}
        counts = {module: 0 for module in needed_modules}
        resolutions: dict[str, int] = {}
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_module_maps(
                sample, step_index, positions, needed_modules, branch=settings.branch
            )
            for module_name, (value, resolution) in step_maps.items():
                if module_name not in sums:
                    sums[module_name] = torch.zeros_like(value)
                sums[module_name] += value
                counts[module_name] += 1
                resolutions[module_name] = resolution

        target = sample.load_target_mask()
        for configuration_name, module_names in configurations.items():
            combined: torch.Tensor | None = None
            count = 0
            for module_name in module_names:
                if counts.get(module_name, 0) == 0:
                    continue
                value = resize_map(sums[module_name], settings.working_size)
                combined = value if combined is None else combined + value
                count += counts[module_name]
            if combined is None or count == 0:
                raise ValueError(
                    f"Configuration {configuration_name} selected no maps for {sample.sample_id}"
                )
            prediction = attention_to_mask(
                combined / count, target.shape, settings.threshold, settings.normalization
            )
            metrics[configuration_name].update(sample.classname, prediction, target)

        if progress is not None:
            progress("cross_attention_layer", sample_index, len(dataset))
        del sums, counts, resolutions, target

    gc.collect()
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def run_threshold_normalization_ablation(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    layer_configurations: dict[str, tuple[str, ...]],
    thresholds: Sequence[float],
    normalizations: Sequence[str],
    settings: SegmentationSettings,
) -> dict[str, dict]:
    """Evaluate postprocessing settings for fixed cross-attention layer sets."""
    keys = {
        (layer_name, normalization, threshold): (
            f"{layer_name}__{normalization}__t{threshold:.2f}"
        )
        for layer_name in layer_configurations
        for normalization in normalizations
        for threshold in thresholds
    }
    metrics = {key: BinarySegmentationMetrics() for key in keys.values()}
    needed_modules = sorted(
        {module for values in layer_configurations.values() for module in values}
    )

    for sample in tqdm(dataset, desc="Threshold-normalization ablation"):
        positions = target_token_positions(sample, tokenizer)
        sums: dict[str, torch.Tensor] = {}
        counts = {module: 0 for module in needed_modules}
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_module_maps(
                sample, step_index, positions, needed_modules, branch=settings.branch
            )
            for module_name, (value, _) in step_maps.items():
                resized = resize_map(value, settings.working_size)
                if module_name not in sums:
                    sums[module_name] = torch.zeros_like(resized)
                layer_sums = sums[module_name]
                layer_sums += resized
                counts[module_name] += 1

        target = sample.load_target_mask()
        for layer_name, module_names in layer_configurations.items():
            combined = torch.stack([sums[name] for name in module_names]).sum(dim=0)
            combined /= sum(counts[name] for name in module_names)
            for normalization in normalizations:
                if normalization == "ovam_max_after_resize":
                    resized = torch.nn.functional.interpolate(
                        combined[None, None], size=target.shape, mode="bilinear", align_corners=False
                    )[0, 0]
                    maximum = resized.max()
                    resized = resized / maximum if maximum > 0 else torch.zeros_like(resized)
                else:
                    normalized = normalize_map(combined, normalization)
                    resized = torch.nn.functional.interpolate(
                        normalized[None, None],
                        size=target.shape,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                for threshold in thresholds:
                    key = keys[(layer_name, normalization, threshold)]
                    comparison = resized > threshold if normalization == "ovam_max_after_resize" else resized >= threshold
                    metrics[key].update(sample.classname, comparison, target)

        del sums, counts, target

    gc.collect()
    return {name: accumulator.result() for name, accumulator in metrics.items()}
