"""Exact QK self-attention propagation for offline segmentation refinement."""

from __future__ import annotations

import gc
import itertools
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer

from .ablation_runner import target_token_positions
from .attention_aggregation import load_step_module_maps, normalize_map, resize_map
from .captured_dataset import AttentionSample, CapturedAttentionDataset
from .metrics import BinarySegmentationMetrics
from .stage1_suite import BEST_12_INDICES


SA_WINDOWS = {
    "all": tuple(range(31)),
    "steps_00_04": tuple(range(0, 5)),
    "steps_05_09": tuple(range(5, 10)),
    "steps_10_14": tuple(range(10, 15)),
    "steps_15_20": tuple(range(15, 21)),
    "steps_21_25": tuple(range(21, 26)),
    "steps_26_30": tuple(range(26, 31)),
}
SEED_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
ITERATIONS = (1, 2, 4)
ALPHAS = (0.10, 0.25, 0.50, 0.75)


def _module_order(sample: AttentionSample) -> tuple[str, ...]:
    return tuple(sorted(name for values in sample.modules_by_resolution.values() for name in values))


def _base_cross_attention(
    sample: AttentionSample, tokenizer: CLIPTokenizer, working_size: int = 64
) -> torch.Tensor:
    modules = _module_order(sample)
    selected = tuple(modules[index] for index in BEST_12_INDICES)
    positions = target_token_positions(sample, tokenizer)
    total = torch.zeros((working_size, working_size))
    count = 0
    for step_index in range(len(sample.timesteps)):
        maps = load_step_module_maps(sample, step_index, positions, selected, branch=1)
        for value, _ in maps.values():
            total += resize_map(value, working_size)
            count += 1
    return total / count


@torch.inference_mode()
def _propagate_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    seeds: torch.Tensor,
    iterations: tuple[int, ...],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Average exact per-head softmax(QK^T) propagation results."""
    maximum_iteration = max(iterations)
    output = {
        iteration: torch.zeros_like(seeds, device=device, dtype=torch.float32)
        for iteration in iterations
    }
    head_count = query.shape[0]
    scale = 1.0 / math.sqrt(query.shape[-1])
    seeds_device = seeds.to(device=device, dtype=torch.float32)
    for head in range(head_count):
        q = query[head].to(device=device, dtype=torch.float32)
        k = key[head].to(device=device, dtype=torch.float32)
        affinity = torch.softmax(torch.matmul(q, k.transpose(0, 1)) * scale, dim=-1)
        current = seeds_device
        for iteration in range(1, maximum_iteration + 1):
            current = torch.matmul(affinity, current)
            if iteration in output:
                output[iteration] += current
        del q, k, affinity, current
    return {iteration: value.div_(head_count).cpu() for iteration, value in output.items()}


def _category_names(
    module_name: str,
    step_index: int,
    selected_self_modules: set[str],
) -> tuple[str, ...]:
    region = "down" if module_name.startswith("down_") else "mid" if module_name.startswith("mid_") else "up"
    categories = ["all_modules_all_steps", f"module__{module_name}", f"region__{region}"]
    if module_name in selected_self_modules:
        categories.append("selected_ca_corresponding")
    categories.extend(
        f"window__{name}" for name, indices in SA_WINDOWS.items() if step_index in indices
    )
    return tuple(categories)


def _metric_update(
    metrics: dict[str, BinarySegmentationMetrics],
    key: str,
    classname: str,
    score: torch.Tensor,
    target,
) -> None:
    normalized = normalize_map(score, "percentile_01_99")
    resized = F.interpolate(
        normalized[None, None], size=target.shape, mode="bilinear", align_corners=False
    )[0, 0]
    metrics.setdefault(key, BinarySegmentationMetrics()).update(classname, resized >= 0.45, target)


def run_self_attention_screen(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    resolution: int,
    device: str = "cuda",
) -> tuple[dict[str, dict], dict]:
    """Screen SA modules, regions, timesteps, strengths, seeds, and iterations."""
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    first = dataset.samples[0]
    self_modules = first.self_modules_by_resolution.get(resolution, ())
    if not self_modules:
        raise ValueError(f"No self-attention modules at resolution {resolution}")
    cross_modules = _module_order(first)
    selected_cross = {cross_modules[index] for index in BEST_12_INDICES}
    selected_self = {name.replace(".attn2", ".attn1") for name in selected_cross}
    metrics: dict[str, BinarySegmentationMetrics] = {}

    for sample in tqdm(dataset, desc=f"SA refinement {resolution}x{resolution}"):
        base = _base_cross_attention(sample, tokenizer)
        base_normalized = normalize_map(base, "percentile_01_99")
        native_base = resize_map(base_normalized, resolution).flatten()
        seed_columns = [native_base]
        seed_names = ["continuous"]
        for threshold in SEED_THRESHOLDS:
            seed_columns.extend(
                [(native_base >= threshold).float(), (native_base <= 1.0 - threshold).float()]
            )
            seed_names.extend([f"foreground_t{threshold:.2f}", f"background_t{threshold:.2f}"])
        seeds = torch.stack(seed_columns, dim=1)

        sums: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        counts: dict[str, int] = defaultdict(int)
        for step_index in range(len(sample.timesteps)):
            with sample.open_attention(step_index) as tensors:
                for module_name in self_modules:
                    query_tensor = tensors.get_tensor(f"{module_name}.q")
                    key_tensor = tensors.get_tensor(f"{module_name}.k")
                    branch = 1 if query_tensor.shape[0] > 1 else 0
                    query = query_tensor[branch]
                    key = key_tensor[branch]
                    propagated = _propagate_qk(query, key, seeds, ITERATIONS, compute_device)
                    for category in _category_names(module_name, step_index, selected_self):
                        counts[category] += 1
                        for iteration, value in propagated.items():
                            if iteration not in sums[category]:
                                sums[category][iteration] = torch.zeros_like(value)
                            sums[category][iteration] += value
                    del query, key, propagated

        target = sample.load_target_mask()
        base_64 = base_normalized
        # Every within-resolution SA module subset is derived from the already
        # accumulated module outputs, so layer selection adds no QK computation.
        if len(self_modules) > 1:
            for width in range(1, len(self_modules) + 1):
                for subset_indices in itertools.combinations(range(len(self_modules)), width):
                    category = "subset__" + "_".join(str(index) for index in subset_indices)
                    member_categories = [f"module__{self_modules[index]}" for index in subset_indices]
                    counts[category] = sum(counts[name] for name in member_categories)
                    for iteration in ITERATIONS:
                        sums[category][iteration] = sum(
                            (sums[name][iteration] for name in member_categories),
                            start=torch.zeros_like(sums[member_categories[0]][iteration]),
                        )
        for category, iteration_values in sums.items():
            for iteration, value in iteration_values.items():
                propagated = value / counts[category]
                refined = resize_map(
                    propagated[:, 0].reshape(resolution, resolution), 64
                )
                prefix = f"{category}__iter{iteration}__continuous"
                _metric_update(metrics, f"{prefix}__non_residual", sample.classname, refined, target)
                for alpha in ALPHAS:
                    mixed = (1.0 - alpha) * base_64 + alpha * refined
                    _metric_update(
                        metrics,
                        f"{prefix}__residual_a{alpha:.2f}",
                        sample.classname,
                        mixed,
                        target,
                    )

                # Seed/background variants are screened with the fully aggregated
                # SA operator. Module, region, and time comparisons use the same
                # continuous seed so their effect remains isolated.
                if category == "all_modules_all_steps":
                    for threshold_index, threshold in enumerate(SEED_THRESHOLDS):
                        foreground_index = 1 + threshold_index * 2
                        background_index = foreground_index + 1
                        foreground = resize_map(
                            propagated[:, foreground_index].reshape(resolution, resolution), 64
                        )
                        background = resize_map(
                            propagated[:, background_index].reshape(resolution, resolution), 64
                        )
                        seed_prefix = f"{category}__iter{iteration}__seed_t{threshold:.2f}"
                        _metric_update(
                            metrics,
                            f"{seed_prefix}__foreground_non_residual",
                            sample.classname,
                            foreground,
                            target,
                        )
                        _metric_update(
                            metrics,
                            f"{seed_prefix}__foreground_residual_a0.50",
                            sample.classname,
                            0.5 * base_64 + 0.5 * foreground,
                            target,
                        )
                        difference = foreground - background
                        ratio = foreground / (foreground + background).clamp_min(1e-8)
                        _metric_update(metrics, f"{seed_prefix}__background_difference", sample.classname, difference, target)
                        _metric_update(metrics, f"{seed_prefix}__background_ratio", sample.classname, ratio, target)

        _metric_update(metrics, "cross_attention_baseline", sample.classname, base_64, target)
        del sums, counts, seeds, base, base_normalized, target
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()
    results = {name: metric.result() for name, metric in metrics.items()}
    metadata = {
        "resolution": resolution,
        "self_modules": self_modules,
        "iterations": ITERATIONS,
        "alphas": ALPHAS,
        "seed_thresholds": SEED_THRESHOLDS,
        "time_windows": SA_WINDOWS,
        "propagation": "exact per-head softmax(QK^T/sqrt(d)); average propagated maps",
        "background_methods": ("foreground_only", "foreground_minus_background", "foreground_ratio"),
    }
    return results, metadata


def run_progressive_self_attention(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    sequence: Sequence[int],
    device: str = "cuda",
) -> tuple[dict[str, dict], dict]:
    """Apply exact SA refinement progressively through a resolution sequence."""
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sequence = tuple(sequence)
    if len(sequence) < 2 or any(value not in (8, 16, 32, 64) for value in sequence):
        raise ValueError(f"Invalid progressive resolution sequence: {sequence}")
    modes: tuple[tuple[str, float | None], ...] = (
        ("non_residual", None),
        ("residual_a0.25", 0.25),
        ("residual_a0.50", 0.50),
        ("residual_a0.75", 0.75),
    )
    metrics: dict[str, BinarySegmentationMetrics] = {}

    for sample in tqdm(dataset, desc=f"Progressive SA {'-'.join(map(str, sequence))}"):
        base = normalize_map(_base_cross_attention(sample, tokenizer), "percentile_01_99")
        initial_native = resize_map(base, sequence[0]).flatten()
        seed_vectors = [initial_native]
        seed_names = ["continuous"]
        for threshold in SEED_THRESHOLDS:
            seed_vectors.extend(
                [(initial_native >= threshold).float(), (initial_native <= 1.0 - threshold).float()]
            )
            seed_names.extend([f"foreground_t{threshold:.2f}", f"background_t{threshold:.2f}"])

        state_keys = [
            (iteration, mode_name, alpha, seed_index)
            for iteration in ITERATIONS
            for mode_name, alpha in modes
            for seed_index in range(len(seed_vectors))
        ]
        state = torch.stack(
            [seed_vectors[seed_index] for _, _, _, seed_index in state_keys], dim=1
        )
        current_resolution = sequence[0]

        for resolution in sequence:
            if resolution != current_resolution:
                resized_columns = [
                    resize_map(
                        state[:, column].reshape(current_resolution, current_resolution), resolution
                    ).flatten()
                    for column in range(state.shape[1])
                ]
                state = torch.stack(resized_columns, dim=1)
                current_resolution = resolution

            modules = sample.self_modules_by_resolution[resolution]
            aggregate = {
                iteration: torch.zeros((resolution * resolution, state.shape[1]))
                for iteration in ITERATIONS
            }
            count = 0
            for step_index in range(len(sample.timesteps)):
                with sample.open_attention(step_index) as tensors:
                    for module_name in modules:
                        query_tensor = tensors.get_tensor(f"{module_name}.q")
                        key_tensor = tensors.get_tensor(f"{module_name}.k")
                        branch = 1 if query_tensor.shape[0] > 1 else 0
                        query = query_tensor[branch]
                        key = key_tensor[branch]
                        propagated = _propagate_qk(query, key, state, ITERATIONS, compute_device)
                        for iteration in ITERATIONS:
                            aggregate[iteration] += propagated[iteration]
                        count += 1
                        del query, key, propagated
            for iteration in ITERATIONS:
                aggregate[iteration] /= count
            next_columns = []
            for column, (iteration, _, alpha, _) in enumerate(state_keys):
                propagated_column = aggregate[iteration][:, column]
                if alpha is None:
                    next_columns.append(propagated_column)
                else:
                    next_columns.append((1.0 - alpha) * state[:, column] + alpha * propagated_column)
            state = torch.stack(next_columns, dim=1)

        target = sample.load_target_mask()
        lookup = {key: column for column, key in enumerate(state_keys)}
        for iteration in ITERATIONS:
            for mode_name, alpha in modes:
                continuous = state[:, lookup[(iteration, mode_name, alpha, 0)]].reshape(
                    current_resolution, current_resolution
                )
                prefix = f"iter{iteration}__{mode_name}"
                _metric_update(metrics, f"{prefix}__continuous", sample.classname, continuous, target)
                for threshold_index, threshold in enumerate(SEED_THRESHOLDS):
                    foreground_index = 1 + threshold_index * 2
                    background_index = foreground_index + 1
                    foreground = state[
                        :, lookup[(iteration, mode_name, alpha, foreground_index)]
                    ].reshape(current_resolution, current_resolution)
                    background = state[
                        :, lookup[(iteration, mode_name, alpha, background_index)]
                    ].reshape(current_resolution, current_resolution)
                    seed_prefix = f"{prefix}__seed_t{threshold:.2f}"
                    _metric_update(metrics, f"{seed_prefix}__foreground", sample.classname, foreground, target)
                    _metric_update(metrics, f"{seed_prefix}__background_difference", sample.classname, foreground - background, target)
                    _metric_update(
                        metrics,
                        f"{seed_prefix}__background_ratio",
                        sample.classname,
                        foreground / (foreground + background).clamp_min(1e-8),
                        target,
                    )
        _metric_update(metrics, "cross_attention_baseline", sample.classname, base, target)
        del state, base, target
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()
    return (
        {name: metric.result() for name, metric in metrics.items()},
        {
            "sequence": sequence,
            "iterations_per_stage": ITERATIONS,
            "modes": modes,
            "seed_thresholds": SEED_THRESHOLDS,
            "propagation": "exact per-head SA, averaged across all modules and timesteps at each stage",
        },
    )
