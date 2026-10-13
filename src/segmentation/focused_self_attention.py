"""Focused full-dataset ablations around the strongest self-attention modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .ablation_runner import target_token_positions
from .attention_aggregation import (
    attention_to_mask,
    load_step_module_head_maps,
    normalize_map,
    resize_map,
)
from .metrics import BinarySegmentationMetrics
from .ovam_baseline import ovam_cross_attention_map
from .postprocessing import postprocess_mask
from .stage1_suite import BEST_12_INDICES


ITERATIONS = (1, 2, 3, 4)
OUTPUT_THRESHOLDS = tuple(value / 100 for value in range(20, 81, 5))
NORMALIZATIONS = ("ovam_max_after_resize", "minmax", "max", "percentile_01_99")
PARALLEL_R64_WEIGHTS = (0.25, 0.50, 0.75)
POSTPROCESS_METHODS = (
    "none",
    "largest_component",
    "remove_fraction_0.001",
    "remove_fraction_0.005",
    "remove_fraction_0.01",
    "fill_holes",
    "closing_fill_r1",
    "closing_fill_r2",
    "closing_fill_r4",
)


@dataclass
class PropagationSummary:
    """Small aggregate outputs from one selected self-attention module."""

    aggregate: dict[int, torch.Tensor]  # iteration -> [tokens, seed]
    per_head: dict[int, torch.Tensor] | None = None  # iteration -> [head, tokens]
    per_step: dict[int, torch.Tensor] | None = None  # iteration -> [step, tokens]
    per_step_head: dict[int, torch.Tensor] | None = None  # iteration -> [step, head, tokens]
    per_step_head_columns: dict[int, torch.Tensor] | None = None  # iteration -> [step, head, tokens, seed]


def _entropy(attention_map: torch.Tensor) -> float:
    values = attention_map.flatten().float()
    values = values - values.min()
    total = values.sum()
    if total <= 0:
        return float("inf")
    probabilities = values / total
    return float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())


def _normalize_seed(attention_map: torch.Tensor, method: str) -> torch.Tensor:
    if method == "ovam_max_after_resize":
        method = "max"
    return normalize_map(attention_map, method)


@torch.inference_mode()
def build_cross_attention_layer_heads(
    sample: Any,
    tokenizer: Any,
) -> tuple[tuple[str, ...], dict[int, int], torch.Tensor]:
    """Return timestep-averaged CA maps as ``[layer, head, 64, 64]``.

    Keeping this extraction separate lets the analysis code inspect individual
    layers and heads while the experiment code continues to build exactly the
    same semantic seeds from the same tensor.
    """
    modules = tuple(sorted(name for names in sample.modules_by_resolution.values() for name in names))
    if len(modules) != 16:
        raise ValueError(f"Expected 16 cross-attention modules, found {len(modules)}")
    module_index = {name: index for index, name in enumerate(modules)}
    module_resolution = {
        module_index[name]: resolution
        for resolution, names in sample.modules_by_resolution.items()
        for name in names
    }
    positions = target_token_positions(sample, tokenizer)
    head_count = 8
    layer_heads = torch.zeros((len(modules), head_count, 64, 64), dtype=torch.float32)
    layer_counts = torch.zeros(len(modules), dtype=torch.float32)
    for step_index in range(len(sample.timesteps)):
        step_maps = load_step_module_head_maps(sample, step_index, positions, modules, branch=1)
        for module_name, (heads, _) in step_maps.items():
            index = module_index[module_name]
            if heads.shape[0] != head_count:
                raise ValueError(f"Expected {head_count} heads for {module_name}, found {heads.shape[0]}")
            layer_heads[index] += torch.stack([resize_map(value, 64) for value in heads])
            layer_counts[index] += 1
    if torch.any(layer_counts == 0):
        raise RuntimeError("At least one cross-attention layer was never observed")
    layer_heads /= layer_counts[:, None, None, None]
    return modules, module_resolution, layer_heads


@torch.inference_mode()
def build_cross_attention_seeds_from_heads(
    modules: Sequence[str],
    module_resolution: dict[int, int],
    layer_heads: torch.Tensor,
    fixed_head_rankings: dict[str, Sequence[int]],
    ovam_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build selected CA seeds from already extracted per-layer head maps."""
    layer_maps = layer_heads.mean(dim=1)
    best12 = torch.tensor(BEST_12_INDICES, dtype=torch.long)
    r8_r16 = torch.tensor(
        [index for index, resolution in module_resolution.items() if resolution in (8, 16)],
        dtype=torch.long,
    )
    mid_up = torch.tensor(
        [index for index, name in enumerate(modules) if name.startswith(("mid_block", "up_blocks"))],
        dtype=torch.long,
    )
    top4_maps = []
    for index in BEST_12_INDICES:
        ranking = fixed_head_rankings.get(str(index))
        if ranking is None or len(ranking) < 4:
            raise ValueError(f"Missing fixed top-four head ranking for CA layer {index}")
        top4_maps.append(layer_heads[index, [int(value) for value in ranking[:4]]].mean(dim=0))

    raw = {
        "best12_percentile": layer_maps[best12].mean(dim=0),
        "all16_percentile": layer_maps.mean(dim=0),
        "r8_r16_max": layer_maps[r8_r16].mean(dim=0),
        "mid_up_max": layer_maps[mid_up].mean(dim=0),
        "top4_heads_percentile": torch.stack(top4_maps).mean(dim=0),
        "ovam_nonmiddle_max": ovam_map,
    }
    methods = {
        "best12_percentile": "percentile_01_99",
        "all16_percentile": "percentile_01_99",
        "r8_r16_max": "max",
        "mid_up_max": "max",
        "top4_heads_percentile": "percentile_01_99",
        "ovam_nonmiddle_max": "max",
    }
    return {name: _normalize_seed(value, methods[name]) for name, value in raw.items()}


@torch.inference_mode()
def build_cross_attention_seeds(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: dict[str, Sequence[int]],
) -> dict[str, torch.Tensor]:
    """Build the six semantic seeds selected by the first full-dataset study."""
    modules, module_resolution, layer_heads = build_cross_attention_layer_heads(sample, tokenizer)
    return build_cross_attention_seeds_from_heads(
        modules, module_resolution, layer_heads, fixed_head_rankings, ovam_cross_attention_map(sample)
    )


@torch.inference_mode()
def propagate_selected_module(
    sample: Any,
    module_name: str,
    seeds: torch.Tensor,
    device: str,
    track_seed_column: int | None = None,
    track_step_heads: bool = False,
    track_seed_columns: Sequence[int] | None = None,
) -> PropagationSummary:
    """Propagate seed columns through one SA module and retain compact aggregates."""
    compute_device = torch.device(device)
    resolution = next(
        size for size, names in sample.self_modules_by_resolution.items() if module_name in names
    )
    token_count = resolution * resolution
    if seeds.shape[0] != token_count:
        raise ValueError(f"Seed has {seeds.shape[0]} tokens; {module_name} expects {token_count}")
    aggregate = {
        iteration: torch.zeros((token_count, seeds.shape[1]), dtype=torch.float32)
        for iteration in ITERATIONS
    }
    per_head: dict[int, torch.Tensor] | None = None
    per_step: dict[int, torch.Tensor] | None = None
    per_step_head: dict[int, torch.Tensor] | None = None
    per_step_head_columns: dict[int, torch.Tensor] | None = None
    if track_step_heads and track_seed_column is None:
        raise ValueError("track_step_heads requires track_seed_column")
    tracked_columns = tuple(int(value) for value in (track_seed_columns or ()))
    if any(value < 0 or value >= seeds.shape[1] for value in tracked_columns):
        raise ValueError("track_seed_columns contains an invalid seed-column index")
    observed_heads: int | None = None
    seeds_device = seeds.to(device=compute_device, dtype=torch.float32)

    for step_index in range(len(sample.timesteps)):
        with sample.open_attention(step_index) as tensors:
            query_tensor = tensors.get_tensor(f"{module_name}.q")
            key_tensor = tensors.get_tensor(f"{module_name}.k")
            branch = 1 if query_tensor.shape[0] > 1 else 0
            query = query_tensor[branch]
            key = key_tensor[branch]
        head_count = int(query.shape[0])
        if observed_heads is None:
            observed_heads = head_count
            if track_seed_column is not None:
                per_head = {
                    iteration: torch.zeros((head_count, token_count), dtype=torch.float32)
                    for iteration in ITERATIONS
                }
                per_step = {
                    iteration: torch.zeros((len(sample.timesteps), token_count), dtype=torch.float32)
                    for iteration in ITERATIONS
                }
                if track_step_heads:
                    per_step_head = {
                        iteration: torch.zeros(
                            (len(sample.timesteps), head_count, token_count),
                            dtype=torch.float32,
                        )
                        for iteration in ITERATIONS
                    }
            if tracked_columns:
                per_step_head_columns = {
                    iteration: torch.zeros(
                        (
                            len(sample.timesteps),
                            head_count,
                            token_count,
                            len(tracked_columns),
                        ),
                        dtype=torch.float32,
                    )
                    for iteration in ITERATIONS
                }
        elif head_count != observed_heads:
            raise RuntimeError("Self-attention head count changed between denoising steps")

        step_sums = {
            iteration: torch.zeros_like(seeds_device) for iteration in ITERATIONS
        }
        scale = 1.0 / math.sqrt(query.shape[-1])
        for head in range(head_count):
            q = query[head].to(device=compute_device, dtype=torch.float32)
            k = key[head].to(device=compute_device, dtype=torch.float32)
            affinity = torch.softmax(torch.matmul(q, k.transpose(0, 1)) * scale, dim=-1)
            current = seeds_device
            for iteration in ITERATIONS:
                current = torch.matmul(affinity, current)
                step_sums[iteration] += current
                if per_head is not None and track_seed_column is not None:
                    tracked = current[:, track_seed_column].cpu()
                    per_head[iteration][head] += tracked
                    if per_step_head is not None:
                        per_step_head[iteration][step_index, head] = tracked
                if per_step_head_columns is not None:
                    per_step_head_columns[iteration][step_index, head] = current[
                        :, list(tracked_columns)
                    ].cpu()
            del q, k, affinity, current
        for iteration in ITERATIONS:
            aggregate[iteration] += step_sums[iteration].cpu()
            if per_step is not None and track_seed_column is not None:
                per_step[iteration][step_index] = (
                    step_sums[iteration][:, track_seed_column] / head_count
                ).cpu()
        del query, key, step_sums

    if observed_heads is None:
        raise RuntimeError(f"No self-attention calls found for {module_name}")
    denominator = len(sample.timesteps) * observed_heads
    for iteration in ITERATIONS:
        aggregate[iteration] /= denominator
        if per_head is not None:
            per_head[iteration] /= len(sample.timesteps)
    return PropagationSummary(
        aggregate=aggregate,
        per_head=per_head,
        per_step=per_step,
        per_step_head=per_step_head,
        per_step_head_columns=per_step_head_columns,
    )


def temporal_weight_schemes(
    timesteps: Sequence[int], alphas_cumprod: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Return normalized, label-free timestep weighting schemes."""
    count = len(timesteps)
    progress = torch.linspace(1 / count, 1.0, count, dtype=torch.float32)
    output = {
        "late_linear": progress,
        "late_quadratic": progress.square(),
        "late_exp_0.05": torch.exp(torch.arange(count, dtype=torch.float32) * 0.05),
        "late_exp_0.10": torch.exp(torch.arange(count, dtype=torch.float32) * 0.10),
        "late_exp_0.20": torch.exp(torch.arange(count, dtype=torch.float32) * 0.20),
    }
    if alphas_cumprod is not None:
        alpha = alphas_cumprod[torch.tensor(tuple(timesteps), dtype=torch.long)].float()
        snr = (alpha / (1.0 - alpha).clamp_min(1e-8)).clamp_max(1e4)
        output.update({"snr": snr, "sqrt_snr": snr.sqrt(), "log1p_snr": snr.log1p()})
    return {name: weights / weights.sum().clamp_min(1e-12) for name, weights in output.items()}


def _update(
    metrics: dict[str, BinarySegmentationMetrics],
    key: str,
    classname: str,
    score_map: torch.Tensor,
    target: np.ndarray,
    normalization: str = "percentile_01_99",
    threshold: float = 0.45,
    postprocessing: str = "none",
) -> None:
    prediction = attention_to_mask(score_map, target.shape, threshold, normalization)
    if postprocessing != "none":
        prediction = torch.from_numpy(postprocess_mask(prediction.cpu().numpy(), postprocessing))
    metrics.setdefault(key, BinarySegmentationMetrics()).update(classname, prediction, target)


@torch.inference_mode()
def run_focused_self_attention_suite(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: dict[str, Sequence[int]],
    device: str = "cuda",
    alphas_cumprod: torch.Tensor | None = None,
) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    """Run priorities 1--4 from the full VOC-Sim follow-up plan on one image."""
    target = sample.load_target_mask()
    seed_maps = build_cross_attention_seeds(sample, tokenizer, fixed_head_rankings)
    seed_names = tuple(seed_maps)
    seed64 = torch.stack([seed_maps[name].flatten() for name in seed_names], dim=1)
    seed32 = torch.stack([resize_map(seed_maps[name], 32).flatten() for name in seed_names], dim=1)

    module32 = "up_blocks.2.attentions.2.transformer_blocks.0.attn1"
    module64 = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
    if module32 not in sample.self_modules_by_resolution.get(32, ()):
        raise ValueError(f"Winning 32x32 module is absent: {module32}")
    if module64 not in sample.self_modules_by_resolution.get(64, ()):
        raise ValueError(f"Winning 64x64 module is absent: {module64}")

    direct32 = propagate_selected_module(sample, module32, seed32, device)
    direct64 = propagate_selected_module(sample, module64, seed64, device, track_seed_column=0)

    # Apply the selected 64x64 operator after each matching 32x32 iteration.
    sequential_columns = []
    sequential_lookup = []
    for iteration in ITERATIONS:
        for seed_index in range(len(seed_names)):
            sequential_columns.append(
                resize_map(direct32.aggregate[iteration][:, seed_index].reshape(32, 32), 64).flatten()
            )
            sequential_lookup.append((iteration, seed_index))
    sequential_input = torch.stack(sequential_columns, dim=1)
    sequential_raw = propagate_selected_module(sample, module64, sequential_input, device)

    groups: dict[str, dict[str, BinarySegmentationMetrics]] = {
        "cross_attention_seeds": {},
        "seed_operator": {},
        "output_calibration": {},
        "postprocessing": {},
        "sa_heads": {},
        "sa_time": {},
    }
    for seed_name, seed_map in seed_maps.items():
        _update(groups["cross_attention_seeds"], seed_name, sample.classname, seed_map, target)

    candidate_maps: dict[str, torch.Tensor] = {}
    for seed_index, seed_name in enumerate(seed_names):
        for iteration in ITERATIONS:
            map32 = direct32.aggregate[iteration][:, seed_index].reshape(32, 32)
            map64 = direct64.aggregate[iteration][:, seed_index].reshape(64, 64)
            name32 = f"{seed_name}__r32__iter{iteration}"
            name64 = f"{seed_name}__r64__iter{iteration}"
            candidate_maps[name32] = map32
            candidate_maps[name64] = map64
            _update(groups["seed_operator"], name32, sample.classname, map32, target)
            _update(groups["seed_operator"], name64, sample.classname, map64, target)
            for weight64 in PARALLEL_R64_WEIGHTS:
                parallel = (1.0 - weight64) * resize_map(map32, 64) + weight64 * map64
                key = f"{seed_name}__parallel32_64__r64w{weight64:.2f}__iter{iteration}"
                candidate_maps[key] = parallel
                _update(groups["seed_operator"], key, sample.classname, parallel, target)

            input_column = sequential_lookup.index((iteration, seed_index))
            sequential = sequential_raw.aggregate[iteration][:, input_column].reshape(64, 64)
            key = f"{seed_name}__sequential32_64__iter{iteration}"
            candidate_maps[key] = sequential
            _update(groups["seed_operator"], key, sample.classname, sequential, target)

    # Calibrate every r64 seed at iterations 2/4, plus the strongest structural alternatives.
    calibration_names = {
        f"{seed_name}__r64__iter{iteration}"
        for seed_name in seed_names
        for iteration in (2, 4)
    }
    calibration_names.update(
        {
            "best12_percentile__r32__iter4",
            "best12_percentile__sequential32_64__iter2",
            "best12_percentile__sequential32_64__iter4",
            *{
                f"best12_percentile__parallel32_64__r64w{weight:.2f}__iter4"
                for weight in PARALLEL_R64_WEIGHTS
            },
        }
    )
    for name in sorted(calibration_names):
        score_map = candidate_maps[name]
        for normalization in NORMALIZATIONS:
            for threshold in OUTPUT_THRESHOLDS:
                key = f"{name}__{normalization}__t{threshold:.2f}"
                _update(
                    groups["output_calibration"], key, sample.classname, score_map, target,
                    normalization=normalization, threshold=threshold,
                )

    for iteration in (2, 4):
        name = f"best12_percentile__r64__iter{iteration}"
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55):
            for method in POSTPROCESS_METHODS:
                key = f"{name}__percentile_01_99__t{threshold:.2f}__{method}"
                _update(
                    groups["postprocessing"], key, sample.classname, candidate_maps[name], target,
                    threshold=threshold, postprocessing=method,
                )

    # Head study: supervised-free per-head, leave-one-out, entropy top-k and weighting.
    assert direct64.per_head is not None
    for iteration, head_maps_flat in direct64.per_head.items():
        head_maps = head_maps_flat.reshape(head_maps_flat.shape[0], 64, 64)
        head_count = head_maps.shape[0]
        _update(groups["sa_heads"], f"iter{iteration}__all_heads", sample.classname, head_maps.mean(0), target)
        entropies = torch.tensor([_entropy(value) for value in head_maps])
        for head in range(head_count):
            _update(groups["sa_heads"], f"iter{iteration}__head_{head}", sample.classname, head_maps[head], target)
            kept = [value for value in range(head_count) if value != head]
            _update(
                groups["sa_heads"], f"iter{iteration}__leave_head_{head}", sample.classname,
                head_maps[kept].mean(0), target,
            )
        low_order = torch.argsort(entropies)
        high_order = torch.argsort(entropies, descending=True)
        for top_k in (1, 2, 4, 8):
            for label, order in (("low_entropy", low_order), ("high_entropy", high_order)):
                selected = order[:top_k]
                _update(
                    groups["sa_heads"], f"iter{iteration}__{label}_top{top_k}",
                    sample.classname, head_maps[selected].mean(0), target,
                )
        finite = torch.where(torch.isfinite(entropies), entropies, torch.full_like(entropies, 1e6))
        inverse_weights = 1.0 / finite.clamp_min(1e-6)
        inverse_weights /= inverse_weights.sum()
        softmax_weights = torch.softmax(-finite, dim=0)
        for label, weights in (("inverse_entropy", inverse_weights), ("softmax_neg_entropy", softmax_weights)):
            weighted = torch.einsum("h,hij->ij", weights, head_maps)
            _update(groups["sa_heads"], f"iter{iteration}__{label}", sample.classname, weighted, target)

    # Timestep study on the winning r64 module with the original continuous seed.
    assert direct64.per_step is not None
    schemes = temporal_weight_schemes(sample.timesteps, alphas_cumprod)
    windows = {
        "steps_00_04": range(0, 5),
        "steps_05_09": range(5, 10),
        "steps_10_14": range(10, 15),
        "steps_15_20": range(15, 21),
        "steps_21_25": range(21, 26),
        "steps_26_30": range(26, 31),
    }
    for iteration, step_maps_flat in direct64.per_step.items():
        step_maps = step_maps_flat.reshape(len(sample.timesteps), 64, 64)
        total = step_maps.sum(0)
        _update(groups["sa_time"], f"iter{iteration}__all_uniform", sample.classname, total / len(step_maps), target)
        for step_index in range(len(step_maps)):
            _update(
                groups["sa_time"], f"iter{iteration}__step_{step_index:02d}",
                sample.classname, step_maps[step_index], target,
            )
            _update(
                groups["sa_time"], f"iter{iteration}__leave_step_{step_index:02d}",
                sample.classname, (total - step_maps[step_index]) / (len(step_maps) - 1), target,
            )
        for name, indices_range in windows.items():
            indices = tuple(indices_range)
            _update(
                groups["sa_time"], f"iter{iteration}__{name}", sample.classname,
                step_maps[list(indices)].mean(0), target,
            )
        for end in (4, 9, 14, 20, 25):
            _update(
                groups["sa_time"], f"iter{iteration}__cumulative_00_{end:02d}",
                sample.classname, step_maps[: end + 1].mean(0), target,
            )
        for start in (5, 10, 15, 21, 26):
            _update(
                groups["sa_time"], f"iter{iteration}__trailing_{start:02d}_30",
                sample.classname, step_maps[start:].mean(0), target,
            )
        for name, weights in schemes.items():
            weighted = torch.einsum("t,tij->ij", weights, step_maps)
            _update(groups["sa_time"], f"iter{iteration}__{name}", sample.classname, weighted, target)

    results = {
        group: {name: metric.result() for name, metric in metrics.items()}
        for group, metrics in groups.items()
    }
    metadata = {
        "seed_names": seed_names,
        "module32": module32,
        "module64": module64,
        "iterations": ITERATIONS,
        "output_thresholds": OUTPUT_THRESHOLDS,
        "normalizations": NORMALIZATIONS,
        "parallel_r64_weights": PARALLEL_R64_WEIGHTS,
        "postprocessing": POSTPROCESS_METHODS,
        "temporal_weight_schemes": tuple(schemes),
        "selection_note": (
            "No dataset-learned head or timestep weights are fit in this evaluation-only run; "
            "head ranking is per-image and label-free, and temporal weights are fixed."
        ),
    }
    return results, metadata


def _selection_weights(step_count: int, indices: Sequence[int]) -> torch.Tensor:
    weights = torch.zeros(step_count, dtype=torch.float32)
    valid = [int(index) for index in indices if 0 <= int(index) < step_count]
    if not valid:
        raise ValueError("Timestep selection is empty")
    weights[valid] = 1.0 / len(valid)
    return weights


def joint_timestep_weights(
    timesteps: Sequence[int], alphas_cumprod: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """Focused timestep candidates motivated by the first follow-up sweep."""
    count = len(timesteps)
    selections = {
        "all": range(count),
        "trailing_05_30": range(5, count),
        "trailing_10_30": range(10, count),
        "trailing_15_30": range(15, count),
        "steps_10_20": range(10, 21),
        "steps_10_25": range(10, 26),
        "steps_15_20": range(15, 21),
        "step_14": (14,),
        "step_15": (15,),
        "step_16": (16,),
    }
    output = {
        name: _selection_weights(count, tuple(indices)) for name, indices in selections.items()
    }
    broad = temporal_weight_schemes(timesteps, alphas_cumprod)
    for name in ("late_linear", "late_exp_0.05", "late_exp_0.10", "sqrt_snr", "log1p_snr"):
        if name in broad:
            output[name] = broad[name]
    return output


def joint_head_selections(head_count: int = 8) -> dict[str, tuple[int, ...]]:
    """Fixed SA head subsets centered on the strongest leave-head findings."""
    all_heads = tuple(range(head_count))
    output = {"all_heads": all_heads}
    for omitted in (2, 3, 4, 6, 7):
        output[f"leave_{omitted}"] = tuple(head for head in all_heads if head != omitted)
    for first, second in ((6, 7), (2, 6), (3, 6), (4, 6), (2, 7), (3, 7), (4, 7)):
        output[f"leave_{first}_{second}"] = tuple(
            head for head in all_heads if head not in (first, second)
        )
    return output


@torch.inference_mode()
def run_joint_sa_head_time_suite(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: dict[str, Sequence[int]],
    device: str = "cuda",
    alphas_cumprod: torch.Tensor | None = None,
) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    """Combine the best seed with SA head and timestep choices factorially."""
    target = sample.load_target_mask()
    seed = build_cross_attention_seeds(sample, tokenizer, fixed_head_rankings)[
        "top4_heads_percentile"
    ]
    module = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
    propagated = propagate_selected_module(
        sample,
        module,
        seed.flatten()[:, None],
        device,
        track_seed_column=0,
        track_step_heads=True,
    )
    assert propagated.per_step_head is not None
    head_selections = joint_head_selections()
    time_weights = joint_timestep_weights(sample.timesteps, alphas_cumprod)
    thresholds = (0.35, 0.40, 0.45, 0.50, 0.55)
    groups: dict[str, dict[str, BinarySegmentationMetrics]] = {
        "joint_head_time_threshold": {},
        "joint_postprocessing": {},
    }

    for iteration in ITERATIONS:
        values = propagated.per_step_head[iteration].reshape(
            len(sample.timesteps), 8, 64, 64
        )
        for head_name, heads in head_selections.items():
            per_step = values[:, list(heads)].mean(dim=1)
            for time_name, weights in time_weights.items():
                score_map = torch.einsum("t,tij->ij", weights, per_step)
                normalized = normalize_map(score_map, "percentile_01_99")
                resized = F.interpolate(
                    normalized[None, None],
                    size=target.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                prefix = f"iter{iteration}__{head_name}__{time_name}"
                for threshold in thresholds:
                    prediction = resized >= threshold
                    key = f"{prefix}__t{threshold:.2f}"
                    groups["joint_head_time_threshold"].setdefault(
                        key, BinarySegmentationMetrics()
                    ).update(sample.classname, prediction, target)
                prediction = (resized >= 0.45).cpu().numpy()
                processed = torch.from_numpy(postprocess_mask(prediction, "remove_fraction_0.01"))
                groups["joint_postprocessing"].setdefault(
                    f"{prefix}__t0.45__remove_fraction_0.01",
                    BinarySegmentationMetrics(),
                ).update(sample.classname, processed, target)

    results = {
        group: {name: metric.result() for name, metric in metrics.items()}
        for group, metrics in groups.items()
    }
    metadata = {
        "seed": "top4_heads_percentile",
        "module": module,
        "iterations": ITERATIONS,
        "head_selections": head_selections,
        "timestep_weights": tuple(time_weights),
        "thresholds": thresholds,
        "normalization": "percentile_01_99",
        "postprocessing": "remove_fraction_0.01 at threshold 0.45",
        "configuration_count": sum(len(configurations) for configurations in results.values()),
    }
    return results, metadata
