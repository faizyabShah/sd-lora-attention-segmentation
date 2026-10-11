"""Comprehensive offline cross-attention ablations on captured tensors."""

from __future__ import annotations

import gc
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer

from .ablation_runner import SegmentationSettings, target_token_positions
from .attention_aggregation import load_step_module_head_maps, normalize_map, resize_map
from .captured_dataset import CapturedAttentionDataset
from .metrics import BinarySegmentationMetrics
from .postprocessing import postprocess_mask


BEST_12_INDICES = (0, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
CLASS_LAYER_INDICES = {
    "aeroplane": (0, 2, 4, 5, 7, 9, 11, 12, 13, 14),
    "bicycle": (0, 2, 4, 7, 9, 11, 12, 13, 14, 15),
    "bird": (2, 6, 7, 8, 9, 10, 11, 12, 13),
    "boat": (0, 7, 8, 9, 10, 11, 12, 13, 14),
}


def _time_windows(step_count: int) -> dict[str, tuple[int, ...]]:
    windows: dict[str, tuple[int, ...]] = {
        "all": tuple(range(step_count)),
        "steps_00_04": tuple(range(0, 5)),
        "steps_05_09": tuple(range(5, 10)),
        "steps_10_14": tuple(range(10, 15)),
        "steps_15_20": tuple(range(15, 21)),
        "steps_21_25": tuple(range(21, 26)),
        "steps_26_30": tuple(range(26, 31)),
    }
    windows.update({f"step_{i:02d}": (i,) for i in range(step_count)})
    for end in (4, 9, 14, 20, 25):
        windows[f"cumulative_00_{end:02d}"] = tuple(range(0, end + 1))
    for start in (5, 10, 15, 21, 26):
        windows[f"trailing_{start:02d}_30"] = tuple(range(start, step_count))
    windows.update(
        {
            f"leave_step_{i:02d}": tuple(j for j in range(step_count) if j != i)
            for i in range(step_count)
        }
    )
    return windows


def _entropy(attention_map: torch.Tensor) -> float:
    values = attention_map.flatten().float()
    values = values - values.min()
    total = values.sum()
    if total <= 0:
        return float("inf")
    probabilities = values / total
    return float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())


def _prediction(attention_map: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    normalized = normalize_map(attention_map, "percentile_01_99")
    resized = F.interpolate(
        normalized[None, None], size=shape, mode="bilinear", align_corners=False
    )[0, 0]
    return resized >= 0.45


def _weighted_sum(layer_maps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("l,lhw->hw", weights.float(), layer_maps)


def run_stage1_suite(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    settings: SegmentationSettings,
    exhaustive_summary: str | Path,
    dataset_adaptive: bool = True,
    fixed_adaptive_metadata: dict | None = None,
) -> tuple[dict[str, dict], dict]:
    """Run time, head, fusion, adaptive, and postprocessing ablations in one pass."""
    first = dataset.samples[0]
    modules = tuple(sorted(name for values in first.modules_by_resolution.values() for name in values))
    if len(modules) != 16:
        raise ValueError(f"Expected 16 cross-attention modules, found {len(modules)}")
    module_index = {name: index for index, name in enumerate(modules)}
    resolution = {
        module_index[name]: value
        for value, names in first.modules_by_resolution.items()
        for name in names
    }
    region = {
        index: ("down" if name.startswith("down_") else "mid" if name.startswith("mid_") else "up")
        for index, name in enumerate(modules)
    }
    layer_sets = {
        "best12": BEST_12_INDICES,
        "all16": tuple(range(16)),
        "previous15": tuple(i for i in range(16) if i != 6),
    }
    windows = _time_windows(len(first.timesteps))
    metric_groups: dict[str, dict[str, BinarySegmentationMetrics]] = {
        "time_layer": {}, "heads": {}, "fusion": {}, "class_adaptive": {}, "postprocessing": {}
    }
    cache: list[tuple[str, torch.Tensor, np.ndarray]] = []

    summary = json.loads(Path(exhaustive_summary).read_text(encoding="utf-8"))
    all_miou = float(summary["all_layers"]["miou"])
    contribution = torch.zeros(16)
    for result in summary["leave_one_out"]:
        omitted = result["omitted_layer_indices"][0]
        contribution[omitted] = all_miou - float(result["miou"])

    for sample in tqdm(dataset, desc="Stage-1 cross-attention suite"):
        positions = target_token_positions(sample, tokenizer)
        layer_heads = torch.zeros((16, 8, settings.working_size, settings.working_size))
        layer_counts = torch.zeros(16)
        time_sums = {
            layer_name: {window: torch.zeros((settings.working_size, settings.working_size)) for window in windows}
            for layer_name in layer_sets
        }
        time_counts = {layer_name: {window: 0 for window in windows} for layer_name in layer_sets}

        membership = {
            step: tuple(name for name, indices in windows.items() if step in indices)
            for step in range(len(sample.timesteps))
        }
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_module_head_maps(
                sample, step_index, positions, modules, branch=settings.branch
            )
            step_layers: dict[int, torch.Tensor] = {}
            for module_name, (head_maps, _) in step_maps.items():
                index = module_index[module_name]
                resized_heads = torch.stack(
                    [resize_map(head_map, settings.working_size) for head_map in head_maps]
                )
                layer_heads[index] += resized_heads
                layer_counts[index] += 1
                step_layers[index] = resized_heads.mean(dim=0)
            for layer_name, indices in layer_sets.items():
                step_sum = torch.stack([step_layers[index] for index in indices]).sum(dim=0)
                for window in membership[step_index]:
                    time_sums[layer_name][window] += step_sum
                    time_counts[layer_name][window] += len(indices)

        layer_heads /= layer_counts[:, None, None, None]
        layer_maps = layer_heads.mean(dim=1)
        target = sample.load_target_mask()
        if dataset_adaptive:
            cache.append((sample.classname, layer_heads.half(), target))

        for layer_name in layer_sets:
            for window in windows:
                key = f"{layer_name}__{window}"
                metric = metric_groups["time_layer"].setdefault(key, BinarySegmentationMetrics())
                attention_map = time_sums[layer_name][window] / time_counts[layer_name][window]
                metric.update(sample.classname, _prediction(attention_map, target.shape), target)

        selected_heads = layer_heads[list(BEST_12_INDICES)]
        selected_maps = layer_maps[list(BEST_12_INDICES)]
        for local_index, layer_index in enumerate(BEST_12_INDICES):
            for head in range(8):
                key = f"layer_{layer_index:02d}__head_{head}"
                metric = metric_groups["heads"].setdefault(key, BinarySegmentationMetrics())
                metric.update(sample.classname, _prediction(selected_heads[local_index, head], target.shape), target)
                key = f"layer_{layer_index:02d}__leave_head_{head}"
                metric = metric_groups["heads"].setdefault(key, BinarySegmentationMetrics())
                kept = [h for h in range(8) if h != head]
                metric.update(
                    sample.classname,
                    _prediction(selected_heads[local_index, kept].mean(dim=0), target.shape),
                    target,
                )
        for head in range(8):
            for prefix, values in (
                ("global_head", selected_heads[:, head]),
                ("global_leave_head", selected_heads[:, [h for h in range(8) if h != head]]),
            ):
                key = f"{prefix}_{head}"
                metric = metric_groups["heads"].setdefault(key, BinarySegmentationMetrics())
                metric.update(sample.classname, _prediction(values.mean(dim=tuple(range(values.ndim - 2))), target.shape), target)

        if not dataset_adaptive and fixed_adaptive_metadata is not None:
            fixed_rankings = fixed_adaptive_metadata["head_rankings"]
            for top_k in (1, 2, 4, 8):
                maps = []
                for local_index, layer_index in enumerate(BEST_12_INDICES):
                    heads = [int(value) for value in fixed_rankings[str(layer_index)][:top_k]]
                    maps.append(selected_heads[local_index, heads].mean(dim=0))
                metric = metric_groups["heads"].setdefault(
                    f"per_layer_top_{top_k}_heads", BinarySegmentationMetrics()
                )
                metric.update(sample.classname, _prediction(torch.stack(maps).mean(dim=0), target.shape), target)

        fusion_maps: dict[str, torch.Tensor] = {"uniform_best12": selected_maps.mean(dim=0)}
        if not dataset_adaptive and fixed_adaptive_metadata is not None:
            fixed_weights = torch.tensor(
                [
                    float(fixed_adaptive_metadata["learned_layer_weights"][str(layer)])
                    for layer in BEST_12_INDICES
                ]
            )
            fusion_maps["learned_nonnegative_soft_iou"] = _weighted_sum(
                selected_maps, fixed_weights
            )
        for group_name, grouping in (
            ("resolution", resolution),
            ("region", region),
        ):
            groups = sorted({grouping[i] for i in BEST_12_INDICES}, key=str)
            group_maps = [
                layer_maps[[i for i in BEST_12_INDICES if grouping[i] == group]].mean(dim=0)
                for group in groups
            ]
            fusion_maps[f"equal_{group_name}"] = torch.stack(group_maps).mean(dim=0)
            for weights in itertools.product((0.5, 1.0, 2.0), repeat=len(groups)):
                name = "_".join(f"{group}-{weight:g}" for group, weight in zip(groups, weights))
                fusion_maps[f"{group_name}_weights__{name}"] = _weighted_sum(
                    torch.stack(group_maps), torch.tensor(weights)
                )
        selected_contribution = contribution[list(BEST_12_INDICES)]
        for temperature in (0.01, 0.025, 0.05, 0.1):
            weights = torch.softmax(selected_contribution / temperature, dim=0)
            fusion_maps[f"loo_softmax_t{temperature:g}"] = _weighted_sum(selected_maps, weights)
        entropies = torch.tensor([_entropy(value) for value in selected_maps])
        inverse_entropy = 1.0 / entropies.clamp_min(1e-6)
        fusion_maps["inverse_entropy_layers"] = _weighted_sum(selected_maps, inverse_entropy)
        flat_heads = selected_heads.reshape(-1, settings.working_size, settings.working_size)
        head_entropies = torch.tensor([_entropy(value) for value in flat_heads])
        fusion_maps["inverse_entropy_heads"] = _weighted_sum(
            flat_heads, 1.0 / head_entropies.clamp_min(1e-6)
        )
        for key, attention_map in fusion_maps.items():
            metric = metric_groups["fusion"].setdefault(key, BinarySegmentationMetrics())
            metric.update(sample.classname, _prediction(attention_map, target.shape), target)

        if dataset_adaptive and sample.classname in CLASS_LAYER_INDICES:
            adaptive_indices = CLASS_LAYER_INDICES[sample.classname]
            adaptive_map = layer_maps[list(adaptive_indices)].mean(dim=0)
            for threshold in (0.35, 0.40, 0.45, 0.50, 0.55):
                key = f"class_layers__percentile__t{threshold:.2f}"
                metric = metric_groups["class_adaptive"].setdefault(key, BinarySegmentationMetrics())
                normalized = normalize_map(adaptive_map, "percentile_01_99")
                resized = F.interpolate(normalized[None, None], size=target.shape, mode="bilinear", align_corners=False)[0, 0]
                metric.update(sample.classname, resized >= threshold, target)

        baseline_prediction = _prediction(selected_maps.mean(dim=0), target.shape).numpy()
        postprocess_methods = (
            "none", "largest_component", "remove_fraction_0.001", "remove_fraction_0.005",
            "remove_fraction_0.01", "opening_r1", "opening_r2", "opening_r4",
            "closing_r1", "closing_r2", "closing_r4",
            "fill_holes", "closing_fill_r1", "closing_fill_r2", "closing_fill_r4",
            "opening_closing_r1", "opening_closing_r2", "opening_closing_r4",
        )
        for method in postprocess_methods:
            key = method
            metric = metric_groups["postprocessing"].setdefault(key, BinarySegmentationMetrics())
            processed = torch.from_numpy(postprocess_mask(baseline_prediction, method))
            metric.update(sample.classname, processed, target)

        del layer_heads, layer_maps, time_sums, time_counts, target

    # Dataset-level head selection: rank heads within each selected layer, then
    # evaluate top-k combinations from the in-memory compact 64x64 cache.
    head_rankings = (
        {} if fixed_adaptive_metadata is None
        else {int(key): value for key, value in fixed_adaptive_metadata.get("head_rankings", {}).items()}
    )
    learned_weights = (
        None if fixed_adaptive_metadata is None
        else torch.tensor([
            float(fixed_adaptive_metadata["learned_layer_weights"][str(layer)])
            for layer in BEST_12_INDICES
        ])
    )
    if dataset_adaptive:
        head_results = {name: metric.result() for name, metric in metric_groups["heads"].items()}
        for layer_index in BEST_12_INDICES:
            head_rankings[layer_index] = sorted(
                range(8),
                key=lambda head: head_results[f"layer_{layer_index:02d}__head_{head}"]["miou"],
                reverse=True,
            )
        for top_k in (1, 2, 4, 8):
            metric = BinarySegmentationMetrics()
            for classname, cached_heads, target in cache:
                maps = []
                for layer_index in BEST_12_INDICES:
                    heads = head_rankings[layer_index][:top_k]
                    maps.append(cached_heads[layer_index, heads].float().mean(dim=0))
                metric.update(classname, _prediction(torch.stack(maps).mean(dim=0), target.shape), target)
            metric_groups["heads"][f"per_layer_top_{top_k}_heads"] = metric

        train_maps = torch.stack(
            [heads[list(BEST_12_INDICES)].float().mean(dim=1) for _, heads, _ in cache]
        )
        train_targets = torch.stack(
            [F.interpolate(torch.from_numpy(target).float()[None, None], size=(64, 64), mode="nearest")[0, 0]
             for _, _, target in cache]
        )
        logits = torch.zeros(len(BEST_12_INDICES), requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.05)
        for _ in range(300):
            weights = torch.softmax(logits, dim=0)
            predictions = torch.einsum("l,nlhw->nhw", weights, train_maps)
            minima = predictions.amin(dim=(1, 2), keepdim=True)
            maxima = predictions.amax(dim=(1, 2), keepdim=True)
            predictions = (predictions - minima) / (maxima - minima).clamp_min(1e-8)
            intersection = (predictions * train_targets).sum(dim=(1, 2))
            union = predictions.sum(dim=(1, 2)) + train_targets.sum(dim=(1, 2)) - intersection
            loss = -(intersection / union.clamp_min(1e-8)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        learned_weights = torch.softmax(logits.detach(), dim=0)
        learned_metric = BinarySegmentationMetrics()
        for classname, heads, target in cache:
            maps = heads[list(BEST_12_INDICES)].float().mean(dim=1)
            learned_metric.update(classname, _prediction(_weighted_sum(maps, learned_weights), target.shape), target)
        metric_groups["fusion"]["learned_nonnegative_soft_iou"] = learned_metric
        del train_maps, train_targets

    results = {
        group: {
            name: value if isinstance(value, dict) else value.result()
            for name, value in metrics.items()
        }
        for group, metrics in metric_groups.items()
    }
    metadata = {
        "module_index": {str(i): name for i, name in enumerate(modules)},
        "resolution_by_index": resolution,
        "region_by_index": region,
        "best_12_indices": BEST_12_INDICES,
        "class_layer_indices": CLASS_LAYER_INDICES,
        "time_windows": windows,
        "head_rankings": {str(k): v for k, v in head_rankings.items()},
        "learned_layer_weights": None if learned_weights is None else {
            str(layer): float(weight)
            for layer, weight in zip(BEST_12_INDICES, learned_weights)
        },
        "dense_crf": "not run: pydensecrf is not installed",
    }
    del cache
    gc.collect()
    return results, metadata
