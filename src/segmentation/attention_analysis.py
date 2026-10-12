"""Compact explanatory measurements for the VOC-Sim attention pipeline.

The routines in this module consume either an in-memory attention sample or a
compatible saved sample.  They return scalars and small visualization maps;
they never persist Q/K tensors, attention matrices, latents, or generated
images.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .attention_aggregation import attention_to_mask, normalize_map, resize_map
from .focused_self_attention import (
    BEST_12_INDICES,
    build_cross_attention_layer_heads,
    build_cross_attention_seeds_from_heads,
    propagate_selected_module,
)
from .ovam_baseline import ovam_cross_attention_map
from .postprocessing import postprocess_mask


WINNING_MODULE = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
ANALYSIS_SIZE = 64


def object_geometry(target: np.ndarray) -> dict[str, float | int]:
    """Describe target size and shape without using model predictions."""
    mask = target.astype(bool, copy=False)
    height, width = mask.shape
    area = int(mask.sum())
    labels, components = ndimage.label(mask)
    del labels
    ys, xs = np.nonzero(mask)
    if area:
        box_height = int(ys.max() - ys.min() + 1)
        box_width = int(xs.max() - xs.min() + 1)
        cy, cx = float(ys.mean()), float(xs.mean())
    else:
        box_height = box_width = 0
        cy = cx = float("nan")
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded
    perimeter = int(boundary.sum())
    bbox_area = box_height * box_width
    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
    return {
        "height": height,
        "width": width,
        "pixels": area,
        "area_fraction": area / mask.size,
        "bbox_area_fraction": bbox_area / mask.size,
        "bbox_width_fraction": box_width / width,
        "bbox_height_fraction": box_height / height,
        "fill_fraction": area / bbox_area if bbox_area else 0.0,
        "components": int(components),
        "boundary_fraction": perimeter / mask.size,
        "compactness": compactness,
        "centroid_x": cx / max(width - 1, 1),
        "centroid_y": cy / max(height - 1, 1),
    }


def _target_at_size(target: np.ndarray, size: int = ANALYSIS_SIZE) -> torch.Tensor:
    value = torch.from_numpy(target.astype(np.float32, copy=False))[None, None]
    return F.interpolate(value, size=(size, size), mode="nearest")[0, 0].bool()


def _average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    truth = target[order].astype(np.float64)
    precision = np.cumsum(truth) / np.arange(1, truth.size + 1)
    return float((precision * truth).sum() / positives)


def _roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[target].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def map_statistics(
    score_map: torch.Tensor,
    target: np.ndarray,
    prediction: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.45,
) -> dict[str, float]:
    """Measure localization, concentration, smoothness, and boundary alignment."""
    score = normalize_map(resize_map(score_map.float().cpu(), ANALYSIS_SIZE), "percentile_01_99")
    truth = _target_at_size(target)
    flat = score.flatten()
    mass = flat.clamp_min(0)
    probabilities = mass / mass.sum().clamp_min(1e-12)
    raw_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    normalized_entropy = raw_entropy / math.log(flat.numel())
    top_count = max(1, math.ceil(flat.numel() * 0.01))
    top_mass = torch.topk(mass, top_count).values.sum() / mass.sum().clamp_min(1e-12)
    foreground = score[truth]
    background = score[~truth]
    fg_mean = foreground.mean()
    bg_mean = background.mean()
    pooled_std = torch.cat((foreground, background)).std().clamp_min(1e-12)
    dy = torch.zeros_like(score)
    dx = torch.zeros_like(score)
    dy[1:] = (score[1:] - score[:-1]).abs()
    dx[:, 1:] = (score[:, 1:] - score[:, :-1]).abs()
    gradient = torch.sqrt(dx.square() + dy.square())
    truth_np = truth.numpy()
    boundary = np.logical_xor(
        ndimage.binary_dilation(truth_np), ndimage.binary_erosion(truth_np)
    )
    boundary_t = torch.from_numpy(boundary)
    yy, xx = torch.meshgrid(
        torch.arange(ANALYSIS_SIZE), torch.arange(ANALYSIS_SIZE), indexing="ij"
    )
    weighted_total = mass.sum().clamp_min(1e-12)
    mass_2d = mass.reshape(ANALYSIS_SIZE, ANALYSIS_SIZE)
    map_cy = float((mass_2d * yy).sum() / weighted_total)
    map_cx = float((mass_2d * xx).sum() / weighted_total)
    ty, tx = torch.nonzero(truth, as_tuple=True)
    target_cy = float(ty.float().mean())
    target_cx = float(tx.float().mean())
    centroid_error = math.hypot(map_cx - target_cx, map_cy - target_cy) / math.hypot(
        ANALYSIS_SIZE - 1, ANALYSIS_SIZE - 1
    )
    if prediction is None:
        pred64 = score >= threshold
        full_pred = attention_to_mask(score_map.cpu(), target.shape, threshold, "percentile_01_99")
    else:
        full_pred = torch.as_tensor(prediction).bool().cpu()
        pred64 = F.interpolate(
            full_pred.float()[None, None], size=(ANALYSIS_SIZE, ANALYSIS_SIZE), mode="nearest"
        )[0, 0].bool()
    full_truth = torch.from_numpy(target.astype(bool, copy=False))
    intersection = int((full_pred & full_truth).sum())
    union = int((full_pred | full_truth).sum())
    soft_intersection = (score * truth.float()).sum()
    soft_union = score.sum() + truth.sum() - soft_intersection
    scores_np = flat.numpy()
    truth_flat = truth.flatten().numpy()
    return {
        "tp": intersection,
        "fp": int((full_pred & ~full_truth).sum()),
        "fn": int((~full_pred & full_truth).sum()),
        "tn": int((~full_pred & ~full_truth).sum()),
        "iou": intersection / union if union else 1.0,
        "iou_at_64": float((pred64 & truth).sum() / (pred64 | truth).sum().clamp_min(1)),
        "soft_iou": float(soft_intersection / soft_union.clamp_min(1e-12)),
        "average_precision": _average_precision(scores_np, truth_flat),
        "roc_auc": _roc_auc(scores_np, truth_flat),
        "entropy": float(normalized_entropy),
        "effective_support_fraction": float(torch.exp(raw_entropy) / flat.numel()),
        "top_1pct_mass": float(top_mass),
        "foreground_mean": float(fg_mean),
        "background_mean": float(bg_mean),
        "foreground_background_gap": float(fg_mean - bg_mean),
        "standardized_gap": float((fg_mean - bg_mean) / pooled_std),
        "total_variation": float((dx.mean() + dy.mean()) / 2),
        "boundary_gradient_ratio": float(
            gradient[boundary_t].mean() / gradient.mean().clamp_min(1e-12)
        ),
        "centroid_error": centroid_error,
        "predicted_area_fraction": float(full_pred.float().mean()),
    }


@torch.inference_mode()
def propagate_module_window(
    sample: Any,
    module_name: str,
    seed_map: torch.Tensor,
    device: str,
    steps: Sequence[int] = tuple(range(15, 31)),
    iterations: int = 4,
) -> torch.Tensor:
    """Apply one SA module over a fixed step window and average its heads."""
    resolution = next(
        size for size, names in sample.self_modules_by_resolution.items() if module_name in names
    )
    seed = resize_map(seed_map, resolution).flatten().to(device=device, dtype=torch.float32)
    output = torch.zeros(resolution * resolution, dtype=torch.float32)
    observations = 0
    for step_index in steps:
        if step_index >= len(sample.timesteps):
            continue
        with sample.open_attention(step_index) as tensors:
            query_tensor = tensors.get_tensor(f"{module_name}.q")
            key_tensor = tensors.get_tensor(f"{module_name}.k")
            branch = 1 if query_tensor.shape[0] > 1 else 0
            query = query_tensor[branch]
            key = key_tensor[branch]
        scale = query.shape[-1] ** -0.5
        for head in range(query.shape[0]):
            q = query[head].to(device=device, dtype=torch.float32)
            k = key[head].to(device=device, dtype=torch.float32)
            affinity = torch.softmax(torch.matmul(q, k.transpose(0, 1)) * scale, dim=-1)
            current = seed
            for _ in range(iterations):
                current = torch.matmul(affinity, current)
            output += current.cpu()
            observations += 1
            del q, k, affinity, current
        del query, key
    if observations == 0:
        raise RuntimeError(f"No attention observations for {module_name}")
    return (output / observations).reshape(resolution, resolution)


@torch.inference_mode()
def analyze_attention_sample(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: dict[str, Sequence[int]],
    implemented_ovam_map: torch.Tensor,
    implemented_ovam_prediction: torch.Tensor,
    released_ovam_map: torch.Tensor,
    released_ovam_prediction: torch.Tensor,
    device: str = "cuda",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Profile one image under the baseline and the selected winning pipeline."""
    target = sample.load_target_mask()
    ovam_raw = ovam_cross_attention_map(sample)
    modules, module_resolutions, layer_heads = build_cross_attention_layer_heads(sample, tokenizer)
    seeds = build_cross_attention_seeds_from_heads(
        modules, module_resolutions, layer_heads, fixed_head_rankings, ovam_raw
    )
    top4 = seeds["top4_heads_percentile"]
    best12 = seeds["best12_percentile"]
    propagated = propagate_selected_module(
        sample,
        WINNING_MODULE,
        torch.stack((top4.flatten(), best12.flatten()), dim=1),
        device,
        track_seed_column=0,
        track_step_heads=True,
    )
    assert propagated.per_step_head is not None
    joint4 = propagated.per_step_head[4].reshape(len(sample.timesteps), 8, 64, 64)
    kept_heads = (0, 1, 2, 4, 5, 6)
    winner_map = joint4[15:, list(kept_heads)].mean(dim=(0, 1))
    winner_pred = attention_to_mask(winner_map, target.shape, 0.45, "percentile_01_99")
    winner_pred = torch.from_numpy(
        postprocess_mask(winner_pred.numpy(), "remove_fraction_0.01")
    )
    joint3 = propagated.per_step_head[3].reshape(len(sample.timesteps), 8, 64, 64)
    nonpost_map = joint3[15:21, list(kept_heads)].mean(dim=(0, 1))
    nonpost_pred = attention_to_mask(nonpost_map, target.shape, 0.40, "percentile_01_99")
    prior_map = propagated.aggregate[2][:, 0].reshape(64, 64)
    prior_pred = attention_to_mask(prior_map, target.shape, 0.45, "percentile_01_99")
    old_map = propagated.aggregate[4][:, 1].reshape(64, 64)
    old_pred = attention_to_mask(old_map, target.shape, 0.45, "percentile_01_99")
    ovam_pred = attention_to_mask(ovam_raw, target.shape, 0.40, "ovam_max_after_resize")

    selected = {
        "ovam_implemented": map_statistics(
            implemented_ovam_map, target, implemented_ovam_prediction, threshold=0.40
        ),
        "ovam_published": map_statistics(
            released_ovam_map, target, released_ovam_prediction, threshold=0.40
        ),
        "ovam_cross_attention_only": map_statistics(ovam_raw, target, ovam_pred),
        "top4_ca_seed": map_statistics(top4, target),
        "best12_ca_seed": map_statistics(best12, target),
        "former_best_0.7777": map_statistics(old_map, target, old_pred),
        "focused_best_0.7922": map_statistics(prior_map, target, prior_pred),
        "joint_best_without_postprocess_0.7991": map_statistics(
            nonpost_map, target, nonpost_pred, threshold=0.40
        ),
        "joint_best_0.7997": map_statistics(winner_map, target, winner_pred),
    }
    ca_layers = {}
    for index, module_name in enumerate(modules):
        ca_layers[module_name] = {
            "index": index,
            "resolution": module_resolutions[index],
            **map_statistics(layer_heads[index].mean(0), target),
        }

    sa_heads = {}
    for head in range(8):
        value = joint4[15:, head].mean(0)
        sa_heads[str(head)] = map_statistics(value, target)
    sa_steps = {}
    for step in range(len(sample.timesteps)):
        value = joint4[step, list(kept_heads)].mean(0)
        sa_steps[str(step)] = {
            "scheduler_timestep": int(sample.timesteps[step]),
            **map_statistics(value, target),
        }

    sa_modules = {}
    for resolution, names in sorted(sample.self_modules_by_resolution.items()):
        for module_name in names:
            value = propagate_module_window(sample, module_name, top4, device)
            sa_modules[module_name] = {
                "resolution": resolution,
                **map_statistics(value, target),
            }

    record = {
        "image_id": sample.sample_id,
        "classname": sample.classname,
        "prompt": sample.prompt,
        "geometry": object_geometry(target),
        "selected_methods": selected,
        "cross_attention_layers": ca_layers,
        "self_attention_modules": sa_modules,
        "winning_module_heads": sa_heads,
        "winning_module_steps": sa_steps,
    }
    display = {
        "ovam": normalize_map(implemented_ovam_map, "max"),
        "ovam_ca_reproduced": normalize_map(ovam_raw, "max"),
        "top4_ca": normalize_map(top4, "percentile_01_99"),
        "focused": normalize_map(prior_map, "percentile_01_99"),
        "winner": normalize_map(winner_map, "percentile_01_99"),
        "winner_mask": winner_pred.float(),
    }
    return record, display
