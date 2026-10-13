"""Frozen best VOC-Sim method and its cumulative presentation stages.

All choices in this module were selected on VOC-Sim.  Keeping them in one
place prevents the COCO-Cap generalization run from adapting any parameter to
the evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .attention_aggregation import attention_to_mask, normalize_map
from .focused_self_attention import (
    build_cross_attention_layer_heads,
    build_cross_attention_seeds_from_heads,
    propagate_selected_module,
)
from .ovam_baseline import ovam_cross_attention_map
from .postprocessing import postprocess_mask
from .stage1_suite import BEST_12_INDICES


BEST_SA_MODULE = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
BEST_SA_HEADS = (0, 1, 2, 4, 5, 6)
BEST_LATE_STEP_START = 15
BEST_ITERATIONS = 4
BEST_THRESHOLD = 0.45
BEST_NORMALIZATION = "percentile_01_99"
BEST_POSTPROCESSING = "remove_fraction_0.01"

COMPONENT_ABLATION_LABELS = {
    "full_system": "Full system",
    "without_ca_layer_filter": "Without CA layer filtering",
    "without_ca_head_filter": "Without CA head filtering",
    "without_sa_propagation": "Without SA propagation",
    "one_sa_pass": "One SA pass instead of four",
    "all_sa_calls": "All SA calls instead of calls 15--30",
    "all_sa_heads": "All SA heads",
    "max_normalization": "Max normalization instead of percentile 1--99",
    "without_component_cleanup": "Without 1% component cleanup",
}

REVISED_COMPONENT_ABLATION_LABELS = {
    "revised_full_system": "Revised full system: one SA pass, all CA layers",
    "without_ca_head_filter": "Without CA head filtering",
    "without_sa_propagation": "Without SA propagation",
    "all_sa_calls": "All SA calls instead of calls 15--30",
    "all_sa_heads": "All SA heads",
    "max_normalization": "Max normalization instead of percentile 1--99",
    "without_component_cleanup": "Without 1% component cleanup",
}


@dataclass(frozen=True)
class MethodStage:
    """One cumulative stage in the route from raw CA to the final method."""

    key: str
    label: str
    score: torch.Tensor
    prediction: torch.Tensor


def _prediction(score: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return attention_to_mask(score, shape, BEST_THRESHOLD, BEST_NORMALIZATION)


def _configured_prediction(
    score: torch.Tensor,
    shape: tuple[int, int],
    normalization: str = BEST_NORMALIZATION,
    postprocessing: str = BEST_POSTPROCESSING,
) -> torch.Tensor:
    prediction = attention_to_mask(score, shape, BEST_THRESHOLD, normalization)
    if postprocessing != "none":
        prediction = torch.from_numpy(
            postprocess_mask(prediction.cpu().numpy(), postprocessing)
        )
    return prediction.bool()


def _stage(
    key: str,
    label: str,
    score: torch.Tensor,
    shape: tuple[int, int],
    postprocessing: str = "none",
) -> MethodStage:
    prediction = _prediction(score, shape)
    if postprocessing != "none":
        prediction = torch.from_numpy(
            postprocess_mask(prediction.cpu().numpy(), postprocessing)
        )
    return MethodStage(key, label, score.detach().cpu(), prediction.bool())


@torch.inference_mode()
def frozen_best_stages(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: Mapping[str, Sequence[int]],
    device: str = "cuda",
) -> tuple[MethodStage, ...]:
    """Return cumulative stages ending at the frozen best VOC-Sim method.

    The returned scores remain small 64x64 tensors.  Predictions are resized
    to the annotation shape and are suitable for metric accumulation.
    """
    target_shape = tuple(sample.load_target_mask().shape)
    modules, module_resolutions, layer_heads = build_cross_attention_layer_heads(
        sample, tokenizer
    )
    seeds = build_cross_attention_seeds_from_heads(
        modules,
        module_resolutions,
        layer_heads,
        dict(fixed_head_rankings),
        ovam_cross_attention_map(sample),
    )
    all16 = seeds["all16_percentile"]
    best12 = seeds["best12_percentile"]
    top4 = seeds["top4_heads_percentile"]

    if BEST_SA_MODULE not in sample.self_modules_by_resolution.get(64, ()):
        raise ValueError(f"Required self-attention module is absent: {BEST_SA_MODULE}")
    propagated = propagate_selected_module(
        sample,
        BEST_SA_MODULE,
        top4.flatten()[:, None],
        device,
        track_seed_column=0,
        track_step_heads=True,
    )
    if propagated.per_step_head is None:
        raise RuntimeError("Per-step/per-head propagation was not recorded")
    one = propagated.per_step_head[1].reshape(len(sample.timesteps), 8, 64, 64)
    four = propagated.per_step_head[BEST_ITERATIONS].reshape(
        len(sample.timesteps), 8, 64, 64
    )
    if len(sample.timesteps) <= BEST_LATE_STEP_START:
        raise ValueError(
            f"Need more than {BEST_LATE_STEP_START} denoising calls; "
            f"found {len(sample.timesteps)}"
        )

    one_all = one.mean(dim=(0, 1))
    four_all = four.mean(dim=(0, 1))
    four_late = four[BEST_LATE_STEP_START:].mean(dim=(0, 1))
    final_score = four[BEST_LATE_STEP_START:, list(BEST_SA_HEADS)].mean(dim=(0, 1))

    return (
        _stage("all_ca", "All 16 CA layers", all16, target_shape),
        _stage("selected_ca_layers", "Select 12 CA layers", best12, target_shape),
        _stage("selected_ca_heads", "Select top 4 CA heads/layer", top4, target_shape),
        _stage("sa_one_iteration", "+ final 64x64 SA (1 pass)", one_all, target_shape),
        _stage("sa_four_iterations", "+ four SA passes", four_all, target_shape),
        _stage("late_sa_steps", "+ SA denoising calls 15-end", four_late, target_shape),
        _stage("selected_sa_heads", "+ SA heads 0,1,2,4,5,6", final_score, target_shape),
        _stage(
            "final_cleanup",
            "+ remove components under 1%",
            final_score,
            target_shape,
            BEST_POSTPROCESSING,
        ),
    )


@torch.inference_mode()
def frozen_component_ablations(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: Mapping[str, Sequence[int]],
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Evaluate the final method and one-change component controls.

    All variants retain the final threshold and component cleanup except where
    that exact operation is being ablated.  For the layer-filter control, the
    four excluded layers are restored using their all-head mean because fixed
    top-four rankings were learned only for the selected 12 layers.
    """
    target_shape = tuple(sample.load_target_mask().shape)
    modules, module_resolutions, layer_heads = build_cross_attention_layer_heads(
        sample, tokenizer
    )
    seeds = build_cross_attention_seeds_from_heads(
        modules,
        module_resolutions,
        layer_heads,
        dict(fixed_head_rankings),
        ovam_cross_attention_map(sample),
    )
    selected = set(BEST_12_INDICES)
    restored_layer_maps = []
    for index in range(len(modules)):
        if index in selected:
            ranking = fixed_head_rankings.get(str(index))
            if ranking is None or len(ranking) < 4:
                raise ValueError(f"Missing fixed top-four head ranking for CA layer {index}")
            restored_layer_maps.append(
                layer_heads[index, [int(value) for value in ranking[:4]]].mean(dim=0)
            )
        else:
            restored_layer_maps.append(layer_heads[index].mean(dim=0))
    restored_all16 = normalize_map(
        torch.stack(restored_layer_maps).mean(dim=0), BEST_NORMALIZATION
    )

    seed_names = ("top4", "selected12_all_heads", "restored_all16")
    seed_maps = (
        seeds["top4_heads_percentile"],
        seeds["best12_percentile"],
        restored_all16,
    )
    seed_matrix = torch.stack([value.flatten() for value in seed_maps], dim=1)
    propagated = propagate_selected_module(
        sample,
        BEST_SA_MODULE,
        seed_matrix,
        device,
        track_seed_columns=tuple(range(len(seed_names))),
    )
    if propagated.per_step_head_columns is None:
        raise RuntimeError("Per-step, per-head, per-seed propagation was not recorded")
    step_count = len(sample.timesteps)
    if step_count <= BEST_LATE_STEP_START:
        raise ValueError(
            f"Need more than {BEST_LATE_STEP_START} denoising calls; found {step_count}"
        )

    def score(iteration: int, seed_index: int, late: bool, selected_heads: bool) -> torch.Tensor:
        values = propagated.per_step_head_columns[iteration].reshape(
            step_count, 8, 64, 64, len(seed_names)
        )[..., seed_index]
        if late:
            values = values[BEST_LATE_STEP_START:]
        if selected_heads:
            values = values[:, list(BEST_SA_HEADS)]
        return values.mean(dim=(0, 1))

    final_score = score(BEST_ITERATIONS, 0, True, True)
    controls = {
        "full_system": _configured_prediction(final_score, target_shape),
        "without_ca_layer_filter": _configured_prediction(
            score(BEST_ITERATIONS, 2, True, True), target_shape
        ),
        "without_ca_head_filter": _configured_prediction(
            score(BEST_ITERATIONS, 1, True, True), target_shape
        ),
        "without_sa_propagation": _configured_prediction(seed_maps[0], target_shape),
        "one_sa_pass": _configured_prediction(score(1, 0, True, True), target_shape),
        "all_sa_calls": _configured_prediction(
            score(BEST_ITERATIONS, 0, False, True), target_shape
        ),
        "all_sa_heads": _configured_prediction(
            score(BEST_ITERATIONS, 0, True, False), target_shape
        ),
        "max_normalization": _configured_prediction(
            final_score, target_shape, normalization="max"
        ),
        "without_component_cleanup": _configured_prediction(
            final_score, target_shape, postprocessing="none"
        ),
    }
    if tuple(controls) != tuple(COMPONENT_ABLATION_LABELS):
        raise RuntimeError("Component-ablation definitions and labels differ")
    return controls


@torch.inference_mode()
def frozen_one_pass_without_ca_layer_filter(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: Mapping[str, Sequence[int]],
    device: str = "cuda",
) -> torch.Tensor:
    """Return the combined one-pass, all-16-layer generalization candidate.

    The selected 12 layers retain their fixed top-four CA heads.  The four
    restored layers use their all-head mean because no fixed rankings were
    learned for layers excluded during VOC-Sim selection.  Every remaining
    final-system choice is unchanged.
    """
    target_shape = tuple(sample.load_target_mask().shape)
    modules, _, layer_heads = build_cross_attention_layer_heads(sample, tokenizer)
    selected = set(BEST_12_INDICES)
    restored_layer_maps = []
    for index in range(len(modules)):
        if index in selected:
            ranking = fixed_head_rankings.get(str(index))
            if ranking is None or len(ranking) < 4:
                raise ValueError(f"Missing fixed top-four head ranking for CA layer {index}")
            restored_layer_maps.append(
                layer_heads[index, [int(value) for value in ranking[:4]]].mean(dim=0)
            )
        else:
            restored_layer_maps.append(layer_heads[index].mean(dim=0))
    seed = normalize_map(
        torch.stack(restored_layer_maps).mean(dim=0), BEST_NORMALIZATION
    )
    propagated = propagate_selected_module(
        sample,
        BEST_SA_MODULE,
        seed.flatten()[:, None],
        device,
        track_seed_column=0,
        track_step_heads=True,
    )
    if propagated.per_step_head is None:
        raise RuntimeError("Per-step/per-head propagation was not recorded")
    step_count = len(sample.timesteps)
    if step_count <= BEST_LATE_STEP_START:
        raise ValueError(
            f"Need more than {BEST_LATE_STEP_START} denoising calls; found {step_count}"
        )
    values = propagated.per_step_head[1].reshape(step_count, 8, 64, 64)
    score = values[BEST_LATE_STEP_START:, list(BEST_SA_HEADS)].mean(dim=(0, 1))
    return _configured_prediction(score, target_shape)


@torch.inference_mode()
def frozen_revised_component_ablations(
    sample: Any,
    tokenizer: Any,
    fixed_head_rankings: Mapping[str, Sequence[int]],
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Leave-one-change controls around the revised generalization candidate.

    The revised system uses one SA pass and restores all CA layers.  The 12
    previously selected layers retain their frozen top-four heads; the four
    restored layers use all-head means because no rankings exist for them.
    """
    target_shape = tuple(sample.load_target_mask().shape)
    modules, _, layer_heads = build_cross_attention_layer_heads(sample, tokenizer)
    selected = set(BEST_12_INDICES)
    restored_layer_maps = []
    for index in range(len(modules)):
        if index in selected:
            ranking = fixed_head_rankings.get(str(index))
            if ranking is None or len(ranking) < 4:
                raise ValueError(f"Missing fixed top-four head ranking for CA layer {index}")
            restored_layer_maps.append(
                layer_heads[index, [int(value) for value in ranking[:4]]].mean(dim=0)
            )
        else:
            restored_layer_maps.append(layer_heads[index].mean(dim=0))
    revised_seed = normalize_map(
        torch.stack(restored_layer_maps).mean(dim=0), BEST_NORMALIZATION
    )
    all_heads_seed = normalize_map(layer_heads.mean(dim=(0, 1)), BEST_NORMALIZATION)
    seed_matrix = torch.stack(
        (revised_seed.flatten(), all_heads_seed.flatten()), dim=1
    )
    propagated = propagate_selected_module(
        sample,
        BEST_SA_MODULE,
        seed_matrix,
        device,
        track_seed_columns=(0, 1),
    )
    if propagated.per_step_head_columns is None:
        raise RuntimeError("Per-step, per-head, per-seed propagation was not recorded")
    step_count = len(sample.timesteps)
    if step_count <= BEST_LATE_STEP_START:
        raise ValueError(
            f"Need more than {BEST_LATE_STEP_START} denoising calls; found {step_count}"
        )
    values = propagated.per_step_head_columns[1].reshape(
        step_count, 8, 64, 64, 2
    )

    def score(seed_index: int, late: bool, selected_heads: bool) -> torch.Tensor:
        chosen = values[..., seed_index]
        if late:
            chosen = chosen[BEST_LATE_STEP_START:]
        if selected_heads:
            chosen = chosen[:, list(BEST_SA_HEADS)]
        return chosen.mean(dim=(0, 1))

    final_score = score(0, True, True)
    controls = {
        "revised_full_system": _configured_prediction(final_score, target_shape),
        "without_ca_head_filter": _configured_prediction(
            score(1, True, True), target_shape
        ),
        "without_sa_propagation": _configured_prediction(revised_seed, target_shape),
        "all_sa_calls": _configured_prediction(score(0, False, True), target_shape),
        "all_sa_heads": _configured_prediction(score(0, True, False), target_shape),
        "max_normalization": _configured_prediction(
            final_score, target_shape, normalization="max"
        ),
        "without_component_cleanup": _configured_prediction(
            final_score, target_shape, postprocessing="none"
        ),
    }
    if tuple(controls) != tuple(REVISED_COMPONENT_ABLATION_LABELS):
        raise RuntimeError("Revised component-ablation definitions and labels differ")
    return controls


def binary_iou(prediction: torch.Tensor | np.ndarray, target: np.ndarray) -> float:
    """Foreground IoU for one image."""
    pred = np.asarray(prediction.cpu() if isinstance(prediction, torch.Tensor) else prediction)
    pred = pred.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    return intersection / union if union else 1.0


def display_score(score: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Percentile-normalized full-size score used only for visualization."""
    import torch.nn.functional as F

    normalized = normalize_map(score.float(), BEST_NORMALIZATION)
    return F.interpolate(
        normalized[None, None], size=shape, mode="bilinear", align_corners=False
    )[0, 0].cpu()

