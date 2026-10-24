"""Observation-wise SG26 coarse affinity gating of fine self-attention."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from src.segmentation.sg25_sa import _apply, _attention, _resize

NAMES = tuple(f"S{i}" for i in range(7))
EDGE_FIELDS = ("gate_mean", "gate_median", "fraction_ratio_below_0p5",
               "fraction_ratio_0p9_to_1p1", "row_entropy_before", "row_entropy_after",
               "effective_neighbors_before", "effective_neighbors_after")
EDGE_GROUPS = ("object_to_object", "object_to_background",
               "background_to_object", "background_to_background")
EDGE_GROUP_FIELDS = ("gate_mean", "fine_probability_before", "fine_probability_after")


def coarse_index(side: int, device: torch.device) -> torch.Tensor:
    if side not in (16, 32) or 64 % side:
        raise ValueError("SG26 only defines 16 and 32 coarse grids")
    pos = torch.arange(4096, device=device)
    factor = 64 // side
    return (pos // 64 // factor) * side + (pos % 64 // factor)


def lift_gate(coarse: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Block-lift a row-stochastic coarse matrix without interpolation or weights."""
    if coarse.shape[0] != coarse.shape[1] or int(index.max()) >= coarse.shape[0]:
        raise ValueError("coarse affinity/index mismatch")
    return coarse.index_select(0, index).index_select(1, index)


def gated_fine(fine: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise gate and row-normalize each fine head independently."""
    if fine.ndim != 3 or fine.shape[1:] != gate.shape:
        raise ValueError("fine attention and lifted gate sizes differ")
    result = fine.clone()
    result.mul_(gate.unsqueeze(0))
    denominator = result.sum(-1)
    if not torch.all(denominator > 0):
        raise ValueError("coarse gate produced an empty fine attention row")
    result.div_(denominator.unsqueeze(-1))
    return result, denominator


def diagnostic_indices(truth64: np.ndarray, image_id: str, device: torch.device) -> tuple[torch.Tensor, bool, bool]:
    """Balanced fixed GT sample used only for post-hoc edge diagnostics."""
    foreground = np.flatnonzero(truth64.reshape(-1))
    background = np.flatnonzero(~truth64.reshape(-1))
    has_foreground, has_background = bool(len(foreground)), bool(len(background))
    if not (has_foreground or has_background):
        raise ValueError("empty edge diagnostic image")
    if not has_foreground:
        foreground = background
    if not has_background:
        background = foreground
    seed = int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    indices = np.r_[rng.choice(foreground, 64, replace=len(foreground) < 64),
                    rng.choice(background, 64, replace=len(background) < 64)]
    return torch.as_tensor(indices, dtype=torch.long, device=device), has_foreground, has_background


def edge_diagnostics(fine: torch.Tensor, gated: torch.Tensor, gate: torch.Tensor,
                     denominator: torch.Tensor, indices: torch.Tensor,
                     has_foreground: bool, has_background: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reduced graph diagnostics on the same balanced GT edge sample."""
    sampled_gate = gate.index_select(0, indices).index_select(1, indices)
    sampled_ratio = sampled_gate.unsqueeze(0) / denominator.index_select(1, indices).unsqueeze(-1)
    before_rows = fine.index_select(1, indices)
    after_rows = gated.index_select(1, indices)
    entropy_before = -(before_rows * before_rows.clamp_min(1e-30).log()).sum(-1)
    entropy_after = -(after_rows * after_rows.clamp_min(1e-30).log()).sum(-1)
    scalar = np.array((float(sampled_gate.mean()), float(sampled_gate.median()),
                       float((sampled_ratio < .5).float().mean()),
                       float(((sampled_ratio >= .9) & (sampled_ratio <= 1.1)).float().mean()),
                       float(entropy_before.mean()), float(entropy_after.mean()),
                       float(entropy_before.exp().mean()), float(entropy_after.exp().mean())),
                      dtype=np.float64)
    before = before_rows.index_select(2, indices)
    after = after_rows.index_select(2, indices)
    slices = ((slice(0,64),slice(0,64)), (slice(0,64),slice(64,128)),
              (slice(64,128),slice(0,64)), (slice(64,128),slice(64,128)))
    groups = np.array([(float(sampled_gate[q,k].mean()),
                        float(before[:,q,k].mean()), float(after[:,q,k].mean()))
                       for q,k in slices], dtype=np.float64)
    if not has_foreground:
        groups[:3] = np.nan
    if not has_background:
        groups[1:] = np.nan
    return scalar, groups


@torch.inference_mode()
def aggregate(sa_qk: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
              seed: torch.Tensor, truth64: np.ndarray, image_id: str):
    """Seven raw maps and compact edge diagnostics, without retaining matrices."""
    if seed.shape != (64, 64) or np.asarray(truth64).shape != (64,64):
        raise ValueError("SG26 requires one 64x64 seed and GT diagnostic mask")
    if any(len(sa_qk.get(side, [])) != 11 for side in (16,32,64)):
        raise ValueError("SG26 requires 11 matching calls per SA resolution")
    c64 = seed.float().reshape(-1)
    c16, c32 = (_resize(c64, 64, side).reshape(-1) for side in (16,32))
    # GT determines only which edges are inspected after inference. None of
    # the seven propagated maps below depend on these diagnostic indices.
    indices,has_foreground,has_background = diagnostic_indices(np.asarray(truth64, bool), image_id, seed.device)
    coarse_indices = {side: coarse_index(side, seed.device) for side in (16,32)}
    sums = torch.zeros((7,64,64), dtype=torch.float32, device=seed.device)
    edge_sums = torch.zeros((2,len(EDGE_FIELDS)), dtype=torch.float64)
    group_sums = torch.zeros((2,4,len(EDGE_GROUP_FIELDS)), dtype=torch.float64)
    for call in range(11):
        a16 = _attention(sa_qk[16][call],16)
        a32 = _attention(sa_qk[32][call],32)
        a64 = _attention(sa_qk[64][call],64)
        first16 = _apply(a16,c16).mean(0)
        first32 = _apply(a32,c32).mean(0)
        first64 = _apply(a64,c64)
        sums[0] += _apply(a64,first64).sum(0).reshape(64,64)
        sums[1] += _apply(a64,_resize(first16,16,64).reshape(-1)).sum(0).reshape(64,64)
        sums[2] += _apply(a64,_resize(first32,32,64).reshape(-1)).sum(0).reshape(64,64)
        for side,coarse,scale_index,both_index,second_index in (
                (16,a16,0,3,5),(32,a32,1,4,6)):
            gate = lift_gate(coarse.mean(0), coarse_indices[side])
            gated, denominator = gated_fine(a64,gate)
            sums[both_index] += _apply(gated,_apply(gated,c64)).sum(0).reshape(64,64)
            sums[second_index] += _apply(gated,first64).sum(0).reshape(64,64)
            scalar,groups = edge_diagnostics(a64,gated,gate,denominator,indices,
                                             has_foreground,has_background)
            edge_sums[scale_index] += torch.as_tensor(scalar)
            group_sums[scale_index] += torch.as_tensor(groups)
            del gate, gated, denominator
        del a16,a32,a64,first16,first32,first64
    maps = {name:sums[i]/88 for i,name in enumerate(NAMES)}
    return maps, (edge_sums/11).numpy(), (group_sums/11).numpy()
