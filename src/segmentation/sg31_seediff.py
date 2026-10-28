"""Locked SeeDiff-inspired re-seeding and background controls for SG31.

SA rows are seed-to-destination affinities. This is deliberately a different
orientation from the historical A @ c continuous propagation controls.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy import ndimage

from src.segmentation import controlled_cv as cv
from src.segmentation.sg25_sa import _attention, _apply, _resize

NAMES = tuple(f"S{i}" for i in range(11))
SEED_FIELDS = ("count", "fraction", "empty_fallback", "components", "expanded_area", "overlap_previous")
BG_FIELDS = ("bg_precision", "bg_inside_gt", "bg_outside_gt", "bg_seed_fraction")


def norm(map_: torch.Tensor) -> torch.Tensor:
    return cv.normalize_batch(map_.reshape(1, *map_.shape[-2:]), "minmax")[0].reshape(map_.shape)


def seeds(map_: torch.Tensor, alpha: float = .5) -> tuple[torch.Tensor, bool]:
    flat = map_.reshape(-1)
    selected = flat >= alpha
    fallback = not bool(selected.any())
    if fallback:
        selected[int(flat.argmax())] = True
    return selected, fallback


def expand(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean selected SA rows, or nonnegative row-weighted soft control."""
    if rows.ndim != 2 or weights.shape != (rows.shape[0],):
        raise ValueError("SA/seed dimensions differ")
    weights = weights.float().clamp_min(0)
    if not bool(weights.sum() > 0):
        weights[int(weights.argmax())] = 1
    return weights @ rows / weights.sum()


def _seed_stats(selected, fallback, expanded, previous):
    side = int(selected.numel() ** .5)
    components = ndimage.label(selected.reshape(side, side).cpu().numpy())[1]
    if previous is None:
        overlap = float("nan")
    else:
        prior = F.interpolate(previous.float().reshape(1, 1, side // 2, side // 2),
                              size=(side, side), mode="nearest").reshape(-1).bool()
        overlap = float((prior & selected).sum() / (prior | selected).sum().clamp_min(1))
    return [int(selected.sum()), float(selected.float().mean()), int(fallback),
            int(components), float((norm(expanded.reshape(side, side)) >= .5).float().mean()), overlap]


def _path(attn, seed, sides, hard):
    state = None
    previous = None
    stats = torch.full((3, len(SEED_FIELDS)), float("nan"))
    for side in sides:
        if state is None:
            state = norm(_resize(seed.reshape(-1), 64, side).reshape(side, side)).reshape(-1)
        else:
            state = norm(_resize(norm(state.reshape(previous_side, previous_side)).reshape(-1),
                                 previous_side, side).reshape(side, side)).reshape(-1)
        if hard:
            chosen, fallback = seeds(state)
            weights = chosen.float()
        else:
            chosen, fallback = None, False
            weights = state
        state = expand(attn[side], weights)
        if hard:
            stats[{16: 0, 32: 1, 64: 2}[side]] = torch.tensor(
                _seed_stats(chosen, fallback, state, previous))
            previous = chosen
        previous_side = side
    return state.reshape(64, 64), stats


@torch.inference_mode()
def aggregate(sa_qk: dict[int, list], seed: torch.Tensor):
    """All SG31 maps and compact per-call seed/background support maps.

    Background support maps are returned only within the image scope for
    post-hoc GT diagnostics and must never be persisted.
    """
    if seed.shape != (64, 64) or any(len(sa_qk.get(s, [])) != 11 for s in (16, 32, 64)):
        raise ValueError("SG31 requires A3 seed and 11 same-call SA observations per scale")
    c = seed.float().reshape(-1)
    sums = {name: torch.zeros((64, 64), device=seed.device) for name in NAMES[:6]}
    seed_stats = torch.full((11, 2, 3, len(SEED_FIELDS)), float("nan"))
    for call in range(11):
        a = {s: _attention(sa_qk[s][call], s) for s in (16, 32, 64)}
        mean = {s: a[s].mean(0) for s in a}
        # S0 and S1 use the exact SG25 per-head implementation.
        first64 = _apply(a[64], c)
        sums["S0"] += _apply(a[64], first64).mean(0).reshape(64, 64) / 11
        first16 = _apply(a[16], _resize(c, 64, 16).reshape(-1)).mean(0)
        sums["S1"] += _apply(a[64], _resize(first16, 16, 64).reshape(-1)).mean(0).reshape(64, 64) / 11
        middle = _apply(a[32], _resize(first16, 16, 32).reshape(-1)).mean(0)
        sums["S2"] += _apply(a[64], _resize(middle, 32, 64).reshape(-1)).mean(0).reshape(64, 64) / 11
        for name, sides, hard, slot in (("S3", (16, 32, 64), True, 0),
                                         ("S4", (32, 64), True, 1),
                                         ("S5", (16, 32, 64), False, 0)):
            value, stats = _path(mean, seed, sides, hard)
            sums[name] += value / 11
            if hard:
                seed_stats[call, slot] = stats
        del a, mean
    maps = {name: norm(value) for name, value in sums.items()}
    bg_sources = (("S6", "S0", "rows"), ("S7", "S0", "p2"),
                  ("S8", "S3", "rows"), ("S9", "S4", "rows"))
    bg_seed = {name: seeds(1 - maps[source], .5)[0].float()
               for name, source, _ in bg_sources}
    bg_sums = {name: torch.zeros((64*64,), device=seed.device) for name in bg_seed}
    # Recompute each fine matrix once; do not retain eleven dense 4096² graphs.
    for call in range(11):
        a64 = _attention(sa_qk[64][call], 64)
        avg = a64.mean(0)
        for name, _, style in bg_sources:
            if style == "rows":
                bg_sums[name] += expand(avg, bg_seed[name]) / 11
            else:
                bg_sums[name] += _apply(a64, _apply(a64, bg_seed[name])).mean(0) / 11
        del a64, avg
    support = {}
    for name, source, _ in bg_sources:
        bg = norm(bg_sums[name].reshape(64, 64))
        maps[name] = norm(maps[source] * (1 - bg))
        support[name] = (bg, float(bg_seed[name].mean()))
    maps["S10"] = norm(maps["S0"].square())
    return maps, seed_stats, support
