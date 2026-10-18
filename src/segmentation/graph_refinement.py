"""Graph refinements used by the SG1/SG3 controlled experiment.

The operators accept a row-stochastic spatial affinity and one or more CA
seeds.  They keep the graph state on the active device and return only maps;
callers are responsible for discarding the affinity after scoring.
"""

from __future__ import annotations

import torch


def minmax_vector(seed: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each row of ``seed`` shaped ``[N, P]``."""
    if seed.ndim != 2:
        raise ValueError("seed must have shape [N, P]")
    low = seed.amin(dim=1, keepdim=True)
    scale = seed.amax(dim=1, keepdim=True) - low
    return torch.where(scale > 0, (seed - low) / scale.clamp_min(1e-12), torch.zeros_like(seed))


def anchor_masks(
    seed: torch.Tensor, foreground_quantile: float, background_quantile: float,
    side: int, border_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint confident foreground/background masks for ``[N,P]`` seeds.

    Foreground anchors take precedence when a high-confidence object pixel
    touches the optional image-border background band.
    """
    if not 0 <= background_quantile < foreground_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= background < foreground <= 1")
    if seed.shape[1] != side * side or border_width < 0 or border_width * 2 >= side:
        raise ValueError("invalid spatial shape or border width")
    fg_cut = torch.quantile(seed, foreground_quantile, dim=1, keepdim=True)
    bg_cut = torch.quantile(seed, background_quantile, dim=1, keepdim=True)
    foreground = seed >= fg_cut
    background = seed <= bg_cut
    if border_width:
        border = torch.zeros((side, side), dtype=torch.bool, device=seed.device)
        border[:border_width] = border[-border_width:] = True
        border[:, :border_width] = border[:, -border_width:] = True
        background = background | border.reshape(1, -1)
    background &= ~foreground
    return foreground, background


@torch.inference_mode()
def confidence_locked_trajectory(
    affinity: torch.Tensor, seed: torch.Tensor, foreground: torch.Tensor,
    background: torch.Tensor, steps: int = 2,
) -> torch.Tensor:
    """SG3: propagate while restoring the original values at confident pixels."""
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    current = seed.clone()
    output = []
    for _ in range(steps):
        current = torch.matmul(affinity, current.T).T
        current = torch.where(foreground | background, seed, current)
        output.append(current.clone())
    return torch.stack(output)


@torch.inference_mode()
def absorbing_walk(
    affinity: torch.Tensor, seed: torch.Tensor, foreground: torch.Tensor,
    background: torch.Tensor, tolerance: float = 1e-4, max_iterations: int = 200,
) -> tuple[torch.Tensor, int, float]:
    """SG1: solve the foreground-before-background hitting probability iteratively.

    Foreground and background anchors are clamped to one and zero.  Uncertain
    pixels repeatedly receive the row-stochastic graph average until the
    registered tolerance or iteration limit is reached.
    """
    current = seed.clone()
    current[foreground] = 1.0
    current[background] = 0.0
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        updated = torch.matmul(affinity, current.T).T
        updated[foreground] = 1.0
        updated[background] = 0.0
        residual = float((updated - current).abs().amax())
        current = updated
        if residual < tolerance:
            break
    return current, iteration, residual
