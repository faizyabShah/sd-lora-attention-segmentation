"""CA/SA fusion and calibration operators for the phase 3 registry (Section 1.2).

All operators take maps already min-max normalized to [0, 1]; ``c`` is the CA
seed and ``y`` the one-pass SA propagation of the frozen method.
"""

from __future__ import annotations

import torch


def minmax(maps: torch.Tensor) -> torch.Tensor:
    """Per-map min-max over the last two dimensions; constant maps become 0."""
    flat = maps.reshape(*maps.shape[:-2], -1)
    low = flat.amin(dim=-1, keepdim=True)
    scale = flat.amax(dim=-1, keepdim=True) - low
    out = torch.where(scale > 0, (flat - low) / torch.where(scale > 0, scale, torch.ones_like(scale)),
                      torch.zeros_like(flat))
    return out.reshape(maps.shape)


def residual_mix(y: torch.Tensor, c: torch.Tensor, lam: float) -> torch.Tensor:
    """P4: ``lam * y + (1 - lam) * c``; ``lam = 1`` is the reference."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    return lam * y + (1.0 - lam) * c


def gate(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """F3: keep only where both the seed and its propagation are high."""
    return c * y


def geometric_mean(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """F4: ``sqrt(c * y)``, a softer form of F3."""
    return torch.sqrt((c * y).clamp_min(0.0))


def max_normalize(maps: torch.Tensor) -> torch.Tensor:
    """N3 (OVAM): divide by the maximum, without subtracting the minimum."""
    flat = maps.reshape(*maps.shape[:-2], -1)
    peak = flat.amax(dim=-1, keepdim=True)
    out = torch.where(peak > 0, flat / torch.where(peak > 0, peak, torch.ones_like(peak)), torch.zeros_like(flat))
    return out.reshape(maps.shape)


def otsu_threshold(values: torch.Tensor, bins: int = 256) -> float:
    """N4: Otsu's threshold of values in [0, 1] (maximizes between-class variance)."""
    flat = values.reshape(-1).float().clamp(0, 1)
    hist = torch.histc(flat, bins=bins, min=0.0, max=1.0)
    prob = hist / hist.sum()
    centers = (torch.arange(bins, device=flat.device, dtype=torch.float32) + 0.5) / bins
    weight0 = prob.cumsum(0)
    mean0 = (prob * centers).cumsum(0)
    total_mean = mean0[-1]
    weight1 = 1.0 - weight0
    between = (total_mean * weight0 - mean0) ** 2 / (weight0 * weight1).clamp_min(1e-12)
    between[(weight0 <= 0) | (weight1 <= 0)] = -1.0
    index = int(torch.argmax(between))
    return float((index + 1) / bins)  # upper edge of the last background bin
