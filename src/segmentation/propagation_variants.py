"""Reusable self-attention propagation operators for controlled experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.inference_mode()
def power_trajectory(
    affinity: torch.Tensor, seed: torch.Tensor, steps: int
) -> torch.Tensor:
    """Return ``A^n seed`` for n=1..steps.

    ``affinity`` may have leading batch dimensions and ends in ``[P, P]``.
    ``seed`` is either ``[P]`` or has the same leading dimensions plus ``[P]``.
    """
    if affinity.ndim < 2 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError("affinity must end in a square [P, P] matrix")
    if seed.shape[-1] != affinity.shape[-1]:
        raise ValueError("affinity and seed pixel counts differ")
    if steps < 1:
        raise ValueError("steps must be positive")
    current = seed
    output = []
    for _ in range(steps):
        current = torch.matmul(affinity, current.unsqueeze(-1)).squeeze(-1)
        output.append(current)
    return torch.stack(output)


@torch.inference_mode()
def restart_trajectory(
    affinity: torch.Tensor,
    seed: torch.Tensor,
    betas: Sequence[float],
    steps: int,
) -> torch.Tensor:
    """Return restart propagation for iterations 1..steps.

    The output is ``[steps, *batch, beta, P]`` and applies
    ``x <- (1-beta) A x + beta seed`` from ``x_0 = seed``.
    """
    if affinity.ndim < 2 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError("affinity must end in a square [P, P] matrix")
    if seed.shape[-1] != affinity.shape[-1]:
        raise ValueError("affinity and seed pixel counts differ")
    if not betas or any(beta < 0 or beta > 1 for beta in betas):
        raise ValueError("betas must be nonempty and lie in [0, 1]")
    beta = torch.as_tensor(tuple(betas), device=affinity.device, dtype=affinity.dtype)
    batch_shape = affinity.shape[:-2]
    expanded_seed = seed.expand(*batch_shape, seed.shape[-1])
    current = expanded_seed.unsqueeze(-2).expand(*batch_shape, len(betas), seed.shape[-1]).clone()
    beta_shape = (1,) * len(batch_shape) + (len(betas), 1)
    beta = beta.reshape(beta_shape)
    restart = expanded_seed.unsqueeze(-2)
    output = []
    for _ in range(steps):
        propagated = torch.matmul(affinity, current.transpose(-1, -2)).transpose(-1, -2)
        current = (1.0 - beta) * propagated + beta * restart
        output.append(current)
    return torch.stack(output)


@torch.inference_mode()
def sparse_one_pass(
    affinity: torch.Tensor, seed: torch.Tensor, topk: Sequence[int]
) -> torch.Tensor:
    """Propagate through row-wise top-k affinities, renormalized per row.

    Returns ``[len(topk), *batch, P]``. Values of k must be smaller than P;
    use the ordinary dense matvec for k=P.
    """
    pixels = affinity.shape[-1]
    choices = tuple(sorted(set(int(k) for k in topk)))
    if not choices or choices[0] < 1 or choices[-1] >= pixels:
        raise ValueError("topk values must be unique integers in [1, P-1]")
    values, indices = torch.topk(affinity, choices[-1], dim=-1, sorted=True)
    selected_seed = seed[indices]
    numerator = (values * selected_seed).cumsum(dim=-1)
    denominator = values.cumsum(dim=-1)
    maps = [numerator[..., k - 1] / denominator[..., k - 1].clamp_min(1e-12) for k in choices]
    return torch.stack(maps)
