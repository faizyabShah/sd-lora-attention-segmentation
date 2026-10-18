"""Training-free reciprocal self-attention operators for SG9."""

from __future__ import annotations

import torch


OPERATORS = ("REF", "R1_geometric", "R2_minimum", "R3_soft_gate")


def configuration_names(tracks: tuple[str, ...] = ("W", "T")) -> tuple[str, ...]:
    return tuple(f"{track}/{operator}" for track in tracks for operator in OPERATORS)


def row_normalize(weights: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Normalize non-negative edge weights into a row-stochastic matrix."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)


@torch.inference_mode()
def reciprocal_affinity(affinity: torch.Tensor, operator: str, epsilon: float = 1e-12) -> torch.Tensor:
    """Construct and row-normalize one registered SG9 reciprocal operator."""
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    reverse = affinity.transpose(0, 1)
    if operator == "R1_geometric":
        weights = torch.sqrt((affinity * reverse).clamp_min(0))
    elif operator == "R2_minimum":
        weights = torch.minimum(affinity, reverse)
    elif operator == "R3_soft_gate":
        weights = affinity * torch.sqrt(reverse.clamp_min(0))
    else:
        raise ValueError(f"unknown reciprocal operator: {operator}")
    return row_normalize(weights, epsilon)

