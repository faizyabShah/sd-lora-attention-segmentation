"""Locked, training-free structural SA-head weighting for SG18/T1."""
from __future__ import annotations

import torch

NAMES = ('SG18/T1', 'SG21-LM', 'SG21-LM-RANK', 'SG21-ENT', 'SG21-DIST')
WEIGHTED = NAMES[1:]
CALLS = tuple(range(10, 21))
HEADS = 8
EPSILON = 1e-8


@torch.inference_mode()
def descriptors(affinity: torch.Tensor, distance: torch.Tensor,
                local_radius_one: torch.Tensor) -> torch.Tensor:
    """Entropy, mean Euclidean distance, radius-1 mass as in Pre-SG21.

    Each descriptor averages over all query rows of one row-stochastic head.
    No GT, class label, or propagated values enter the weights.
    """
    if affinity.ndim != 3 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError('expected [heads, queries, keys] square affinity')
    n = affinity.shape[-1]
    if distance.shape != (n, n) or local_radius_one.shape != (n, n):
        raise ValueError('geometry disagrees with affinity size')
    entropy = -(affinity * affinity.clamp_min(1e-30).log()).sum(-1).mean(-1)
    mean_distance = (affinity * distance).sum((-1, -2)) / n
    local_mass = (affinity * local_radius_one).sum((-1, -2)) / n
    return torch.stack((entropy, mean_distance, local_mass), dim=-1)


@torch.inference_mode()
def weights_from_descriptors(values: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """Four [8]-head distributions; stable ties rank by physical head ID."""
    if values.shape != (HEADS, 3) or not torch.isfinite(values).all():
        raise ValueError('expected finite eight-head structural descriptors')
    entropy, distance, local = values.unbind(-1)
    order = torch.argsort(local, stable=True)
    ranks = torch.empty(HEADS, dtype=local.dtype, device=local.device)
    ranks[order] = torch.arange(1, HEADS + 1, dtype=local.dtype, device=local.device)
    raw = torch.stack((local,
                       ranks,
                       entropy.max() - entropy + epsilon,
                       1 / (distance + epsilon)))
    if not torch.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError('nonpositive or nonfinite structural reliability')
    normalized = raw / raw.sum(-1, keepdim=True)
    if not torch.allclose(normalized.sum(-1), torch.ones(len(WEIGHTED), device=values.device), atol=1e-6):
        raise ValueError('weights do not sum to one within each call')
    return normalized


@torch.inference_mode()
def aggregate(calls, seed: torch.Tensor, geometry, *, return_observations: bool = True):
    """Apply all four locked rules on calls 10–20, averaging calls equally.

    Returns five 64x64 maps, four [11,8] weight tables, descriptors and
    optionally the 88 propagated observations for fixed-threshold diagnostics.
    """
    c = seed.reshape(-1).float()
    if len(c) != 4096 or len(calls) != 31:
        raise ValueError('expected 31 SA calls and a 64x64 CA seed')
    distance = geometry.distance
    local = geometry.local[1]
    maps = torch.zeros((len(NAMES), len(c)), device=c.device, dtype=torch.float32)
    all_weights, all_descriptors, observations = [], [], []
    for t in CALLS:
        query, key = calls[t]
        if query.shape[0] != HEADS or key.shape[0] != HEADS:
            raise ValueError('expected eight heads per SA call')
        q = query.to(c.device).float()
        k = key.to(c.device).float()
        affinity = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** .5, dim=-1)
        z = affinity @ c
        desc = descriptors(affinity, distance, local)
        weights = weights_from_descriptors(desc)
        maps[0] += z.mean(0) / len(CALLS)
        maps[1:] += (weights @ z) / len(CALLS)
        all_weights.append(weights.cpu())
        all_descriptors.append(desc.cpu())
        if return_observations:
            observations.append(z.cpu())
        del q, k, affinity, z, desc, weights
    weights = torch.stack(all_weights, dim=1)  # [candidate, call, head]
    descriptor_values = torch.stack(all_descriptors)  # [call, head, descriptor]
    obs = torch.stack(observations) if return_observations else None
    if not torch.isfinite(maps).all() or not torch.isfinite(weights).all():
        raise ValueError('nonfinite SG21 map or weight')
    if obs is not None:
        torch.testing.assert_close(maps[0].cpu(), obs.mean((0, 1)), rtol=1e-5, atol=2e-6)
    return maps, weights, descriptor_values, obs
