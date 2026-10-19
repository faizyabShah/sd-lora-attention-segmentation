"""Locked SG2 consensus graph with ordinary one/two-pass controls.

Implements Claude's selected G_cv_g20_n1 operator without computing the
remaining SG2 grid or loading learned tokens. Affinities are streamed by call.
"""
from __future__ import annotations
import math
import torch


@torch.inference_mode()
def locked_sg2_maps(calls, seed: torch.Tensor, gamma: float = 2.0):
    """Return [mean_i A_i c, mean_i A_i^2 c, rownorm(Abar*r^gamma)c]."""
    if not calls or gamma <= 0:
        raise ValueError('nonempty QK calls and positive gamma required')
    seed = seed.reshape(-1).float()
    p = seed.numel()
    mean_sum = torch.zeros((p, p), device=seed.device)
    square_sum = torch.zeros_like(mean_sum)
    one, two = torch.zeros_like(seed), torch.zeros_like(seed)
    heads = calls[0][0].shape[0]
    for query, key in calls:
        q, k = query.to(seed.device).float(), key.to(seed.device).float()
        if q.shape[0] != heads or q.shape[-2] != p:
            raise ValueError('call/head/resolution dimensions differ')
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        first = a @ seed
        one += first.mean(0)
        two += (a @ first[..., None])[..., 0].mean(0)
        mean_sum += a.sum(0)
        square_sum += (a*a).sum(0)
        del a, first, q, k
    observations = len(calls)*heads
    mean = mean_sum/observations
    reliability = mean/(square_sum/observations).sqrt().clamp_min(1e-30)
    graph = mean*reliability.pow(gamma)
    graph /= graph.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.stack((one/len(calls), two/len(calls), graph @ seed))
