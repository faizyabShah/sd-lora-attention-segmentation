"""SG11/SG12 operators on transient SA propagations [observation, pixel].

No labels enter these operators. Observation order is call-major, head-minor.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def configuration_names(spec: dict) -> tuple[str, ...]:
    names = ['W/REF', 'W/C/rho0p80', 'W/A/trim20', 'W/A/median']
    names += [f'W/SG11/{agg}/rho{rho:.2f}'.replace('.', 'p')
              for agg in spec['sg11']['aggregators'] for rho in spec['sg11']['rhos']]
    names += [f'W/SG12/top{round(100 * fraction):02d}' for fraction in spec['sg12']['fractions']]
    return tuple(names)


@torch.inference_mode()
def reliability_maps(observations: torch.Tensor, seed: torch.Tensor, spec: dict):
    """Compute raw SG8 controls, SG11, SG12 and label-free diagnostics.

    Cosine is measured on raw propagated vectors, before map normalization.
    The self diagonal is excluded. Zero-vector cosine is defined as zero.
    Hard selection uses ceil(fraction*N), breaking score ties by call/head order.
    """
    if observations.ndim != 2 or seed.numel() != observations.shape[1]:
        raise ValueError('expected observations [N,P] and seed [P]')
    if observations.shape[0] < 5 or not torch.isfinite(observations).all():
        raise ValueError('need at least five finite observations')
    seed = seed.reshape(-1)
    n = len(observations)
    ordered = observations.sort(dim=0).values
    mid = n // 2
    median = (ordered[mid-1] + ordered[mid]) / 2 if n % 2 == 0 else ordered[mid]
    cut = int(.2 * n)
    trim20 = ordered[cut:n-cut].mean(0)
    mean = observations.mean(0)
    up = (observations > seed).float().mean(0)
    down = (observations < seed).float().mean(0)
    consensus = torch.maximum(up, down)
    robust = {'median': median, 'trim20': trim20}
    maps = [mean, torch.where(consensus >= .8, mean, seed), trim20, median]
    for agg in spec['sg11']['aggregators']:
        for rho in spec['sg11']['rhos']:
            maps.append(torch.where(consensus >= rho, robust[agg], seed))
    unit = F.normalize(observations, p=2, dim=1, eps=1e-12)
    pairwise = unit @ unit.T
    scores = (pairwise.sum(1) - pairwise.diagonal()) / (n - 1)
    ranking = torch.argsort(scores, descending=True, stable=True)
    selected = []
    for fraction in spec['sg12']['fractions']:
        k = math.ceil(fraction * n)
        keep = torch.zeros(n, dtype=torch.bool, device=seed.device)
        keep[ranking[:k]] = True
        selected.append(keep)
        maps.append(observations[ranking[:k]].mean(0))
    diagnostics = {
        'mean': mean, 'median': median, 'trim20': trim20,
        'r_up': up, 'r_down': down, 'consensus': consensus,
        'std': observations.std(0, correction=0),
        'mad': torch.quantile((observations - median).abs(), .5, dim=0),
        'iqr': torch.quantile(observations, .75, dim=0) - torch.quantile(observations, .25, dim=0),
        'median_minus_seed': median - seed, 'mean_minus_seed': mean - seed,
        'mean_median_disagreement': (mean - median).abs(),
        'fraction_close_to_seed': ((observations-seed).abs() <= spec['diagnostic']['near_seed_epsilon']).float().mean(0),
        'reliability': scores, 'selected': torch.stack(selected), 'pairwise': pairwise,
    }
    return torch.stack(maps), diagnostics


@torch.inference_mode()
def propagate_qk(calls, seed: torch.Tensor, sa_spec: dict, device: str):
    """Reconstruct only one call's affinities at a time; retain A_i c vectors."""
    if len(calls) != sa_spec['calls']:
        raise ValueError(f"expected {sa_spec['calls']} calls, found {len(calls)}")
    outputs = []
    reference = torch.zeros_like(seed.reshape(-1).float(), device=device)
    for query, key in calls:
        q, k = query.to(device).float(), key.to(device).float()
        if q.shape[0] != sa_spec['heads']:
            raise ValueError('unexpected head count')
        probability = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        propagated = probability @ seed.reshape(-1).float()
        outputs.append(propagated)
        reference += propagated.sum(0)
        del probability, propagated, q, k
    observations = torch.cat(outputs)
    if len(observations) != sa_spec['observations']:
        raise ValueError('unexpected observation count')
    return observations, reference / len(observations)
