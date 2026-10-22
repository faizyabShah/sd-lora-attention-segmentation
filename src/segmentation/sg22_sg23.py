"""Locked SG22 depth and SG23 reliability factorials on transient SA Q/K."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

WINDOWS = {'ALL': tuple(range(31)), 'MID': tuple(range(10, 21))}
SG22_RAW = tuple(f'{w}-P{d}' for w in WINDOWS for d in (1, 2, 3))
SG23_RAW = tuple(f'{w}-{r}' for w in WINDOWS for r in ('R0', 'SG2', 'SG11', 'SG12'))
SG23_AUX = 'ALL-P2'  # Same-generation P1+R2 final-pipeline reference.


@torch.inference_mode()
def aggregate(calls, seed: torch.Tensor, experiment: str, *, expected_pixels: int = 4096) -> dict[str, torch.Tensor]:
    """All affinities are reconstructed one call at a time; no averaged A is powered."""
    if experiment not in ('sg22', 'sg23'):
        raise ValueError(experiment)
    c = seed.reshape(-1).float()
    if len(c) != expected_pixels or len(calls) != 31:
        raise ValueError('expected 31 calls and the registered CA seed size')
    sums = {w: torch.zeros((3 if experiment == 'sg22' else 2, len(c)), device=c.device)
            for w in WINDOWS}
    if experiment == 'sg23':
        moments = {w: (torch.zeros((len(c), len(c)), device=c.device),
                       torch.zeros((len(c), len(c)), device=c.device)) for w in WINDOWS}
        vectors = {w: [] for w in WINDOWS}
    for t, (query, key) in enumerate(calls):
        if query.shape[0] != 8 or query.shape[-2] != len(c):
            raise ValueError('unexpected SA head/resolution')
        q = query.to(c.device).float()
        k = key.to(c.device).float()
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        z1 = a @ c
        z2 = (a @ z1[..., None])[..., 0]
        zs = [z1, z2]
        if experiment == 'sg22':
            zs.append((a @ z2[..., None])[..., 0])
        for w, calls_in_window in WINDOWS.items():
            if t not in calls_in_window:
                continue
            for d, z in enumerate(zs):
                sums[w][d] += z.sum(0)
            if experiment == 'sg23':
                mean_sum, square_sum = moments[w]
                mean_sum += a.sum(0)
                square_sum += (a*a).sum(0)
                vectors[w].append(z1.cpu())
        del q, k, a, z1, z2, zs
    output = {}
    for w, calls_in_window in WINDOWS.items():
        n = len(calls_in_window) * 8
        if experiment == 'sg22':
            for d in (1, 2, 3):
                output[f'{w}-P{d}'] = sums[w][d-1] / n
            continue
        obs = torch.cat(vectors[w]).to(c.device)
        if len(obs) != n:
            raise ValueError('observation count differs from window')
        mean_sum, square_sum = moments[w]
        mean = mean_sum / n
        reliability = mean / (square_sum / n).sqrt().clamp_min(1e-30)
        graph = mean * reliability.square()  # Exact G_cv_g20_n1 gamma=2.
        graph /= graph.sum(-1, keepdim=True).clamp_min(1e-30)
        ordered = obs.sort(dim=0).values
        cut = int(.2*n)
        trimmed = ordered[cut:n-cut].mean(0)
        consensus = torch.maximum((obs > c).float().mean(0), (obs < c).float().mean(0))
        unit = F.normalize(obs, p=2, dim=1, eps=1e-12)
        pairwise = unit @ unit.T
        agreement = (pairwise.sum(1)-pairwise.diagonal())/(n-1)
        ranking = torch.argsort(agreement, descending=True, stable=True)
        output[f'{w}-R0'] = sums[w][0] / n
        output[f'{w}-SG2'] = graph @ c
        output[f'{w}-SG11'] = torch.where(consensus >= .8, trimmed, c)
        output[f'{w}-SG12'] = obs[ranking[:n//2]].mean(0)
        if w == 'ALL':
            output[SG23_AUX] = sums[w][1] / n
        del obs, mean_sum, square_sum, mean, reliability, graph, ordered, trimmed
        del consensus, unit, pairwise, agreement, ranking
    expected = set(SG22_RAW if experiment == 'sg22' else (*SG23_RAW, SG23_AUX))
    if set(output) != expected or not all(torch.isfinite(v).all() for v in output.values()):
        raise ValueError('invalid experiment maps')
    return output
