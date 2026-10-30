"""SG-combined-v1 C10 add-on: frozen SG33 M2 coarse source, three cells.

C10a is the exact SG33 M2 path. C10b/c use the same coarse source but
transport it as a probability distribution before optional completion.
"""
from __future__ import annotations

import torch

from src.segmentation.sg25_sa import _apply, _attention, _resize
from src.segmentation.sg32_adaptive import background_support
from src.segmentation.sg33_hierarchy import ipf, markov, propagate
from src.segmentation.sg_combined_v1 import completion, distribution, project, score

NAMES = ("C10a", "C10b", "C10c")


@torch.inference_mode()
def aggregate(sa_qk: dict[int, list], seed: torch.Tensor):
    if seed.shape != (64, 64) or any(len(sa_qk.get(s, ())) != 11 for s in (16, 32, 64)):
        raise ValueError("C10 requires 11 matched SA calls at 16/32/64")
    c64 = seed.float().reshape(-1)
    c16 = _resize(c64, 64, 16).reshape(-1)
    sums = {name: torch.zeros((64, 64), device=seed.device) for name in NAMES}
    stages = torch.zeros((2, 3, 64, 64), device=seed.device)
    completion_maps = torch.zeros((4, 64, 64), device=seed.device)  # P1,P2,B,G
    arrivals = []
    transport = torch.zeros((2, 4), dtype=torch.float64)
    ipf_log = torch.zeros(3, dtype=torch.float64)  # row, column, iterations
    for call in range(11):
        a16 = _attention(sa_qk[16][call], 16).mean(0)
        a32 = _attention(sa_qk[32][call], 32).mean(0)
        fine_heads = _attention(sa_qk[64][call], 64)
        a64 = fine_heads.mean(0)
        graph, _, after, iterations, converged = ipf(a16, 1e-5, 100)
        if not converged:
            raise ValueError("SG33 M2 IPF did not meet its frozen tolerance")
        ipf_log += torch.tensor((after[0], after[1], iterations), dtype=torch.float64) / 11
        coarse, arrival, _ = markov(graph, c16 / c16.sum().clamp_min(1e-30), .3, 256)
        arrivals.append(arrival.cpu())

        # The raw ordinary path matches SG33 M2 before across-call min-max.
        ordinary32, ordinary64 = propagate(a32[None], a64[None], coarse)
        sums["C10a"] += ordinary64 / 11

        x16 = distribution(coarse)
        q32 = distribution(a32 @ _resize(x16, 16, 32).reshape(-1))
        p32, stats32 = project(x16, q32, 16, rho=1.)
        q64 = distribution(a64 @ _resize(p32, 32, 64).reshape(-1))
        p64, stats64 = project(p32, q64, 32, rho=1.)
        sums["C10b"] += p64.reshape(64, 64) / 11
        for j, stats in enumerate((stats32, stats64)):
            transport[j] += torch.tensor(stats["error"], dtype=torch.float64) / 11

        ordinary = (coarse, ordinary32, ordinary64)
        constrained = (x16.reshape(16, 16), p32.reshape(32, 32), p64.reshape(64, 64))
        for i, route in enumerate((ordinary, constrained)):
            for j, (state, side) in enumerate(zip(route, (16, 32, 64))):
                stages[i, j] += _resize(state.reshape(-1), side, 64).reshape(64, 64) / 11

        first_fine = score(_apply(fine_heads, c64).mean(0))
        background, _, _ = background_support(fine_heads, first_fine)
        budgeted, p1, p2, _, odds = completion(p64, a64, background, "odds")
        sums["C10c"] += budgeted.reshape(64, 64) / 11
        completion_maps += torch.stack((p1, p2, background, odds)).reshape(4, 64, 64) / 11
        del a16, a32, a64, fine_heads, graph
    return sums, {"stages": stages.cpu(), "arrival": torch.stack(arrivals),
                  "completion": completion_maps.cpu(), "transport": transport.numpy(),
                  "ipf": ipf_log.numpy()}
