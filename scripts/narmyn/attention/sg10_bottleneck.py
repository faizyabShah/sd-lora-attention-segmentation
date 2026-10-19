"""SG10 seed-connected bottleneck propagation: operators (protocol Section 77; docs/SG10_SEED_CONNECTED_BOTTLENECK.md).

All maps are 64x64 (P = 4096 nodes). ``W[u, v]`` is the weight of the directed
edge u -> v. The bottleneck connectivity of node q to a source set S is

    B(q) = max over paths S -> q of the minimum edge weight on the path

(widest path). Sources start at the largest graph weight. It is computed
exactly on the dense graph with the max-min analogue of Dijkstra's algorithm,
batched over several source sets; no edge is pruned.

    python scripts/narmyn/attention/sg10_bottleneck.py   # self-test against brute force
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SIDE = 64


@torch.no_grad()
def widest_path(weights: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Exact widest-path values from each source set. ``weights`` [P, P], ``sources`` bool [S, P] -> [S, P]."""
    count, nodes = sources.shape
    top = weights.max()
    best = torch.where(sources, top, torch.zeros((), device=weights.device, dtype=weights.dtype))
    done = torch.zeros_like(sources)
    rows = torch.arange(count, device=weights.device)
    for _ in range(nodes):
        u = torch.where(done, torch.full_like(best, -1.0), best).argmax(dim=1)  # [S]: widest unsettled node
        done[rows, u] = True
        reach = torch.minimum(best[rows, u][:, None], weights[u])  # [S, P]
        best = torch.maximum(best, reach)  # settled nodes cannot increase: their value is >= best[u]
    return best


def symmetric_graph(mean_affinity: torch.Tensor) -> torch.Tensor:
    """G2: sqrt(A(p, q) A(q, p))."""
    return torch.sqrt(mean_affinity * mean_affinity.T)


def local_graph(weights: torch.Tensor) -> torch.Tensor:
    """Diagnostic: the same weights restricted to 4-neighbour edges of the 64x64 grid (others 0)."""
    idx = torch.arange(SIDE * SIDE, device=weights.device).reshape(SIDE, SIDE)
    mask = torch.zeros_like(weights, dtype=torch.bool)
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        mask[a.reshape(-1), b.reshape(-1)] = True
        mask[b.reshape(-1), a.reshape(-1)] = True
    return torch.where(mask, weights, torch.zeros_like(weights))


def seed_sets(seed: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """S_F = {p : c(p) >= Q_q(c)} for each quantile; ``seed`` [P] -> bool [len(quantiles), P]."""
    q = torch.quantile(seed.float(), torch.tensor(quantiles, device=seed.device))
    return seed[None] >= q[:, None]


def normalize_bottleneck(raw: torch.Tensor, sources: torch.Tensor, top: torch.Tensor, convention: str) -> torch.Tensor:
    """``raw`` -> [0, 1]: ``ratio`` divides by the source value (largest graph weight); ``rank`` is the
    empirical percentile rank over the 4096 pixels (ties averaged)."""
    if convention == "ratio":
        return (raw / top).clamp(0, 1)
    if convention == "rank":
        order = raw.argsort(dim=-1, stable=True)
        ranks = torch.empty_like(raw)
        ranks.scatter_(-1, order, torch.arange(raw.shape[-1], device=raw.device, dtype=raw.dtype).expand_as(raw))
        # average ranks over ties (sources share the top value)
        out = torch.empty_like(raw)
        for i in range(raw.shape[0]):
            values, inverse = torch.unique(raw[i], return_inverse=True)
            mean_rank = torch.zeros_like(values).scatter_add_(0, inverse, ranks[i]) / torch.bincount(inverse).to(raw.dtype)
            out[i] = mean_rank[inverse] / (raw.shape[-1] - 1)
        return out
    raise ValueError(convention)


def operators(mu: torch.Tensor, seed: torch.Tensor, connectivity: torch.Tensor, alphas=(0.25, 0.50)) -> dict[str, torch.Tensor]:
    """B1, B2 (each alpha) and B3 for one connectivity map; all [P]."""
    out = {"B1": mu * connectivity}
    for a in alphas:
        out[f"B2a{int(a * 100):02d}"] = mu * (a + (1 - a) * connectivity)
    out["B3"] = torch.where(mu > seed, seed + connectivity * (mu - seed), mu)
    return out


def _brute_force(weights: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Widest-path closure by repeated relaxation (Bellman-Ford style), for the self-test."""
    best = torch.where(source, weights.max(), torch.zeros(()))
    for _ in range(weights.shape[0]):
        best = torch.maximum(best, torch.minimum(best[:, None], weights).max(dim=0).values)
    return best


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    for trial in range(20):
        n = 40
        w = torch.rand(n, n, generator=g) * (torch.rand(n, n, generator=g) > 0.6)
        src = torch.zeros(3, n, dtype=torch.bool)
        for s in range(3):
            src[s, torch.randperm(n, generator=g)[: s + 1]] = True
        fast = widest_path(w, src)
        slow = torch.stack([_brute_force(w, src[s]) for s in range(3)])
        assert torch.allclose(fast, slow), (trial, (fast - slow).abs().max())
    r = torch.tensor([[0.1, 0.3, 0.3, 0.9]])
    assert torch.allclose(normalize_bottleneck(r, None, None, "rank"), torch.tensor([[0.0, 1.5 / 3, 1.5 / 3, 1.0]]))
    lg = local_graph(torch.ones(SIDE * SIDE, SIDE * SIDE))
    assert int(lg.sum()) == 2 * 2 * SIDE * (SIDE - 1)
    print("widest_path matches brute force on 20 random directed graphs; rank and local graph checks pass")
