"""Reusable, deterministic attention partitions and region aggregation from SG4."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def kmeans(x: torch.Tensor, k: int, restarts: int = 3, iterations: int = 30, seed: int = 0) -> torch.Tensor:
    """Deterministic k-means (squared Euclidean) with k-means++ initialization; returns labels [N]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_inertia = None, float("inf")
    n = x.shape[0]
    for _ in range(restarts):
        centers = x[torch.randint(n, (1,), generator=g).item()][None]
        for _ in range(k - 1):
            d = torch.cdist(x, centers).min(dim=1).values ** 2
            probs = (d / d.sum().clamp_min(1e-30)).cpu()
            centers = torch.cat([centers, x[torch.multinomial(probs, 1, generator=g).item()][None]])
        for _ in range(iterations):
            labels = torch.cdist(x, centers).argmin(dim=1)
            new = torch.zeros_like(centers).index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=k).to(x.dtype)[:, None]
            centers = torch.where(counts > 0, new / counts.clamp_min(1), centers)
        labels = torch.cdist(x, centers).argmin(dim=1)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    return best


def region_map(labels: torch.Tensor, values: torch.Tensor, how: str) -> torch.Tensor:
    k = int(labels.max()) + 1
    if how == "mean":
        sums = torch.zeros(k, device=values.device).index_add_(0, labels, values)
        score = sums / torch.bincount(labels, minlength=k).clamp_min(1).to(values.dtype)
    else:
        score = torch.zeros(k, device=values.device)
        for r in range(k):
            sel = values[labels == r]
            if sel.numel():
                score[r] = sel.median() if sel.numel() % 2 else sel.sort().values[sel.numel() // 2 - 1: sel.numel() // 2 + 1].mean()
    return score[labels]


@torch.inference_mode()
def attention_partitions(mean: torch.Tensor, ks=(8, 16, 32), include_rows=True):
    """Canonical SG4 implementation, optionally evaluating only required partitions.

    The default preserves SG4's full order and arithmetic. SG13 requests SPEC32
    only; the eigenspace and k-means path are exactly the same as SG4 SPEC32.
    """
    s = (mean + mean.T) / 2
    d = s.sum(1)
    norm = s / torch.sqrt(d[:, None] * d[None, :]).clamp_min(1e-30)
    _, vecs = torch.linalg.eigh(norm.double())
    lead = vecs[:, -max(ks):].flip(1).float()
    rows = F.normalize(mean, dim=1) if include_rows else None
    out = {}
    for k in ks:
        out[f"SPEC{k}"] = kmeans(F.normalize(lead[:, :k], dim=1), k)
        if include_rows:
            out[f"ROWS{k}"] = kmeans(rows, k)
    return out


@torch.inference_mode()
def spectral_partition(mean: torch.Tensor, regions: int = 32):
    return attention_partitions(mean, ks=(regions,), include_rows=False)[f"SPEC{regions}"]
