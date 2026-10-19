"""Pure operators for the registered training-free backlog screen.

All functions consume tensors already captured from frozen Stable Diffusion.
They do not load a model, read annotations, or retain per-image artifacts.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F


def minmax(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().reshape(-1)
    lo, hi = flat.min(), flat.max()
    return ((x.float() - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)


def row_normalize(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def heat_kernel(graph: torch.Tensor, seed: torch.Tensor, t: float, terms: int = 8) -> torch.Tensor:
    """Truncated exp(t(A-I)) seed, including the k=0 term."""
    x = seed.reshape(-1).float()
    out = math.exp(-t) * x
    power = x
    coefficient = math.exp(-t)
    for k in range(1, terms + 1):
        power = graph @ power
        coefficient *= t / k
        out = out + coefficient * power
    return out


def symmetric_power(graph: torch.Tensor, seed: torch.Tensor, powers: tuple[int, ...]) -> list[torch.Tensor]:
    w = 0.5 * (graph + graph.T)
    degree = w.sum(1).clamp_min(1e-12)
    norm = degree.rsqrt()[:, None] * w * degree.rsqrt()[None, :]
    x, output = seed.reshape(-1).float(), []
    for n in range(1, max(powers) + 1):
        x = norm @ x
        if n in powers:
            output.append(x.clone())
    return output


def entropy_weighted(observations: torch.Tensor, temperature: float) -> torch.Tensor:
    z = observations.float().clamp_min(0)
    probability = z / z.sum(1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
    weights = entropy.clamp_min(1e-12).reciprocal().pow(1.0 / temperature)
    weights = weights / weights.sum()
    return weights @ z


def agreement_heads(observations: torch.Tensor, seed: torch.Tensor, keep: int) -> torch.Tensor:
    """Rank heads by correlation of their call-mean propagation with the CA seed."""
    # observations [calls, heads, pixels]
    maps = observations.float().mean(0)
    c = seed.reshape(-1).float()
    centered_c = c - c.mean()
    centered = maps - maps.mean(1, keepdim=True)
    correlation = (centered * centered_c).sum(1) / (
        centered.square().sum(1).sqrt() * centered_c.square().sum().sqrt()
    ).clamp_min(1e-12)
    chosen = correlation.topk(keep).indices
    return maps[chosen].mean(0)


def iterate_to_convergence(graph: torch.Tensor, seed: torch.Tensor, tolerance: float, max_iterations: int):
    x = seed.reshape(-1).float()
    for iteration in range(1, max_iterations + 1):
        updated = minmax(graph @ x)
        delta = float((updated - x).abs().max())
        x = updated
        if delta < tolerance:
            return x, iteration, delta
    return x, max_iterations, delta


def deterministic_kmeans(features: torch.Tensor, clusters: int, iterations: int = 20) -> torch.Tensor:
    """Deterministic farthest-point initialized Lloyd clustering."""
    x = features.float()
    first = int(x.square().sum(1).argmax())
    centers = [x[first]]
    distance = (x - centers[0]).square().sum(1)
    for _ in range(1, clusters):
        index = int(distance.argmax())
        centers.append(x[index])
        distance = torch.minimum(distance, (x - centers[-1]).square().sum(1))
    centers_t = torch.stack(centers)
    labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(iterations):
        x2 = x.square().sum(1, keepdim=True)
        c2 = centers_t.square().sum(1)[None]
        new_labels = (x2 + c2 - 2 * x @ centers_t.T).argmin(1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        next_centers = []
        for k in range(clusters):
            members = x[labels == k]
            next_centers.append(members.mean(0) if len(members) else centers_t[k])
        centers_t = torch.stack(next_centers)
    return labels


def region_scores(labels: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    flat = score.reshape(-1).float()
    output = torch.empty_like(flat)
    for label in labels.unique(sorted=True):
        region = labels == label
        output[region] = flat[region].mean()
    return output


def diffseg_labels(graph: torch.Tensor, threshold: float, anchors_per_axis: int = 16, iterations: int = 3):
    """PyTorch transcription of official DiffSeg's iterative symmetric-KL merge."""
    side = math.isqrt(graph.shape[0])
    if side * side != graph.shape[0]:
        raise ValueError("DiffSeg graph must be square in spatial coordinates")
    step = (side - 1) // (anchors_per_axis - 1)
    total = step * (anchors_per_axis - 1)
    start = (side - 1 - total) // 2
    coordinates = torch.linspace(start, start + total, anchors_per_axis, device=graph.device).long()
    anchors = (coordinates[:, None] * side + coordinates[None, :]).reshape(-1)
    maps = row_normalize(graph.float().clamp_min(1e-12))

    def symmetric_kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # [A,P] against [B,P], returned [A,B], in bounded chunks.
        values = []
        for begin in range(0, x.shape[0], 8):
            a = x[begin : begin + 8, None].clamp_min(1e-8)
            b = y[None].clamp_min(1e-8)
            values.append((0.5 * (a - b) * (a.log() - b.log())).sum(-1))
        return torch.cat(values)

    membership = symmetric_kl(maps[anchors], maps) < threshold
    merged = torch.stack([
        maps[group].mean(0) if bool(group.any()) else maps[anchor]
        for group, anchor in zip(membership, anchors)
    ])
    for _ in range(1, iterations):
        used = torch.zeros(len(merged), dtype=torch.bool, device=merged.device)
        groups = []
        distances = symmetric_kl(merged, merged)
        for index in range(len(merged)):
            if bool(used[index]):
                continue
            group = (distances[index] < threshold) & ~used
            group[index] = True
            used |= group
            groups.append(merged[group].mean(0))
        merged = torch.stack(groups)
    return merged.argmax(0)


def query_features(query_sum: torch.Tensor, calls: int) -> torch.Tensor:
    """[heads,pixels,dim] -> L2-normalized [pixels,heads*dim]."""
    value = query_sum.float() / calls
    value = F.normalize(value, dim=-1).permute(1, 0, 2).reshape(value.shape[1], -1)
    return F.normalize(value, dim=1)


def prototype_score(features: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    weights = seed.reshape(-1).float().clamp_min(0)
    prototype = (weights[:, None] * features).sum(0) / weights.sum().clamp_min(1e-12)
    return features @ F.normalize(prototype, dim=0)


def multiple_prototype_score(features: torch.Tensor, seed: torch.Tensor, clusters: int, iterations: int = 20):
    flat = seed.reshape(-1).float()
    selected = flat >= torch.quantile(flat, 0.8)
    subset = features[selected]
    labels = deterministic_kmeans(subset, clusters, iterations)
    prototypes = torch.stack([subset[labels == k].mean(0) for k in range(clusters)])
    prototypes = F.normalize(prototypes, dim=1)
    return (features @ prototypes.T).amax(1)


def seeded_region_growing(features: torch.Tensor, seed: torch.Tensor, similarity: float) -> torch.Tensor:
    """4-connected components of feature-similar edges retained iff they contain a CA peak."""
    side = math.isqrt(features.shape[0])
    feat = features.reshape(side, side, -1)
    right = (feat[:, :-1] * feat[:, 1:]).sum(-1) >= similarity
    down = (feat[:-1] * feat[1:]).sum(-1) >= similarity
    peak = minmax(seed).reshape(side, side) >= 0.9
    seen = torch.zeros((side, side), dtype=torch.bool, device=features.device)
    output = torch.zeros_like(seen)
    right_cpu, down_cpu, peak_cpu = right.cpu(), down.cpu(), peak.cpu()
    seen_cpu, output_cpu = seen.cpu(), output.cpu()
    for r in range(side):
        for c in range(side):
            if seen_cpu[r, c]:
                continue
            queue, component, keep = deque([(r, c)]), [], False
            seen_cpu[r, c] = True
            while queue:
                i, j = queue.popleft()
                component.append((i, j)); keep = keep or bool(peak_cpu[i, j])
                candidates = []
                if j + 1 < side and right_cpu[i, j]: candidates.append((i, j + 1))
                if j and right_cpu[i, j - 1]: candidates.append((i, j - 1))
                if i + 1 < side and down_cpu[i, j]: candidates.append((i + 1, j))
                if i and down_cpu[i - 1, j]: candidates.append((i - 1, j))
                for ni, nj in candidates:
                    if not seen_cpu[ni, nj]:
                        seen_cpu[ni, nj] = True; queue.append((ni, nj))
            if keep:
                for i, j in component: output_cpu[i, j] = True
    return output_cpu.to(features.device).reshape(-1)


def sa_mean_field(graph: torch.Tensor, unary: torch.Tensor, weight: float, iterations: int = 5):
    probability = minmax(unary).reshape(-1).clamp(1e-4, 1 - 1e-4)
    logit = torch.logit(probability)
    q = probability
    for _ in range(iterations):
        q = torch.sigmoid(logit + weight * (graph @ (2 * q - 1)))
    return q


def self_test() -> None:
    generator = torch.Generator().manual_seed(7)
    graph = row_normalize(torch.rand((16, 16), generator=generator))
    seed = torch.rand(16, generator=generator)
    observations = torch.stack((graph @ seed, graph @ seed * 0.9)).reshape(1, 2, 16)
    assert heat_kernel(graph, seed, 1.0).shape == seed.shape
    assert len(symmetric_power(graph, seed, (1, 2))) == 2
    assert entropy_weighted(observations.reshape(2, 16), 1.0).shape == seed.shape
    assert agreement_heads(observations, seed, 1).shape == seed.shape
    assert iterate_to_convergence(graph, seed, 1e-4, 3)[0].shape == seed.shape
    labels = deterministic_kmeans(graph, 2, 3)
    assert region_scores(labels, seed).shape == seed.shape
    assert diffseg_labels(graph, 1.1, 2, 2).shape == seed.shape
    features = F.normalize(torch.rand((16, 8), generator=generator), dim=1)
    assert prototype_score(features, seed).shape == seed.shape
    assert multiple_prototype_score(features, seed, 2, 3).shape == seed.shape
    assert seeded_region_growing(features, seed, 0.5).shape == seed.shape
    assert sa_mean_field(graph, seed, 1.0, 2).shape == seed.shape


if __name__ == "__main__":
    self_test()
