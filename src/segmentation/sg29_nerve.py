"""SG29: observation-wise NERVE-inspired propagation on frozen SD1.5 SA.

The paper's class-distribution entropy is adapted to single-class *spatial*
entropy. Global transitions remain the historical SD softmax SA matrices.
No annotation enters any inference path.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.segmentation.sg25_sa import _apply, _attention

MAIN = tuple(f"N{i}" for i in range(7))
DIAGNOSTIC = ("N2-L1", "N2-L2", "N2-L5")
ALL = MAIN + DIAGNOSTIC


def neighbor_index(side: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight spatial neighbors plus self, with invalid sites masked."""
    ids = torch.arange(side * side, device=device).reshape(side, side)
    padded = F.pad(ids, (1, 1, 1, 1), value=-1)
    index = torch.stack([padded[1 + dy:1 + dy + side, 1 + dx:1 + dx + side].reshape(-1)
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dim=-1)
    valid = index >= 0
    return index.clamp_min(0), valid


def local_transition(q: torch.Tensor, k: torch.Tensor, index: torch.Tensor,
                     valid: torch.Tensor, self_weight: float, epsilon: float) -> torch.Tensor:
    """Row-stochastic nine-entry Q/K cosine graph; center receives self_weight."""
    q = F.normalize(q.float(), dim=-1, eps=epsilon)
    k = F.normalize(k.float(), dim=-1, eps=epsilon)
    # Gather only 9 keys per query; no dense local matrix is created.
    neighbors = k[:, index.reshape(-1), :].reshape(k.shape[0], k.shape[1], 9, k.shape[2])
    cosine = (q[:, :, None, :] * neighbors).sum(-1)
    weights = ((cosine + 1.0) * .5 + epsilon) * valid[None]
    weights[:, :, 4] += self_weight
    return weights / weights.sum(-1, keepdim=True)


def local_apply(weights: torch.Tensor, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    gathered = value[:, index.reshape(-1)].reshape(value.shape[0], value.shape[1], 9)
    return (weights * gathered).sum(-1)


def spatial_entropy_weights(one_step: torch.Tensor, concentration: float,
                            epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.any(one_step < -1e-6):
        raise ValueError("SG29 spatial entropy needs nonnegative CA evidence")
    probability = (one_step.clamp_min(0) + epsilon)
    probability = probability / probability.sum(-1, keepdim=True)
    entropy = -(probability * probability.log()).sum(-1) / math.log(one_step.shape[-1])
    return entropy, torch.softmax(-concentration * entropy, dim=0)


def walk_mass(alpha: float, length: int) -> torch.Tensor:
    powers = torch.tensor([alpha ** i for i in range(length + 1)], dtype=torch.float64)
    return powers / powers.sum()


def _walk(initial: torch.Tensor, step, alpha: float, length: int,
          checkpoints: tuple[int, ...] = ()) -> dict[int, torch.Tensor]:
    """Normalized finite stopped walk; every map includes depth zero."""
    state = initial
    total = initial.clone()
    divisor = 1.0
    out = {}
    for depth in range(1, length + 1):
        state = step(state)
        total = total + (alpha ** depth) * state
        divisor += alpha ** depth
        if depth in checkpoints or depth == length:
            out[depth] = total / divisor
    return out


@torch.inference_mode()
def aggregate(sa_qk: list[tuple[torch.Tensor, torch.Tensor]], seed: torch.Tensor,
              *, alpha: float, beta: float, length: int, concentration: float,
              self_weight: float, epsilon: float,
              truth64: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if len(sa_qk) != 11 or seed.shape != (64, 64) or length < 5:
        raise ValueError("SG29 requires frozen 11-call module-15 64x64 observations")
    if not (0 < alpha < 1 and 0 <= beta <= 1 and self_weight > 0 and epsilon > 0):
        raise ValueError("Invalid locked SG29 transition constants")
    c = seed.float().reshape(-1)
    index, valid = neighbor_index(64, c.device)
    sums = {name: torch.zeros_like(c) for name in ALL}
    entropy = torch.zeros((11, 8), dtype=torch.float32)
    fused_entropy = torch.zeros_like(entropy)
    weights = torch.zeros_like(entropy)
    fused_weights = torch.zeros_like(entropy)
    ap = torch.full_like(entropy, float("nan"))
    # Distances and transition types are deterministic sampled-row diagnostics.
    transitions = torch.zeros((11, 8, 3, 7), dtype=torch.float32)
    row_ids = torch.linspace(0, 4095, 64, device=c.device).long()
    xx = torch.arange(4096, device=c.device) % 64
    yy = torch.div(torch.arange(4096, device=c.device), 64, rounding_mode="floor")
    for call, (q, k) in enumerate(sa_qk):
        a = _attention((q, k), 64)
        local = local_transition(q, k, index, valid, self_weight, epsilon)
        global_step = lambda x: _apply(a, x)
        local_step = lambda x: local_apply(local, index, x)
        fused_step = lambda x: beta * global_step(x) + (1-beta) * local_step(x)
        one = global_step(c)
        h, w = spatial_entropy_weights(one, concentration, epsilon)
        fused_one = fused_step(c)
        fh, fw = spatial_entropy_weights(fused_one, concentration, epsilon)
        two = global_step(one)
        global_walk = _walk(c[None].expand(8, -1), global_step, alpha, length, (1, 2, 5))
        local_walk = _walk(c[None].expand(8, -1), local_step, alpha, length)
        fused_walk = _walk(c[None].expand(8, -1), fused_step, alpha, length)
        results = {"N0": two.mean(0), "N1": (two * w[:, None]).sum(0),
                   "N2": global_walk[length].mean(0), "N3": local_walk[length].mean(0),
                   "N4": fused_walk[length].mean(0),
                   "N5": (global_walk[length] * w[:, None]).sum(0),
                   "N6": (fused_walk[length] * fw[:, None]).sum(0),
                   "N2-L1": global_walk[1].mean(0),
                   "N2-L2": global_walk[2].mean(0),
                   "N2-L5": global_walk[5].mean(0)}
        for name in ALL:
            sums[name] += results[name] / 11
        entropy[call] = h.cpu(); fused_entropy[call] = fh.cpu()
        weights[call] = w.cpu(); fused_weights[call] = fw.cpu()
        if truth64 is not None:
            # Threshold-independent head utility, post hoc; never used in weights.
            from src.segmentation.controlled_cv import average_precision_batch
            ap[call] = average_precision_batch(one.cpu(), truth64.cpu().reshape(-1)).float()
        for kind, matrix in enumerate((a, None, None)):
            if kind == 0:
                selected = a[:, row_ids]
                row_entropy = -(selected * selected.clamp_min(epsilon).log()).sum(-1)
                target_x, target_y = xx[None, None], yy[None, None]
                distance = torch.sqrt((xx[row_ids][None, :, None]-target_x).float().square() +
                                      (yy[row_ids][None, :, None]-target_y).float().square())
                neighbor_mass = (selected * (distance <= math.sqrt(2)).float()).sum(-1)
                mean_distance = (selected * distance).sum(-1)
                support = (row_entropy.exp()).clamp_max(4096)
                if truth64 is not None:
                    gt = truth64.to(c.device).reshape(-1)
                    fg_mass = (selected * gt[None, None].float()).sum(-1)
                else:
                    fg_mass = torch.zeros_like(mean_distance)
            else:
                # Local and fused row diagnostics computed from their sparse/dense parts.
                loc = local[:, row_ids]
                loc_distance = torch.sqrt((xx[row_ids,None]-xx[index[row_ids]]).float().square() +
                                          (yy[row_ids,None]-yy[index[row_ids]]).float().square())
                loc_entropy = -(loc * loc.clamp_min(epsilon).log()).sum(-1)
                loc_distance_mean = (loc * loc_distance[None]).sum(-1)
                loc_mass = torch.ones_like(loc_distance_mean)
                if truth64 is not None:
                    gt = truth64.to(c.device).reshape(-1)
                    loc_fg = (loc * gt[index[row_ids]][None].float()).sum(-1)
                else:
                    loc_fg = torch.zeros_like(loc_distance_mean)
                if kind == 1:
                    row_entropy = loc_entropy; mean_distance = loc_distance_mean
                    neighbor_mass = loc_mass; support = loc_entropy.exp(); fg_mass = loc_fg
                else:
                    # Exact fused entropy is sampled directly, without full fused matrix.
                    fused_rows = beta * a[:, row_ids].clone()
                    fused_rows.scatter_add_(2, index[row_ids][None].expand(8,-1,-1), (1-beta)*loc)
                    row_entropy = -(fused_rows * fused_rows.clamp_min(epsilon).log()).sum(-1)
                    mean_distance = beta * (selected*distance).sum(-1)+(1-beta)*loc_distance_mean
                    neighbor_mass = beta*(selected*(distance<=math.sqrt(2))).sum(-1)+(1-beta)
                    support = row_entropy.exp()
                    fg_mass = beta*(selected*gt[None,None].float()).sum(-1)+(1-beta)*loc_fg if truth64 is not None else torch.zeros_like(mean_distance)
            # Mean across fixed sampled rows. Last columns split GT-positive and negative source rows.
            record = torch.stack((mean_distance.mean(1), row_entropy.mean(1),
                                  neighbor_mass.mean(1), support.mean(1),fg_mass.mean(1)),dim=-1)
            transitions[call,:,kind,:5] = record.cpu()
            if truth64 is not None:
                positive = gt[row_ids]
                transitions[call,:,kind,5] = fg_mass[:,positive].mean(1).cpu() if positive.any() else float("nan")
                transitions[call,:,kind,6] = fg_mass[:,~positive].mean(1).cpu() if (~positive).any() else float("nan")
        del a, local, one, two, global_walk, local_walk, fused_walk
    diagnostics = {"entropy":entropy,"fused_entropy":fused_entropy,
                   "weights":weights,"fused_weights":fused_weights,
                   "head_ap":ap,"transitions":transitions}
    return {name:value.reshape(64,64) for name,value in sums.items()}, diagnostics
