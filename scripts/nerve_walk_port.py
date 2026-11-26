"""NERVE's self-attention fusion and closed-form walk, copied from its released code for use outside its environment.

Source: kunal-mahatha/nerve @ 1ddc5f9, ovs/pipeline_clean_mean_final_run_correct.py, lines 17-134 and 190-258, with
side N's patch (K=1 guard in the entropy fusion; head dim passed in). Function bodies are unchanged; only unused
imports were dropped. `walk` reproduces Pipeline.refinement after the attention capture.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F

ITERATIONS = 30
DELTA = 0.9
BETA = 0.8
ALPHA = 3.0
EPS = 1e-12


def weighted_attention_fusion_by_entropy_joint(A_self_list: List[torch.Tensor], A_cross: torch.Tensor,
                                               alpha: float = ALPHA, eps: float = EPS):
    K = A_cross.shape[-1]
    logK = math.log2(K) if K > 1 else float("inf")      # side N's K=1 guard: uniform head weights
    entropies = []
    A_cross = A_cross.to(A_self_list[0].dtype)
    for A_self in A_self_list:
        P = A_self @ A_cross
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        H_row = -(P * (P + eps).log()).sum(dim=-1)
        entropies.append(H_row.mean())
    entropies = torch.stack(entropies)
    raw_w = torch.exp(-alpha * entropies / logK)
    weights = raw_w / raw_w.sum()
    A_weighted = sum(w * A_self for w, A_self in zip(weights, A_self_list))
    return A_weighted, weights


def build_cosine_local_mask(q, k, H, W, kernel, eps=1e-5):
    N, D = q.shape
    assert N == H * W
    kernel = kernel.to(q.device)
    coords = torch.stack(torch.meshgrid(torch.arange(H, device=q.device), torch.arange(W, device=q.device),
                                        indexing="ij"), dim=-1).view(-1, 2)
    offsets = [(i - kernel.shape[0] // 2, j - kernel.shape[1] // 2)
               for i in range(kernel.shape[0]) for j in range(kernel.shape[1]) if kernel[i, j] != 0]
    M = torch.zeros((N, N), device=q.device, dtype=q.dtype)
    for dy, dx in offsets:
        nbr = coords + torch.tensor([dy, dx], device=q.device)
        valid = (nbr[:, 0] >= 0) & (nbr[:, 0] < H) & (nbr[:, 1] >= 0) & (nbr[:, 1] < W)
        idx0 = (coords[valid][:, 0] * W + coords[valid][:, 1]).long()
        idx1 = (nbr[valid][:, 0] * W + nbr[valid][:, 1]).long()
        raw_sim = F.cosine_similarity(q[idx0], k[idx1], dim=-1)
        sim = (raw_sim + 1.0) * 0.5 + eps
        M[idx0, idx1] = sim
    return M


def chained_attention_product(qk_store: List[Dict[str, torch.Tensor]], A_cross: torch.Tensor, d: int):
    """qk_store: one {"query", "key"} [heads, N, d] per layer (NERVE's layers -4, -6)."""
    A_list_g, A_list_l = [], []
    kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    for entry in qk_store:
        q, k = entry["query"], entry["key"]
        for h in range(q.shape[0]):
            qh, kh = q[h].float(), k[h].float()
            H = W = int(math.sqrt(qh.shape[0]))
            M = build_cosine_local_mask(qh, kh, H, W, kernel)
            M += torch.eye(M.size(0), device=M.device) * 1e-2
            logits = (qh @ kh.T) / math.sqrt(d)
            A_list_l.append(torch.softmax(logits * M, dim=-1))
            A_list_g.append(torch.softmax(logits, dim=-1))

    def preprocess(A_list):
        A = torch.stack(A_list, dim=0)
        A = A / (A.amax(dim=-2, keepdim=True) + 1e-5)
        A = A + torch.where(A == 0, 0, 0.01 * (torch.log10(torch.e * A)))
        A = torch.clamp(A, min=0)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-5)
        return list(A.unbind(dim=0))

    A_w_g, _ = weighted_attention_fusion_by_entropy_joint(preprocess(A_list_g), A_cross)
    A_w_l, _ = weighted_attention_fusion_by_entropy_joint(preprocess(A_list_l), A_cross)
    return BETA * A_w_g + (1 - BETA) * A_w_l


def seed_column(pred_mask: torch.Tensor) -> torch.Tensor:
    """pred_mask [1, h, w] -> cross_att [4096, 1], as in Pipeline.refinement."""
    pred = F.interpolate(pred_mask[None], size=(64, 64), mode="bilinear", align_corners=False)[0]
    return pred.flatten(-2).T


def walk_operator(fused: torch.Tensor) -> torch.Tensor:
    """fused @ (1 - DELTA) (I - DELTA fused)^-1, the closed-form walk of Pipeline.refinement (seed-independent)."""
    N = fused.size(0)
    I = torch.eye(N, device=fused.device, dtype=fused.dtype)
    return fused @ ((1 - DELTA) * torch.linalg.inv(I - DELTA * fused))


def apply_walk(op: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    cross = seed_column(pred_mask).to(op.dtype)
    return (op @ cross).T.reshape(-1, 64, 64)
