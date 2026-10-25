"""SG28: PEARL-inspired Q/K alignment and local binary-score refinement.

These are adaptations to SD1.5 attention, not PEARL's CLIP pipeline.
No ground truth enters either operator.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from src.segmentation.sg25_sa import _apply, _attention

PA_NAMES = ("P1", "P2", "P3")
LAP_NAMES = ("B1", "B2", "C1", "C2", "C3", "D1", "D2", "D3")
NAMES = ("P0",) + PA_NAMES + LAP_NAMES
PA_FIELDS = ("alignment_error_before", "alignment_error_after", "row_entropy_before",
             "row_entropy_after", "effective_neighbors_before", "effective_neighbors_after")
LAP_FIELDS = ("cg_iterations", "relative_residual", "mean_abs_score_change",
              "pixels_increased", "pixels_decreased", "signed_score_change")


def _sample_rows(image_id: str, device: torch.device, count: int) -> torch.Tensor:
    digest = hashlib.sha256(image_id.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return torch.as_tensor(rng.choice(4096, count, replace=False), device=device, dtype=torch.long)


def _neighborhood(attention: torch.Tensor, rows: torch.Tensor) -> tuple[float, float]:
    probs = attention.index_select(1, rows)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    return float(entropy.mean()), float(entropy.exp().mean())


def _pa_attention(q: torch.Tensor, k: torch.Tensor, qc: torch.Tensor,
                  kc: torch.Tensor, rotate: bool) -> tuple[torch.Tensor, float]:
    if rotate:
        covariance = torch.bmm(kc.transpose(1, 2), qc)
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        rotation = torch.bmm(u, vh)
        key = torch.bmm(k, rotation)
        error = float((torch.bmm(kc, rotation) - qc).square().mean().sqrt())
        query = q
    else:
        query, key = qc, kc
        error = float((kc - qc).square().mean().sqrt())
    attention = torch.softmax(torch.bmm(query, key.transpose(1, 2)) /
                              math.sqrt(q.shape[-1]), dim=-1)
    return attention, error


@torch.inference_mode()
def pa_maps(sa_qk: list[tuple[torch.Tensor, torch.Tensor]], seed: torch.Tensor,
            image_id: str, diagnostic_rows: int = 64) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Apply each PA variant to every head/call before its two SA passes."""
    if seed.shape != (64, 64) or len(sa_qk) != 11:
        raise ValueError("SG28 PA requires one 64x64 seed and 11 fine SA calls")
    rows = _sample_rows(image_id, seed.device, diagnostic_rows)
    c = seed.float().reshape(-1)
    sums = torch.zeros((3, 64, 64), device=seed.device, dtype=torch.float32)
    diagnostics = np.zeros((3, len(PA_FIELDS)), np.float64)
    for call, pair in enumerate(sa_qk):
        q, k = pair
        if q.shape != k.shape or q.shape[:2] != (8, 4096):
            raise ValueError(f"SG28 fine Q/K dimensions changed at call {call}")
        q, k = q.float(), k.float()
        ordinary = _attention(pair, 64)
        old_entropy, old_effective = _neighborhood(ordinary, rows)
        del ordinary
        weight = q.norm(dim=-1)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        weighted_q = q - (weight[..., None] * q).sum(dim=1, keepdim=True)
        weighted_k = k - (weight[..., None] * k).sum(dim=1, keepdim=True)
        uniform_q, uniform_k = q - q.mean(dim=1, keepdim=True), k - k.mean(dim=1, keepdim=True)
        before = ((weighted_k - weighted_q).square().mean().sqrt(),
                  (k - q).square().mean().sqrt(),
                  (uniform_k - uniform_q).square().mean().sqrt())
        for index, (name, qc, kc, rotate) in enumerate((
                ("P1", weighted_q, weighted_k, True),
                ("P2", weighted_q, weighted_k, False),
                ("P3", uniform_q, uniform_k, True))):
            attention, error = _pa_attention(q, k, qc, kc, rotate)
            after_entropy, after_effective = _neighborhood(attention, rows)
            sums[index] += _apply(attention, _apply(attention, c)).sum(0).reshape(64, 64)
            diagnostics[index] += (float(before[index]), error, old_entropy,
                                   after_entropy, old_effective, after_effective)
            del attention
    return {name: sums[index] / 88 for index, name in enumerate(PA_NAMES)}, diagnostics / 11


def _grid_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("SG28 boundary graph requires an RGB generated image")
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    return F.interpolate(torch.from_numpy(gray)[None, None], size=(64, 64),
                         mode="area")[0, 0].numpy().astype(np.float64).reshape(-1)


def _edges(gray: np.ndarray, boundary: bool, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(4096, dtype=np.int32).reshape(64, 64)
    source = np.concatenate((grid[:, :-1].reshape(-1), grid[:-1, :].reshape(-1)))
    target = np.concatenate((grid[:, 1:].reshape(-1), grid[1:, :].reshape(-1)))
    weight = (np.exp(-kappa * np.abs(gray[source] - gray[target])) if boundary
              else np.ones(len(source), np.float64))
    return source, target, weight


def _system(source: np.ndarray, target: np.ndarray, weight: np.ndarray,
            rho: np.ndarray) -> sparse.csr_matrix:
    degree = np.bincount(source, weights=weight, minlength=4096)
    degree += np.bincount(target, weights=weight, minlength=4096)
    row = np.concatenate((source, target, np.arange(4096)))
    col = np.concatenate((target, source, np.arange(4096)))
    values = np.concatenate((-weight, -weight, degree + rho))
    return sparse.coo_matrix((values, (row, col)), shape=(4096, 4096)).tocsr()


def _solve(graph: tuple[np.ndarray, np.ndarray, np.ndarray], rho: np.ndarray,
           seed: np.ndarray, rtol: float, maxiter: int) -> tuple[np.ndarray, int, float]:
    matrix = _system(*graph, rho)
    rhs = rho * seed
    iterations = 0
    def callback(_):
        nonlocal iterations
        iterations += 1
    inv_diagonal = 1.0 / matrix.diagonal()
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x: inv_diagonal*x)
    solution, info = cg(matrix, rhs, x0=seed, rtol=rtol, atol=0.0,
                        maxiter=maxiter, M=preconditioner, callback=callback)
    residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-30))
    if info != 0 or not np.isfinite(solution).all() or residual > 5*rtol:
        raise RuntimeError(f"SG28 CG accuracy failure: info={info}, iterations={iterations}, residual={residual:.3g}")
    return solution, iterations, residual


def laplacian_maps(fine: torch.Tensor, c16: torch.Tensor, c32: torch.Tensor,
                   image: np.ndarray, *, kappa: float = 5.0, tau: float = 1.0,
                   epsilon: float = 1e-6, rtol: float = 1e-8,
                   maxiter: int = 2000) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, np.ndarray]]:
    """Solve eight fixed local systems on normalized 64x64 SG27 base maps."""
    if any(x.shape != (64, 64) for x in (fine, c16, c32)) or tau != 1.0:
        raise ValueError("SG28 Laplacian graph is frozen at 64x64 with tau=1")
    f, c16, c32 = (x.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
                   for x in (fine, c16, c32))
    if any(np.min(x) < -1e-5 or np.max(x) > 1.00001 for x in (f, c16, c32)):
        raise ValueError("SG28 Laplacian expects independently normalized base maps")
    gray = _grid_image(image)
    graphs = {False: _edges(gray, False, kappa), True: _edges(gray, True, kappa)}
    disagreement = (np.abs(f-c16) + np.abs(f-c32) + np.abs(c16-c32)) / 3.0
    trust = {"C16": np.clip(1-np.abs(f-c16), epsilon, 1),
             "C32": np.clip(1-np.abs(f-c32), epsilon, 1),
             "C3": np.clip(1-disagreement, epsilon, 1)}
    methods = (("B1", False, None), ("B2", True, None),
               ("C1", False, "C16"), ("C2", False, "C32"), ("C3", False, "C3"),
               ("D1", True, "C16"), ("D2", True, "C32"), ("D3", True, "C3"))
    maps = {}
    diagnostics = np.zeros((len(methods), len(LAP_FIELDS)), np.float64)
    for index, (name, boundary, confidence) in enumerate(methods):
        rho = np.ones(4096, np.float64) if confidence is None else trust[confidence]
        solution, iterations, residual = _solve(graphs[boundary], rho, f, rtol, maxiter)
        diff = solution - f
        diagnostics[index] = (iterations, residual, float(np.abs(diff).mean()),
                              int((diff > 1e-7).sum()), int((diff < -1e-7).sum()), float(diff.sum()))
        maps[name] = torch.from_numpy(solution.reshape(64, 64).astype(np.float32)).to(fine.device)
    return maps, diagnostics, trust
