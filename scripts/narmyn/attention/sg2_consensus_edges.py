"""SG2 consensus-edge SA graph (protocol Section 81; ``configs/sg2_consensus_edges.json``). Standalone family.

For the 248 observations ``A_i`` (31 calls x 8 heads) of frozen SA module 15,
each edge (p, q) gets a reliability computed exactly and densely, streaming
over the observations (no top-k, no pruning):

  cv   r = mean_i A_i / sqrt(mean_i A_i^2)   (= 1 / sqrt(1 + CV^2); 1 when all observations agree)
  occ  r = (#{i : A_i(p, q) > 1/P} + 0.5) / (N + 1)   (share of observations above uniform; never 0)

Edges are reweighted, never removed: ``A~ = rownorm(A . r^gamma)``, gamma in {0.5, 1, 2}, in two forms:

  G   one consensus graph  W = rownorm(A_bar . r^gamma); n1 = W c, n2 = W (W c)
  O   per observation      A~_i = rownorm(A_i . r^gamma); n1 = mean_i A~_i c, n2 = mean_i A~_i^2 c

References (recomputed in the package): ``seed`` (c), ``REF`` (mean_i A_i c), ``P1``
(mean_i A_i^2 c) and the diagnostic ``MG2`` (A_bar^2 c, the mean-graph two-pass).

    python scripts/narmyn/attention/sg2_consensus_edges.py   # self-test
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
NAME = "sg2"
CONFIG = ROOT / "configs" / "sg2_consensus_edges.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg2_consensus_edges"
TRACKS = ("W", "T")
RELIABILITY = ("cv", "occ")
GAMMAS = (0.5, 1.0, 2.0)
FORMS = ("G", "O")


def _tag(g: float) -> str:
    return f"g{int(round(g * 10)):02d}"


def variants() -> list[tuple[str, str, float]]:
    return [(form, rel, g) for form in FORMS for rel in RELIABILITY for g in GAMMAS]


def cell_names(track: str) -> list[str]:
    names = [f"{track}/seed", f"{track}/REF", f"{track}/P1", f"{track}/MG2"]
    for n in (1, 2):
        names += [f"{track}/{form}_{rel}_{_tag(g)}_n{n}" for form, rel, g in variants()]
    return names


CONFIGURATIONS = tuple(n for t in TRACKS for n in cell_names(t))


def reference_for(cell: str) -> str:
    track, rest = cell.split("/", 1)
    return f"{track}/P1" if rest.endswith("_n2") or rest in ("P1", "MG2") else f"{track}/REF"


def rownorm(w: torch.Tensor) -> torch.Tensor:
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _affinity(query, key, device):
    q, k = query.to(device).float(), key.to(device).float()
    return torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)  # [heads, P, P]


@torch.inference_mode()
def compute(recorder, module_name: str, seeds: torch.Tensor, truth64: torch.Tensor, device: str):
    """Maps [len(CONFIGURATIONS), P] in CONFIGURATIONS order, and per-image diagnostics."""
    calls = recorder.sa_qk[module_name]
    heads = calls[0][0].shape[0]
    if len(calls) != 31 or heads != 8:
        raise RuntimeError(f"Expected 31 calls x 8 heads, got {len(calls)} x {heads}")
    flat = seeds.T  # [P, 2]
    p = flat.shape[0]
    s1 = torch.zeros((p, p), device=device); s2 = torch.zeros_like(s1); cnt = torch.zeros_like(s1)
    one, two = torch.zeros_like(flat), torch.zeros_like(flat)
    for query, key in calls:
        a = _affinity(query, key, device)
        first = torch.matmul(a, flat)
        one += first.mean(0); two += torch.matmul(a, first).mean(0)
        s1 += a.sum(0); s2 += (a * a).sum(0); cnt += (a > 1.0 / p).float().sum(0)
        del a, first
    n_obs = len(calls) * heads
    mean = s1 / n_obs
    rel = {"cv": mean / torch.sqrt(s2 / n_obs).clamp_min(1e-30), "occ": (cnt + 0.5) / (n_obs + 1)}
    del s1, s2, cnt
    ref, p1 = one / len(calls), two / len(calls)
    mg2 = torch.matmul(mean, torch.matmul(mean, flat))
    weights = {(rel_name, g): rel[rel_name] ** g for rel_name in RELIABILITY for g in GAMMAS}
    out = {}
    for (rel_name, g), w in weights.items():
        graph = rownorm(mean * w)
        g1 = torch.matmul(graph, flat)
        out[("G", rel_name, g, 1)], out[("G", rel_name, g, 2)] = g1, torch.matmul(graph, g1)
        del graph
    o1 = {k: torch.zeros_like(flat) for k in weights}; o2 = {k: torch.zeros_like(flat) for k in weights}
    for query, key in calls:
        a = _affinity(query, key, device)
        for k, w in weights.items():
            at = rownorm(a * w)
            f = torch.matmul(at, flat)
            o1[k] += f.mean(0); o2[k] += torch.matmul(at, f).mean(0)
            del at, f
        del a
    for k in weights:
        out[("O", k[0], k[1], 1)], out[("O", k[0], k[1], 2)] = o1[k] / len(calls), o2[k] / len(calls)
    maps = []
    for t in range(len(TRACKS)):
        maps += [flat[:, t], ref[:, t], p1[:, t], mg2[:, t]]
        for n in (1, 2):
            maps += [out[(form, r, g, n)][:, t] for form, r, g in variants()]
    diag = {"edge_reliability": edge_reliability(mean, rel, truth64)}
    return torch.stack(maps), diag


def edge_reliability(mean: torch.Tensor, rel: dict, truth64: torch.Tensor) -> dict:
    """A_bar-weighted mean reliability of edges object->object, object->background, background->object,
    background->background (row pixel -> column pixel; leakage into a background pixel uses background->object)."""
    obj = truth64.reshape(-1).bool()
    out = {}
    for name, r in rel.items():
        cell = {}
        for a_name, rows in (("obj", obj), ("bg", ~obj)):
            for b_name, cols in (("obj", obj), ("bg", ~obj)):
                if rows.any() and cols.any():
                    w = mean[rows][:, cols]
                    cell[f"{a_name}->{b_name}"] = [float((w * r[rows][:, cols]).sum()), float(w.sum())]
        out[name] = cell
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(5, 6, 6).softmax(-1)
    m = a.mean(0)
    r = m / torch.sqrt((a * a).mean(0))
    assert (r <= 1 + 1e-6).all() and (r > 0).all()
    same = torch.stack([a[0]] * 5)
    assert torch.allclose(same.mean(0) / torch.sqrt((same * same).mean(0)), torch.ones(6, 6))
    w = rownorm(m * r)
    assert torch.allclose(w.sum(-1), torch.ones(6))
    assert len(CONFIGURATIONS) == 56 and reference_for("W/O_cv_g10_n2") == "W/P1" and reference_for("T/G_occ_g05_n1") == "T/REF"
    print("SG2 self-test passed: reliability in (0, 1], 1 for identical observations; row-stochastic graphs; 56 cells")
