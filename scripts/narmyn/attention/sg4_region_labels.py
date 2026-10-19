"""SG4 segment with SA, label with CA (protocol Section 81; ``configs/sg4_region_labels.json``). Standalone family.

The 64x64 grid is partitioned into K regions from self-attention alone, then
every region gets one score from its evidence; the map is piecewise constant.

Partitions (shared by both tracks; deterministic k-means: k-means++ from seed 0,
3 restarts, 30 iterations, lowest inertia kept):
  SPEC   spectral clustering of S = (A_bar + A_bar^T) / 2: the K leading eigenvectors of
         D^-1/2 S D^-1/2, rows L2-normalized, then k-means
  ROWS   k-means of the L2-normalized rows of A_bar (pixels with similar attention distributions)
  K in {8, 16, 32}
Region labels: ``cmean`` (mean of the CA seed c), ``cmed`` (median of c),
``rmean`` (mean of REF = mean_i A_i c).

Diagnostic (never selected): the oracle mask that marks a region as object when
at least half of its 64x64 pixels are object, scored at annotation resolution.

    python scripts/narmyn/attention/sg4_region_labels.py   # self-test
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.segmentation.attention_regions import kmeans, region_map, attention_partitions
NAME = "sg4"
CONFIG = ROOT / "configs" / "sg4_region_labels.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg4_region_labels"
TRACKS = ("W", "T")
PARTITIONS = ("SPEC", "ROWS")
KS = (8, 16, 32)
LABELS = ("cmean", "cmed", "rmean")


def cell_names(track: str) -> list[str]:
    return [f"{track}/seed", f"{track}/REF"] + [f"{track}/{p}{k}_{lab}" for p in PARTITIONS for k in KS for lab in LABELS]


CONFIGURATIONS = tuple(n for t in TRACKS for n in cell_names(t))


def reference_for(cell: str) -> str:
    return f"{cell.split('/')[0]}/REF"



def partitions(mean: torch.Tensor) -> dict[str, torch.Tensor]:
    return attention_partitions(mean, ks=KS)


@torch.inference_mode()
def compute(recorder, module_name: str, seeds: torch.Tensor, truth64: torch.Tensor, device: str):
    calls = recorder.sa_qk[module_name]
    heads = calls[0][0].shape[0]
    if len(calls) != 31 or heads != 8:
        raise RuntimeError(f"Expected 31 calls x 8 heads, got {len(calls)} x {heads}")
    flat = seeds.T
    p = flat.shape[0]
    mean, one = torch.zeros((p, p), device=device), torch.zeros_like(flat)
    for query, key in calls:
        q, kk = query.to(device).float(), key.to(device).float()
        a = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        one += torch.matmul(a, flat).mean(0); mean += a.mean(0)
        del a, q, kk
    mean /= len(calls); ref = one / len(calls)
    parts = partitions(mean)
    maps = []
    for t in range(len(TRACKS)):
        maps += [flat[:, t], ref[:, t]]
        for name in (f"{pp}{k}" for pp in PARTITIONS for k in KS):
            lab = parts[name]
            maps += [region_map(lab, flat[:, t], "mean"), region_map(lab, flat[:, t], "median"), region_map(lab, ref[:, t], "mean")]
    obj = truth64.reshape(-1).float()
    diag = {"oracle": {}, "regions": {}}
    for name, lab in parts.items():
        k = int(lab.max()) + 1
        frac = torch.zeros(k, device=device).index_add_(0, lab, obj) / torch.bincount(lab, minlength=k).clamp_min(1)
        diag["oracle"][name] = (frac[lab] >= 0.5).float()  # 64x64 oracle mask, scored by the runner
        diag["regions"][name] = int(torch.unique(lab).numel())
    return torch.stack(maps), diag


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.cat([torch.randn(50, 2) + 5, torch.randn(50, 2) - 5])
    lab = kmeans(x, 2)
    assert len(set(lab[:50].tolist())) == 1 and len(set(lab[50:].tolist())) == 1 and lab[0] != lab[-1]
    assert torch.equal(kmeans(x, 2), lab), "k-means must be deterministic"
    m = region_map(torch.tensor([0, 0, 1, 1, 1]), torch.tensor([1.0, 3.0, 5.0, 6.0, 10.0]), "median")
    assert torch.allclose(m, torch.tensor([2.0, 2.0, 6.0, 6.0, 6.0]))
    assert len(CONFIGURATIONS) == 40 and reference_for("T/SPEC16_cmed") == "T/REF"
    print("SG4 self-test passed: deterministic k-means separates two clusters; region median; 40 cells")
