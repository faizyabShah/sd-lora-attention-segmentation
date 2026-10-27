#!/usr/bin/env python3
"""Select and regenerate the images for the paper's propagation-depth figure (one, two, three SA passes).

Selection (fixed rule, no image inspected), from the SG22 all-call raw cells at threshold 0.45:

  completion   the image at the 90th percentile of IoU(ALL-P2) - IoU(ALL-P1)   (VOC-Sim, COCO-Cap)
  leakage      the image at the 10th percentile of IoU(ALL-P3) - IoU(ALL-P2)   (VOC-Sim, COCO-Cap)

For each image, the frozen CA seed is propagated through SA module 15 with all 31 calls x 8 heads,
p_k = mean_i(A_i^k c) for k = 1, 2, 3, and saved with the GT to ``--cache`` (outside the repository).
Each raw-map IoU at 0.45 is checked against the SG22 per-image counts. Writes the selection to
``outputs/segmentation/paper_figures/passes_selection.json``.

    python scripts/narmyn/attention/figure_passes_maps.py --cache /path/to/cache --crf-site /path/to/crf
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.narmyn.attention.run_sg10_bottleneck as base  # noqa: E402  (recorder, setup, class keys)
from scripts.run_fusion_calibration import upsampled  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402

SA, transfer = base.sa_common, base.transfer
SG22 = ROOT / "outputs/segmentation/controlled_attention_cv/sg22"
T45 = 6  # index of 0.45 in the SG22 17-point grid
RULES = (("completion", "ALL-P2", "ALL-P1", 0.90), ("leakage", "ALL-P3", "ALL-P2", 0.10))


def image_ious(ds: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(SG22 / ds / "shards" / "*.npz"))):
        if ".diagnostics." in f or ".arrival." in f:
            continue
        d = np.load(f)
        names = [str(n) for n in d["configurations"]]
        c = d["counts"][:, :, T45, :3].astype(float)
        iou = c[..., 0] / np.maximum(c.sum(-1), 1)
        for j, img in enumerate(d["image_ids"]):
            out[str(img)] = {"class": str(d["classes"][j]), **{n: float(iou[i, j]) for i, n in enumerate(names)}}
    return out


def select() -> dict:
    picks = {}
    for ds in ("voc", "coco"):
        ious = image_ious(ds)
        for label, a, b, q in RULES:
            ids = sorted(ious)
            delta = np.array([ious[i][a] - ious[i][b] for i in ids])
            target = np.quantile(delta, q)
            k = int(np.argmin(np.abs(delta - target) + 1e-9 * np.arange(len(ids))))
            r = ious[ids[k]]
            picks[f"{ds}_{label}"] = {"dataset": ds, "rule": f"{label}: q{q:.2f} of IoU({a})-IoU({b}) at 0.45",
                                      "image_id": ids[k], "class": r["class"], "delta": float(delta[k]),
                                      "sg22_iou": {n: r[n] for n in ("ALL-P1", "ALL-P2", "ALL-P3")}}
    return picks


@torch.inference_mode()
def passes(recorder, module, seed, device, depth=3):
    flat = seed.reshape(-1, 1)
    acc = [torch.zeros_like(flat) for _ in range(depth)]
    calls = recorder.sa_qk[module]
    for q, k in calls:
        q, k = q.to(device).float(), k.to(device).float()
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        x = flat
        for d in range(depth):
            x = a @ x
            acc[d] += x.mean(0)
    return [(m / len(calls)).reshape(64, 64) for m in acc]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--crf-site", type=Path, required=True)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    picks = select()
    out = ROOT / "outputs/segmentation/paper_figures/passes_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "outputs/segmentation/controlled_attention_cv/sg22 (raw, threshold 0.45)",
                               "picks": picks}, indent=2) + "\n")
    pipe, ca_modules, selected, seed_spec = base.setup(args)
    args.cache.mkdir(parents=True, exist_ok=True)
    for key, pick in picks.items():
        voc = pick["dataset"] == "voc"
        for s in range(20):
            name, rows = (transfer.voc_rows if voc else transfer.coco_rows)(s)
            if name == pick["class"] or name.replace(" ", "_") == pick["class"] or name.replace(" ", "-") == pick["class"]:
                break
        row = next(r for r in rows if r["image_id"] == pick["image_id"])
        base.class_keys(pipe, name, row["attribution_prompt"], voc, args.device)
        manifest = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
        SA._RECORDERS.clear()
        sample = SA.live.generate_attention_sample(pipe, row, manifest, "float16", cross_map_kind="conditional")
        rec = SA._RECORDERS[-1]
        ca_res = {n: r for r, ns in sample.modules_by_resolution.items() for n in ns}
        seed = SA.ca_seed(rec, ca_modules, ca_res, seed_spec, args.device)
        maps = passes(rec, selected, seed, args.device)
        truth = np.asarray(sample.target_mask, bool)
        report = []
        for k, m in enumerate(maps, 1):
            mask = (upsampled(minmax(m), truth.shape) >= 0.45).cpu().numpy()
            iou = float((mask & truth).sum() / max((mask | truth).sum(), 1))
            ref = pick["sg22_iou"][f"ALL-P{k}"]
            report.append(f"P{k} {iou:.4f} (SG22 {ref:.4f}){'' if abs(iou - ref) < 5e-3 else ' MISMATCH'}")
        np.savez_compressed(args.cache / f"passes_{pick['image_id']}.npz", image=sample.generated_image, seed=seed.cpu().numpy(),
                            p1=maps[0].cpu().numpy(), p2=maps[1].cpu().numpy(), p3=maps[2].cpu().numpy(), gt=truth)
        print(f"{key:16s} {pick['class']:12s} {pick['image_id']}: " + "; ".join(report), flush=True)
        rec.ca_calls.clear(); rec.sa_qk.clear(); rec.token_calls.clear(); sample.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
