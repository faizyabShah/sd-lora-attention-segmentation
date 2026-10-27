#!/usr/bin/env python3
"""Regenerate the images chosen by ``select_figure_images.py`` and save every FTF stage for the paper figures.

For each image (class word, frozen CA seed, SA module 15, all 31 calls x 8 heads; the
same recorder and inputs as SG10/SG2, bit-identical to SG5 in env ``gen``):

  seed   CA seed c (min-max)
  REF    one observation-wise SA pass  mean_i(A_i c)
  P1     two passes                    mean_i(A_i^2 c)
  FTF    OVAM DenseCRF on minmax(P1) >= 0.45 (the frozen training-free pipeline)
  gt     annotation

Saves one compressed ``.npz`` per image to ``--cache`` (outside the repository) and checks
that the FTF image IoU equals the value recorded by SG5 for that image.

    python scripts/narmyn/attention/figure_maps.py --cache /path/to/cache --crf-site /path/to/crf
"""
from __future__ import annotations

import argparse
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
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402

SA, transfer = base.sa_common, base.transfer


@torch.inference_mode()
def passes(recorder, module, seed, device):
    flat = seed.reshape(-1, 1)
    one, two = torch.zeros_like(flat), torch.zeros_like(flat)
    calls = recorder.sa_qk[module]
    for q, k in calls:
        q, k = q.to(device).float(), k.to(device).float()
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        first = a @ flat
        one += first.mean(0); two += (a @ first).mean(0)
    return (one / len(calls)).reshape(64, 64), (two / len(calls)).reshape(64, 64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--crf-site", type=Path, required=True)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--selection", type=Path, default=ROOT / "outputs/segmentation/paper_figures/selection.json")
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    pipe, ca_modules, selected, seed_spec = base.setup(args)
    args.cache.mkdir(parents=True, exist_ok=True)
    for ds, picks in selection["datasets"].items():
        voc = ds == "voc"
        for pick in picks:
            for s in range(20):
                name, rows = (transfer.voc_rows if voc else transfer.coco_rows)(s)
                if name == pick["class"] or name.replace(" ", "-") == pick["class"]:
                    break
            row = next(r for r in rows if r["image_id"] == pick["image_id"])
            base.class_keys(pipe, name, row["attribution_prompt"], voc, args.device)
            manifest = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
            SA._RECORDERS.clear()
            sample = SA.live.generate_attention_sample(pipe, row, manifest, "float16", cross_map_kind="conditional")
            rec = SA._RECORDERS[-1]
            ca_res = {n: r for r, ns in sample.modules_by_resolution.items() for n in ns}
            seed = SA.ca_seed(rec, ca_modules, ca_res, seed_spec, args.device)
            ref, p1 = passes(rec, selected, seed, args.device)
            truth = np.asarray(sample.target_mask, bool)
            full = upsampled(minmax(p1), truth.shape)
            ftf = apply_ovam_densecrf(sample.generated_image, (full >= 0.45).cpu().numpy(), args.crf_site).astype(bool)
            iou = float((ftf & truth).sum() / max((ftf | truth).sum(), 1))
            np.savez_compressed(args.cache / f"{pick['image_id']}.npz", image=sample.generated_image, seed=seed.cpu().numpy(),
                                ref=ref.cpu().numpy(), p1=p1.cpu().numpy(), ftf=ftf, gt=truth)
            ok = abs(iou - pick["image_iou"]) < 1e-3
            print(f"{ds} {pick['class']:12s} {pick['image_id']}: FTF IoU {iou:.4f} (SG5 {pick['image_iou']:.4f}) {'OK' if ok else 'MISMATCH'}", flush=True)
            rec.ca_calls.clear(); rec.sa_qk.clear(); rec.token_calls.clear(); sample.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
