#!/usr/bin/env python3
"""Stream one image at a time through locked SG31 maps and compact diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_coco_cap_transfer as transfer, sg27_runtime
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv, sg31_seediff
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg31"
SPEC_PATH = ROOT / "configs/sg31.json"
RAW = sg31_seediff.NAMES
NAMES = tuple(n + suffix for n in RAW for suffix in ("", "+R2"))
SIDE_FIELDS = ("seeds", "changes", "pair_changes", "background")
CHANGE_PAIRS = (("S1", "S3"), ("S2", "S3"), ("S1", "S4"), ("S2", "S4"))
BG_DIAG_FIELDS = ("bg_precision_at_0p5", "mean_bg_inside_gt", "mean_bg_outside_gt",
                  "bg_seed_fraction", "tp_removed", "fp_removed", "fp_removed_0_16",
                  "fp_removed_16_64", "fp_removed_gt64", "separate_fp_removed")


def read_side(path):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        ids = [str(i) for i in data["image_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate SG31 sidecar image")
        return {image_id: tuple(data[field][i] for field in SIDE_FIELDS)
                for i, image_id in enumerate(ids)}


def save_side(path, rows):
    atomic_npz(path, image_ids=np.asarray(list(rows)),
               **{field: np.stack([row[j] for row in rows.values()])
                  for j, field in enumerate(SIDE_FIELDS)})


def background_diagnostics(maps, support, truth, threshold):
    gt = cv.target_at_size(truth).cpu().numpy().astype(bool)
    distance = ndimage.distance_transform_edt(~gt)
    out = []
    for name, source in (("S6", "S0"), ("S7", "S0"), ("S8", "S3"), ("S9", "S4")):
        bg, fraction = support[name]
        b = bg.cpu().numpy()
        original = (maps[source] >= threshold).cpu().numpy()
        refined = (maps[name] >= threshold).cpu().numpy()
        removed = original & ~refined
        labels, _ = ndimage.label(original)
        touched = np.unique(labels[original & gt])
        separate = original & ~gt & ~np.isin(labels, touched[touched > 0])
        selected = b >= .5
        out.append((float((selected & ~gt).sum() / max(selected.sum(), 1)),
                    float(b[gt].mean()) if gt.any() else float("nan"),
                    float(b[~gt].mean()) if (~gt).any() else float("nan"), fraction,
                    int((removed & gt).sum()), int((removed & ~gt).sum()),
                    int((removed & ~gt & (distance <= 16)).sum()),
                    int((removed & ~gt & (distance > 16) & (distance <= 64)).sum()),
                    int((removed & ~gt & (distance > 64)).sum()),
                    int((removed & separate).sum())))
    return np.asarray(out, np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--class-shard", type=int, choices=range(20), required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--verify-controls", action="store_true")
    p.add_argument("--output-dir", type=Path, default=BASE)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    args = p.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if tuple(spec["cells"]) != RAW or spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("SG31 locked cell/threshold list changed")
    runtime = sg27_runtime.make_runtime(args.model)
    _import_densecrf(args.crf_site)
    classname, all_rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if args.dataset == "voc" else {
        r["image_id"]: 0 for r in all_rows}
    stem = f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}'
    path = args.output_dir / args.dataset / "shards" / (stem + ".npz")
    side_path = path.with_suffix(".diagnostics.npz")
    settings = {"spec": spec, "dataset": args.dataset, "model": args.model,
                "ca_modules": list(runtime["ca_modules"]), "sa_modules": runtime["modules"],
                "crf": list(OVAM_CRF_PARAMETERS)}
    writer = cv.ShardWriter(path, NAMES, folds, settings, spec["thresholds"])
    side = read_side(side_path)
    if writer.completed - set(side):
        raise ValueError("SG31 metric shard ahead of diagnostic sidecar")
    for row in rows:
        image_id = row["image_id"]
        if image_id in writer.completed:
            continue
        with sg27_runtime.live_maps(runtime, row, args.dataset) as (sample, sa, seed, base, previous):
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            maps, seed_stats, support = sg31_seediff.aggregate(sa, seed)
            if args.verify_controls:
                torch.testing.assert_close(maps["S0"], previous["S0"], rtol=1e-5, atol=2e-6)
                # S1 is independently accumulated before minmax; a few GPU
                # rounding units can differ from SG27's 16->64 accumulator.
                torch.testing.assert_close(maps["S1"], previous["S1"], rtol=1e-5, atol=1e-5)
            maps["S0"] = previous["S0"]
            maps["S1"] = previous["S1"]
            raw_counts, raw_ap = cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),
                                                truth, "minmax", spec["thresholds"])
            counts = np.zeros((len(NAMES), len(spec["thresholds"]), 4), np.int64)
            ap = np.full(len(NAMES), np.nan)
            for i, name in enumerate(RAW):
                counts[2*i], ap[2*i] = raw_counts[i], raw_ap[i]
                counts[2*i+1] = transfer.crf_counts(maps[name], truth, image, args.crf_site)[0]
            changes = change_diagnostics(maps, list(RAW), truth, spec["diagnostic_threshold"])
            pair_changes = np.stack([change_diagnostics({"S0": maps[a], "S1": maps[b]},
                        ["S0", "S1"], truth, spec["diagnostic_threshold"])[0]
                        for a, b in CHANGE_PAIRS])
            bg = background_diagnostics(maps, support, truth, spec["diagnostic_threshold"])
            side[image_id] = (seed_stats.numpy(), changes, pair_changes, bg)
            save_side(side_path, side)
            writer.add(image_id, classname, counts, ap)
            print(f"[sg31:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}", flush=True)
    if args.max_samples is None:
        if writer.completed != {r["image_id"] for r in all_rows} or set(side) != writer.completed:
            raise ValueError("incomplete SG31 class shard")
        writer.finish()


if __name__ == "__main__":
    main()
