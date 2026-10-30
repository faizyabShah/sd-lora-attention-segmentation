#!/usr/bin/env python3
"""Resumable SG-combined-v1 C10 add-on, one VOC/COCO class per shard."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts import run_coco_cap_transfer as transfer, sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv, sg33_hierarchy, sg_combined_c10
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg_combined_c10"
SPEC = ROOT / "configs/sg_combined_c10.json"
RAW = ("S0",) + sg_combined_c10.NAMES
NAMES = tuple(n + suffix for n in RAW for suffix in ("", "+R2"))
SIDE_FIELDS = ("arrival_auc", "stage_auc", "retention", "change", "transport", "ipf")


def auc(score, truth):
    values = np.asarray(score).reshape(-1)
    labels = np.asarray(truth, bool).reshape(-1)
    pos = int(labels.sum()); neg = len(labels) - pos
    if not pos or not neg: return float("nan")
    ranks = rankdata(values, method="average")
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def diagnostic(maps, transient, truth, threshold):
    gt16 = cv.target_at_size(truth, 16).numpy().astype(bool)
    gt64 = cv.target_at_size(truth, 64).numpy().astype(bool)
    arrivals = transient["arrival"].numpy()
    arrival_auc = np.asarray([auc(-a, gt16) for a in arrivals], np.float64)
    stages = cv.normalize_batch(transient["stages"].reshape(6, 64, 64), "minmax").reshape(2, 3, 64, 64)
    stage_auc = np.asarray([[auc(stages[i, j].numpy(), gt64) for j in range(3)] for i in range(2)])
    coarse_supported = stages[0, 0].numpy() >= threshold
    retention = np.zeros((2, 2, 2), np.int64)  # route, GT FG/BG, retained/total
    for route in range(2):
        final_supported = stages[route, 2].numpy() >= threshold
        for category, mask in enumerate((gt64, ~gt64)):
            selected = coarse_supported & mask
            retention[route, category] = (int((selected & final_supported).sum()), int(selected.sum()))
    p2 = cv.normalize_batch(transient["completion"][1][None], "minmax")[0]
    change = np.stack((
        change_diagnostics(maps, ["C10a", "C10b"], truth, threshold)[0],
        change_diagnostics(maps, ["C10b", "C10c"], truth, threshold)[0],
        change_diagnostics({"C10b": maps["C10b"], "P2": p2}, ["C10b", "P2"], truth, threshold)[0]))
    return {"arrival_auc": arrival_auc, "stage_auc": stage_auc,
            "retention": retention, "change": change,
            "transport": transient["transport"], "ipf": transient["ipf"]}


def read_side(path):
    if not path.exists(): return {}
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["image_ids"]]
        if len(ids) != len(set(ids)): raise ValueError("duplicate C10 diagnostic image")
        return {key: tuple(data[field][i] for field in SIDE_FIELDS) for i, key in enumerate(ids)}


def save_side(path, rows):
    atomic_npz(path, image_ids=np.asarray(list(rows)),
               **{field: np.stack([row[i] for row in rows.values()])
                  for i, field in enumerate(SIDE_FIELDS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-controls", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    if tuple(spec["cells"]) != sg_combined_c10.NAMES or spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("C10 locked cells/thresholds changed")
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
    if writer.completed - set(side): raise ValueError("C10 metric shard ahead of diagnostics")
    for row in rows:
        image_id = row["image_id"]
        if image_id in writer.completed: continue
        with sg27_runtime.live_maps(runtime, row, args.dataset) as (sample, sa, seed, base, previous):
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            raw, transient = sg_combined_c10.aggregate(sa, seed)
            maps = {name: cv.normalize_batch(raw[name][None], "minmax")[0] for name in sg_combined_c10.NAMES}
            maps["S0"] = previous["S0"]
            if args.verify_controls and args.max_samples:
                sg33_maps, _ = sg33_hierarchy.aggregate(sa, seed)
                expected = cv.normalize_batch(sg33_maps["M2"][None], "minmax")[0]
                assert_control_close(maps["C10a"], expected, "C10a/SG33 M2 same-generation")
            raw_counts, raw_ap = cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),
                                               truth, "minmax", spec["thresholds"])
            counts = np.zeros((len(NAMES), len(spec["thresholds"]), 4), np.int64)
            ap = np.full(len(NAMES), np.nan)
            for i, name in enumerate(RAW):
                counts[2*i], ap[2*i] = raw_counts[i], raw_ap[i]
                counts[2*i+1] = transfer.crf_counts(maps[name], truth, image, args.crf_site)[0]
            diag = diagnostic(maps, transient, truth, spec["diagnostic_threshold"])
            side[image_id] = tuple(diag[field] for field in SIDE_FIELDS)
            save_side(side_path, side)
            writer.add(image_id, classname, counts, ap)
            print(f"[C10:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}", flush=True)
    if args.max_samples is None:
        if writer.completed != {r["image_id"] for r in all_rows} or set(side) != writer.completed:
            raise ValueError("incomplete C10 class shard")
        writer.finish()


if __name__ == "__main__": main()
