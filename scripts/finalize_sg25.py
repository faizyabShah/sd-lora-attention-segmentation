#!/usr/bin/env python3
"""Five-fold SG25 selection and Pre-SG26 first-arrival diagnostic summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.run_sg25 import ARRIVAL_FIELDS, BASE, NAMES, SPEC
from scripts.select_propagation_cv import best_threshold, run_screen
from src.segmentation import controlled_cv as cv


def load(out: Path, dataset: str, spec: dict, smoke: bool):
    files = sorted(p for p in (out / dataset / "shards").glob("class_*.npz")
                   if not p.name.endswith((".arrival.npz", ".tmp.npz")))
    result = cv.load_results(files)
    if result.configurations != NAMES or result.thresholds != tuple(spec["thresholds"]):
        raise ValueError("SG25 cells or threshold grid differ")
    if result.settings["spec"] != spec or result.settings["dataset"] != dataset:
        raise ValueError("foreign SG25 shard")
    expected = 490 if dataset == "voc" else 707
    if not smoke and (len(files) != 20 or len(result.image_ids) != expected or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} shards")
    arrivals = {}
    for file in files:
        with np.load(file.with_suffix(".arrival.npz"), allow_pickle=False) as data:
            for image_id, counts in zip(data["image_ids"], data["counts"]):
                if str(image_id) in arrivals:
                    raise ValueError("duplicate arrival image")
                arrivals[str(image_id)] = counts
    if set(arrivals) != set(result.image_ids):
        raise ValueError("arrival sidecar coverage mismatch")
    if dataset == "voc":
        folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv")
        if any(result.folds[i] != folds[image_id] for i, image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold assignment mismatch")
    return result, np.stack([arrivals[x] for x in result.image_ids])


def arrival_summary(result, arrivals):
    classes = np.asarray(result.classes)
    def describe(counts):
        out = []
        for k, values in enumerate(counts):
            x = dict(zip(ARRIVAL_FIELDS, (int(v) for v in values)))
            x["depth"] = k + 1
            x["tp_rate"] = x["tp"] / max(x["pixels"], 1)
            x["fp_rate"] = x["fp"] / max(x["pixels"], 1)
            x["tp_fp_ratio"] = x["tp"] / max(x["fp"], 1)
            out.append(x)
        return out
    return {"overall": describe(arrivals.sum(0)),
            "per_class": {name: describe(arrivals[classes == name].sum(0))
                          for name in sorted(set(classes))}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=BASE)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    spec = json.loads(SPEC.read_text())
    voc, va = load(args.output_dir, "voc", spec, args.smoke)
    coco, ca = load(args.output_dir, "coco", spec, args.smoke)
    if args.smoke:
        if len(voc.image_ids) != 1 or len(coco.image_ids) != 1:
            raise ValueError("smoke requires one VOC and one COCO image")
        print("SG25 + Pre-SG26 live VOC/COCO smoke passed", flush=True)
        return
    keep = np.ones(len(voc.image_ids), bool)
    thresholds = {name: best_threshold(voc, voc.index(name), keep)[0] for name in NAMES}
    screens = {
        "raw": run_screen(voc, "FINE-P2", [name for name in NAMES if not name.endswith("+R2") and name != "FINE-P2"],
                          {name: 1 for name in NAMES}),
        "final": run_screen(voc, "FINE-P2+R2", [name for name in NAMES if name.endswith("+R2") and name != "FINE-P2+R2"],
                            {name: 1 for name in NAMES}),
    }
    oof = {name: forced_oof(voc, name, "FINE-P2+R2" if name.endswith("+R2") else "FINE-P2") for name in NAMES}
    cells = {dataset: {name: metrics(result, name, "FINE-P2+R2" if name.endswith("+R2") else "FINE-P2", thresholds)
                       for name in NAMES} for dataset, result in (("voc", voc), ("coco", coco))}
    arrival = {"voc": arrival_summary(voc, va), "coco": arrival_summary(coco, ca)}
    summary = {"spec": spec, "selection": screens, "forced_oof": oof, "cells": cells,
               "first_arrival": arrival, "voc_thresholds_frozen_for_coco": True}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "per_class_iou.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "cell", "class", "iou", "delta_vs_fine"))
        for dataset in ("voc", "coco"):
            for name in NAMES:
                item = cells[dataset][name]
                for classname, iou in item["per_class_iou"].items():
                    writer.writerow((dataset, name, classname, iou, item["per_class_delta_vs_reference"][classname]))
    with (out / "first_arrival.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "class", "depth", *ARRIVAL_FIELDS, "tp_rate", "fp_rate", "tp_fp_ratio"))
        for dataset in ("voc", "coco"):
            for name, rows in (("ALL", arrival[dataset]["overall"]), *arrival[dataset]["per_class"].items()):
                for row in rows:
                    writer.writerow((dataset, name, row["depth"], *(row[x] for x in ARRIVAL_FIELDS),
                                     row["tp_rate"], row["fp_rate"], row["tp_fp_ratio"]))
    lines = ["# SG25 + Pre-SG26 results", "", "A3 CA frozen; five-fold VOC selection; COCO frozen transfer.", "",
             "| Cell | VOC OOF | OOF delta | Held-out wins | Full VOC | VOC threshold | COCO |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item = oof[name]; v = cells["voc"][name]; c = cells["coco"][name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{v['miou']:.4f} | {v['voc_frozen_threshold']:.3f} | {c['miou']:.4f} |")
    lines.extend(["", "## First-arrival diagnostic", "", "Fixed threshold .425 after per-depth minmax; diagnostic only.", "",
                  "| Dataset | Depth | Pixels | TP rate | TP/FP | FP >16 | FP >64 | Separate FP |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset in ("voc", "coco"):
        for row in arrival[dataset]["overall"]:
            lines.append(f"| {dataset} | {row['depth']} | {row['pixels']} | {row['tp_rate']:.3f} | "
                         f"{row['tp_fp_ratio']:.3f} | {row['fp_gt16']} | {row['fp_gt64']} | {row['fp_separate']} |")
    lines.extend(["", "Per-class IoUs and arrival counts are in the CSV files; full fold and paired statistics are in `results.json`.", ""])
    (out / "report.md").write_text("\n".join(lines))
    (out / "completed.json").write_text(json.dumps({"voc_images": len(voc.image_ids),
        "coco_images": len(coco.image_ids), "all_required_stages_complete": True}, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
