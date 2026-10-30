#!/usr/bin/env python3
"""Merge SG-combined-v1 C10 add-on with VOC selection and frozen COCO transfer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg_combined_c10 import BASE, NAMES, RAW, SIDE_FIELDS, SPEC
from scripts.select_propagation_cv import best_threshold, run_screen
from src.segmentation import controlled_cv as cv


def load(out, dataset, spec, smoke):
    files = sorted(p for p in (out / dataset / "shards").glob("class_*.npz")
                   if not p.name.endswith((".diagnostics.npz", ".tmp.npz")))
    result = cv.load_results(files)
    if result.configurations != NAMES or result.thresholds != tuple(spec["thresholds"]):
        raise ValueError("C10 cell/threshold mismatch")
    if result.settings["spec"] != spec or result.settings["dataset"] != dataset:
        raise ValueError("foreign C10 shard")
    if not smoke and (len(files) != 20 or len(result.image_ids) != (490 if dataset == "voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} C10 coverage")
    side = {}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"), allow_pickle=False) as data:
            for i, image_id in enumerate(data["image_ids"]):
                key = str(image_id)
                if key in side: raise ValueError("duplicate C10 diagnostic image")
                side[key] = tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side) != set(result.image_ids): raise ValueError("C10 diagnostic coverage mismatch")
    return result, {field: np.stack([side[x][i] for x in result.image_ids])
                    for i, field in enumerate(SIDE_FIELDS)}


def finite_mean(values):
    values = np.asarray(values, np.float64)
    keep = np.isfinite(values)
    return float(values[keep].mean()) if keep.any() else None


def changes(values):
    total = np.asarray(values).sum(0)
    d = {name: int(total[i]) for i, name in enumerate(CHANGE_FIELDS)}
    d["fp_0_16_added"] = d["fp_gained"] - d["fp_gt16_gained"]
    d["fp_16_64_added"] = d["fp_gt16_gained"] - d["fp_gt64_gained"]
    d["fp_gt64_added"] = d["fp_gt64_gained"]
    d["eta_tp_per_fp_added"] = d["tp_gained"] / max(d["fp_gained"], 1)
    d["eta_far_tp_per_fp_gt64_added"] = d["tp_gained"] / max(d["fp_gt64_gained"], 1)
    return d


def mechanism(side):
    ret = side["retention"].sum(0)
    retention = {route: {
        category: {"retained": int(ret[i, j, 0]), "coarse_supported": int(ret[i, j, 1]),
                   "fraction": float(ret[i, j, 0] / max(ret[i, j, 1], 1))}
        for j, category in enumerate(("GT_foreground", "GT_background"))}
        for i, route in enumerate(("C10a_ordinary", "C10b_constrained"))}
    return {
        "coarse_M2_minus_arrival_AUC_native16_mean_image_call": finite_mean(side["arrival_auc"]),
        "scale_AUC_64x64_sampled_GT_mean_image": {
            route: {stage: finite_mean(side["stage_auc"][:, i, j])
                    for j, stage in enumerate(("16", "32", "64"))}
            for i, route in enumerate(("C10a_ordinary", "C10b_constrained"))},
        "coarse_supported_retention_at_0p45": retention,
        "changes_at_full_resolution_0p45": {
            contrast: changes(side["change"][:, i])
            for i, contrast in enumerate(("C10b_vs_C10a", "C10c_vs_C10b",
                                          "ordinary_P2_vs_C10b_diagnostic_only"))},
        "transport": {transition: {field: finite_mean(side["transport"][:, i, j])
                         for j, field in enumerate(("mean_error", "max_error", "p95_error", "fallback_parents"))}
                      for i, transition in enumerate(("16_to_32", "32_to_64"))},
        "M2_IPF": {field: finite_mean(side["ipf"][:, i])
                   for i, field in enumerate(("row_error", "column_error", "iterations"))},
        "diagnostic_note": "GT enters diagnostics only; AUC is mean per image, not pooled pixel AUC; "
                           "fixed 0.45 comparisons do not use cell-selected mask thresholds"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); spec = json.loads(SPEC.read_text())
    voc, vd = load(args.output_dir, "voc", spec, args.smoke)
    coco, cd = load(args.output_dir, "coco", spec, args.smoke)
    if args.smoke:
        if len(voc.image_ids) != 1 or len(coco.image_ids) != 1:
            raise ValueError("C10 smoke needs one image per dataset")
        for side in (vd, cd):
            if side["stage_auc"].shape[1:] != (2, 3) or side["retention"].shape[1:] != (2, 2, 2):
                raise ValueError("C10 diagnostic sidecar shape mismatch")
        print("SG-combined C10 smoke passed: three raw/R2 cells and diagnostics", flush=True)
        return
    keep = np.ones(len(voc.image_ids), bool)
    thresholds = {name: best_threshold(voc, voc.index(name), keep)[0] for name in NAMES}
    screens = {stage: run_screen(voc, "S0" + suffix,
                                [n + suffix for n in RAW[1:]],
                                {n + suffix: 1 for n in RAW[1:]})
               for stage, suffix in (("raw", ""), ("final", "+R2"))}
    oof = {name: forced_oof(voc, name, "S0+R2" if name.endswith("+R2") else "S0")
           for name in NAMES}
    contrasts = {f"{a}_vs_{b}": forced_oof(voc, a, b)
                 for a, b in (("C10b", "C10a"), ("C10c", "C10b"),
                              ("C10c", "C10a"), ("C10b+R2", "C10a+R2"),
                              ("C10c+R2", "C10b+R2"))}
    cells = {dataset: {name: metrics(result, name,
                                    "S0+R2" if name.endswith("+R2") else "S0", thresholds)
                       for name in NAMES}
             for dataset, result in (("voc", voc), ("coco", coco))}
    output = {"spec": spec, "selection": screens, "forced_oof": oof,
              "causal_contrasts": contrasts, "cells": cells,
              "mechanism": {"voc": mechanism(vd), "coco": mechanism(cd)},
              "voc_thresholds_frozen_for_coco": True}
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (out / "per_class_iou.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("dataset", "cell", "class", "iou", "delta_vs_same_stage_S0"))
        for dataset in ("voc", "coco"):
            for name in NAMES:
                for cls, iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset, name, cls, iou,
                                     cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines = ["# SG-combined-v1 C10 add-on", "",
             "Three selectable Markov-source cells, 40 complete class shards, 490 VOC and 707 COCO images.",
             "VOC-only selection; COCO thresholds frozen. S0 is a same-generation reference only.", "",
             "| Cell | VOC forced OOF | Δ vs S0 | Direct fold wins | Full VOC | VOC threshold | Frozen COCO |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item = oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | "
                     f"{cells['coco'][name]['miou']:.4f} |")
    lines += ["", f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}", "",
              "Scale AUC, M2 coarse retention, TP/FP distance bands, completion efficiency, and "
              "all per-class values are in results.json and per_class_iou.csv.", ""]
    (out / "report.md").write_text("\n".join(lines))
    (out / "completed.json").write_text(json.dumps({"voc_images": len(voc.image_ids),
        "coco_images": len(coco.image_ids), "all_required_stages_complete": True,
        "raw_frozen": screens["raw"]["frozen"], "final_frozen": screens["final"]["frozen"]},
        indent=2) + "\n")
    print("\n".join(lines[-5:]), flush=True)


if __name__ == "__main__": main()
