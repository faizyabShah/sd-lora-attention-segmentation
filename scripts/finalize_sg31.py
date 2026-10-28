#!/usr/bin/env python3
"""Merge SG31, run fixed VOC selection and report seed/background mechanisms."""
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

from scripts import run_coco_cap_transfer as transfer
from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg31 import (BASE, BG_DIAG_FIELDS, CHANGE_PAIRS, NAMES, RAW,
                              SIDE_FIELDS, SPEC_PATH)
from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen
from src.segmentation import controlled_cv as cv, sg31_seediff


def load(out: Path, dataset: str, spec: dict, smoke: bool):
    files = sorted(p for p in (out / dataset / "shards").glob("class_*.npz")
                   if not p.name.endswith((".diagnostics.npz", ".tmp.npz")))
    result = cv.load_results(files)
    if result.configurations != NAMES or result.thresholds != tuple(spec["thresholds"]):
        raise ValueError("SG31 cell/threshold mismatch")
    if result.settings["spec"] != spec or result.settings["dataset"] != dataset:
        raise ValueError("foreign SG31 shard")
    if not smoke and (len(files) != 20 or len(result.image_ids) != (490 if dataset == "voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG31 {dataset} shards")
    side = {}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"), allow_pickle=False) as data:
            for i, image_id in enumerate(data["image_ids"]):
                key = str(image_id)
                if key in side:
                    raise ValueError("duplicate SG31 diagnostic image")
                side[key] = tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side) != set(result.image_ids):
        raise ValueError("SG31 diagnostic coverage mismatch")
    if dataset == "voc":
        folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv")
        if any(result.folds[i] != folds[image_id] for i, image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold assignment changed")
    return result, {field: np.stack([side[x][j] for x in result.image_ids])
                    for j, field in enumerate(SIDE_FIELDS)}


def _changes(rows):
    total = rows.sum(axis=0)
    output = {field: int(value) for field, value in zip(CHANGE_FIELDS, total)}
    output["fp_removed_per_tp_lost"] = (output["fp_removed"] / output["tp_lost"]
                                         if output["tp_lost"] else None)
    return output


def mechanisms(result, side):
    seed = side["seeds"]  # image, call, path S3/S4, scale, field
    classes = np.asarray(result.classes)
    seed_output = {}
    for path, slot, scales in (("S3/S8", 0, (16, 32, 64)), ("S4/S9", 1, (32, 64))):
        for scale in scales:
            s = seed[:, :, slot, {16:0, 32:1, 64:2}[scale], :].reshape(-1, len(sg31_seediff.SEED_FIELDS))
            seed_output[f"{path}/{scale}"] = {
                "observations": len(s), "fallbacks": int(np.nansum(s[:,2])),
                "seed_count_p10_p50_p90": np.nanpercentile(s[:,0], [10,50,90]).tolist(),
                "seed_fraction_p10_p50_p90": np.nanpercentile(s[:,1], [10,50,90]).tolist(),
                "components_p10_p50_p90": np.nanpercentile(s[:,3], [10,50,90]).tolist(),
                "expanded_area_mean": float(np.nanmean(s[:,4])),
                "overlap_previous_mean": float(np.nanmean(s[:,5])) if scale != scales[0] else None}
    changes = {name: _changes(side["changes"][:, i]) for i, name in enumerate(RAW[1:])}
    pairs = {f"{a}->{b}": _changes(side["pair_changes"][:, i])
             for i, (a,b) in enumerate(CHANGE_PAIRS)}
    by_class = {name: {cell: _changes(side["changes"][classes == name, i])
                       for i, cell in enumerate(RAW[1:])}
                for name in sorted(set(classes))}
    backgrounds = {}
    for i, name in enumerate(("S6", "S7", "S8", "S9")):
        data = side["background"][:, i]
        fields = {field: ((float(np.nanmean(data[:,j])) if np.isfinite(data[:,j]).any() else None)
                          if j < 4 else int(np.nansum(data[:,j])))
                  for j,field in enumerate(BG_DIAG_FIELDS)}
        fields["fp_removed_per_tp_removed"] = (fields["fp_removed"] / fields["tp_removed"]
                                                if fields["tp_removed"] else None)
        backgrounds[name] = fields
    return {"seed_fields": sg31_seediff.SEED_FIELDS, "seed_distributions": seed_output,
            "changes_vs_S0_at_0p45": changes, "hard_vs_continuous_at_0p45": pairs,
            "changes_by_class_at_0p45": by_class,
            "background_fields": BG_DIAG_FIELDS, "background_at_0p45": backgrounds,
            "GT_usage": "post-hoc diagnostics only; never used for inference or selection"}


def caption_presence(coco, thresholds, candidates):
    rows = {r["image_id"]: r for shard in range(20) for r in transfer.coco_rows(shard)[1]}
    if set(rows) != set(coco.image_ids):
        raise ValueError("COCO caption join mismatch")
    present = np.asarray([rows[x]["word_included"].strip().lower() == "true"
                          for x in coco.image_ids])
    classes = np.asarray(coco.classes)
    output = {}
    for name in candidates:
        output[name] = {}
        for label,mask in (("present",present), ("absent",~present)):
            a = coco.counts[coco.index(name),mask,thresholds[name]]
            b = coco.counts[coco.index("S0+R2"),mask,thresholds["S0+R2"]]
            output[name][label] = {"images": int(mask.sum()),
                "miou": pooled_miou(a, classes[mask]),
                "reference_miou": pooled_miou(b, classes[mask]),
                "delta": pooled_miou(a, classes[mask]) - pooled_miou(b, classes[mask]),
                "paired_mean_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b))}
    return {"field": "released word_included in generation caption; descriptive only",
            "cells": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=BASE)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    voc, vd = load(args.output_dir, "voc", spec, args.smoke)
    coco, cd = load(args.output_dir, "coco", spec, args.smoke)
    if args.smoke:
        if len(voc.image_ids) != 1 or len(coco.image_ids) != 1:
            raise ValueError("SG31 smoke requires one image per dataset")
        for data in (vd,cd):
            if data["seeds"].shape[1:] != (11,2,3,6) or data["changes"].shape[1:] != (10,11) or data["pair_changes"].shape[1:] != (4,11) or data["background"].shape[1:] != (4,10):
                raise ValueError("SG31 diagnostic sidecar shape changed")
        print("SG31 VOC/COCO live smoke passed: 11 raw + 11 R2 maps and diagnostics", flush=True)
        return
    thresholds = {n: best_threshold(voc, voc.index(n), np.ones(len(voc.image_ids), bool))[0]
                  for n in NAMES}
    screens = {stage: run_screen(voc, "S0"+suffix, [n+suffix for n in RAW[1:]],
                    {n+suffix: 1 for n in RAW[1:]})
               for stage,suffix in (("raw",""),("final","+R2"))}
    oof = {n: forced_oof(voc, n, "S0+R2" if n.endswith("+R2") else "S0") for n in NAMES}
    cells = {dataset: {n: metrics(result, n, "S0+R2" if n.endswith("+R2") else "S0", thresholds)
                       for n in NAMES}
             for dataset,result in (("voc",voc),("coco",coco))}
    best = sorted((n for n in NAMES if n.endswith("+R2") and n != "S0+R2"),
                  key=lambda n: cells["voc"][n]["miou"], reverse=True)[:3]
    summary = {"spec": spec, "selection": screens, "forced_oof": oof, "cells": cells,
               "mechanism": {"voc": mechanisms(voc,vd), "coco": mechanisms(coco,cd)},
               "caption_presence_posthoc": caption_presence(coco,thresholds,best),
               "voc_thresholds_frozen_for_coco": True}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w", newline="") as handle:
        writer = csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_S0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,
                                     cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines = ["# SG31 SeeDiff-inspired re-seeding and background expansion", "",
             "490 VOC and 707 COCO images; VOC-only five-fold selection; frozen COCO transfer.", "",
             "| Cell | VOC forced OOF | Δ vs S0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item = oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | "
                     f"{cells['coco'][name]['miou']:.4f} |")
    lines += ["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
              "Full per-class, paired 20k, seed, foreground/background and caption diagnostics are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
         "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
         "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__ == "__main__":
    main()
