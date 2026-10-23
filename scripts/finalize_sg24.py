#!/usr/bin/env python3
"""SG24 five-fold VOC selection, per-class analysis, and frozen COCO transfer."""
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

from scripts import run_sg24 as runner
from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.select_propagation_cv import best_threshold, image_ious, run_screen
from src.segmentation import controlled_cv as cv

BASE = runner.BASE
DIAG_FIELDS = ("tp", "fp", "foreground", "fp_gt16", "fp_gt64", "fp_separate",
               "changed", "useful_change", "harmful_change")


def load(out, dataset, names, spec, smoke):
    files = sorted(p for p in (out / dataset / "shards").glob("class_*.npz")
                   if not p.name.endswith((".diagnostics.npz", ".tmp.npz")))
    result = cv.load_results(files)
    if result.configurations != names or result.thresholds != tuple(spec["thresholds"]):
        raise ValueError("unexpected SG24 cells or thresholds")
    if result.settings["spec"] != spec or result.settings["dataset"] != dataset:
        raise ValueError("foreign SG24 shard")
    if not smoke and (len(files) != 20 or len(result.image_ids) != (490 if dataset == "voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"{dataset} missing complete SG24 class shards")
    sidecars = {}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"), allow_pickle=False) as data:
            for i, image_id in enumerate(data["image_ids"]):
                if str(image_id) in sidecars:
                    raise ValueError("duplicate SG24 image diagnostics")
                sidecars[str(image_id)] = (data["ca_diagnostics"][i], data["layer_weights"][i],
                                            data["content_mass"][i], data["layer_weights_b0"][i],
                                            data["content_detail"][i])
    if set(sidecars) != set(result.image_ids):
        raise ValueError("SG24 diagnostic coverage mismatch")
    if dataset == "voc":
        folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv")
        if any(result.folds[i] != folds[x] for i,x in enumerate(result.image_ids)):
            raise ValueError("VOC fold mismatch")
    return result, [sidecars[x] for x in result.image_ids]


def layer_summary(result, sidecars, selected=False):
    w = np.stack([x[3 if selected else 1] for x in sidecars])
    classes = np.asarray(result.classes)
    modules = result.settings["ca_modules"]
    if selected and len(modules) == 16:
        modules = [modules[i] for i in result.settings["spec"]["selected_ca_modules"]]
    if w.shape[-1] != len(modules):
        raise ValueError("module/weight dimension mismatch")
    top = np.argsort(w, axis=-1)[..., ::-1]
    rows = []
    for i,module in enumerate(modules):
        vals = w[:,:,i]
        resolution = (64 if module.startswith(("down_blocks.0", "up_blocks.3")) else
                      32 if module.startswith(("down_blocks.1", "up_blocks.2")) else
                      16 if module.startswith(("down_blocks.2", "up_blocks.1")) else 8)
        rows.append({"module": module, "index": i, "resolution": resolution,
                     "mean": float(vals.mean()), "median": float(np.median(vals)),
                     "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
                     "top1_fraction": float((top[:,:,0] == i).mean()),
                     "top3_fraction": float((top[:,:,:3] == i).any(-1).mean()),
                     "mean_by_call": vals.mean(0).tolist(),
                     "mean_by_class": {c: float(vals[classes == c].mean()) for c in sorted(set(classes))}})
    return rows


def restored_weight_relationship(result, sidecars, thresholds, cells):
    if len(cells) != 2:
        return None
    weights = np.stack([x[1] for x in sidecars])
    restored = [i for i in range(weights.shape[-1]) if i not in result.settings["spec"]["selected_ca_modules"]]
    share = weights[:,:,restored].sum(-1).mean(1)
    output = {"restored_mean_weight":float(share.mean()),"restored_median_weight":float(np.median(share))}
    for level in ("CA","FINAL"):
        before,after=(f"{cells[0]}/{level}",f"{cells[1]}/{level}")
        a=image_ious(result.counts[result.index(after),:,thresholds[after]])
        b=image_ious(result.counts[result.index(before),:,thresholds[before]])
        delta=a-b
        output[level]={"mean_image_delta":float(delta.mean()),
                       "pearson_restored_weight_vs_image_delta":float(np.corrcoef(share,delta)[0,1])
                       if np.std(share)>0 and np.std(delta)>0 else None}
    return output


def resolution_summary(layers):
    return {str(r):{"mean_total_weight":float(sum(x["mean"] for x in layers if x["resolution"]==r)),
                    "modules":sum(x["resolution"]==r for x in layers)} for r in (8,16,32,64)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("a", "b"), required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--b-method", choices=("A3", "A4"), help="smoke only")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    spec = json.loads(runner.SPEC.read_text())
    out = args.output_dir or BASE / f"sg24_{args.experiment}"
    method = None
    if args.experiment == "b":
        method = args.b_method or json.loads((BASE / "sg24_a/selection_for_b.json").read_text())["automatic_method"]
    names = runner.names_for(args.experiment, method)
    voc, vd = load(out, "voc", names, spec, args.smoke)
    coco, cd = load(out, "coco", names, spec, args.smoke)
    if args.smoke:
        if len(voc.image_ids) != 1 or len(coco.image_ids) != 1:
            raise ValueError("SG24 smoke requires one VOC and one COCO image")
        print(f"SG24-{args.experiment.upper()} live VOC/COCO smoke passed", flush=True)
        return
    cells = ("A0", "A1", "A2", "A3", "A4") if args.experiment == "a" else (f"B0-{method}", f"B1-{method}")
    ca_ref, final_ref = f"{cells[0]}/CA", f"{cells[0]}/FINAL"
    allmask = np.ones(len(voc.image_ids), bool)
    thresholds = {n: best_threshold(voc, voc.index(n), allmask)[0] for n in names}
    screens = {level: run_screen(voc, f"{cells[0]}/{level}",
                                 [f"{n}/{level}" for n in cells[1:]],
                                 {f"{n}/{level}": 1 for n in cells[1:]})
               for level in ("CA", "FINAL")}
    oof = {n: forced_oof(voc, n, ca_ref if n.endswith("/CA") else final_ref) for n in names}
    results = {dataset: {n: metrics(r,n,ca_ref if n.endswith("/CA") else final_ref,thresholds)
                         for n in names} for dataset,r in (("voc",voc),("coco",coco))}
    diag = {}
    for dataset, sidecars in (("voc",vd),("coco",cd)):
        all_diag = np.stack([x[0] for x in sidecars])
        diag[dataset] = {cells[i]: dict(zip(DIAG_FIELDS, map(int, all_diag[:,i].sum(0))))
                         for i in range(len(cells))}
        mass = np.stack([x[2] for x in sidecars])
        diag[dataset]["mean_content_mass"] = float(mass.mean())
        diag[dataset]["fraction_non_content_mass"] = float(1 - mass.mean())
        detail = np.stack([x[4] for x in sidecars])
        diag[dataset]["content_fg_bg"] = {
            name: float(np.nanmean(detail[:,i])) for i,name in enumerate((
                "content_mass_fg", "content_mass_bg", "target_top_content_fg",
                "target_top_content_bg"))}
        diag[dataset]["content_rank_change_fraction"] = 0.0  # Same positive denominator for every content token.
    relations = {dataset: {cell: {"delta_ca": results[dataset][f"{cell}/CA"]["delta_vs_reference"],
                                    "delta_final": results[dataset][f"{cell}/FINAL"]["delta_vs_reference"]}
                           for cell in cells[1:]} for dataset in ("voc", "coco")}
    per_class_relations = {dataset:{cell:{cls:{"delta_ca":results[dataset][f"{cell}/CA"]["per_class_delta_vs_reference"][cls],
                                            "delta_final":results[dataset][f"{cell}/FINAL"]["per_class_delta_vs_reference"][cls]}
                                     for cls in results[dataset][f"{cell}/CA"]["per_class_iou"]}
                                    for cell in cells[1:]}
                           for dataset in ("voc","coco")}
    summary = {"spec":spec,"experiment":args.experiment,"method":method,"selection":screens,
               "forced_oof":oof,"cells":results,"diagnostics_fixed_0p45":diag,
               "ca_to_final":relations,"ca_to_final_per_class":per_class_relations,
               "restored_weight_relationship":({ds:restored_weight_relationship(res,d,thresholds,cells)
                   for ds,res,d in (("voc",voc,vd),("coco",coco,cd))} if args.experiment == "b" else None),
               "layer_weights":{ds:(
                   {"B0":layer_summary(res,d,selected=True),"B1":layer_summary(res,d)}
                   if args.experiment == "b" else {"A":layer_summary(res,d)})
                   for ds,res,d in (("voc",voc,vd),("coco",coco,cd))},
               "voc_thresholds_frozen_for_coco":True}
    summary["layer_resolution_weights"]={ds:{name:resolution_summary(rows) for name,rows in sets.items()}
                                          for ds,sets in summary["layer_weights"].items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out / "per_class_iou.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("dataset","cell","class","iou","delta_vs_reference"))
        for dataset in ("voc","coco"):
            for n in names:
                metric=results[dataset][n]
                for cls,iou in metric["per_class_iou"].items():
                    writer.writerow((dataset,n,cls,iou,metric["per_class_delta_vs_reference"][cls]))
    lines = [f"# SG24-{args.experiment.upper()} results", "", "VOC-only thresholds; COCO frozen transfer.", "",
             "| Cell | VOC OOF | OOF delta | Fold wins | Full VOC | Threshold | Frozen COCO |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        item=oof[n]; v=results["voc"][n]; c=results["coco"][n]
        lines.append(f"| {n} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{v['miou']:.4f} | {v['voc_frozen_threshold']:.3f} | {c['miou']:.4f} |")
    lines += ["", "Per-class IoUs: `per_class_iou.csv`. Full metrics, 20k paired intervals, CA diagnostics and layer weights: `results.json`.", ""]
    (out / "report.md").write_text("\n".join(lines))
    if args.experiment == "a":
        frozen = screens["FINAL"]["frozen"].split("/")[0]
        automatic = frozen if frozen in ("A3","A4") else "A3"
        selection = {"automatic_method":automatic,"sg24_a_final_frozen":frozen,
                     "b_is_diagnostic":frozen not in ("A3","A4"),
                     "reason":"VOC-only final-pipeline selection; A3 preregistered fallback if no automatic method promoted"}
        (out / "selection_for_b.json").write_text(json.dumps(selection,indent=2)+"\n")
    (out / "completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"all_required_stages_complete":True},indent=2)+"\n")
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()
