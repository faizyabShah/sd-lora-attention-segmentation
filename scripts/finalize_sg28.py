#!/usr/bin/env python3
"""Merge SG28 shards; select on VOC folds and transfer frozen to COCO."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg28 import BASE, NAMES, SIDE_FIELDS, SPEC_PATH
from scripts.select_propagation_cv import best_threshold, image_ious, run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation import sg28_pearl

BRANCHES = {"PA":["P1","P2","P3"], "plain_and_boundary":["B1","B2"],
            "confidence":["C1","C2","C3"], "confidence_and_boundary":["D1","D2","D3"]}


def load(out: Path, dataset: str, spec: dict, smoke: bool):
    files = sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                   if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result = cv.load_results(files)
    if result.configurations != NAMES or result.thresholds != tuple(spec["thresholds"]):
        raise ValueError("SG28 cell or threshold grid mismatch")
    if result.settings["spec"] != spec or result.settings["dataset"] != dataset:
        raise ValueError("foreign SG28 shard settings")
    if not smoke and (len(files) != 20 or len(result.image_ids) != (490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG28 {dataset} shards")
    sidecars = {}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for index,image_id in enumerate(data["image_ids"]):
                key = str(image_id)
                if key in sidecars:
                    raise ValueError("SG28 duplicate diagnostic image ID")
                sidecars[key] = tuple(data[field][index] for field in SIDE_FIELDS)
    if set(sidecars) != set(result.image_ids):
        raise ValueError("SG28 metric/diagnostic coverage mismatch")
    if dataset == "voc":
        folds = cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i] != folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("SG28 VOC fold mismatch")
    return result,tuple(np.stack([sidecars[image_id][j] for image_id in result.image_ids])
                        for j in range(len(SIDE_FIELDS)))


def _correlation(a: np.ndarray,b: np.ndarray) -> dict[str,float | int | None]:
    valid = np.isfinite(a)&np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return {"images":int(valid.sum()),"pearson":None,"spearman":None}
    return {"images":int(valid.sum()),"pearson":float(np.corrcoef(a[valid],b[valid])[0,1]),
            "spearman":float(spearmanr(a[valid],b[valid]).statistic)}


def pa_summary(result,records,thresholds):
    output = {}
    for index,name in enumerate(sg28_pearl.PA_NAMES):
        values = records[:,index,:]
        candidate = result.counts[result.index(name),:,thresholds[name]]
        reference = result.counts[result.index("P0"),:,thresholds["P0"]]
        delta = image_ious(candidate)-image_ious(reference)
        improvement = values[:,0]-values[:,1]
        output[name] = {field:float(np.mean(values[:,i])) for i,field in enumerate(sg28_pearl.PA_FIELDS)}
        output[name]["mean_alignment_error_reduction"] = float(improvement.mean())
        output[name]["alignment_reduction_vs_raw_image_iou_delta"] = _correlation(improvement,delta)
    return output


def lap_summary(records,trust):
    output = {}
    for index,name in enumerate(sg28_pearl.LAP_NAMES):
        values=records[:,index,:]
        output[name]={field:float(np.mean(values[:,i])) for i,field in enumerate(sg28_pearl.LAP_FIELDS)}
        output[name]["max_relative_residual"] = float(values[:,1].max())
        output[name]["max_cg_iterations"] = int(values[:,0].max())
    trust_result={}
    for index,name in enumerate(("C16","C32","three_scale")):
        bins=trust[:,index].sum(0)
        trust_result[name]={key:{"pixels":int(item[0]),"gt_positive":int(item[1]),
                                 "gt_positive_fraction":float(item[1]/max(item[0],1))}
                            for key,item in zip(("low","medium","high"),bins)}
    return output,trust_result


def change_summary(result,records):
    classes=np.asarray(result.classes)
    def reduced(batch):
        total=batch.sum(0)
        return {name:{field:int(v) for field,v in zip(CHANGE_FIELDS,total[index])}
                for index,name in enumerate(sg28_pearl.NAMES[1:])}
    return {"threshold":0.45,"stage":"raw normalized score maps before R2",
            "overall":reduced(records),
            "per_class":{name:reduced(records[classes==name]) for name in sorted(set(classes))}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    spec=json.loads(SPEC_PATH.read_text())
    voc,vdiag=load(args.output_dir,"voc",spec,args.smoke)
    coco,cdiag=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG28 live smoke requires one VOC and one COCO image")
        expected=((11,len(CHANGE_FIELDS)),(3,len(sg28_pearl.PA_FIELDS)),
                  (8,len(sg28_pearl.LAP_FIELDS)),(3,3,2))
        if any(x.shape[1:]!=shape for arrays in (vdiag,cdiag) for x,shape in zip(arrays,expected)):
            raise ValueError("SG28 diagnostic sidecar shapes differ")
        print("SG28 live VOC/COCO smoke passed: 12 raw and 12 R2 cells",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={}
    for stage,suffix in (("raw",""),("final","+R2")):
        choices={"all":list(sg28_pearl.NAMES[1:]),**BRANCHES}
        screens[stage]={branch:run_screen(voc,"P0"+suffix,[name+suffix for name in cells],
                           {name+suffix:1 for name in cells}) for branch,cells in choices.items()}
    oof={name:forced_oof(voc,name,"P0+R2" if name.endswith("+R2") else "P0") for name in NAMES}
    cells={dataset:{name:metrics(result,name,"P0+R2" if name.endswith("+R2") else "P0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    diagnostics={}
    for dataset,result,data in (("voc",voc,vdiag),("coco",coco,cdiag)):
        change,pa,lap,trust=data
        lap_stats,trust_stats=lap_summary(lap,trust)
        diagnostics[dataset]={"fixed_threshold_changes":change_summary(result,change),
                              "pa":pa_summary(result,pa,thresholds),
                              "laplacian":lap_stats,"trust_bins":trust_stats}
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "diagnostics":diagnostics,"voc_thresholds_frozen_for_coco":True,
             "pa_and_laplacian_never_combined":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_P0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                item=cells[dataset][name]
                for cls,iou in item["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,item["per_class_delta_vs_reference"][cls]))
    with (out/"pa_diagnostics.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell",*sg28_pearl.PA_FIELDS,"mean_alignment_error_reduction",
                         "pearson_error_gain_vs_image_iou_gain","spearman_error_gain_vs_image_iou_gain"))
        for dataset in ("voc","coco"):
            for name,values in diagnostics[dataset]["pa"].items():
                correlation=values["alignment_reduction_vs_raw_image_iou_delta"]
                writer.writerow((dataset,name,*(values[f] for f in sg28_pearl.PA_FIELDS),
                                 values["mean_alignment_error_reduction"],
                                 correlation["pearson"],correlation["spearman"]))
    lines=["# SG28 PEARL-inspired independent components","",
           "40 complete class shards; 490 VOC and 707 COCO images; VOC-only selection; COCO frozen.","",
           "A3 and R2 remain as implemented historically; PA and Laplacian are not combined.",""]
    for stage,suffix in (("Raw",""),("After R2","+R2")):
        lines += [f"## {stage}","",
                  "| Cell | VOC forced OOF | Δ vs P0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name in sg28_pearl.NAMES:
            key=name+suffix; item=oof[key]; v=cells["voc"][key]; c=cells["coco"][key]
            lines.append(f"| {key} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                         f"{v['miou']:.4f} | {v['voc_frozen_threshold']:.3f} | {c['miou']:.4f} |")
        lines.append("")
    lines += ["## Separate VOC selectors","",
              "| Stage | Branch | Fold votes | Frozen cell | OOF gain |",
              "|---|---|---|---|---:|"]
    for stage,branches in screens.items():
        for branch,item in branches.items():
            lines.append(f"| {stage} | {branch} | {item['selection_counts']} | "
                         f"{item['frozen']} | {item['oof_gain']:+.4f} |")
    lines += ["","Full per-class results, precision/recall, paired intervals, fixed-threshold mechanism, "
              "PA alignment/entropy correlations, CG accuracy, and confidence bins are in `results.json`; "
              "per-class and PA summary tables are also exported as CSV.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"cells":len(sg28_pearl.NAMES),
        "all_required_stages_complete":True},indent=2)+"\n")
    print("\n".join(lines[-11:]),flush=True)


if __name__ == "__main__":
    main()
