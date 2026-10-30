#!/usr/bin/env python3
"""Diagnostic calibration/ranking audit of committed SG25-SG-combined-v1 packages.

CPU only; reads the compact metric shards and never regenerates an image.
For every cell it reports, next to the registered frozen-threshold result:

* best-threshold mIoU on VOC and on COCO (the threshold is chosen on that
  dataset's own labels, so both are diagnostic, never results);
* raw-map AP, mean over images (threshold free; undefined after R2).

Every delta is against the same package's same-stage reference (raw versus
raw, R2 versus R2), so no comparison crosses separate regenerations.  No
registered selection, threshold or promotion is recomputed or changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from src.segmentation import controlled_cv as cv

BASE=ROOT/"outputs/segmentation/controlled_attention_cv"
OUT=ROOT/"outputs/segmentation/analysis/calibration_ap_audit"
PACKAGES={"sg25":"FINE-P2","sg27":"S0","sg30":"S0","sg31":"S0",
          "sg32":"S0","sg33":"S0","sg_combined_v1":"C0"}
# Cells named in the owners' audit request; every other cell is still audited.
HIGHLIGHT={"sg25":("16-to-64","32-to-64","COARSE32-P2"),"sg27":("S8","S9","S38"),
           "sg30":("S2","S6"),"sg31":("S1","S2","S6","S7","S10"),"sg32":("S1","S7"),
           "sg33":("H0","H2","H3","M2"),"sg_combined_v1":("C1","C2","C5","C6","C8")}
MATERIAL=.015  # descriptive "large COCO change" marker, not a decision rule


def load(package:str,dataset:str)->cv.Results:
    files=sorted(p for p in (BASE/package/dataset/"shards").glob("class_*.npz")
                 if p.name.count(".")==1)
    result=cv.load_results(files)
    expected=490 if dataset=="voc" else 707
    if len(result.image_ids)!=expected:raise ValueError(f"{package}/{dataset}: {len(result.image_ids)} images")
    return result


def curve(result:cv.Results,c:int)->np.ndarray:
    return np.array([cv.miou(result,c,t) for t in range(len(result.thresholds))])


def class_mean_ap(result:cv.Results,c:int)->float:
    classes=np.array(result.classes)
    return float(np.mean([result.ap[c,classes==k].mean() for k in sorted(set(classes))]))


def audit(package:str,reference:str)->list[dict]:
    registered=json.loads((BASE/package/"results.json").read_text())
    data={d:load(package,d) for d in ("voc","coco")}
    thresholds=data["voc"].thresholds
    if data["coco"].thresholds!=thresholds or data["coco"].configurations!=data["voc"].configurations:
        raise ValueError(f"{package}: VOC/COCO shard layout differs")
    names=data["voc"].configurations
    curves={d:{n:curve(r,i) for i,n in enumerate(names)} for d,r in data.items()}
    rows=[]
    for i,name in enumerate(names):
        r2=name.endswith("+R2")
        ref=reference+("+R2" if r2 else "")
        j=names.index(ref)
        frozen=registered["cells"]["voc"][name]["voc_frozen_threshold"]
        t=thresholds.index(frozen)
        row={"package":package,"cell":name,"stage":"R2" if r2 else "raw","reference":ref,
             "highlight":name.removesuffix("+R2") in HIGHLIGHT[package],
             "voc_frozen_threshold":frozen,
             "forced_oof_voc":registered["forced_oof"][name]["miou"],
             "forced_oof_voc_delta":registered["forced_oof"][name]["miou"]-registered["forced_oof"][ref]["miou"]}
        for d in ("voc","coco"):
            got=curves[d][name][t];want=registered["cells"][d][name]["miou"]
            if abs(got-want)>1e-9:raise ValueError(f"{package}/{d}/{name}: {got} != registered {want}")
            tref=thresholds.index(registered["cells"]["voc"][ref]["voc_frozen_threshold"])
            best=int(curves[d][name].argmax());best_ref=int(curves[d][ref].argmax())
            row|={f"{d}_frozen":got,f"{d}_frozen_delta":got-curves[d][ref][tref],
                  f"{d}_best":curves[d][name][best],f"{d}_best_threshold":thresholds[best],
                  f"{d}_best_delta":curves[d][name][best]-curves[d][ref][best_ref],
                  f"{d}_best_threshold_reference":thresholds[best_ref]}
            if r2:
                row|={f"{d}_ap":None,f"{d}_ap_class_mean":None,f"{d}_ap_delta":None,
                      f"{d}_ap_ci_low":None,f"{d}_ap_ci_high":None}
                continue
            res=data[d];a=res.ap[i];b=res.ap[j]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError(f"{package}/{d}/{name}: non-finite AP")
            boot=cv.paired_bootstrap(a,b)
            row|={f"{d}_ap":float(a.mean()),f"{d}_ap_class_mean":class_mean_ap(res,i),
                  f"{d}_ap_delta":boot["mean"],f"{d}_ap_ci_low":boot["ci_low"],f"{d}_ap_ci_high":boot["ci_high"]}
        rows.append(row)
    return rows


def fmt(value,digits=4,signed=False):
    if value is None:return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def report(rows:list[dict])->str:
    lines=["# Calibration and ranking audit, SG25–SG-combined-v1 (diagnostic only)","",
           "Computed from the committed compact shards by `scripts/audit_calibration_ap.py`.",
           "No image was regenerated and no registered result, threshold or decision is changed.","",
           "**How to read the columns.** *Frozen* is the registered result: VOC-selected threshold, transferred unchanged to COCO.",
           "*Best-threshold* chooses each cell's threshold on that dataset's own labels, separately for the cell and for its reference.",
           "It is an optimistic, label-using diagnostic and never a result. *AP* is the mean per-image average precision of the raw",
           "64×64 map: threshold free, so it measures spatial ranking rather than calibration. AP is undefined for binary R2 masks.",
           "Every Δ is against the same package's same-stage reference (raw vs raw, R2 vs R2). The VOC best-threshold column is in-sample.",
           "Comparing frozen and best-threshold deltas is descriptive. Threshold and map geometry interact, so the difference is not an exact causal split.",""]
    lines+=["## Requested cells","",
            "| Package | Cell | VOC forced-OOF Δ | COCO frozen Δ | COCO best-thr Δ (thr cell/ref) | VOC best-thr Δ | COCO raw AP Δ [95% CI] | VOC raw AP Δ [95% CI] |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if not r["highlight"]:continue
        def ap(d):
            if r[f"{d}_ap_delta"] is None:return "—"
            return f"{r[f'{d}_ap_delta']:+.4f} [{r[f'{d}_ap_ci_low']:+.4f}, {r[f'{d}_ap_ci_high']:+.4f}]"
        lines.append(f"| {r['package']} | {r['cell']} | {fmt(r['forced_oof_voc_delta'],signed=True)} | "
                     f"{fmt(r['coco_frozen_delta'],signed=True)} | {fmt(r['coco_best_delta'],signed=True)} "
                     f"({r['coco_best_threshold']:.3f}/{r['coco_best_threshold_reference']:.3f}) | "
                     f"{fmt(r['voc_best_delta'],signed=True)} | {ap('coco')} | {ap('voc')} |")
    lines+=["","## Reference calibration","",
            "| Package | Reference | VOC-frozen thr | COCO frozen | COCO best-thr (thr) | Gap |","|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["cell"]!=r["reference"]:continue
        lines.append(f"| {r['package']} | {r['cell']} | {r['voc_frozen_threshold']:.3f} | {fmt(r['coco_frozen'])} | "
                     f"{fmt(r['coco_best'])} ({r['coco_best_threshold']:.3f}) | {fmt(r['coco_best']-r['coco_frozen'],signed=True)} |")
    lines+=["","## All cells with a large frozen COCO gain","",
            f"Cells whose registered frozen COCO Δ is at least +{MATERIAL:.3f} over their same-stage reference (a descriptive marker only).",""]
    for stage in ("raw","R2"):
        big=[r for r in rows if r["stage"]==stage and r["coco_frozen_delta"]>=MATERIAL]
        kept=[r for r in big if r["coco_best_delta"]>=MATERIAL]
        lines.append(f"- **{stage}:** {len(big)} cells; {len(kept)} still gain at least +{MATERIAL:.3f} at matched best thresholds; "
                     f"{sum(r['coco_best_delta']>0 for r in big)} keep a positive best-threshold Δ.")
        if stage=="raw":
            ap_pos=[r for r in big if r["coco_ap_ci_low"]>0]
            ap_neg=[r for r in big if r["coco_ap_ci_high"]<0]
            lines.append(f"  COCO raw AP: {len(ap_pos)} significantly higher, {len(ap_neg)} significantly lower, "
                         f"{len(big)-len(ap_pos)-len(ap_neg)} interval spanning zero.")
    lines+=["","The complete per-cell table, including every non-requested cell, is in `cells.csv` and `results.json`.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    rows=[row for package,reference in PACKAGES.items() for row in audit(package,reference)]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"results.json").write_text(json.dumps(
        {"diagnostic_only":True,"packages":PACKAGES,"highlight":HIGHLIGHT,"material_marker":MATERIAL,
         "bootstrap_resamples":cv.BOOTSTRAP_RESAMPLES,"cells":rows},indent=1)+"\n")
    with (args.output_dir/"cells.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    (args.output_dir/"report.md").write_text(report(rows))
    print(f"{len(rows)} cells audited -> {args.output_dir}")


if __name__=="__main__":main()
