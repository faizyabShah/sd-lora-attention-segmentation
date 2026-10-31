#!/usr/bin/env python3
"""Merge SG-A3G: VOC selection, frozen COCO, paired orientation contrasts, size diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.finalize_sg_combined_c10 import finite_mean
from scripts.run_sg_a3g import BASE,NAMES,RAW,SIDE_FIELDS,SPEC
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv

CONTRASTS=(("V0","G0"),("V3","G3"),("G0","G3"),("V0","V3"))
TIE_BAND=.002
SIZE_BINS=(0.,.1,.25,.5,.75,1.01)


def load(out,dataset,spec,smoke):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG-A3G cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG-A3G shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} SG-A3G coverage")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG-A3G diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("SG-A3G diagnostic coverage mismatch")
    return result,{field:np.stack([side[x][i] for x in result.image_ids])
                   for i,field in enumerate(SIDE_FIELDS)}


def reference(name):
    return "V0+R2" if name.endswith("+R2") else "V0"


def ranking(result):
    """Diagnostic only: raw-map AP and best-threshold mIoU versus same-stage V0."""
    out={};keep=np.ones(len(result.image_ids),bool)
    best={n:best_threshold(result,result.index(n),keep) for n in NAMES}
    for name in NAMES:
        item={"best_threshold":result.thresholds[best[name][0]],"best_threshold_miou":best[name][1],
              "best_threshold_delta":best[name][1]-best[reference(name)][1]}
        if not name.endswith("+R2"):
            a=result.ap[result.index(name)];b=result.ap[result.index("V0")]
            item|={"raw_ap":float(a.mean()),"raw_ap_vs_V0":cv.paired_bootstrap(a,b)}
        out[name]=item
    return out


def size_bins(result,thresholds,pairs):
    """Mean-image IoU change by GT area fraction at full-VOC frozen thresholds."""
    ref=result.counts[result.index("V0"),:,thresholds["V0"]]
    gt=(ref[:,0]+ref[:,2])/np.maximum(ref.sum(1),1)
    out={}
    for a,b in pairs:
        diff=(image_ious(result.counts[result.index(b),:,thresholds[b]])
              -image_ious(result.counts[result.index(a),:,thresholds[a]]))
        rows=[]
        for lo,hi in zip(SIZE_BINS[:-1],SIZE_BINS[1:]):
            m=(gt>=lo)&(gt<hi)
            rows.append({"gt_fraction":[lo,min(hi,1.)],"images":int(m.sum()),
                         "mean_image_iou_change":float(diff[m].mean()) if m.any() else None})
        out[f"{b}_vs_{a}"]=rows
    return out


def seed_summary(side):
    s=side["seed"]
    return {"mean_abs_seed_difference":finite_mean(s[:,0]),"max_abs_seed_difference":float(s[:,0].max()),
            "seed_pearson_mean":finite_mean(s[:,1]),"seed_pearson_min":float(s[:,1].min()),
            "call_mean_abs_layer_weight_difference":finite_mean(s[:,2])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG-A3G smoke needs one image per dataset")
        for side in (vd,cd):
            if side["seed"].shape[1:]!=(3,) or not np.isfinite(side["seed"]).all():
                raise ValueError("SG-A3G seed diagnostic shape or value mismatch")
        print("SG-A3G smoke passed: four raw/R2 cells and paired seed diagnostics;",
              "VOC seed |d|, r, |dw| =",vd["seed"][0].round(5).tolist(),
              "COCO =",cd["seed"][0].round(5).tolist(),flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"V0"+suffix,[n+suffix for n in RAW[1:]],
                              {n+suffix:1 for n in RAW[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,reference(name)) for name in NAMES}
    contrasts={f"{b}{s}_vs_{a}{s}":forced_oof(voc,b+s,a+s) for a,b in CONTRASTS for s in ("","+R2")}
    cells={dataset:{name:metrics(result,name,reference(name),thresholds) for name in NAMES}
           for dataset,result in (("voc",voc),("coco",coco))}
    move=contrasts["G0+R2_vs_V0+R2"]["delta"]
    control={"forced_oof_delta_G0R2_vs_V0R2":move,"tie_band":TIE_BAND,
             "outside_tie_band":bool(abs(move)>TIE_BAND),
             "coco_frozen_delta_G0R2_vs_V0R2":cells["coco"]["G0+R2"]["miou"]-cells["coco"]["V0+R2"]["miou"]}
    pairs=[(a+s,b+s) for a,b in CONTRASTS for s in ("","+R2")]
    worst={dataset:{name:min(cells[dataset][name]["per_class_delta_vs_reference"].items(),key=lambda x:x[1])
                    for name in NAMES if not name.startswith("V0")}
           for dataset in ("voc","coco")}
    diagnostics={"ranking_diagnostic_only":{"voc":ranking(voc),"coco":ranking(coco)},
                 "seed":{"voc":seed_summary(vd),"coco":seed_summary(cd)},
                 "size_bins_diagnostic_only":{"voc":size_bins(voc,thresholds,pairs),
                                              "coco":size_bins(coco,thresholds,pairs)},
                 "worst_class_vs_same_stage_V0":worst}
    output={"spec":spec,"selection":screens,"forced_oof":oof,"contrasts":contrasts,"control_check":control,
            "cells":cells,**diagnostics,"voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_V0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    rank=diagnostics["ranking_diagnostic_only"]
    lines=["# SG-A3G","",
           "Historical (V) vs GoCA-oriented (G) A3 seed from the same diffusion run; FINE-P2 (0) and V3 (3).",
           "40 class shards, 490 VOC and 707 COCO images. VOC-only selection; COCO thresholds frozen.","",
           "| Cell | VOC forced OOF | Δ vs V0 | Fold wins | Full VOC (thr) | Frozen COCO | COCO best-thr Δ | Worst VOC class |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name in NAMES:
        item=oof[name];w=worst["voc"].get(name)
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} ({cells['voc'][name]['voc_frozen_threshold']:.3f}) | "
                     f"{cells['coco'][name]['miou']:.4f} | {rank['coco'][name]['best_threshold_delta']:+.4f} | "
                     f"{'—' if w is None else f'{w[0]} {w[1]:+.3f}'} |")
    lines+=["","| Paired contrast | VOC forced OOF Δ | Fold wins | Paired mean-image CI |","|---|---:|---:|---|"]
    for key,item in contrasts.items():
        pb=item["paired_mean_image_iou"]
        lines.append(f"| {key} | {item['delta']:+.4f} | {item['fold_wins']}/5 | [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}] |")
    lines+=["",f"Control check (Section 107): G0+R2 − V0+R2 = {move:+.4f} VOC forced OOF; "
            f"{'outside' if control['outside_tie_band'] else 'inside'} the ±{TIE_BAND} tie band; "
            f"frozen COCO {control['coco_frozen_delta_G0R2_vs_V0R2']:+.4f}.",
            f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
            f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}",
            f"Seed difference (VOC): {json.dumps(diagnostics['seed']['voc'])}","",
            "AP, best-threshold, size-bin and per-class values are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
        "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"],
        "control_outside_tie_band":control["outside_tie_band"]},indent=2)+"\n")
    print("\n".join(lines[-8:]),flush=True)


if __name__=="__main__":main()
