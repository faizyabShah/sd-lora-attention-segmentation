#!/usr/bin/env python3
"""Merge SG-combined-v3: registered selection vs R0, mechanism table vs R1, cap-loss diagnostics."""
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
from scripts.run_sg_combined_v3 import BASE,NAMES,RAW,SIDE_FIELDS,SPEC
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation.sg_combined_v3 import CAP_ROWS,GT_ROWS

MECHANISM=(("R1","W1"),("R1","W2"),("R1","W3"),("R0","R1"))
SIZE_BINS=(0.,.1,.25,.5,.75,1.01)
GT_FIELDS=("gamma_GT","L_GT","L_background")


def load(out,dataset,spec,smoke):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG-combined-v3 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG-combined-v3 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} SG-combined-v3 coverage")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG-combined-v3 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS if field!="maps")
    if set(side)!=set(result.image_ids):raise ValueError("SG-combined-v3 diagnostic coverage mismatch")
    fields=[f for f in SIDE_FIELDS if f!="maps"]
    return result,{field:np.stack([side[x][i] for x in result.image_ids]) for i,field in enumerate(fields)}


def stage(name):
    return "+R2" if name.endswith("+R2") else ""


def ranking(result,ref):
    """Diagnostic only: raw AP and best-threshold mIoU versus same-stage ``ref``."""
    out={};keep=np.ones(len(result.image_ids),bool)
    best={n:best_threshold(result,result.index(n),keep) for n in NAMES}
    for name in NAMES:
        r=ref+stage(name)
        item={"best_threshold":result.thresholds[best[name][0]],"best_threshold_miou":best[name][1],
              "best_threshold_delta":best[name][1]-best[r][1]}
        if not stage(name):
            a=result.ap[result.index(name)];b=result.ap[result.index(ref)]
            item|={"raw_ap":float(a.mean()),f"raw_ap_vs_{ref}":cv.paired_bootstrap(a,b)}
        out[name]=item
    return out


def size_bins(result,thresholds):
    ref=result.counts[result.index("R0"),:,thresholds["R0"]]
    gt=(ref[:,0]+ref[:,2])/np.maximum(ref.sum(1),1)
    out={}
    for a,b in MECHANISM:
        for s in ("","+R2"):
            diff=(image_ious(result.counts[result.index(b+s),:,thresholds[b+s]])
                  -image_ious(result.counts[result.index(a+s),:,thresholds[a+s]]))
            out[f"{b}{s}_vs_{a}{s}"]=[{"gt_fraction":[lo,min(hi,1.)],"images":int(m.sum()),
                                       "mean_image_iou_change":float(diff[m].mean()) if m.any() else None}
                                      for lo,hi in zip(SIZE_BINS[:-1],SIZE_BINS[1:])
                                      for m in [(gt>=lo)&(gt<hi)]]
    return out


def diagnostics(result,side):
    classes=np.asarray(result.classes)
    cap={row:{f:finite_mean(side["cap"][:,i,j]) for j,f in enumerate(
             ("binding_group_fraction","removed_mass_fraction","mean_factor_binding_groups"))}
         for i,row in enumerate(CAP_ROWS)}
    gt={}
    for i,row in enumerate(GT_ROWS):
        values=side["gt_cap"][:,i]
        by_class={c:{f:finite_mean(values[classes==c,j]) for j,f in enumerate(GT_FIELDS)}
                  for c in sorted(set(classes))}
        others=classes!="sheep"
        gt[row]={"all":{f:finite_mean(values[:,j]) for j,f in enumerate(GT_FIELDS)},
                 "sheep":by_class.get("sheep"),
                 "non_sheep":{f:finite_mean(values[others,j]) for j,f in enumerate(GT_FIELDS)},
                 "by_class":by_class}
    seed=side["seed"]
    return {"cap":cap,"gt_cap_loss_16_to_32":gt,
            "c16_native_vs_downsampled":{"pearson_mean":finite_mean(seed[:,0]),
                                         "pearson_min":float(np.nanmin(seed[:,0])),
                                         "mean_abs_difference":finite_mean(seed[:,1])},
            "note":"GT enters diagnostics only"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG-combined-v3 smoke needs one image per dataset")
        for side in (vd,cd):
            if side["cap"].shape[1:]!=(len(CAP_ROWS),3) or side["gt_cap"].shape[1:]!=(len(GT_ROWS),3) \
               or side["seed"].shape[1:]!=(2,) or not np.isfinite(side["cap"]).all():
                raise ValueError("SG-combined-v3 diagnostic sidecar mismatch")
        for file in (args.output_dir/"voc"/"shards").glob("*.diagnostics.npz"):
            with np.load(file) as data:
                if data["maps"].shape[1:]!=(len(RAW),64,64) or data["maps"].dtype!=np.float16:
                    raise ValueError("SG-combined-v3 dense map sidecar mismatch")
        print("SG-combined-v3 smoke passed: five raw/R2 cells, cap/GT-loss/seed diagnostics and dense maps;",
              "VOC c16* r, |d| =",vd["seed"][0].round(4).tolist(),"COCO =",cd["seed"][0].round(4).tolist(),flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={st:run_screen(voc,"R0"+s,[n+s for n in RAW[1:]],{n+s:1 for n in RAW[1:]})
             for st,s in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"R0"+stage(name)) for name in NAMES}
    mech={f"{b}{s}_vs_{a}{s}":forced_oof(voc,b+s,a+s) for a,b in MECHANISM for s in ("","+R2")}
    cells={ds:{name:metrics(r,name,"R0"+stage(name),thresholds) for name in NAMES}
           for ds,r in (("voc",voc),("coco",coco))}
    vs_r1={ds:{name:metrics(r,name,"R1"+stage(name),thresholds) for name in NAMES if name[0]=="W"}
           for ds,r in (("voc",voc),("coco",coco))}
    rank={ds:{"vs_R0":ranking(r,"R0"),"vs_R1":ranking(r,"R1")} for ds,r in (("voc",voc),("coco",coco))}
    decisions={}
    for w in ("W1","W2","W3"):
        rows={s or "raw":mech[f"{w}{s}_vs_R1{s}"] for s in ("","+R2")}
        wins=any(r["fold_wins"]>=4 and r["paired_mean_image_iou"]["ci_low"]>0 for r in rows.values())
        sheep={s or "raw":vs_r1["voc"][w+s]["per_class_delta_vs_reference"]["sheep"] for s in ("","+R2")}
        decisions[w]={"beats_R1_4of5_with_interval_above_zero":wins,"voc_sheep_delta_vs_R1":sheep}
    output={"spec":spec,"selection":screens,"forced_oof_vs_R0":oof,"mechanism_vs_R1":mech,
            "cells_vs_R0":cells,"cells_vs_R1":vs_r1,"ranking_diagnostic_only":rank,
            "size_bins_diagnostic_only":{"voc":size_bins(voc,thresholds),"coco":size_bins(coco,thresholds)},
            "diagnostics":{"voc":diagnostics(voc,vd),"coco":diagnostics(coco,cd)},
            "registered_decision_inputs":decisions,"voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_R0"))
        for ds in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[ds][name]["per_class_iou"].items():
                    writer.writerow((ds,name,cls,iou,cells[ds][name]["per_class_delta_vs_reference"][cls]))
    lines=["# SG-combined-v3","",
           "Five cells raw and with unchanged R2, 40 class shards, 490 VOC and 707 COCO images; historical A3 seed.",
           "VOC-only selection; COCO thresholds frozen. AP, best-threshold and GT cap-loss values are diagnostic only.","",
           "## Selection table (registered, reference R0)","",
           "| Cell | VOC forced OOF | Δ vs R0 | Fold wins | Full VOC (thr) | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        o=oof[name]
        lines.append(f"| {name} | {o['miou']:.4f} | {o['delta']:+.4f} | {o['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} ({cells['voc'][name]['voc_frozen_threshold']:.3f}) | "
                     f"{cells['coco'][name]['miou']:.4f} |")
    lines+=["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
            f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
            "## Mechanism table (reference R1 = v2 V3)","",
            "| Contrast | VOC OOF Δ | Fold wins | Paired CI | COCO frozen Δ | COCO best-thr Δ | COCO raw AP Δ | VOC sheep Δ |",
            "|---|---:|---:|---|---:|---:|---:|---:|"]
    for a,b in MECHANISM:
        for s in ("","+R2"):
            m=mech[f"{b}{s}_vs_{a}{s}"];pb=m["paired_mean_image_iou"]
            ref=metrics(coco,a+s,a+s,thresholds)["miou"]
            rk=ranking(coco,a)[b+s];ap=rk.get(f"raw_ap_vs_{a}")
            voc_cell=metrics(voc,b+s,a+s,thresholds)["per_class_delta_vs_reference"]["sheep"]
            lines.append(f"| {b}{s} − {a}{s} | {m['delta']:+.4f} | {m['fold_wins']}/5 | "
                         f"[{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}] | {cells['coco'][b+s]['miou']-ref:+.4f} | "
                         f"{rk['best_threshold_delta']:+.4f} | {'—' if ap is None else format(ap['mean'],'+.4f')} | "
                         f"{voc_cell:+.3f} |")
    d=output["diagnostics"]["voc"]
    lines+=["","GT cap loss at 16→32 (VOC; sheep vs non-sheep): "+"; ".join(
                f"{row} γ_GT {d['gt_cap_loss_16_to_32'][row]['sheep']['gamma_GT']:.3f}/"
                f"{d['gt_cap_loss_16_to_32'][row]['non_sheep']['gamma_GT']:.3f}, L_GT "
                f"{d['gt_cap_loss_16_to_32'][row]['sheep']['L_GT']:.3f}/{d['gt_cap_loss_16_to_32'][row]['non_sheep']['L_GT']:.3f}"
                for row in GT_ROWS),
            f"W2 native coarse seed vs downsampled (VOC): {json.dumps(d['c16_native_vs_downsampled'])}",
            f"Registered decision inputs: {json.dumps(decisions)}","",
            "Size bins, cap binding, per-class GT cap loss and per-class IoU are in results.json; dense maps are in the shard sidecars.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),"coco_images":len(coco.image_ids),
        "all_required_stages_complete":True,"raw_frozen":screens["raw"]["frozen"],
        "final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":main()
