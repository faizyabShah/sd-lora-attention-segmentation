#!/usr/bin/env python3
"""SG-threshold: per-image threshold rules on the SG-combined-v3 counts and dense maps (CPU only).

Every rule's tau_I is snapped to the registered grid, so the v3 confusion
counts (raw and R2) score it exactly. Parameterised rules put their parameter
on the 17-slot axis, which the unchanged selector chooses per training fold.
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

from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.run_sg_combined_v3 import RAW
from scripts.select_propagation_cv import best_threshold,image_ious,pooled_miou,run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation.sg_threshold import D_AXIS,GRID,K_AXIS,RULES,rule_slots

SPEC=ROOT/"configs/sg_threshold.json"
SOURCE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_combined_v3"
BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_threshold"
MAPS=("R0","R1")
STAGES=("","+R2")
SIZE_BINS=(0.,.1,.25,.5,.75,1.01)
AXES={"T0":("threshold",GRID),"T3":("k",K_AXIS),"T5":("R0_threshold",GRID),"T6":("offset",D_AXIS)}


def cells():
    return tuple(f"{m}.{r}{s}" for s in STAGES for m in MAPS for r in RULES if not (m=="R0" and r=="T5"))


def split(name):
    stage="+R2" if name.endswith("+R2") else ""
    m,r=name.removesuffix("+R2").split(".")
    return m,r,stage


def load(dataset,smoke):
    files=sorted(p for p in (SOURCE/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    if not smoke and len(files)!=20:raise ValueError(f"{dataset}: expected 20 v3 shards")
    result=cv.load_results(files)
    if tuple(result.thresholds)!=tuple(GRID):raise ValueError("v3 threshold grid differs")
    maps={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                maps[str(image_id)]=data["maps"][i].astype(np.float64)
    if set(maps)!=set(result.image_ids):raise ValueError(f"{dataset}: dense-map coverage mismatch")
    return result,np.stack([maps[x] for x in result.image_ids])


def alignment(result,maps):
    """Stored 64x64 maps must reproduce the counted predicted area (bilinear 512 vs 64)."""
    out={}
    for k,name in enumerate(RAW[:2]):
        c=result.counts[result.index(name)]
        counted=(c[...,0]+c[...,1])/c.sum(-1)
        stored=np.stack([[(cv_map>=t).mean() for t in GRID] for cv_map in maps[:,k].reshape(len(maps),-1)])
        out[name]={"pearson":float(np.corrcoef(counted.ravel(),stored.ravel())[0,1]),
                   "mean_abs_area_difference":float(np.abs(counted-stored).mean())}
        if out[name]["pearson"]<.99:raise ValueError(f"{name}: stored maps do not match counts {out[name]}")
    return out


def build(result,maps):
    """Synthetic Results whose 17-slot axis is each rule's parameter axis."""
    names=cells();n=len(result.image_ids)
    counts=np.zeros((len(names),n,17,4),np.int64);edges=np.zeros((len(names),n,17),bool)
    cache={}
    for i in range(n):
        r0=maps[i,RAW.index("R0")]
        for m in MAPS:
            y=maps[i,RAW.index(m)]
            for r in RULES:
                if m=="R0" and r=="T5":continue
                cache[m,r,i]=rule_slots(r,y,r0)
    for c,name in enumerate(names):
        m,r,stage=split(name);src=result.counts[result.index(m+stage)]
        for i in range(n):
            slots,edge=cache[m,r,i]
            counts[c,i]=src[i,slots];edges[c,i]=edge
    synthetic=cv.Results(names,tuple(GRID),result.image_ids,result.classes,result.folds,counts,
                         np.zeros((len(names),n)),result.settings,result.complete)
    return synthetic,edges


def ceiling(result):
    out={}
    for name in ("R0","R1","R0+R2","R1+R2"):
        c=result.counts[result.index(name)]
        iou=c[...,0]/np.maximum(c[...,:3].sum(-1),1)
        best=c[np.arange(len(c)),iou.argmax(1)]
        out[name]={"pooled_miou":pooled_miou(best,result.classes),"mean_image_iou":float(image_ious(best).mean())}
    return out


def size_bins(result,thresholds,reference):
    ref=result.counts[result.index(reference),:,thresholds[reference]]
    gt=(ref[:,0]+ref[:,2])/np.maximum(ref.sum(1),1);out={}
    for name in result.configurations:
        diff=image_ious(result.counts[result.index(name),:,thresholds[name]])-image_ious(ref)
        out[name]=[{"gt_fraction":[lo,min(hi,1.)],"images":int(k.sum()),
                    "mean_image_iou_change":float(diff[k].mean()) if k.any() else None}
                   for lo,hi in zip(SIZE_BINS[:-1],SIZE_BINS[1:]) for k in [(gt>=lo)&(gt<hi)]]
    return out


def parameter(name,slot):
    _,r,_=split(name)
    if r not in AXES:return None
    label,axis=AXES[r];return {label:float(axis[slot])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    voc,vm=load("voc",args.smoke);coco,cm=load("coco",args.smoke)
    if args.smoke:
        keep=np.arange(len(voc.image_ids))<40
        voc,vm=voc.subset(keep),vm[keep]
    align={"voc":alignment(voc,vm),"coco":alignment(coco,cm)}
    sv,ev=build(voc,vm);sc,ec=build(coco,cm)
    names=sv.configurations
    keep=np.ones(len(sv.image_ids),bool)
    thresholds={n:best_threshold(sv,sv.index(n),keep)[0] for n in names}
    ref={s:"R0.T0"+s for s in STAGES}
    screens={st:run_screen(sv,ref[s],[n for n in names if split(n)[2]==s and n!=ref[s]],
                           {n:1 for n in names}) for st,s in (("raw",""),("final","+R2"))}
    vs_ref={n:forced_oof(sv,n,ref[split(n)[2]]) for n in names if n not in ref.values()}
    vs_t0={n:forced_oof(sv,n,f"{split(n)[0]}.T0{split(n)[2]}") for n in names if split(n)[1]!="T0"}
    full={ds:{n:metrics(r,n,ref[split(n)[2]],thresholds) for n in names} for ds,r in (("voc",sv),("coco",sc))}
    clip={ds:{n:float(e[sv.index(n),:,thresholds[n]].mean()) for n in names} for ds,e in (("voc",ev),("coco",ec))}
    params={n:parameter(n,thresholds[n]) for n in names}
    output={"spec":spec,"alignment":align,"selection":screens,"forced_oof_vs_R0_T0":vs_ref,
            "forced_oof_vs_same_map_T0":vs_t0,"cells":full,"selected_parameter_full_voc":params,
            "grid_edge_fraction":clip,"ceiling_diagnostic_only":{"voc":ceiling(voc),"coco":ceiling(coco)},
            "size_bins_diagnostic_only":{s or "raw":size_bins(sv,thresholds,ref[s]) for s in STAGES},
            "voc_parameters_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_R0.T0"))
        for ds in ("voc","coco"):
            for n in names:
                for cls,iou in full[ds][n]["per_class_iou"].items():
                    writer.writerow((ds,n,cls,iou,full[ds][n]["per_class_delta_vs_reference"][cls]))
    lines=["# SG-threshold","",
           "Per-image threshold rules on the SG-combined-v3 counts and dense maps; CPU only, no generation.",
           f"{len(voc.image_ids)} VOC / {len(coco.image_ids)} COCO images. VOC-only selection; VOC parameters frozen for COCO.","",
           "| Cell | VOC forced OOF | Δ vs R0.T0 | Wins | Δ vs same-map T0 (wins) | Full VOC (param) | Sheep Δ | Worst class | Edge % | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for n in names:
        o=vs_ref.get(n);t=vs_t0.get(n);v=full["voc"][n];pc=v["per_class_delta_vs_reference"]
        worst=min(pc,key=pc.get)
        lines.append(f"| {n} | {(o or {'miou':forced_oof(sv,n,n)['miou']})['miou']:.4f} | "
                     f"{'—' if o is None else format(o['delta'],'+.4f')} | {'—' if o is None else str(o['fold_wins'])+'/5'} | "
                     f"{'—' if t is None else format(t['delta'],'+.4f')+' ('+str(t['fold_wins'])+'/5)'} | "
                     f"{v['miou']:.4f} ({json.dumps(params[n]) if params[n] else 'free'}) | {pc.get('sheep',float('nan')):+.3f} | "
                     f"{worst} {pc[worst]:+.3f} | {100*clip['voc'][n]:.0f} | {full['coco'][n]['miou']:.4f} |")
    lines+=["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']} ({screens['raw']['freeze_reason']})",
            f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']} ({screens['final']['freeze_reason']})",
            f"Ceiling (per-image best grid threshold, diagnostic): {json.dumps(output['ceiling_diagnostic_only']['voc'])}",
            f"Map/count alignment: {json.dumps(align)}",""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)


if __name__=="__main__":main()
