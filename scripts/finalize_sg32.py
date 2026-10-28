#!/usr/bin/env python3
"""Merge SG32 shards; separate VOC selection from reliability mechanisms."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.finalize_sg31 import caption_presence
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg32 import (BASE,SPEC,NAMES,RAW,SIDE_FIELDS,BIN_FIELDS,
                              EXP_FIELDS,REGIONS,TEMPERATURE_SIGNALS)
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv,sg32_adaptive


def load(out:Path,dataset:str,spec:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG32 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG32 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG32 {dataset} shards")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG32 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("SG32 diagnostic coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold assignment changed")
    return result,{field:np.stack([side[x][j] for x in result.image_ids])
                   for j,field in enumerate(SIDE_FIELDS)}


def change_summary(values):
    summed=values.sum(axis=0)
    d={field:int(value) for field,value in zip(CHANGE_FIELDS,summed)}
    d["fp_removed_per_tp_lost"]=(d["fp_removed"]/d["tp_lost"] if d["tp_lost"] else None)
    return d


def _finite_mean(values):
    finite=np.isfinite(values)
    return float(np.asarray(values)[finite].mean()) if finite.any() else None


def mechanism(result,side):
    classes=np.asarray(result.classes)
    changes={name:change_summary(side["change"][:,i]) for i,name in enumerate(RAW[1:])}
    by_class={cls:{name:change_summary(side["change"][classes==cls,i])
                   for i,name in enumerate(RAW[1:])} for cls in sorted(set(classes))}
    bins=side["bins"].sum(axis=0)
    reliability_bins={}
    for si,name in enumerate(sg32_adaptive.SIGNALS):
        rows=[]
        for k,v in enumerate(bins[si]):
            n,fg,fp,bg,distance,separate,first,p2=v
            rows.append({"interval":f"[{k*.2:.1f},{(k+1)*.2:.1f}{']' if k==4 else ')'}",
                "pixels":int(n),"gt_fg_probability":float(fg/max(n,1)),
                "baseline_fp_probability":float(fp/max(n,1)),
                "baseline_bg_fp_rate":float(fp/max(bg,1)),
                "mean_bg_distance_from_gt":float(distance/max(bg,1)),
                "separate_fraction_of_fp":float(separate/max(fp,1)),
                "mean_first_pass_score":float(first/max(n,1)),
                "mean_p2_score":float(p2/max(n,1))})
        reliability_bins[name]=rows
    regions=side["regions"].sum(axis=0)
    region_means={name:{region:float(regions[si,k,0]/max(regions[si,k,1],1))
                       for k,region in enumerate(REGIONS)}
                  for si,name in enumerate(sg32_adaptive.SIGNALS)}
    exp={}
    for si,signal in enumerate(TEMPERATURE_SIGNALS):
        exp[signal]={}
        for k,region in enumerate(REGIONS):
            exp[signal][region]={field:_finite_mean(side["exponents"][:,si,k,j])
                                  for j,field in enumerate(EXP_FIELDS)}
    entropy={name:{field:_finite_mean(side["entropy"][:,i,j]) for j,field in enumerate(
                ("original_row_entropy","adapted_row_entropy","delta_entropy",
                 "correlation_sharp_strength_vs_entropy_reduction"))}
             for i,name in enumerate(("S7","S8","S9","S10","S11"))}
    temporal={region:{field:_finite_mean(side["temporal"][:,k,j])
                      for j,field in enumerate(("mean_R_comb","temporal_variance_R_comb",
                                               "fraction_mean_R_below_0p5"))}
              for k,region in enumerate(REGIONS)}
    cross=side["cross"].sum(axis=0)
    cross_summary={f"pair_low_{bool(k//2)}_bg_low_{bool(k%2)}":{
        "gt_fg_pixels":int(cross[k,0]),"bg_pixels":int(cross[k,1])} for k in range(4)}
    return {"reliability_bins":reliability_bins,"bin_fields":BIN_FIELDS,
        "region_reliability_means":region_means,"adaptive_exponent_by_region":exp,
        "exponent_summary_note":"mean of per-image distribution statistics; no GT in inference",
        "sampled_64_row_entropy":entropy,"temporal_combined_reliability":temporal,
        "cross_scale_bg_low_at_0p5":cross_summary,
        "background_seed_fallbacks":int(side["bg_log"][:,:,0].sum()),
        "mean_background_seed_fraction":float(side["bg_log"][:,:,1].mean()),
        "changes_vs_S0_at_0p45":changes,
        "s10_vs_s1_at_0p45":change_summary(side["s10_vs_s1"]),
        "changes_by_class_at_0p45":by_class,
        "GT_usage":"post-hoc diagnostics only; pixel distances are full annotation-image pixels"}


def historical_replay(result,dataset):
    root=ROOT/"outputs/segmentation/controlled_attention_cv/sg30"/dataset/"shards"
    files=sorted(p for p in root.glob("class_*.npz") if not p.name.endswith(".diagnostics.npz"))
    old=cv.load_results(files)
    if len(files)!=20 or set(old.image_ids)!=set(result.image_ids) or old.thresholds!=result.thresholds:
        raise ValueError("SG30 replay coverage differs from SG32")
    lookup={image_id:i for i,image_id in enumerate(old.image_ids)}
    order=np.asarray([lookup[x] for x in result.image_ids])
    out={}
    for current,prior in (("S0","S0"),("S0+R2","S0+R2"),("S1","S2"),("S1+R2","S2+R2")):
        a=result.counts[result.index(current)]
        b=old.counts[old.index(prior),order]
        diff=np.abs(a.astype(np.int64)-b.astype(np.int64))
        at=result.thresholds.index(.45)
        out[current]={"SG30_prior_cell":prior,
            "exact_all_threshold_images":int(np.all(diff==0,axis=(1,2)).sum()),
            "mean_abs_image_IoU_difference_at_0p45":float(np.mean(np.abs(
                image_ious(a[:,at])-image_ious(b[:,at])))),
            "max_count_difference":int(diff.max())}
    out["interpretation"]="separate regenerations; all SG32 comparisons use same-generation S0/S1"
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG32 live smoke requires one image per dataset")
        for data in (vd,cd):
            if (data["change"].shape[1:]!=(11,len(CHANGE_FIELDS)) or
                data["bins"].shape[1:]!=(6,5,len(BIN_FIELDS)) or
                data["exponents"].shape[1:]!=(5,4,len(EXP_FIELDS)) or
                data["entropy"].shape[1:]!=(5,4)):
                raise ValueError("SG32 diagnostic sidecar shape changed")
        print("SG32 live VOC/COCO smoke passed: 12 raw + 12 R2 and mechanisms",flush=True)
        return
    mask=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),mask)[0] for name in NAMES}
    screens={stage:run_screen(voc,"S0"+suffix,[n+suffix for n in RAW[1:]],
                              {n+suffix:1 for n in RAW[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"S0+R2" if name.endswith("+R2") else "S0")
         for name in NAMES}
    cells={dataset:{name:metrics(result,name,"S0+R2" if name.endswith("+R2") else "S0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    best=sorted((n for n in NAMES if n.endswith("+R2") and n not in ("S0+R2","S1+R2")),
                key=lambda n:cells["voc"][n]["miou"],reverse=True)[:3]
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "mechanism":{"voc":mechanism(voc,vd),"coco":mechanism(coco,cd)},
             "historical_replay":{"voc":historical_replay(voc,"voc"),
                                   "coco":historical_replay(coco,"coco")},
             "caption_presence_posthoc":caption_presence(coco,thresholds,best),
             "voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_S0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,
                                     cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines=["# SG32 reliability-adaptive second-pass propagation","",
           "40 complete class shards, 490 VOC and 707 COCO; VOC-only selection, COCO frozen.","",
           "| Cell | VOC forced OOF | Δ vs S0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item=oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | "
                     f"{cells['coco'][name]['miou']:.4f} |")
    lines += ["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
              "Full per-class, paired 20k and reliability diagnostics are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
       "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
       "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__=="__main__":main()
