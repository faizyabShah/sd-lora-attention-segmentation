#!/usr/bin/env python3
"""Merge SG33 shards and audit hierarchy/Markov mechanisms separately."""
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
from scripts.finalize_sg31 import caption_presence
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg33 import (BASE,SPEC,NAMES,RAW,SIDE_FIELDS,MARKOV_FIELDS,REGIONS)
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv,sg33_hierarchy


def load(out:Path,dataset:str,spec:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG33 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG33 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG33 {dataset} shards")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG33 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("SG33 diagnostic coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold assignment changed")
    return result,{field:np.stack([side[x][j] for x in result.image_ids])
                   for j,field in enumerate(SIDE_FIELDS)}


def finite_mean(values):
    a=np.asarray(values);keep=np.isfinite(a)
    return float(a[keep].mean()) if keep.any() else None


def change_summary(rows):
    total=rows.sum(0)
    d={field:int(value) for field,value in zip(CHANGE_FIELDS,total)}
    d["fp_removed_per_tp_lost"]=(d["fp_removed"]/d["tp_lost"] if d["tp_lost"] else None)
    return d


def stage_class_miou(result,counts):
    classes=np.asarray(result.classes)
    per={}
    for i,name in enumerate(("r16","r32","r64")):
        score=[];by_class={}
        for cls in sorted(set(classes)):
            pooled=counts[classes==cls,i,:3].sum(0)
            value=float(pooled[0]/max(pooled.sum(),1))
            score.append(value);by_class[cls]=value
        per[name]={"macro_iou_at_0p45":float(np.mean(score)),"per_class_iou":by_class}
    return per


def mechanism(result,side):
    classes=np.asarray(result.classes)
    changes={name:change_summary(side["change"][:,i]) for i,name in enumerate(RAW[1:])}
    by_class={cls:{name:change_summary(side["change"][classes==cls,i])
                   for i,name in enumerate(RAW[1:])} for cls in sorted(set(classes))}
    stage={"stage_iou":stage_class_miou(result,side["stage_counts"]),
           "correlation_mean":[[finite_mean(side["stage_correlations"][:,i,j])
                                for j in range(3)] for i in range(3)],
           "changes_r16_to_r32":change_summary(side["stage_change"][:,0]),
           "changes_r32_to_r64":change_summary(side["stage_change"][:,1]),
           "earlier_higher_than_fine_counts":{
               source:{region:int(side["rescue"][:,i,k].sum()) for k,region in enumerate(REGIONS)}
               for i,source in enumerate(("r16","r32"))}}
    markov={}
    for mi,name in enumerate(sg33_hierarchy.MARKOV):
        region_stats={}
        for k,region in enumerate(REGIONS):
            region_stats[region]={field:finite_mean(side["markov"][:,mi,k,j])
                                  for j,field in enumerate(MARKOV_FIELDS)}
        global_fields=("roc_auc_minus_arrival_vs_gt","mean_arrival_bg_minus_fg",
                       "unsaturated_fraction_at_256","conditional_unsaturated_at_512")
        markov[name]={"by_region":region_stats,
            **{field:finite_mean(side["markov_global"][:,mi,j])
               for j,field in enumerate(global_fields)},
            "amplitude_vs_reachability_delta_gt_0p2":{
               direction:{region:int(side["amplitude_disagreement"][:,mi,d,k].sum())
                          for k,region in enumerate(REGIONS)}
               for d,direction in enumerate(("reachability_higher","amplitude_higher"))}}
    ipf={name:{field:finite_mean(side["ipf"][:,i,j])
               for j,field in enumerate(sg33_hierarchy.IPF_FIELDS)}
         for i,name in enumerate(sg33_hierarchy.MARKOV)}
    temp={field:finite_mean(side["temperature"][:,j])
          for j,field in enumerate(sg33_hierarchy.TEMP_FIELDS)}
    return {"hierarchy":stage,"markov":markov,"ipf":ipf,"temperature_M3":temp,
            "prediction_changes_at_0p45":changes,"changes_by_class_at_0p45":by_class,
            "markov_distribution_note":"means of per-image/call 16x16 arrival-time statistics; GT sampled at coarse resolution only for diagnostics",
            "stationary_entropy_note":"256-step estimate from uniform start, not an exact eigensolver",
            "GT_usage":"post-hoc diagnostics only; no GT in inference or selection except VOC threshold/folds"}


def historical_replay(result,dataset):
    root=ROOT/"outputs/segmentation/controlled_attention_cv/sg31"/dataset/"shards"
    files=sorted(p for p in root.glob("class_*.npz") if not p.name.endswith(".diagnostics.npz"))
    old=cv.load_results(files)
    if len(files)!=20 or set(old.image_ids)!=set(result.image_ids) or old.thresholds!=result.thresholds:
        raise ValueError("SG31 replay coverage differs from SG33")
    lookup={image_id:i for i,image_id in enumerate(old.image_ids)}
    ordered=np.asarray([lookup[x] for x in result.image_ids])
    out={}
    for current,prior in (("S0","S0"),("S0+R2","S0+R2"),
                          ("H0","S2"),("H0+R2","S2+R2")):
        a=result.counts[result.index(current)];b=old.counts[old.index(prior),ordered]
        diff=np.abs(a.astype(np.int64)-b.astype(np.int64))
        at=result.thresholds.index(.45)
        out[current]={"SG31_prior_cell":prior,
                      "exact_all_threshold_images":int(np.all(diff==0,axis=(1,2)).sum()),
                      "mean_abs_image_IoU_difference_at_0p45":float(np.mean(np.abs(
                          image_ious(a[:,at])-image_ious(b[:,at])))),
                      "max_count_difference":int(diff.max())}
    out["interpretation"]="separate regenerations; SG33 comparisons use same-generation controls"
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
            raise ValueError("SG33 smoke needs one image per dataset")
        for data in (vd,cd):
            if (data["stage_counts"].shape[1:]!=(3,4) or
                data["markov"].shape[1:]!=(3,4,len(MARKOV_FIELDS)) or
                data["ipf"].shape[1:]!=(3,len(sg33_hierarchy.IPF_FIELDS))):
                raise ValueError("SG33 diagnostic sidecar shape changed")
        print("SG33 live VOC/COCO smoke passed: 12 raw + 12 R2, Markov/IPF diagnostics",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"S0"+suffix,[n+suffix for n in RAW[1:]],
                              {n+suffix:1 for n in RAW[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"S0+R2" if name.endswith("+R2") else "S0")
         for name in NAMES}
    cells={dataset:{name:metrics(result,name,"S0+R2" if name.endswith("+R2") else "S0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    residual=max((n+"+R2" for n in ("H1","H2","H3","H4")),key=lambda n:cells["voc"][n]["miou"])
    markov=max((n+"+R2" for n in ("M1","M2","M3","C1","C2","C3")),key=lambda n:cells["voc"][n]["miou"])
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "mechanism":{"voc":mechanism(voc,vd),"coco":mechanism(coco,cd)},
             "historical_replay":{"voc":historical_replay(voc,"voc"),
                                   "coco":historical_replay(coco,"coco")},
             "caption_presence_posthoc":caption_presence(coco,thresholds,[residual,markov]),
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
    lines=["# SG33 hierarchy residual and coarse Markov reachability","",
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
              "Full per-class, paired 20k, hierarchy, Markov/IPF and visual diagnostics are in results.json and visuals/.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
       "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
       "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__=="__main__":main()
