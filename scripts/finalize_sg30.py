#!/usr/bin/env python3
"""Merge complete SG30 shards, select on VOC folds, audit graph mechanisms."""
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
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg30 import BASE,NAMES,RAW,SIDE_FIELDS,SPEC_PATH
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv,sg30_spark


def load(out:Path,dataset:str,spec:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG30 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG30 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG30 {dataset} shards")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG30 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("SG30 diagnostic coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("SG30 VOC folds changed")
    return result,{field:np.stack([side[i][j] for i in result.image_ids])
                   for j,field in enumerate(SIDE_FIELDS)}


def mechanism(result,data):
    graph=data["graph"]
    transition=data["transition"]
    change=data["change"]
    classes=np.asarray(result.classes)
    graph_mean=np.nanmean(graph,axis=(0,1,2))
    transition_mean=np.nanmean(transition,axis=(0,1,2))
    output={"sampled_graph_rows_note":"32 uniform plus up to 16 GT foreground and 16 background rows; GT never enters inference",
            "graph_fields":sg30_spark.GRAPH_FIELDS,
            "transition_fields":sg30_spark.TRANSITION_FIELDS,
            "graph_by_cell":{name:{field:float(value) for field,value in zip(sg30_spark.GRAPH_FIELDS,graph_mean[i])}
                             for i,name in enumerate(sg30_spark.MAIN)},
            "gt_transition_mass_by_cell":{name:{field:float(value) for field,value in zip(sg30_spark.TRANSITION_FIELDS,transition_mean[i])}
                                          for i,name in enumerate(sg30_spark.MAIN)},
            "edge_quartile_mass":{"original_vs_inflated_A":np.nanmean(data["edge_quartile_mass"][:,:,:,0],axis=(0,1,2)).tolist(),
                                  "original_vs_inflated_A2":np.nanmean(data["edge_quartile_mass"][:,:,:,1],axis=(0,1,2)).tolist(),
                                  "order":["original_mass_Q1_to_Q4","transformed_mass_Q1_to_Q4"]},
            "iterations":{"fields":sg30_spark.ITER_FIELDS,
                          "per_update_mean":np.nanmean(data["iteration"],axis=(0,1,2)).tolist()},
            "changes_at_0p45":{name:{field:int(value) for field,value in zip(CHANGE_FIELDS,change[:,i].sum(0))}
                               for i,name in enumerate(RAW[1:])},
            "changes_by_class":{name:{cell:{field:int(value) for field,value in zip(CHANGE_FIELDS,change[classes==name,i].sum(0))}
                                      for i,cell in enumerate(RAW[1:])} for name in sorted(set(classes))}}
    for name in RAW[1:]:
        record=output["changes_at_0p45"][name]
        record["fp_removed_per_tp_lost"]=float(record["fp_removed"]/record["tp_lost"]) if record["tp_lost"] else None
    return output


def historical_replay(result,dataset:str):
    root=ROOT/"outputs/segmentation/controlled_attention_cv/sg29"/dataset/"shards"
    files=sorted(p for p in root.glob("class_*.npz") if not p.name.endswith(".diagnostics.npz"))
    old=cv.load_results(files)
    if len(files)!=20 or set(old.image_ids)!=set(result.image_ids) or old.thresholds!=result.thresholds:
        raise ValueError("SG29 baseline replay coverage differs from SG30")
    if tuple(old.configurations[:2])!=("N0","N0+R2"):
        raise ValueError("SG29 historical baseline changed")
    lookup={image_id:i for i,image_id in enumerate(old.image_ids)}
    ordered=np.asarray([lookup[image_id] for image_id in result.image_ids])
    prior=old.counts[:2,ordered];current=result.counts[:2]
    diff=np.abs(current.astype(np.int64)-prior.astype(np.int64))
    at=result.thresholds.index(.45)
    return {"exact_raw_images":int(np.all(diff[0]==0,axis=(1,2)).sum()),
            "exact_R2_images":int(np.all(diff[1]==0,axis=(1,2)).sum()),
            "max_count_difference":int(diff.max()),
            "mean_absolute_raw_image_IoU_difference_at_0p45":float(np.mean(np.abs(
                image_ious(current[0,:,at])-image_ious(prior[0,:,at])))),
            "interpretation":"separate regenerations; all SG30 candidate comparisons use current-generation S0"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();spec=json.loads(SPEC_PATH.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG30 smoke expects one image per dataset")
        for data in (vd,cd):
            if (data["graph"].shape[1:]!=(11,8,8,len(sg30_spark.GRAPH_FIELDS)) or
                data["transition"].shape[1:]!=(11,8,8,len(sg30_spark.TRANSITION_FIELDS)) or
                data["iteration"].shape[1:]!=(11,8,3,len(sg30_spark.ITER_FIELDS))):
                raise ValueError("SG30 diagnostic sidecar shape changed")
        print("SG30 live VOC/COCO smoke passed: 8 raw + 8 R2 + 2 diagnostic cells",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"S0"+suffix,[n+suffix for n in sg30_spark.MAIN[1:]],
                              {n+suffix:1 for n in sg30_spark.MAIN[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"S0+R2" if name.endswith("+R2") else "S0") for name in NAMES}
    cells={dataset:{name:metrics(result,name,"S0+R2" if name.endswith("+R2") else "S0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "mechanism":{"voc":mechanism(voc,vd),"coco":mechanism(coco,cd)},
             "historical_replay":{"voc":historical_replay(voc,"voc"),
                                   "coco":historical_replay(coco,"coco")},
             "voc_thresholds_frozen_for_coco":True,"iterated_S5_diagnostics_not_selectable":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_S0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines=["# SG30 SPARK-inspired nonlinear affinity sharpening","",
           "40 complete class shards, 490 VOC and 707 COCO images; VOC-only selection, COCO frozen.","",
           "| Cell | VOC forced OOF | Δ vs S0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item=oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | {cells['coco'][name]['miou']:.4f} |")
    lines += ["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
              "Full per-class, paired 20k intervals, graph/GT/edge/iteration diagnostics and replay audit are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
      "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
      "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__=="__main__":main()
