#!/usr/bin/env python3
"""Validate SG29 shards and report VOC selection with frozen COCO transfer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof, metrics
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg29 import BASE, NAMES, RAW, SIDE_FIELDS, SPEC_PATH
from scripts.select_propagation_cv import best_threshold, run_screen
from src.segmentation import controlled_cv as cv, sg29_nerve


def load(out: Path, dataset: str, spec: dict, smoke: bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG29 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG29 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG29 {dataset} shards")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:
                    raise ValueError("duplicate SG29 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):
        raise ValueError("SG29 diagnostic coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC folds changed")
    return result,{field:np.stack([side[i][j] for i in result.image_ids])
                   for j,field in enumerate(SIDE_FIELDS)}


def corr(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    if mask.sum()<3 or np.std(x[mask])==0 or np.std(y[mask])==0:
        return {"n":int(mask.sum()),"pearson":None,"spearman":None}
    return {"n":int(mask.sum()),"pearson":float(np.corrcoef(x[mask],y[mask])[0,1]),
            "spearman":float(spearmanr(x[mask],y[mask]).statistic)}


def diagnostics(result,data):
    weights=data["weights"];fused=data["fused_weights"]
    entropy=data["entropy"];utility=data["head_ap"]
    change=data["change"]
    classes=np.asarray(result.classes)
    def weight_stats(w):
        flat=w.reshape(-1,8)
        return {"per_head_mean":flat.mean(0).tolist(),"per_head_median":np.median(flat,axis=0).tolist(),
                "per_head_min":flat.min(0).tolist(),"per_head_max":flat.max(0).tolist(),
                "effective_heads_mean":float(np.mean(1/np.sum(flat**2,axis=1))),
                "argmax_frequency":np.bincount(flat.argmax(1),minlength=8).tolist()}
    def head_corr(choose):
        x=(-entropy[choose]).reshape(-1,8); y=utility[choose].reshape(-1,8)
        rows=[corr(xx,yy) for xx,yy in zip(x,y)]
        valid=[r for r in rows if r["spearman"] is not None]
        return {"observations":len(valid),"mean_per_call_pearson":float(np.mean([r["pearson"] for r in valid])) if valid else None,
                "mean_per_call_spearman":float(np.mean([r["spearman"] for r in valid])) if valid else None,
                "pooled":corr(x.reshape(-1),y.reshape(-1))}
    temporal=[]
    for per_image in entropy:
        rankings=[spearmanr(per_image[i],per_image[j]).statistic for i in range(11) for j in range(i+1,11)]
        temporal.append(float(np.nanmean(rankings)))
    transitions=data["transitions"]
    output={"weights_global":weight_stats(weights),"weights_fused":weight_stats(fused),
            "head_entropy_vs_AP":{"overall":head_corr(np.ones(len(classes),bool)),
                                  "per_class":{name:head_corr(classes==name) for name in sorted(set(classes))}},
            "within_image_between_call_entropy_rank_spearman_mean":float(np.nanmean(temporal)),
            "per_head_entropy_mean":entropy.mean((0,1)).tolist(),
            "transition_sampled_rows":{"global":np.nanmean(transitions[:,:,:,0,:],axis=(0,1,2)).tolist(),
                                       "local":np.nanmean(transitions[:,:,:,1,:],axis=(0,1,2)).tolist(),
                                       "fused":np.nanmean(transitions[:,:,:,2,:],axis=(0,1,2)).tolist()},
            "transition_fields":["mean_distance","row_entropy","neighbor_mass","effective_neighborhood",
                                 "target_mass_all_sources","target_mass_target_sources","target_mass_background_sources"],
            "changes_at_0p45":{name:{field:int(value) for field,value in zip(CHANGE_FIELDS,change[:,i].sum(0))}
                                 for i,name in enumerate(RAW[1:])},
            "changes_by_class":{name:{cell:{field:int(value) for field,value in zip(CHANGE_FIELDS,change[classes==name,i].sum(0))}
                                      for i,cell in enumerate(RAW[1:])} for name in sorted(set(classes))}}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args(); spec=json.loads(SPEC_PATH.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG29 live smoke needs one image per dataset")
        print("SG29 live VOC/COCO smoke passed",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"N0"+suffix,[n+suffix for n in sg29_nerve.MAIN[1:]],
                              {n+suffix:1 for n in sg29_nerve.MAIN[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"N0+R2" if name.endswith("+R2") else "N0") for name in NAMES}
    cells={dataset:{name:metrics(result,name,"N0+R2" if name.endswith("+R2") else "N0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "walk_depth_mass":sg29_nerve.walk_mass(spec["alpha"],spec["default_walk_length"]).tolist(),
             "diagnostics":{"voc":diagnostics(voc,vd),"coco":diagnostics(coco,cd)},
             "voc_thresholds_frozen_for_coco":True,"N2_short_lengths_diagnostic_only":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_N0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines=["# SG29 NERVE-inspired semantic weighting and stopped walk","",
           "490 VOC / 707 COCO; VOC-only selection; COCO frozen transfer. NERVE adaptation, not reproduction.","",
           "| Cell | VOC forced OOF | Δ vs N0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item=oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | {cells['coco'][name]['miou']:.4f} |")
    lines += ["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
              "Full per-class, precision/recall, bootstrap intervals, and head/local/walk diagnostics are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),"coco_images":len(coco.image_ids),
      "all_required_stages_complete":True,"raw_frozen":screens["raw"]["frozen"],
      "final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__=="__main__":main()
