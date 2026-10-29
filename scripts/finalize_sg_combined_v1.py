#!/usr/bin/env python3
"""Merge SG-combined-v1 shards and report registered selection and mechanisms."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer
from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.run_sg27 import CHANGE_FIELDS
from scripts.run_sg_combined_v1 import BASE,SPEC,RAW,NAMES,SIDE_FIELDS,REGIONS,BIN_FIELDS
from scripts.select_propagation_cv import best_threshold,image_ious,pooled_miou,run_screen
from src.segmentation import controlled_cv as cv


def load(out:Path,dataset:str,spec:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG-combined-v1 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG-combined-v1 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} shards")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("diagnostic coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold assignment changed")
    return result,{field:np.stack([side[x][j] for x in result.image_ids])
                   for j,field in enumerate(SIDE_FIELDS)}


def finite_mean(values):
    values=np.asarray(values);keep=np.isfinite(values)
    return float(values[keep].mean()) if keep.any() else None


def correlation(a,b):
    a=np.asarray(a,np.float64).reshape(-1);b=np.asarray(b,np.float64).reshape(-1)
    good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<3 or np.std(a[good])<1e-12 or np.std(b[good])<1e-12:return None
    return float(np.corrcoef(a[good],b[good])[0,1])


def change_summary(rows):
    total=np.asarray(rows).sum(0)
    return {field:int(total[i]) for i,field in enumerate(CHANGE_FIELDS)}


def macro_stage(result,counts,route,stage):
    classes=np.asarray(result.classes)
    per={}
    for label in sorted(set(classes)):
        tp,fp,fn=counts[classes==label,route,stage,:3].sum(0)
        per[label]=float(tp/max(tp+fp+fn,1))
    return {"macro_miou":float(np.mean(list(per.values()))),"per_class":per}


def mechanism(result,side):
    classes=np.asarray(result.classes)
    transport={}
    for i,name in enumerate(("C2","C3","C4","C5")):
        transport[name]={}
        for j,transition in enumerate(("16_to_32","32_to_64")):
            v=side["transport"][:,i,j]
            if not np.isfinite(v).any():continue
            transport[name][transition]={"mean_parent_error":finite_mean(v[:,0]),
                "max_parent_error":finite_mean(v[:,1]),"p95_parent_error":finite_mean(v[:,2]),
                "fallback_parents_total":int(np.nansum(v[:,3])*11)}
    ratios={transition:{field:finite_mean(side["ratio"][:,i,k])
                        for k,field in enumerate(("mean_abs_log_ratio_error",
                                                  "max_abs_log_ratio_error","p95_abs_log_ratio_error"))}
            for i,transition in enumerate(("16_to_32","32_to_64"))}
    entropy={transition:{field:finite_mean(side["entropy"][:,i,k])
                         for k,field in enumerate(("before_SA","SA_proposal","KL_projected"))}
             for i,transition in enumerate(("16_to_32","32_to_64"))}
    stages={route:{stage:macro_stage(result,side["stage_counts"],i,j)
                   for j,stage in enumerate(("16","32","64"))}
            for i,route in enumerate(("C1","C4","C5"))}
    stage_changes={route:{transition:change_summary(side["stage_changes"][:,i,j])
                          for j,transition in enumerate(("16_to_32","32_to_64"))}
                   for i,route in enumerate(("C1","C4","C5"))}
    wrong=side["wrong_coarse"].sum(0)
    categories=("coarse_correct_fine_correct","coarse_correct_fine_wrong",
                "coarse_wrong_fine_correct","coarse_wrong_fine_wrong")
    error_types={candidate:{cat:{"pixels":int(values[0]),"corrected":int(values[1]),
                                 "newly_wrong":int(values[2])}
                            for cat,values in zip(categories,side["error_types"][:,i].sum(0))}
                 for i,candidate in enumerate(("C4","C5"))}
    bins={}
    for name,field in (("odds_G","completion_bins"),("positive_residual","residual_bins")):
        pooled=side[field].sum(0)
        bins[name]=[{**{column:(int(row[k]) if k<6 else float(row[k]))
                         for k,column in enumerate(BIN_FIELDS)},
                      "TP_fraction":float(row[1]/max(row[0],1)),
                      "TP_per_FP":float(row[1]/max(row[2]+row[3]+row[4],1))}
                     for row in pooled]
    efficiency={}
    for i,name in enumerate(("C6","C7","C8")):
        v=side["completion_efficiency"][:,i,:5].sum(0)
        efficiency[name]={"TP_added":int(v[0]),"FP_added":int(v[1]),
                          "FP_more_than_64_added":int(v[2]),
                          "TP_lost":int(v[3]),"FP_removed":int(v[4]),
                          "TP_added_per_FP_added":float(v[0]/max(v[1],1)),
                          "TP_added_per_distant_FP_added":float(v[0]/max(v[2],1))}
    beta={name:{"mean":finite_mean(side["beta"][:,i]),
                "p10":float(np.quantile(side["beta"][:,i],.1)),
                "median":float(np.median(side["beta"][:,i])),
                "p90":float(np.quantile(side["beta"][:,i],.9)),
                "correlation_with_unbudgeted_P2_image_IoU_gain":correlation(
                    side["beta"][:,i],side["beta_context"][:,0]),
                "correlation_with_unbudgeted_P2_FP_increase":correlation(
                    side["beta"][:,i],side["beta_context"][:,1]),
                "per_class":{cls:finite_mean(side["beta"][classes==cls,i])
                             for cls in sorted(set(classes))}}
          for i,name in enumerate(("C7","C8","C9"))}
    background=side["background_log"]
    return {"transport_mass":transport,"KL_ratio_preservation":ratios,
            "child_distribution_entropy":entropy,"stage_at_fixed_0p45":stages,
            "stage_changes_at_fixed_0p45":stage_changes,
            "wrong_coarse_parent_diagnostic":{"high_seed_gt_background_parents":int(wrong[0]),
                 "C1_FP_in_those_parents":int(wrong[1]),"C4_FP":int(wrong[2]),"C5_FP":int(wrong[3])},
            "coarse_fine_error_types_at_fixed_0p45":error_types,
            "completion_candidate_bins":bins,"completion_efficiency_at_fixed_0p45":efficiency,
            "completion_budget_beta":beta,
            "SG32_background_map":{"fallback_image_calls":int(background[:,0].sum()),
                 "mean_selected_row_fraction":float(background[:,1].mean())},
            "prediction_changes_vs_C0_at_fixed_0p45":{
                 name:change_summary(side["change"][:,i]) for i,name in enumerate(RAW[1:])},
            "GT_usage":"all GT-derived quantities are post-hoc diagnostics only"}


def caption_presence(coco,thresholds,candidates):
    rows={r["image_id"]:r for shard in range(20) for r in transfer.coco_rows(shard)[1]}
    if set(rows)!=set(coco.image_ids):raise ValueError("COCO caption join mismatch")
    present=np.asarray([rows[x]["word_included"].strip().lower()=="true" for x in coco.image_ids])
    classes=np.asarray(coco.classes)
    output={}
    for name in dict.fromkeys(candidates):
        output[name]={}
        for label,mask in (("present",present),("absent",~present)):
            a=coco.counts[coco.index(name),mask,thresholds[name]]
            b=coco.counts[coco.index("C0+R2"),mask,thresholds["C0+R2"]]
            output[name][label]={"images":int(mask.sum()),"miou":pooled_miou(a,classes[mask]),
                "reference_miou":pooled_miou(b,classes[mask]),
                "delta":pooled_miou(a,classes[mask])-pooled_miou(b,classes[mask]),
                "paired_mean_image_iou":cv.paired_bootstrap(image_ious(a),image_ious(b))}
    return {"field":"class word present in generation caption; descriptive only","cells":output}


def historical_replay(result,dataset):
    files=sorted(p for p in (ROOT/"outputs/segmentation/controlled_attention_cv/sg31"/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith(".diagnostics.npz"))
    prior=cv.load_results(files)
    if len(files)!=20 or set(prior.image_ids)!=set(result.image_ids) or prior.thresholds!=result.thresholds:
        raise ValueError("SG31 replay coverage mismatch")
    ordered=np.asarray([list(prior.image_ids).index(x) for x in result.image_ids])
    out={}
    for current,old in (("C0","S0"),("C0+R2","S0+R2"),("C1","S2"),("C1+R2","S2+R2")):
        a=result.counts[result.index(current)]
        b=prior.counts[prior.index(old),ordered]
        diff=np.abs(a.astype(np.int64)-b.astype(np.int64))
        at=result.thresholds.index(.45)
        out[current]={"SG31_prior_cell":old,"exact_all_threshold_images":int(np.all(diff==0,axis=(1,2)).sum()),
            "mean_abs_image_IoU_difference_at_0p45":float(np.mean(np.abs(image_ious(a[:,at])-image_ious(b[:,at])))),
            "max_count_difference":int(diff.max())}
    out["note"]="separate regenerated runs; all combined-v1 comparisons use same-generation controls"
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:raise ValueError("smoke needs one image per dataset")
        for side in (vd,cd):
            if (side["transport"].shape[1:]!=(4,2,4) or side["stage_counts"].shape[1:]!=(3,3,4)
                or side["completion_bins"].shape[1:]!=(5,len(BIN_FIELDS))):
                raise ValueError("diagnostic sidecar shape mismatch")
        print("SG-combined-v1 live smoke passed: 10 raw + 10 R2 and diagnostics",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"C0"+suffix,[n+suffix for n in RAW[1:]],
                              {n+suffix:1 for n in RAW[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"C0+R2" if name.endswith("+R2") else "C0") for name in NAMES}
    cells={dataset:{name:metrics(result,name,"C0+R2" if name.endswith("+R2") else "C0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    contrasts={f"{a}_vs_{b}":forced_oof(voc,a,b) for a,b in
               (("C2","C1"),("C3","C1"),("C4","C1"),("C5","C4"),
                ("C6","C4"),("C7","C6"),("C8","C6"),("C8","C7"),("C9","C8"),
                ("C4+R2","C1+R2"),("C5+R2","C4+R2"),("C8+R2","C0+R2"),
                ("C8+R2","C4+R2"),("C9+R2","C0+R2"))}
    frozen=screens["final"]["frozen"]
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"theoretical_contrasts":contrasts,
        "cells":cells,"mechanism":{"voc":mechanism(voc,vd),"coco":mechanism(coco,cd)},
        "historical_replay":{"voc":historical_replay(voc,"voc"),"coco":historical_replay(coco,"coco")},
        "caption_presence_posthoc":caption_presence(coco,thresholds,[frozen,"C4+R2","C5+R2","C8+R2","C9+R2"]),
        "voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_C0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                for cls,iou in cells[dataset][name]["per_class_iou"].items():
                    writer.writerow((dataset,name,cls,iou,cells[dataset][name]["per_class_delta_vs_reference"][cls]))
    lines=["# SG-combined-v1 constrained transport and budgeted completion","",
           "40 complete class shards, 490 VOC and 707 COCO; VOC-only selection, COCO frozen.","",
           "| Cell | VOC forced OOF | Δ vs C0 | Pairwise wins | Full VOC | VOC threshold | Frozen COCO |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item=oof[name]
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | "
                     f"{cells['coco'][name]['miou']:.4f} |")
    lines += ["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
              f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
              "Full per-class, paired 20k, transport, completion and visual diagnostics are in results.json and visuals/.",""]
    (out/"report.md").write_text("\n".join(lines))
    visuals=sorted((out/"voc"/"visuals").glob("class_*.png"))
    if len(visuals)!=20:raise ValueError(f"expected 20 VOC visual sheets, found {len(visuals)}")
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
        "visual_sheets":len(visuals),"raw_frozen":screens["raw"]["frozen"],
        "final_frozen":screens["final"]["frozen"]},indent=2)+"\n")
    print("\n".join(lines[-5:]),flush=True)


if __name__=="__main__":main()
