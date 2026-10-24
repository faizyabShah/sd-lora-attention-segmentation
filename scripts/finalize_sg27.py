#!/usr/bin/env python3
"""SG27 VOC selection, frozen COCO transfer and reduced fusion diagnostics."""
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

from scripts import run_coco_cap_transfer as transfer
from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.run_sg27 import BASE,CHANGE_FIELDS,MANIFEST,names_for
from scripts.select_propagation_cv import best_threshold,image_ious,pooled_miou,run_screen
from src.segmentation import controlled_cv as cv

SPEC=ROOT/"configs/sg27.json"


def load(out:Path,dataset:str,spec:dict,manifest:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    names=names_for(manifest["unique_cells"])
    if result.configurations!=names or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG27 cells or threshold grid differ")
    if (result.settings["spec"]!=spec or result.settings["manifest"]!=manifest
            or result.settings["dataset"]!=dataset):
        raise ValueError("foreign SG27 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete SG27 {dataset} shards")
    sidecars={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for image_id,change,pair,triple in zip(data["image_ids"],data["change"],
                                                   data["pair"],data["triple"]):
                if str(image_id) in sidecars:
                    raise ValueError("duplicate SG27 diagnostic image")
                sidecars[str(image_id)]=(change,pair,triple)
    if set(sidecars)!=set(result.image_ids):
        raise ValueError("SG27 diagnostic sidecar coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold mismatch")
    return result,tuple(np.stack([sidecars[x][j] for x in result.image_ids]) for j in range(3))


def change_summary(result,records,unique,selected):
    classes=np.asarray(result.classes)
    indices={name:i-1 for i,name in enumerate(unique) if i>0 and name in selected}
    def reduce(values):
        total=values.sum(0)
        output={}
        for name,i in indices.items():
            item={field:int(v) for field,v in zip(CHANGE_FIELDS,total[i])}
            item["net_tp"]=item["tp_gained"]-item["tp_lost"]
            item["net_fp"]=item["fp_gained"]-item["fp_removed"]
            item["fp_removed_per_tp_lost"]=item["fp_removed"]/max(item["tp_lost"],1)
            output[name]=item
        return output
    return {"selection_rule":"final full VOC within 0.005 mIoU of S0 OR frozen COCO gain >=0.015",
            "selected_cells":list(indices),"overall":reduce(records),
            "per_class":{name:reduce(records[classes==name]) for name in sorted(set(classes))}}


def disagreement_summary(result,pair,triple):
    classes=np.asarray(result.classes)
    def describe(counts):
        total=counts.sum(0)
        width=(len(total)-1).bit_length()
        return {format(i,f"0{width}b"):{"gt_positive":int(v[0]),"gt_negative":int(v[1]),
                "tp_probability":float(v[0]/max(v.sum(),1)),"pixels":int(v.sum())}
                for i,v in enumerate(total)}
    def reduce(p,t):
        return {"fine_c16":describe(p[:,0]),"fine_c32":describe(p[:,1]),
                "fine_c16_c32":describe(t)}
    return {"bit_order":{"fine_c16":"F,C16","fine_c32":"F,C32","fine_c16_c32":"F,C16,C32"},
            "threshold":0.5,"overall":reduce(pair,triple),
            "per_class":{name:reduce(pair[classes==name],triple[classes==name])
                         for name in sorted(set(classes))}}


def family_highlights(unique,cells):
    final=[name for name in unique]
    ranks={dataset:{name:rank+1 for rank,name in enumerate(sorted(final,key=lambda x:cells[dataset][x+"+R2"]["miou"],reverse=True))}
           for dataset in ("voc","coco")}
    score=lambda n: cells["voc"][n+"+R2"]["miou"]
    coco_score=lambda n: cells["coco"][n+"+R2"]["miou"]
    families={"fine_c16":[n for n in unique if 3<=int(n[1:])<=15],
              "fine_c32":[n for n in unique if 16<=int(n[1:])<=26],
              "coarse_pair":[n for n in unique if 27<=int(n[1:])<=32],
              "three_way_and_consensus":[n for n in unique if 33<=int(n[1:])<=45]}
    return {"best_full_voc":max(final,key=score),"best_frozen_coco":max(final,key=coco_score),
            "best_average_rank_descriptive":min(final,key=lambda n:(ranks["voc"][n]+ranks["coco"][n],ranks["voc"][n])),
            "ranks":ranks,"families":{key:{"best_voc":max(names,key=score),
                                            "best_coco":max(names,key=coco_score)}
                                      for key,names in families.items()}}


def caption_presence_summary(result,thresholds,cells):
    rows={row["image_id"]:row for shard in range(20) for row in transfer.coco_rows(shard)[1]}
    if set(rows)!=set(result.image_ids):
        raise ValueError("SG27 COCO caption-presence join mismatch")
    flag=np.asarray([rows[x]["word_included"].strip().lower()=="true" for x in result.image_ids])
    classes=np.asarray(result.classes)
    output={}
    for name in cells:
        idx=thresholds[name+"+R2"]
        ref=thresholds["S0+R2"]
        for label,mask in (("present",flag),("absent",~flag)):
            count=result.counts[result.index(name+"+R2"),mask,idx]
            baseline=result.counts[result.index("S0+R2"),mask,ref]
            group=classes[mask]
            base=pooled_miou(baseline,group);new=pooled_miou(count,group)
            output.setdefault(name,{})[label]={"images":int(mask.sum()),"classes":len(set(group)),
                "reference_miou":base,"miou":new,"delta":new-base,
                "paired_mean_image_iou":cv.paired_bootstrap(image_ious(count),image_ious(baseline))}
    return {"field":"released word_included for generation caption; post-hoc only",
            "methods":output}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    spec=json.loads(SPEC.read_text())
    manifest=json.loads(MANIFEST.read_text())
    unique=manifest["unique_cells"]
    voc,(vc,vp,vt)=load(args.output_dir,"voc",spec,manifest,args.smoke)
    coco,(cc,cp,ct)=load(args.output_dir,"coco",spec,manifest,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG27 full-runner smoke expects one VOC and one COCO image")
        if vc.shape[1]!=len(unique)-1 or vp.shape[1:]!=(2,4,2) or vt.shape[1:]!=(8,2):
            raise ValueError("SG27 reduced diagnostic shape mismatch")
        print(f"SG27 live VOC/COCO scoring smoke passed: {len(unique)} unique cells",flush=True)
        return
    names=names_for(unique)
    mask=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),mask)[0] for name in names}
    screens={stage:run_screen(voc,"S0"+suffix,[name+suffix for name in unique if name!="S0"],
                              {name+suffix:1 for name in unique if name!="S0"})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"S0+R2" if name.endswith("+R2") else "S0") for name in names}
    cells={dataset:{name:metrics(result,name,"S0+R2" if name.endswith("+R2") else "S0",thresholds)
                    for name in names} for dataset,result in (("voc",voc),("coco",coco))}
    highlights=family_highlights(unique,cells)
    highlights["formal_final_promotion"]=(screens["final"]["frozen"]
                                            if screens["final"]["frozen"]!="S0+R2" else None)
    reference_voc=cells["voc"]["S0+R2"]["miou"]
    selected=[name for name in unique if name!="S0" and (
              cells["voc"][name+"+R2"]["miou"]>=reference_voc-.005 or
              cells["coco"][name+"+R2"]["delta_vs_reference"]>=.015)]
    changes={dataset:change_summary(result,records,unique,selected)
             for dataset,result,records in (("voc",voc,vc),("coco",coco,cc))}
    disagreements={dataset:disagreement_summary(result,pair,triple)
                   for dataset,result,pair,triple in (("voc",voc,vp,vt),("coco",coco,cp,ct))}
    top_coco=[name for name in sorted(unique,key=lambda n:cells["coco"][n+"+R2"]["miou"],reverse=True)
              if name!="S0"][:3]
    top_set=list(dict.fromkeys(["S0","S1",highlights["best_full_voc"],
                                highlights["families"]["fine_c16"]["best_coco"],
                                highlights["families"]["fine_c32"]["best_coco"],
                                highlights["families"]["three_way_and_consensus"]["best_coco"],
                                *top_coco]))
    captions=caption_presence_summary(coco,thresholds,top_set)
    summary={"spec":spec,"duplicate_manifest":manifest,"selection":screens,"forced_oof":oof,
             "cells":cells,"highlights":highlights,"mechanism_fixed_0p45":changes,
             "disagreement":disagreements,"caption_presence_posthoc":captions,
             "voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_s0"))
        for dataset in ("voc","coco"):
            for name in names:
                item=cells[dataset][name]
                for classname,iou in item["per_class_iou"].items():
                    writer.writerow((dataset,name,classname,iou,item["per_class_delta_vs_reference"][classname]))
    with (out/"mechanism.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","class","cell",*CHANGE_FIELDS,"net_tp","net_fp","fp_removed_per_tp_lost"))
        for dataset in ("voc","coco"):
            for cls,methods in (("ALL",changes[dataset]["overall"]),*changes[dataset]["per_class"].items()):
                for name,item in methods.items():
                    writer.writerow((dataset,cls,name,*(item[f] for f in CHANGE_FIELDS),
                                     item["net_tp"],item["net_fp"],item["fp_removed_per_tp_lost"]))
    with (out/"disagreement.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","class","pattern_family","high_bits","gt_positive","gt_negative","tp_probability","pixels"))
        for dataset in ("voc","coco"):
            for cls,families in (("ALL",disagreements[dataset]["overall"]),*disagreements[dataset]["per_class"].items()):
                for family,patterns in families.items():
                    for bits,item in patterns.items():
                        writer.writerow((dataset,cls,family,bits,item["gt_positive"],item["gt_negative"],
                                         item["tp_probability"],item["pixels"]))
    lines=["# SG27 comprehensive late-fusion results","",
           f"{len(unique)} unique cells after four-image duplicate scan; 490 VOC and 707 COCO images. VOC-only selection, COCO frozen.",""]
    for stage,suffix in (("Raw",""),("After R2","+R2")):
        lines += [f"## {stage}","",
                  "| Fusion | VOC forced OOF | Δ vs FINE | Fold wins | Full VOC | VOC threshold | Frozen COCO |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for cell in unique:
            name=cell+suffix;item=oof[name];v=cells["voc"][name];c=cells["coco"][name]
            lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                         f"{v['miou']:.4f} | {v['voc_frozen_threshold']:.3f} | {c['miou']:.4f} |")
        lines.append("")
    lines += ["## Highlights","",
              f"VOC selector froze: `{screens['final']['frozen']}`; reason: {screens['final']['freeze_reason']}.",
              f"Best full VOC: `{highlights['best_full_voc']}`; best frozen COCO: `{highlights['best_frozen_coco']}`; best descriptive average rank: `{highlights['best_average_rank_descriptive']}`.",
              f"Detailed mechanism cells: {', '.join(selected) if selected else 'none'}.",
              f"Smoke duplicate aliases: {manifest['aliases']}.","",
              "Per-class IoUs: `per_class_iou.csv`; selected-cell changes: `mechanism.csv`; two/three-scale GT disagreement bins: `disagreement.csv`; full folds, intervals, rankings and caption-presence diagnostics: `results.json`.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"unique_cells":len(unique),
        "all_required_stages_complete":True},indent=2)+"\n")
    print("\n".join(lines[-8:]),flush=True)


if __name__=="__main__":
    main()
