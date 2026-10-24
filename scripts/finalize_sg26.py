#!/usr/bin/env python3
"""SG26 VOC-only selection, frozen COCO transfer and gate mechanism summaries."""
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
from scripts.run_sg26 import BASE,MECHANISM_FIELDS,NAMES,SPEC
from scripts.select_propagation_cv import best_threshold,run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation import sg26_sa


def load(out:Path,dataset:str,spec:dict,smoke:bool):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG26 cell or threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG26 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"{dataset} incomplete SG26 shards")
    sidecars={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for image_id,mechanism,edge,groups in zip(data["image_ids"],data["mechanism"],
                                                       data["edge"],data["edge_groups"]):
                if str(image_id) in sidecars:
                    raise ValueError("duplicate SG26 diagnostic image")
                sidecars[str(image_id)]=(mechanism,edge,groups)
    if set(sidecars)!=set(result.image_ids):
        raise ValueError("SG26 sidecar coverage mismatch")
    if dataset=="voc":
        folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv")
        if any(result.folds[i]!=folds[image_id] for i,image_id in enumerate(result.image_ids)):
            raise ValueError("VOC fold mismatch")
    return result,tuple(np.stack([sidecars[x][j] for x in result.image_ids]) for j in range(3))


def mechanism_summary(result,records):
    classes=np.asarray(result.classes)
    def reduce(values):
        total=values.sum(0)
        output={}
        for i,name in enumerate(sg26_sa.NAMES[3:]):
            item={field:int(v) for field,v in zip(MECHANISM_FIELDS,total[i])}
            item["fp_removed_per_tp_lost"]=item["fp_removed"]/max(item["tp_lost"],1)
            item["distant_fp16_removed_per_tp_lost"]=item["fp_gt16_removed"]/max(item["tp_lost"],1)
            output[name]=item
        return output
    return {"overall":reduce(records),
            "per_class":{name:reduce(records[classes==name]) for name in sorted(set(classes))}}


def edge_summary(result,edge,groups):
    classes=np.asarray(result.classes)
    def reduce(e,g):
        output={}
        for i,side in enumerate((16,32)):
            scalars={name:float(v) for name,v in zip(sg26_sa.EDGE_FIELDS,e[:,i].mean(0))}
            categories={}
            for j,name in enumerate(sg26_sa.EDGE_GROUPS):
                with np.errstate(invalid="ignore"):
                    values=np.nanmean(g[:,i,j],axis=0)
                if np.isnan(values).all():
                    categories[name]=None
                else:
                    row={field:float(v) for field,v in zip(sg26_sa.EDGE_GROUP_FIELDS,values)}
                    row["after_before_ratio"]=row["fine_probability_after"]/max(row["fine_probability_before"],1e-30)
                    categories[name]=row
            output[str(side)]={"scalar":scalars,"gt_edge_groups":categories}
        return output
    return {"overall":reduce(edge,groups),
            "per_class":{name:reduce(edge[classes==name],groups[classes==name])
                         for name in sorted(set(classes))}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    spec=json.loads(SPEC.read_text())
    voc,(vm,ve,vg)=load(args.output_dir,"voc",spec,args.smoke)
    coco,(cm,ce,cg)=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG26 smoke requires one VOC and one COCO sample")
        if vm.shape!=(1,4,len(MECHANISM_FIELDS)) or ve.shape!=(1,2,len(sg26_sa.EDGE_FIELDS)):
            raise ValueError("SG26 smoke diagnostic shape mismatch")
        print("SG26 live VOC/COCO smoke passed",flush=True)
        return
    mask=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),mask)[0] for name in NAMES}
    raw=[name for name in NAMES if not name.endswith("+R2")]
    final=[name for name in NAMES if name.endswith("+R2")]
    screens={"raw":run_screen(voc,"S0",raw[1:],{name:1 for name in raw[1:]}),
             "final":run_screen(voc,"S0+R2",final[1:],{name:1 for name in final[1:]})}
    oof={name:forced_oof(voc,name,"S0+R2" if name.endswith("+R2") else "S0") for name in NAMES}
    cells={dataset:{name:metrics(result,name,"S0+R2" if name.endswith("+R2") else "S0",thresholds)
                    for name in NAMES} for dataset,result in (("voc",voc),("coco",coco))}
    mechanisms={"voc":mechanism_summary(voc,vm),"coco":mechanism_summary(coco,cm)}
    edges={"voc":edge_summary(voc,ve,vg),"coco":edge_summary(coco,ce,cg)}
    summary={"spec":spec,"selection":screens,"forced_oof":oof,"cells":cells,
             "mechanism_fixed_0p45":mechanisms,"edge_diagnostics":edges,
             "voc_thresholds_frozen_for_coco":True}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out/"per_class_iou.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","cell","class","iou","delta_vs_same_stage_s0"))
        for dataset in ("voc","coco"):
            for name in NAMES:
                item=cells[dataset][name]
                for classname,iou in item["per_class_iou"].items():
                    writer.writerow((dataset,name,classname,iou,item["per_class_delta_vs_reference"][classname]))
    with (out/"mechanism.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("dataset","class","cell",*MECHANISM_FIELDS,"fp_removed_per_tp_lost","distant_fp16_removed_per_tp_lost"))
        for dataset in ("voc","coco"):
            for cls,methods in (("ALL",mechanisms[dataset]["overall"]),*mechanisms[dataset]["per_class"].items()):
                for name,item in methods.items():
                    writer.writerow((dataset,cls,name,*(item[f] for f in MECHANISM_FIELDS),
                                     item["fp_removed_per_tp_lost"],item["distant_fp16_removed_per_tp_lost"]))
    lines=["# SG26 results","","A3 CA frozen; same-call coarse gates; VOC five-fold selection; frozen COCO.",""]
    for stage,names in (("Raw",raw),("After R2",final)):
        lines += [f"## {stage}","",
                  "| Cell | VOC forced OOF | Δ OOF | Held-out wins | Full VOC | VOC threshold | Frozen COCO |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name in names:
            item=oof[name];v=cells["voc"][name];c=cells["coco"][name]
            lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                         f"{v['miou']:.4f} | {v['voc_frozen_threshold']:.3f} | {c['miou']:.4f} |")
        lines.append("")
    lines += ["## Mechanism at fixed raw threshold .45","",
              "| Dataset | Cell | TP gained | TP lost | FP gained | FP removed | FP removed / TP lost | FP >16 gained/removed | FP >64 gained/removed | Separate FP gained/removed |",
              "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for dataset in ("voc","coco"):
        for name,item in mechanisms[dataset]["overall"].items():
            lines.append(f"| {dataset} | {name} | {item['tp_gained']} | {item['tp_lost']} | "
                         f"{item['fp_gained']} | {item['fp_removed']} | {item['fp_removed_per_tp_lost']:.3f} | "
                         f"{item['fp_gt16_gained']}/{item['fp_gt16_removed']} | "
                         f"{item['fp_gt64_gained']}/{item['fp_gt64_removed']} | "
                         f"{item['separate_fp_gained']}/{item['separate_fp_removed']} |")
    lines += ["","## Edge diagnostics","",
              "Per-image means on a deterministic balanced 64-foreground/64-background query/key sample; ratios compare each gated fine edge to its original value.","",
              "| Dataset | Coarse gate | Strongly suppressed (<0.5×) | Nearly unchanged (0.9–1.1×) | Row entropy before→after | Effective neighbors before→after |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("voc","coco"):
        for side,item in edges[dataset]["overall"].items():
            s=item["scalar"]
            lines.append(f"| {dataset} | {side} | {s['fraction_ratio_below_0p5']:.3f} | "
                         f"{s['fraction_ratio_0p9_to_1p1']:.3f} | "
                         f"{s['row_entropy_before']:.3f}→{s['row_entropy_after']:.3f} | "
                         f"{s['effective_neighbors_before']:.1f}→{s['effective_neighbors_after']:.1f} |")
    lines += ["","All per-class IoUs: `per_class_iou.csv`. Per-class mask-change counts: `mechanism.csv`. Full GT-edge-type, per-class edge and bootstrap diagnostics: `results.json`.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"all_required_stages_complete":True},indent=2)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
