#!/usr/bin/env python3
"""Merge SG-combined-v2: VOC selection, frozen COCO, cap/AP diagnostics, route AUC."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.finalize_sg22_sg23 import forced_oof,metrics
from scripts.finalize_sg_combined_c10 import changes,finite_mean
from scripts.run_sg_combined_v2 import BASE,CONTRASTS,NAMES,RAW,SIDE_FIELDS,SPEC
from scripts.select_propagation_cv import best_threshold,image_ious,run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation.sg_combined_v2 import ROUTE_PAIRS

ROUTE_SCORES=("R_background_weighted_fine_exclusive","E_fine_exclusive_mean","D_mean_abs_disagreement")
GO_AUC=.65


def load(out,dataset,spec,smoke):
    files=sorted(p for p in (out/dataset/"shards").glob("class_*.npz")
                 if not p.name.endswith((".diagnostics.npz",".tmp.npz")))
    result=cv.load_results(files)
    if result.configurations!=NAMES or result.thresholds!=tuple(spec["thresholds"]):
        raise ValueError("SG-combined-v2 cell/threshold mismatch")
    if result.settings["spec"]!=spec or result.settings["dataset"]!=dataset:
        raise ValueError("foreign SG-combined-v2 shard")
    if not smoke and (len(files)!=20 or len(result.image_ids)!=(490 if dataset=="voc" else 707)
                      or not all(result.complete)):
        raise ValueError(f"incomplete {dataset} SG-combined-v2 coverage")
    side={}
    for file in files:
        with np.load(file.with_suffix(".diagnostics.npz"),allow_pickle=False) as data:
            for i,image_id in enumerate(data["image_ids"]):
                key=str(image_id)
                if key in side:raise ValueError("duplicate SG-combined-v2 diagnostic image")
                side[key]=tuple(data[field][i] for field in SIDE_FIELDS)
    if set(side)!=set(result.image_ids):raise ValueError("SG-combined-v2 diagnostic coverage mismatch")
    return result,{field:np.stack([side[x][i] for x in result.image_ids])
                   for i,field in enumerate(SIDE_FIELDS)}


def auc(score,label):
    pos=int(label.sum());neg=len(label)-pos
    if not pos or not neg:return None
    ranks=rankdata(score,method="average")
    return float((ranks[label].sum()-pos*(pos+1)/2)/(pos*neg))


def route_target(result,name,reference,thresholds=None):
    """1[IoU(H)>IoU(F)] per image; VOC uses each held-out fold's training thresholds."""
    h=result.index(name);f=result.index(reference)
    iou_h=np.zeros(len(result.image_ids));iou_f=np.zeros_like(iou_h)
    if thresholds is None:
        for fold in range(5):
            held=result.folds==fold
            th,_=best_threshold(result,h,~held);tf,_=best_threshold(result,f,~held)
            iou_h[held]=image_ious(result.counts[h,held,th]);iou_f[held]=image_ious(result.counts[f,held,tf])
    else:
        iou_h=image_ious(result.counts[h,:,thresholds[name]])
        iou_f=image_ious(result.counts[f,:,thresholds[reference]])
    return iou_h-iou_f


def route(voc,coco,vd,cd,thresholds):
    out={}
    for p,(fine,hier) in enumerate(ROUTE_PAIRS):
        for stage,suffix in (("R2","+R2"),("raw","")):
            dv=route_target(voc,hier+suffix,fine+suffix)
            dc=route_target(coco,hier+suffix,fine+suffix,thresholds)
            item={"primary":p==0 and stage=="R2","fine":fine+suffix,"hierarchy":hier+suffix}
            for d,diff,side in (("voc_oof",dv,vd),("coco_frozen",dc,cd)):
                keep=diff!=0;label=diff[keep]>0
                item[d]={"images":int(len(diff)),"ties_excluded":int((~keep).sum()),
                         "hierarchy_better_fraction":float(label.mean()) if keep.any() else None,
                         "auc":{s:auc(side["route"][keep,p,k],label) for k,s in enumerate(ROUTE_SCORES)},
                         "oracle_mean_image_iou_gain_over_fine":float(np.maximum(diff,0).mean())}
            # VOC-defined R deciles applied unchanged to COCO (calibration view).
            edges=np.quantile(vd["route"][:,p,0],np.linspace(0,1,11))
            edges[0]=-np.inf;edges[-1]=np.inf
            for d,diff,side in (("voc_oof",dv,vd),("coco_frozen",dc,cd)):
                bins=np.digitize(side["route"][:,p,0],edges[1:-1])
                item[d]["R_deciles_voc_edges"]=[
                    {"images":int((bins==b).sum()),
                     "hierarchy_better_fraction":float((diff[bins==b]>0).mean()) if (bins==b).any() else None,
                     "mean_image_iou_difference":float(diff[bins==b].mean()) if (bins==b).any() else None}
                    for b in range(10)]
            primary=item["voc_oof"]["auc"][ROUTE_SCORES[0]];transfer=item["coco_frozen"]["auc"][ROUTE_SCORES[0]]
            item["go_no_go_R"]=("go" if primary is not None and transfer is not None
                                and primary>=GO_AUC and transfer>=GO_AUC else "no-go")
            out[f"{hier}_vs_{fine}_{stage}"]=item
    return out


def ranking(result,thresholds):
    """Diagnostic only: raw-map AP and best-threshold mIoU versus same-stage V0."""
    out={};keep=np.ones(len(result.image_ids),bool)
    best={n:best_threshold(result,result.index(n),keep) for n in NAMES}
    for name in NAMES:
        ref="V0+R2" if name.endswith("+R2") else "V0"
        item={"best_threshold":result.thresholds[best[name][0]],"best_threshold_miou":best[name][1],
              "best_threshold_delta":best[name][1]-best[ref][1]}
        if not name.endswith("+R2"):
            a=result.ap[result.index(name)];b=result.ap[result.index(ref)]
            item|={"raw_ap":float(a.mean()),"raw_ap_vs_V0":cv.paired_bootstrap(a,b)}
        out[name]=item
    return out


def mechanism(side):
    cap=side["cap"]
    return {"cap":{transition:{field:finite_mean(cap[:,i,j]) for j,field in enumerate(
                ("binding_group_fraction","removed_mass_fraction","mean_factor_binding_groups"))}
            for i,transition in enumerate(("16_to_32_V3_V4","32_to_64_V4"))},
            "changes_at_full_resolution_0p45":{f"{b}_vs_{a}":changes(side["change"][:,i])
                                                for i,(a,b) in enumerate(CONTRASTS)},
            "background_fallback_calls":int(side["background_log"][:,0].sum()),
            "background_seed_fraction_mean":finite_mean(side["background_log"][:,1]),
            "diagnostic_note":"GT enters diagnostics only; fixed 0.45 comparisons do not use cell-selected thresholds"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    voc,vd=load(args.output_dir,"voc",spec,args.smoke)
    coco,cd=load(args.output_dir,"coco",spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1:
            raise ValueError("SG-combined-v2 smoke needs one image per dataset")
        for side in (vd,cd):
            if side["cap"].shape[1:]!=(2,3) or side["route"].shape[1:]!=(len(ROUTE_PAIRS),3) \
               or side["change"].shape[1:]!=(len(CONTRASTS),11):
                raise ValueError("SG-combined-v2 diagnostic sidecar shape mismatch")
            if not np.isfinite(side["route"]).all() or not np.isfinite(side["cap"]).all():
                raise ValueError("non-finite SG-combined-v2 diagnostic")
        print("SG-combined-v2 smoke passed: seven raw/R2 cells, cap and route diagnostics",flush=True)
        return
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={name:best_threshold(voc,voc.index(name),keep)[0] for name in NAMES}
    screens={stage:run_screen(voc,"V0"+suffix,[n+suffix for n in RAW[1:]],
                              {n+suffix:1 for n in RAW[1:]})
             for stage,suffix in (("raw",""),("final","+R2"))}
    oof={name:forced_oof(voc,name,"V0+R2" if name.endswith("+R2") else "V0") for name in NAMES}
    contrasts={f"{b}{s}_vs_{a}{s}":forced_oof(voc,b+s,a+s)
               for a,b in CONTRASTS for s in ("","+R2")}
    cells={dataset:{name:metrics(result,name,"V0+R2" if name.endswith("+R2") else "V0",thresholds)
                    for name in NAMES}
           for dataset,result in (("voc",voc),("coco",coco))}
    diagnostics={"ranking_diagnostic_only":{"voc":ranking(voc,thresholds),"coco":ranking(coco,thresholds)},
                 "mechanism":{"voc":mechanism(vd),"coco":mechanism(cd)},
                 "route_non_selectable":route(voc,coco,vd,cd,thresholds)}
    output={"spec":spec,"selection":screens,"forced_oof":oof,"contrasts":contrasts,"cells":cells,
            **diagnostics,"voc_thresholds_frozen_for_coco":True}
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
    lines=["# SG-combined-v2","",
           "Seven cells raw and with unchanged R2, 40 class shards, 490 VOC and 707 COCO images.",
           "VOC-only selection; COCO thresholds frozen. AP and best-threshold columns are diagnostic only.","",
           "| Cell | VOC forced OOF | Δ vs V0 | Fold wins | Full VOC (thr) | Frozen COCO | COCO best-thr Δ | COCO raw AP Δ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item=oof[name];ap=rank["coco"][name].get("raw_ap_vs_V0")
        lines.append(f"| {name} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][name]['miou']:.4f} ({cells['voc'][name]['voc_frozen_threshold']:.3f}) | "
                     f"{cells['coco'][name]['miou']:.4f} | {rank['coco'][name]['best_threshold_delta']:+.4f} | "
                     f"{'—' if ap is None else format(ap['mean'],'+.4f')} |")
    primary=diagnostics["route_non_selectable"]["V2_vs_V0_R2"]
    lines+=["",f"Raw selector: {screens['raw']['frozen']} {screens['raw']['selection_counts']}",
            f"R2 selector: {screens['final']['frozen']} {screens['final']['selection_counts']}","",
            f"Primary route diagnostic (V2+R2 vs V0+R2, non-selectable): R AUC VOC OOF "
            f"{primary['voc_oof']['auc'][ROUTE_SCORES[0]]}, frozen COCO {primary['coco_frozen']['auc'][ROUTE_SCORES[0]]}; "
            f"{primary['go_no_go_R']} at {GO_AUC}.","",
            "Contrasts, cap binding, change accounting, route deciles and per-class values are in results.json.",""]
    (out/"report.md").write_text("\n".join(lines))
    (out/"completed.json").write_text(json.dumps({"voc_images":len(voc.image_ids),
        "coco_images":len(coco.image_ids),"all_required_stages_complete":True,
        "raw_frozen":screens["raw"]["frozen"],"final_frozen":screens["final"]["frozen"],
        "route_primary":primary["go_no_go_R"]},indent=2)+"\n")
    print("\n".join(lines[-7:]),flush=True)


if __name__=="__main__":main()
