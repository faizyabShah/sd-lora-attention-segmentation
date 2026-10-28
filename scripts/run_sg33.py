#!/usr/bin/env python3
"""Resumable SG33 class shard: hierarchy, Markov maps and compact diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer,sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv,sg33_hierarchy,sg31_seediff
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS,_import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg33"
SPEC=ROOT/"configs/sg33.json"
RAW=sg33_hierarchy.NAMES
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
SIDE_FIELDS=("change","stage_counts","stage_correlations","stage_change",
             "rescue","markov","markov_global","ipf","temperature",
             "amplitude_disagreement")
MARKOV_FIELDS=("mean","median","p10","p25","p75","p90",
               "saturated_by_1","saturated_by_2","saturated_by_5",
               "saturated_by_10","saturated_by_25","saturated_by_50",
               "saturated_by_cap","unsaturated_at_cap")
REGIONS=("gt_fg","bg_0_16","bg_16_64","bg_gt64")


def _full(map64,shape):
    return F.interpolate(map64.reshape(1,1,64,64).float(),size=shape,
                         mode="bilinear",align_corners=False)[0,0].numpy()


def auc(scores,truth):
    scores=np.asarray(scores).reshape(-1)
    truth=np.asarray(truth,bool).reshape(-1)
    positives=int(truth.sum());negatives=len(truth)-positives
    if not positives or not negatives:return float("nan")
    ranks=rankdata(scores,method="average")
    return float((ranks[truth].sum()-positives*(positives+1)/2)/(positives*negatives))


def diagnostic(maps,transient,truth,threshold):
    shape=truth.shape
    stages=cv.normalize_batch(transient["stages"].float(),"minmax").cpu()
    stage_counts,_=cv.score_maps(stages,truth,"minmax",(threshold,))
    flat=stages.reshape(3,-1).numpy()
    correlations=np.corrcoef(flat)
    stage_maps={"S0":stages[0],"S1":stages[1],"S2":stages[2]}
    stage_change=np.stack((change_diagnostics(stage_maps,["S0","S1"],truth,threshold)[0],
        change_diagnostics({"S0":stages[1],"S1":stages[2]},["S0","S1"],truth,threshold)[0]))
    distance=ndimage.distance_transform_edt(~truth)
    region=np.where(truth,0,np.where(distance<=16,1,np.where(distance<=64,2,3)))
    resized=[_full(x,shape) for x in stages]
    rescue=np.zeros((2,4),np.int64)
    for i in range(2):
        selected=resized[i]>resized[2]
        for k in range(4):rescue[i,k]=int((selected&(region==k)).sum())
    # Markov maps are 16², so use the corresponding sampled GT and full-image
    # distance at those cell centers. No GT enters Markov inference.
    gt16=cv.target_at_size(truth,16).numpy().astype(bool)
    dist16=F.interpolate(torch.from_numpy(distance).float()[None,None],
                         size=(16,16),mode="nearest")[0,0].numpy()
    region16=np.where(gt16,0,np.where(dist16<=16,1,np.where(dist16<=64,2,3)))
    markov=np.full((3,4,len(MARKOV_FIELDS)),np.nan,np.float64)
    global_stats=np.full((3,4),np.nan,np.float64)  # AUC, bg-fg gap, unsat, cap512
    disagreement=np.zeros((3,2,4),np.int64)
    for mi,name in enumerate(sg33_hierarchy.MARKOV):
        arrival=np.stack([transient["markov"][call][mi][0].numpy() for call in range(11)])
        unreached=np.stack([transient["markov"][call][mi][1].numpy() for call in range(11)])
        truth_stack=np.broadcast_to(gt16,arrival.shape)
        global_stats[mi,0]=auc(-arrival,truth_stack)
        global_stats[mi,1]=float(arrival[:,~gt16].mean()-arrival[:,gt16].mean()) if gt16.any() and (~gt16).any() else np.nan
        global_stats[mi,2]=float(unreached.mean())
        cap512=[transient["markov"][call][mi][2] for call in range(11)]
        valid=[v for v in cap512 if np.isfinite(v)]
        if valid:global_stats[mi,3]=float(np.mean(valid))
        for k in range(4):
            mask=np.broadcast_to(region16==k,arrival.shape)
            values=arrival[mask].astype(float)
            unseen=unreached[mask]
            if len(values):
                markov[mi,k]=[float(values.mean()),float(np.median(values)),
                    *np.percentile(values,[10,25,75,90]).tolist(),
                    *(float((values<=t).mean()) for t in (1,2,5,10,25,50,256)),
                    float(unseen.mean())]
        for call in range(11):
            amplitude=sg33_hierarchy.norm(transient["coarse"][call][0]).numpy()
            support=transient["coarse"][call][mi+1].numpy()
            for direction,selected in enumerate((support-amplitude>.2,amplitude-support>.2)):
                for k in range(4):disagreement[mi,direction,k]+=int((selected&(region16==k)).sum())
    return {"stage_counts":stage_counts[:,0],"stage_correlations":correlations,
            "stage_change":stage_change,"rescue":rescue,"markov":markov,
            "markov_global":global_stats,"ipf":transient["ipf"].numpy().mean(0),
            "temperature":transient["temperature"].numpy().mean(0),
            "amplitude_disagreement":disagreement}


def read_side(path):
    if not path.exists():return {}
    with np.load(path,allow_pickle=False) as data:
        ids=[str(x) for x in data["image_ids"]]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate SG33 diagnostic image")
        return {image_id:tuple(data[field][i] for field in SIDE_FIELDS)
                for i,image_id in enumerate(ids)}


def save_side(path,rows):
    atomic_npz(path,image_ids=np.asarray(list(rows)),
               **{field:np.stack([row[i] for row in rows.values()])
                  for i,field in enumerate(SIDE_FIELDS)})


def visual_sheet(path,image,truth,visual,classname,image_id):
    from matplotlib import colormaps
    palette=colormaps["magma"]
    source=Image.fromarray(np.asarray(image,np.uint8)).resize((256,256))
    panels=[("image",source)]
    for title,key in (("A3 CA","seed"),("A16 c","ordinary16"),
                      ("M1 reach","M1_16"),("M2 IPF","M2_16"),
                      ("M3 T+IPF","M3_16"),("M3 final","M3_final64")):
        tensor=visual[key].float()
        normalized=sg33_hierarchy.norm(tensor).numpy()
        rgb=(palette(normalized)[...,:3]*255).astype(np.uint8)
        panels.append((title,Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR)))
    panels.append(("GT",Image.fromarray((truth.astype(np.uint8)*255)).convert("RGB").resize((256,256))))
    sheet=Image.new("RGB",(256*len(panels),290),"white")
    draw=ImageDraw.Draw(sheet)
    for i,(title,panel) in enumerate(panels):
        sheet.paste(panel,(256*i,34));draw.text((256*i+5,7),title,fill="black")
    path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path,optimize=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("voc","coco"),required=True)
    p.add_argument("--class-shard",type=int,choices=range(20),required=True)
    p.add_argument("--max-samples",type=int)
    p.add_argument("--verify-controls",action="store_true")
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--model",default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site",type=Path,default=ROOT/"outputs/dependencies/simplecrf_py310")
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    if tuple(spec["cells"])!=RAW or spec["thresholds"]!=list(transfer.THRESHOLDS):
        raise ValueError("SG33 locked cells/thresholds changed")
    runtime=sg27_runtime.make_runtime(args.model)
    _import_densecrf(args.crf_site)
    classname,all_rows=(transfer.voc_rows if args.dataset=="voc" else transfer.coco_rows)(args.class_shard)
    rows=all_rows[:args.max_samples] if args.max_samples else all_rows
    folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset=="voc" else {
        r["image_id"]:0 for r in all_rows}
    stem=f'class_{args.class_shard:02d}_{classname.replace(" ","_")}'
    path=args.output_dir/args.dataset/"shards"/(stem+".npz")
    side_path=path.with_suffix(".diagnostics.npz")
    meta_path=args.output_dir/args.dataset/"visuals"/(stem+".json")
    settings={"spec":spec,"dataset":args.dataset,"model":args.model,
              "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
              "crf":list(OVAM_CRF_PARAMETERS)}
    writer=cv.ShardWriter(path,NAMES,folds,settings,spec["thresholds"])
    side=read_side(side_path)
    if writer.completed-set(side):raise ValueError("SG33 metric shard ahead of diagnostics")
    visual_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {"worst_distant_fp":-1}
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw,transient=sg33_hierarchy.aggregate(sa,seed,tau=spec["markov_tau"],
                temperature_value=spec["temperature"],cap=spec["markov_cap"])
            maps={name:cv.normalize_batch(raw[name][None],"minmax")[0] for name in RAW}
            if args.verify_controls:
                assert_control_close(maps["S0"],previous["S0"],"SG33 S0/FINE")
                if args.max_samples:
                    sg31_maps,_,_=sg31_seediff.aggregate(sa,seed)
                    assert_control_close(maps["H0"],sg31_maps["S2"],"SG33 H0/SG31 continuous hierarchy")
            maps["S0"]=previous["S0"]
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(RAW):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            diag=diagnostic(maps,transient,truth,spec["diagnostic_threshold"])
            change=change_diagnostics(maps,list(RAW),truth,spec["diagnostic_threshold"])
            side[image_id]=(change,*(diag[field] for field in SIDE_FIELDS[1:]))
            save_side(side_path,side)
            if args.dataset=="voc":
                pred=_full(maps["S0"].detach().cpu(),truth.shape)>=spec["diagnostic_threshold"]
                far=ndimage.distance_transform_edt(~truth)>64
                distant_fp=int((pred&~truth&far).sum())
                first_path=meta_path.with_name(stem+"_first.png")
                if not first_path.exists():visual_sheet(first_path,image,truth,transient["visual"],classname,image_id)
                if distant_fp>visual_meta["worst_distant_fp"]:
                    visual_sheet(meta_path.with_name(stem+"_leakage.png"),image,truth,
                                 transient["visual"],classname,image_id)
                    visual_meta={"worst_distant_fp":distant_fp,"worst_image_id":image_id}
                    meta_path.parent.mkdir(parents=True,exist_ok=True)
                    meta_path.write_text(json.dumps(visual_meta,indent=2)+"\n")
            writer.add(image_id,classname,counts,ap)
            print(f"[sg33:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG33 class shard")
        writer.finish()


if __name__=="__main__":main()
