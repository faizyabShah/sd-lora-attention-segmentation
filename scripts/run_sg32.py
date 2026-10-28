#!/usr/bin/env python3
"""Resumable SG32 per-class shard with live maps and compact mechanisms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer, sg27_runtime
from scripts.control_checks import assert_control_close, CONTROL_ATOL, CONTROL_RTOL
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics, _separate_fp
from src.segmentation import controlled_cv as cv, sg32_adaptive
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg32"
SPEC=ROOT/"configs/sg32.json"
RAW=sg32_adaptive.NAMES
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
SIDE_FIELDS=("change","s10_vs_s1","bins","regions","exponents","entropy","temporal","cross","bg_log")
BIN_FIELDS=("pixels","gt_fg","baseline_fp","bg_pixels","bg_distance_sum",
            "separate_fp","first_score_sum","p2_score_sum")
EXP_FIELDS=("mean","median","p10","p25","p75","p90",
            "fraction_ordinary","fraction_moderate","fraction_strong")
REGIONS=("gt_fg","bg_0_16","bg_16_64","bg_gt64")
TEMPERATURE_SIGNALS=("c16","pair","bg","pair_bg","range_bg")


def up(map64,shape):
    return F.interpolate(map64.reshape(1,1,64,64).float(),size=shape,
                         mode="bilinear",align_corners=False)[0,0].numpy()


def posthoc(maps,transient,truth,threshold):
    """Full-resolution reliability analysis, using GT only after inference."""
    truth=np.asarray(truth,bool)
    shape=truth.shape
    distance=ndimage.distance_transform_edt(~truth)
    region=np.where(truth,0,np.where(distance<=16,1,np.where(distance<=64,2,3)))
    p2=up(maps["S0"].detach().cpu(),shape)
    baseline=p2>=threshold
    separate=_separate_fp(baseline,truth)
    fp=baseline & ~truth
    bins=np.zeros((6,5,len(BIN_FIELDS)),np.float64)
    regions=np.zeros((6,4,2),np.float64)  # reliability sum, pixel count
    exponent_values={(signal,k):[] for signal in TEMPERATURE_SIGNALS for k in range(4)}
    cross=np.zeros((4,2),np.int64)  # pair low/bg low codes; GT/BG
    combined=[]
    rel=transient["reliability"]
    for call in range(11):
        first=up(transient["first"][call],shape)
        maps_full=[up(rel[call,s],shape).clip(0,1) for s in range(6)]
        pair_low=maps_full[1]<.5
        bg_low=maps_full[3]<.5
        code=2*pair_low.astype(np.int8)+bg_low.astype(np.int8)
        for k in range(4):
            selected=code==k
            cross[k]=(cross[k,0]+int((selected & truth).sum()),
                      cross[k,1]+int((selected & ~truth).sum()))
        combined.append(maps_full[4])
        for si,r in enumerate(maps_full):
            bin_id=np.minimum((r*5).astype(np.int8),4)
            for j in range(5):
                selected=bin_id==j
                bg=selected & ~truth
                bins[si,j]=(int(selected.sum()),int((selected & truth).sum()),
                    int((selected & fp).sum()),int(bg.sum()),float(distance[bg].sum()),
                    int((selected & separate).sum()),float(first[selected].sum()),
                    float(p2[selected].sum()))+bins[si,j]
            for k in range(4):
                values=r[region==k]
                regions[si,k,0]+=float(values.sum());regions[si,k,1]+=len(values)
            signal=sg32_adaptive.SIGNALS[si]
            if signal in TEMPERATURE_SIGNALS:
                exponent=1+1.6*(1-r)
                for k in range(4):
                    exponent_values[signal,k].append(exponent[region==k].astype(np.float32))
    exp=np.full((5,4,len(EXP_FIELDS)),np.nan,np.float64)
    for si,signal in enumerate(TEMPERATURE_SIGNALS):
        for k in range(4):
            values=np.concatenate(exponent_values[signal,k])
            if len(values):
                exp[si,k]=[float(values.mean()),float(np.median(values)),
                    *np.percentile(values,[10,25,75,90]).tolist(),
                    float((values<1.2).mean()),float(((values>=1.2)&(values<2)).mean()),
                    float((values>=2).mean())]
    stack=np.stack(combined)
    temporal=np.full((4,4),np.nan,np.float64)
    average=stack.mean(0);variance=stack.var(0)
    for k in range(4):
        selected=region==k
        if selected.any():
            temporal[k]=(float(average[selected].mean()),float(variance[selected].mean()),
                         float((average[selected] < .5).mean()),int(selected.sum()))
    return {"bins":bins,"regions":regions,"exponents":exp,
            "entropy":transient["entropy"].numpy().mean(0),"temporal":temporal,
            "cross":cross,"bg_log":transient["background_log"].numpy()}


def read_side(path):
    if not path.exists():return {}
    with np.load(path,allow_pickle=False) as data:
        ids=[str(x) for x in data["image_ids"]]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate SG32 diagnostic image")
        return {image_id:tuple(data[field][i] for field in SIDE_FIELDS)
                for i,image_id in enumerate(ids)}


def save_side(path,rows):
    atomic_npz(path,image_ids=np.asarray(list(rows)),
               **{field:np.stack([row[i] for row in rows.values()])
                  for i,field in enumerate(SIDE_FIELDS)})


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
        raise ValueError("SG32 locked cells/thresholds changed")
    if spec["control_atol"]!=CONTROL_ATOL or spec["control_rtol"]!=CONTROL_RTOL:
        raise ValueError("SG32 control tolerances changed")
    runtime=sg27_runtime.make_runtime(args.model)
    _import_densecrf(args.crf_site)
    classname,all_rows=(transfer.voc_rows if args.dataset=="voc" else transfer.coco_rows)(args.class_shard)
    rows=all_rows[:args.max_samples] if args.max_samples else all_rows
    folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset=="voc" else {
        r["image_id"]:0 for r in all_rows}
    stem=f'class_{args.class_shard:02d}_{classname.replace(" ","_")}'
    path=args.output_dir/args.dataset/"shards"/(stem+".npz")
    side_path=path.with_suffix(".diagnostics.npz")
    settings={"spec":spec,"dataset":args.dataset,"model":args.model,
              "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
              "crf":list(OVAM_CRF_PARAMETERS)}
    writer=cv.ShardWriter(path,NAMES,folds,settings,spec["thresholds"])
    side=read_side(side_path)
    if writer.completed-set(side):raise ValueError("SG32 metric shard ahead of diagnostics")
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw,transient=sg32_adaptive.aggregate(sa,seed,r_max=spec["r_max"])
            maps={name:cv.normalize_batch(raw[name][None],"minmax")[0] for name in RAW}
            if args.verify_controls:
                assert_control_close(maps["S0"],previous["S0"],"SG32 S0/FINE")
            maps["S0"]=previous["S0"]
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(RAW):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            diag=posthoc(maps,transient,truth,spec["diagnostic_threshold"])
            changes=change_diagnostics(maps,list(RAW),truth,spec["diagnostic_threshold"])
            s10_vs_s1=change_diagnostics({"S0":maps["S1"],"S1":maps["S10"]},
                                         ["S0","S1"],truth,spec["diagnostic_threshold"])[0]
            side[image_id]=(changes,s10_vs_s1,*(diag[field] for field in SIDE_FIELDS[2:]))
            save_side(side_path,side)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg32:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG32 class shard")
        writer.finish()


if __name__=="__main__":main()
