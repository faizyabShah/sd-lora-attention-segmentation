#!/usr/bin/env python3
"""Resumable SG-A3G class shard: historical vs GoCA-oriented A3 seed, FINE-P2 and V3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer,sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from src.segmentation import controlled_cv as cv,sg_combined_v2
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS,_import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_a3g"
SPEC=ROOT/"configs/sg_a3g.json"
RAW=("V0","V3","G0","G3")
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
SIDE_FIELDS=("seed",)  # mean |A3G-A3|, Pearson r, call-mean |weight difference|


def seed_diagnostic(recorder,seed,seed_g):
    a,b=seed.float().reshape(-1),seed_g.float().reshape(-1)
    r=torch.corrcoef(torch.stack((a,b)))[0,1]
    old=[w for _,label,w in recorder.weights if label=="selected"]
    new=[w for _,label,w in recorder.weights if label=="selected-goca"]
    if len(old)!=31 or len(new)!=31:raise ValueError("SG-A3G expects 31 weighted calls per orientation")
    return np.asarray((float((a-b).abs().mean()),float(r),
                       float(np.mean([np.abs(x-y).mean() for x,y in zip(old,new)]))),np.float64)


def read_side(path):
    if not path.exists():return {}
    with np.load(path,allow_pickle=False) as data:
        ids=[str(x) for x in data["image_ids"]]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate diagnostic image")
        return {image_id:tuple(data[field][i] for field in SIDE_FIELDS) for i,image_id in enumerate(ids)}


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
        raise ValueError("SG-A3G locked cells/thresholds changed")
    runtime=sg27_runtime.make_runtime(args.model)
    runtime["recorder_class"].dual_orientation=True
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
    if writer.completed-set(side):raise ValueError("metric shard ahead of diagnostics")
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            recorder=runtime["recorders"][-1]
            seed_g=recorder.seeds()["selected/A3G"]
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw,_=sg_combined_v2.aggregate(sa,seed)
            raw_g,_=sg_combined_v2.aggregate(sa,seed_g)
            maps={"V0":raw["V0"],"V3":raw["V3"],"G0":raw_g["V0"],"G3":raw_g["V3"]}
            maps={name:cv.normalize_batch(value[None],"minmax")[0] for name,value in maps.items()}
            if args.verify_controls:
                assert_control_close(maps["V0"],previous["S0"],"A3G V0/FINE-P2")
            maps["V0"]=previous["S0"]
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(RAW):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            side[image_id]=(seed_diagnostic(recorder,seed,seed_g),)
            save_side(side_path,side)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg-a3g:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)} "
                  f"seed|d|={side[image_id][0][0]:.4f} r={side[image_id][0][1]:.4f}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG-A3G class shard")
        writer.finish()


if __name__=="__main__":main()
