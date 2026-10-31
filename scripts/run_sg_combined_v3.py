#!/usr/bin/env python3
"""Resumable SG-combined-v3 class shard: V3 replay and three single-change cells."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer,sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from src.segmentation import controlled_cv as cv,sg_combined_v2,sg_combined_v3
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS,_import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_combined_v3"
SPEC=ROOT/"configs/sg_combined_v3.json"
RAW=sg_combined_v3.NAMES
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
SIDE_FIELDS=("cap","gt_cap","seed","maps")


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


def verify(raw,sa,seed,coarse):
    """Smoke-only same-generation identities."""
    old,_=sg_combined_v2.aggregate(sa,seed)
    for mine,theirs in (("R0","V0"),("R1","V3")):
        torch.testing.assert_close(raw[mine],old[theirs],atol=1e-6,rtol=1e-5,msg=f"v3 {mine}/v2 {theirs}")
    uncapped,_=sg_combined_v3.aggregate(sa,seed,coarse,cap_scale=math.inf)
    for name in ("R1","W1","W3"):
        torch.testing.assert_close(uncapped[name],old["V1"],atol=1e-6,rtol=1e-5,msg=f"uncapped {name} must equal V1")
    if not torch.isfinite(coarse).all() or float(coarse.max()-coarse.min())<=0:
        raise ValueError("degenerate native coarse seed")


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
        raise ValueError("SG-combined-v3 locked cells/thresholds changed")
    runtime=sg27_runtime.make_runtime(args.model)
    runtime["recorder_class"].coarse_seed=True
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
            coarse=runtime["recorders"][-1].coarse_seeds()
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            gt32=F.interpolate(torch.from_numpy(truth).float()[None,None],size=(32,32),mode="area")[0,0]
            raw,info=sg_combined_v3.aggregate(sa,seed,coarse,gt32=gt32)
            maps={name:cv.normalize_batch(raw[name][None],"minmax")[0] for name in RAW}
            if args.verify_controls:
                assert_control_close(maps["R0"],previous["S0"],"v3 R0/FINE-P2")
                if args.max_samples:verify(raw,sa,seed,coarse)
            maps["R0"]=previous["S0"]
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(RAW):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            side[image_id]=(info["cap"],info["gt_cap"],np.asarray(info["seed"],np.float64),
                            torch.stack([maps[n] for n in RAW]).cpu().numpy().astype(np.float16))
            save_side(side_path,side)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg-combined-v3:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)} "
                  f"c16* r={info['seed'][0]:.3f}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG-combined-v3 class shard")
        writer.finish()


if __name__=="__main__":main()
