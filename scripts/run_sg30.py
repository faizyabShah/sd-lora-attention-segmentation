#!/usr/bin/env python3
"""Resumable SG30 class shard: exact dense graph updates and reduced diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer, sg27_runtime
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv, sg30_spark
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg30"
SPEC_PATH=ROOT/"configs/sg30.json"
RAW=sg30_spark.ALL
NAMES=tuple(n+s for n in sg30_spark.MAIN for s in ("","+R2"))+sg30_spark.DIAGNOSTIC
SIDE_FIELDS=("change","graph","transition","edge_quartile_mass","iteration")


def read_side(path:Path):
    if not path.exists():return {}
    with np.load(path,allow_pickle=False) as data:
        ids=[str(x) for x in data["image_ids"]]
        values={key:data[key] for key in SIDE_FIELDS}
    if len(ids)!=len(set(ids)):raise ValueError("duplicate SG30 diagnostic image")
    return {image_id:tuple(values[key][i] for key in SIDE_FIELDS) for i,image_id in enumerate(ids)}


def save_side(path:Path,rows):
    atomic_npz(path,image_ids=np.asarray(list(rows)),
               **{key:np.stack([row[i] for row in rows.values()])
                  for i,key in enumerate(SIDE_FIELDS)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("voc","coco"),required=True)
    p.add_argument("--class-shard",type=int,choices=range(20),required=True)
    p.add_argument("--max-samples",type=int)
    p.add_argument("--verify-controls",action="store_true")
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--model",default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site",type=Path,default=ROOT/"outputs/dependencies/simplecrf_py310")
    args=p.parse_args()
    spec=json.loads(SPEC_PATH.read_text())
    if tuple(spec["cells"])!=sg30_spark.MAIN or tuple(spec["diagnostic_cells"])!=sg30_spark.DIAGNOSTIC:
        raise ValueError("SG30 locked cell list changed")
    if spec["thresholds"]!=list(transfer.THRESHOLDS) or spec["expansion"]!=2:
        raise ValueError("SG30 threshold/expansion control changed")
    runtime=sg27_runtime.make_runtime(args.model,recorder_class=sg30_spark.SG30Recorder)
    _import_densecrf(args.crf_site)
    classname,all_rows=(transfer.voc_rows if args.dataset=="voc" else transfer.coco_rows)(args.class_shard)
    rows=all_rows[:args.max_samples] if args.max_samples else all_rows
    folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset=="voc" else {r["image_id"]:0 for r in all_rows}
    stem=f'class_{args.class_shard:02d}_{classname.replace(" ","_")}'
    path=args.output_dir/args.dataset/"shards"/(stem+".npz")
    side=path.with_suffix(".diagnostics.npz")
    settings={"spec":spec,"dataset":args.dataset,"model":args.model,
              "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
              "crf":list(OVAM_CRF_PARAMETERS)}
    writer=cv.ShardWriter(path,NAMES,folds,settings,spec["thresholds"])
    sidecars=read_side(side)
    if writer.completed-set(sidecars):raise ValueError("SG30 metric shard ahead of diagnostics")
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            features=runtime["recorders"][-1].sa_features
            raw,diag=sg30_spark.aggregate(sa[64],features,seed,power=spec["inflation_r"],
               threshold=spec["prune_tau"],beta=spec["local_global_beta"],
               truth64=cv.target_at_size(truth).to(seed.device))
            if args.verify_controls:
                torch.testing.assert_close(raw["S0"],base["FINE"],rtol=1e-5,atol=2e-6)
            norm=cv.normalize_batch(torch.stack([raw[name] for name in RAW]),"minmax")
            maps={name:norm[i] for i,name in enumerate(RAW)}
            maps["S0"]=previous["S0"]  # exact same-generation historical control
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(sg30_spark.MAIN):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            for j,name in enumerate(sg30_spark.DIAGNOSTIC):
                counts[16+j],ap[16+j]=raw_counts[8+j],raw_ap[8+j]
            changes=change_diagnostics(maps,list(RAW),truth,spec["diagnostic_threshold"])
            sidecars[image_id]=(changes,*(diag[field].numpy() for field in SIDE_FIELDS[1:]))
            save_side(side,sidecars)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg30:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(sidecars)!=writer.completed:
            raise ValueError("incomplete SG30 class shard")
        writer.finish()


if __name__=="__main__":main()
