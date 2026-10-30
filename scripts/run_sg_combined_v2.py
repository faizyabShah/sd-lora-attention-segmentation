#!/usr/bin/env python3
"""Resumable SG-combined-v2 class shard: capped hierarchy cells and route diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer,sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv,sg31_seediff,sg_combined_v1,sg_combined_v2
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS,_import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_combined_v2"
SPEC=ROOT/"configs/sg_combined_v2.json"
RAW=sg_combined_v2.NAMES
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
CONTRASTS=(("V1","V3"),("V3","V4"),("V3","V5"),("V5","V6"),("V2","V5"))
SIDE_FIELDS=("cap","route","change","background_log")


def diagnostic(maps,transient,truth,threshold):
    background=transient["background"].float()
    route=np.asarray([sg_combined_v2.route_scores(maps[f],maps[h],background)
                      for f,h in sg_combined_v2.ROUTE_PAIRS],np.float64)
    change=np.stack([change_diagnostics(maps,[a,b],truth,threshold)[0] for a,b in CONTRASTS])
    return {"cap":transient["cap"],"route":route,"change":change,
            "background_log":np.asarray((transient["background_fallbacks"],
                                          transient["background_seed_fraction"]),np.float64)}


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


def verify(maps,raw,sa,seed,previous):
    """Smoke-only same-generation identities; production only checks V0."""
    sg31_maps,_,_=sg31_seediff.aggregate(sa,seed)
    assert_control_close(maps["V1"],sg31_maps["S2"],"v2 V1/SG31 S2 hierarchy")
    v1_raw,_=sg_combined_v1.aggregate(sa,seed)
    for mine,theirs in (("V1","C1"),("V2","C6")):
        expected=cv.normalize_batch(v1_raw[theirs][None],"minmax")[0]
        assert_control_close(maps[mine],expected,f"v2 {mine}/SG-combined-v1 {theirs}")
    uncapped,_=sg_combined_v2.aggregate(sa,seed,cap_scale=math.inf)
    for name in ("V3","V4"):
        torch.testing.assert_close(uncapped[name],raw["V1"],atol=1e-6,rtol=1e-5,
                                   msg=f"uncapped {name} must equal V1")


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
        raise ValueError("SG-combined-v2 locked cells/thresholds changed")
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
    if writer.completed-set(side):raise ValueError("metric shard ahead of diagnostics")
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw,transient=sg_combined_v2.aggregate(sa,seed)
            maps={name:cv.normalize_batch(raw[name][None],"minmax")[0] for name in RAW}
            if args.verify_controls:
                assert_control_close(maps["V0"],previous["S0"],"v2 V0/FINE-P2")
                if args.max_samples:verify(maps,raw,sa,seed,previous)
            maps["V0"]=previous["S0"]
            raw_counts,raw_ap=cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                              "minmax",spec["thresholds"])
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for i,name in enumerate(RAW):
                counts[2*i],ap[2*i]=raw_counts[i],raw_ap[i]
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            diag=diagnostic(maps,transient,truth,spec["diagnostic_threshold"])
            side[image_id]=tuple(diag[field] for field in SIDE_FIELDS)
            save_side(side_path,side)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg-combined-v2:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG-combined-v2 class shard")
        writer.finish()


if __name__=="__main__":main()
