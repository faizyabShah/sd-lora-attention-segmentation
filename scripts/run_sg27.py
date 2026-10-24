#!/usr/bin/env python3
"""Resumable SG27 class shard: score unique fixed late fusions per live image."""
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

from scripts import run_coco_cap_transfer as transfer
from scripts import sg27_runtime
from scripts.run_sg24 import atomic_npz
from src.segmentation import controlled_cv as cv
from src.segmentation import sg25_sa,sg27_fusion
from src.segmentation.ovam_full import _import_densecrf,OVAM_CRF_PARAMETERS

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg27"
MANIFEST=ROOT/"configs/sg27_unique.json"
CHANGE_FIELDS=("tp_gained","tp_lost","fp_gained","fp_removed",
               "fp_gt16_gained","fp_gt16_removed","fp_gt64_gained","fp_gt64_removed",
               "separate_fp_gained","separate_fp_removed","foreground_area_delta")


def names_for(unique:list[str]) -> tuple[str,...]:
    return tuple(name+suffix for name in unique for suffix in ("","+R2"))


def _separate_fp(prediction:np.ndarray,truth:np.ndarray) -> np.ndarray:
    labels,_=ndimage.label(prediction)
    touching=np.unique(labels[prediction & truth])
    return prediction & ~truth & ~np.isin(labels,touching[touching>0])


def change_diagnostics(maps:dict[str,torch.Tensor],unique:list[str],truth:np.ndarray,threshold:float) -> np.ndarray:
    stack=torch.stack([maps[name] for name in unique])
    full=F.interpolate(stack[:,None],size=truth.shape,mode="bilinear",align_corners=False)[:,0]
    masks=(full>=threshold).cpu().numpy()
    reference=masks[0]
    ref_separate=_separate_fp(reference,truth)
    distance=ndimage.distance_transform_edt(~truth)
    rows=[]
    for candidate in masks[1:]:
        gained=candidate & ~reference
        removed=reference & ~candidate
        candidate_separate=_separate_fp(candidate,truth)
        rows.append((int((gained & truth).sum()),int((removed & truth).sum()),
                     int((gained & ~truth).sum()),int((removed & ~truth).sum()),
                     int((gained & ~truth & (distance>16)).sum()),
                     int((removed & ~truth & (distance>16)).sum()),
                     int((gained & ~truth & (distance>64)).sum()),
                     int((removed & ~truth & (distance>64)).sum()),
                     int((gained & candidate_separate).sum()),
                     int((removed & ref_separate).sum()),
                     int(candidate.sum()-reference.sum())))
    return np.asarray(rows,dtype=np.int64)


def disagreement_diagnostics(maps:dict[str,torch.Tensor],truth:np.ndarray,threshold:float):
    base=torch.stack([maps[n] for n in ("S0","S1","S2")])>=threshold
    bits=base.cpu().numpy().astype(np.uint8)
    gt=cv.target_at_size(truth).cpu().numpy()
    pair=np.zeros((2,4,2),np.int64)
    triple=np.zeros((8,2),np.int64)
    for j in range(2):
        code=2*bits[0]+bits[j+1]
        for value in range(4):
            selected=code==value
            pair[j,value]=(int((selected & gt).sum()),int((selected & ~gt).sum()))
    code=4*bits[0]+2*bits[1]+bits[2]
    for value in range(8):
        selected=code==value
        triple[value]=(int((selected & gt).sum()),int((selected & ~gt).sum()))
    return pair,triple


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("voc","coco"),required=True)
    p.add_argument("--class-shard",type=int,choices=range(20),required=True)
    p.add_argument("--max-samples",type=int)
    p.add_argument("--verify-controls",action="store_true",help="smoke only: compare live F/C16/C32 with SG25")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--model",default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site",type=Path,default=ROOT/"outputs/dependencies/simplecrf_py310")
    args=p.parse_args()
    if not MANIFEST.is_file():
        raise FileNotFoundError("SG27 duplicate smoke manifest must be frozen before production")
    manifest=json.loads(MANIFEST.read_text())
    runtime=sg27_runtime.make_runtime(args.model)
    spec=runtime["spec"]
    unique=manifest["unique_cells"]
    if (manifest["spec_version"]!=spec["version"] or unique[0]!="S0" or
            len(set(unique))!=len(unique) or not set(unique)<=set(sg27_fusion.NAMES)):
        raise ValueError("SG27 duplicate manifest incompatible with locked spec")
    names=names_for(unique)
    classname,all_rows=(transfer.voc_rows if args.dataset=="voc" else transfer.coco_rows)(args.class_shard)
    rows=all_rows[:args.max_samples] if args.max_samples else all_rows
    _import_densecrf(args.crf_site)
    folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset=="voc" else {r["image_id"]:0 for r in all_rows}
    out=args.output_dir or BASE
    stem=f'class_{args.class_shard:02d}_{classname.replace(" ","_")}'
    path=out/args.dataset/"shards"/(stem+".npz")
    side=path.with_suffix(".diagnostics.npz")
    settings={"spec":spec,"manifest":manifest,"dataset":args.dataset,"model":args.model,
              "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
              "crf":list(OVAM_CRF_PARAMETERS)}
    writer=cv.ShardWriter(path,names,folds,settings,spec["thresholds"])
    sidecars={}
    if side.exists():
        with np.load(side,allow_pickle=False) as data:
            sidecars={str(image_id):(change,pair,triple)
                      for image_id,change,pair,triple in zip(data["image_ids"],data["change"],
                                                             data["pair"],data["triple"])}
    if writer.completed-set(sidecars):
        raise ValueError("SG27 metric shard ahead of diagnostics")
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,maps):
            if args.verify_controls:
                old,_=sg25_sa.aggregate(sa,seed)
                for new,prior in (("FINE","FINE-P2"),("C16","16-to-64"),("C32","32-to-64")):
                    torch.testing.assert_close(base[new],old[prior],rtol=1e-5,atol=2e-6)
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw=torch.stack([maps[name] for name in unique])
            raw_counts,raw_ap=cv.score_maps(raw.cpu(),truth,"minmax",spec["thresholds"])
            counts=np.zeros((len(names),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(names),np.nan)
            counts[0::2]=raw_counts
            ap[0::2]=raw_ap
            for i,name in enumerate(unique):
                counts[2*i+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            changes=change_diagnostics(maps,unique,truth,spec["diagnostic_threshold"])
            pair,triple=disagreement_diagnostics(maps,truth,spec["disagreement_threshold"])
            sidecars[row["image_id"]]=(changes,pair,triple)
            atomic_npz(side,image_ids=np.asarray(list(sidecars)),
                       change=np.stack([x[0] for x in sidecars.values()]),
                       pair=np.stack([x[1] for x in sidecars.values()]),
                       triple=np.stack([x[2] for x in sidecars.values()]))
            writer.add(row["image_id"],classname,counts,ap)
            print(f"[sg27:{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(sidecars)!=writer.completed:
            raise ValueError("incomplete SG27 class shard")
        writer.finish()


if __name__=="__main__":
    main()
