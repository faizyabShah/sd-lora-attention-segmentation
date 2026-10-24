#!/usr/bin/env python3
"""Four live SG27 map checks; freeze all smoke-detected mathematical duplicates."""
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

from scripts import run_coco_cap_transfer as transfer
from scripts import sg27_runtime
from src.segmentation import controlled_cv as cv
from src.segmentation import sg25_sa,sg27_fusion

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg27"
MANIFEST=ROOT/"configs/sg27_unique.json"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE)
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--model",default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args=p.parse_args()
    runtime=sg27_runtime.make_runtime(args.model)
    maximum=np.zeros((46,46),np.float64)
    sampled=[]
    for dataset,shard in runtime["spec"]["duplicate_smoke"]:
        _,rows=(transfer.voc_rows if dataset=="voc" else transfer.coco_rows)(shard)
        row=rows[0]
        with sg27_runtime.live_maps(runtime,row,dataset) as (_,sa,seed,base,fused):
            historical,_=sg25_sa.aggregate(sa,seed)
            base_diff={}
            for name,old in (("FINE","FINE-P2"),("C16","16-to-64"),("C32","32-to-64")):
                torch.testing.assert_close(base[name],historical[old],rtol=1e-5,atol=2e-6)
                base_diff[name]=float((base[name]-historical[old]).abs().max())
            controls=cv.normalize_batch(torch.stack([historical[n] for n in ("FINE-P2","16-to-64","32-to-64")]),"minmax")
            controls=cv.normalize_batch(controls,"minmax")
            for i,name in enumerate(("S0","S1","S2")):
                torch.testing.assert_close(fused[name],controls[i],rtol=1e-5,atol=2e-6)
            stacked=torch.stack([fused[name] for name in sg27_fusion.NAMES]).reshape(46,-1)
            diff=(stacked[:,None,:]-stacked[None,:,:]).abs().amax(-1).cpu().numpy()
            maximum=np.maximum(maximum,diff)
            sampled.append({"dataset":dataset,"class":row["classname"],"image_id":row["image_id"],
                            "base_max_abs_difference_vs_sg25":base_diff})
        print(f"[sg27 duplicate smoke] {dataset} {row['classname']} {row['image_id']}",flush=True)
    unique,aliases,groups=sg27_fusion.duplicate_groups(maximum,runtime["spec"]["duplicate_threshold"])
    expected={"S11":"S7","S12":"S8","S15":"S7","S24":"S20"}
    if any(aliases.get(name)!=representative for name,representative in expected.items()):
        raise ValueError(f"expected SG27 algebraic duplicates missing: {aliases}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    scan={"spec":runtime["spec"],"samples":sampled,"unique_cells":unique,
          "aliases":aliases,"duplicate_groups":groups,"max_abs_differences":maximum.tolist()}
    (args.output_dir/"duplicate_scan.json").write_text(json.dumps(scan,indent=2)+"\n")
    manifest={"spec_version":runtime["spec"]["version"],"unique_cells":unique,
              "aliases":aliases,"duplicate_groups":groups,"samples":[x["image_id"] for x in sampled],
              "threshold":runtime["spec"]["duplicate_threshold"],
              "source_scan":str((args.output_dir/"duplicate_scan.json").relative_to(ROOT))}
    args.manifest.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"SG27 duplicate scan passed: {len(unique)} unique of 46 proposed; aliases={aliases}",flush=True)


if __name__=="__main__":
    main()
