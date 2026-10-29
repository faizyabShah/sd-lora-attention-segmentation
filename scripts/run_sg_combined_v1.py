#!/usr/bin/env python3
"""Resumable SG-combined-v1 class shard with compact mechanism diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts import run_coco_cap_transfer as transfer,sg27_runtime
from scripts.control_checks import assert_control_close
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics,_separate_fp
from src.segmentation import controlled_cv as cv,sg31_seediff,sg_combined_v1
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS,_import_densecrf

BASE=ROOT/"outputs/segmentation/controlled_attention_cv/sg_combined_v1"
SPEC=ROOT/"configs/sg_combined_v1.json"
RAW=sg_combined_v1.NAMES
NAMES=tuple(n+s for n in RAW for s in ("","+R2"))
SIDE_FIELDS=("change","transport","ratio","entropy","stage_counts",
             "stage_changes","wrong_coarse","error_types","completion_bins",
             "residual_bins","completion_efficiency","beta","beta_context",
             "background_log")
REGIONS=("gt_fg","bg_0_16","bg_16_64","bg_gt64")
BIN_FIELDS=("pixels","gt_fg","bg_0_16","bg_16_64","bg_gt64",
            "separate_fp","sum_positive_residual")


def full(value:torch.Tensor,shape):
    return F.interpolate(value.reshape(1,1,64,64).float(),size=shape,
                         mode="bilinear",align_corners=False)[0,0].cpu().numpy()


def per_image_iou(mask,truth):
    union=np.logical_or(mask,truth).sum()
    return float(np.logical_and(mask,truth).sum()/union) if union else 1.


def diagnostic(maps,transient,truth,threshold):
    shape=truth.shape
    distance=ndimage.distance_transform_edt(~truth)
    region=np.where(truth,0,np.where(distance<=16,1,np.where(distance<=64,2,3)))
    stages=cv.normalize_batch(transient["stages"].reshape(9,64,64).cpu(),"minmax").reshape(3,3,64,64)
    stage_counts,_=cv.score_maps(stages.reshape(9,64,64),truth,"minmax",(threshold,))
    stage_changes=np.zeros((3,2,11),np.int64)
    for i in range(3):
        for j in range(2):
            stage_changes[i,j]=change_diagnostics({"prior":stages[i,j],
                "next":stages[i,j+1]},["prior","next"],truth,threshold)[0]
    c1=full(maps["C1"].cpu(),shape)>=threshold
    c4=full(maps["C4"].cpu(),shape)>=threshold
    c5=full(maps["C5"].cpu(),shape)>=threshold
    coarse=full(stages[0,0],shape)>=threshold
    correct_coarse=coarse==truth;correct_fine=c1==truth
    error_types=np.zeros((2,4,3),np.int64)
    for index,candidate in enumerate((c4,c5)):
        for category,(a,b) in enumerate(((True,True),(True,False),(False,True),(False,False))):
            selected=(correct_coarse==a)&(correct_fine==b)
            error_types[index,category]=(int(selected.sum()),
                int((selected&~correct_fine&(candidate==truth)).sum()),
                int((selected&correct_fine&(candidate!=truth)).sum()))
    # A "wrong coarse parent" means high coarse seed score and <25% GT area.
    gt_fraction=F.interpolate(torch.from_numpy(truth.astype(np.float32))[None,None],
                              size=(16,16),mode="area")[0,0].numpy()
    coarse16=F.interpolate(stages[0,0].reshape(1,1,64,64),size=(16,16),
                           mode="nearest")[0,0].numpy()
    wrong16=(gt_fraction<.25)&(coarse16>=.7)
    wrong=ndimage.zoom(wrong16.astype(np.uint8),(shape[0]/16,shape[1]/16),order=0)
    if wrong.shape!=shape:
        wrong=F.interpolate(torch.from_numpy(wrong16.astype(np.float32))[None,None],
                            size=shape,mode="nearest")[0,0].numpy()
    wrong=wrong.astype(bool)
    wrong_coarse=np.asarray((int(wrong16.sum()),int((wrong&c1&~truth).sum()),
                            int((wrong&c4&~truth).sum()),int((wrong&c5&~truth).sum())),np.int64)
    core,p2,delta,bg,odds=(full(x,shape) for x in transient["completion_maps"])
    candidate=full(maps["C6"].cpu(),shape)>=threshold
    separate=_separate_fp(candidate,truth)
    positive=delta>1e-6
    bins=np.zeros((2,5,len(BIN_FIELDS)),np.float64)
    for family,values,edges in ((0,odds,np.linspace(0,1,6)),
                                (1,delta,np.asarray((0,.05,.1,.2,.4,1.00001)))):
        for k in range(5):
            selected=positive&(values>=edges[k])&((values<edges[k+1]) if k<4 else (values<=edges[k+1]))
            bins[family,k]=(int(selected.sum()),int((selected&(region==0)).sum()),
                int((selected&(region==1)).sum()),int((selected&(region==2)).sum()),
                int((selected&(region==3)).sum()),int((selected&separate).sum()),
                float(delta[selected].sum()))
    changes=change_diagnostics({"C4":maps["C4"],**{n:maps[n] for n in ("C6","C7","C8")}},
                               ["C4","C6","C7","C8"],truth,threshold)
    efficiency=np.asarray([(r[0],r[2],r[6],r[1],r[3],
                            r[0]/max(r[2],1),r[0]/max(r[6],1)) for r in changes],np.float64)
    beta=transient["beta"]
    beta_context=np.asarray((per_image_iou(candidate,truth)-per_image_iou(c4,truth),
                             int((candidate&~truth).sum())-int((c4&~truth).sum())),np.float64)
    return {"change":change_diagnostics(maps,list(RAW),truth,threshold),
        "transport":transient["transport"],"ratio":transient["ratio"],
        "entropy":transient["entropy"],"stage_counts":stage_counts[:,0].reshape(3,3,4),
        "stage_changes":stage_changes,"wrong_coarse":wrong_coarse,
        "error_types":error_types,"completion_bins":bins[0],"residual_bins":bins[1],
        "completion_efficiency":efficiency,"beta":beta,"beta_context":beta_context,
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


def sheet(path,image,truth,seed,transient,maps,classname,image_id):
    stops=np.asarray(((0.,0,0,4),(.25,74,16,112),(.5,183,55,121),
                      (.75,251,139,75),(1.,252,253,191)),np.float32)
    def palette(value):
        x=sg_combined_v1.score(value).reshape(64,64).cpu().numpy()
        return np.stack([np.interp(x,stops[:,0],stops[:,j]) for j in (1,2,3)],-1).astype(np.uint8)
    stages=transient["stages"].cpu()
    core,p2,delta,bg,odds=transient["completion_maps"]
    items=[("RGB",Image.fromarray(np.asarray(image,np.uint8))),
           ("A3 CA",seed.cpu()),("coarse16",stages[0,0]),
           ("q32",stages[0,1]),("projected32",stages[1,1]),
           ("q64",stages[0,2]),("core C4",stages[1,2]),
           ("positive residual",delta),("background B",bg),
           ("odds G",odds),("C6",maps["C6"].cpu()),
           ("C8",maps["C8"].cpu()),
           ("GT",Image.fromarray((truth.astype(np.uint8)*255)).convert("RGB"))]
    width=160;columns=7;rows=2
    out=Image.new("RGB",(columns*width,rows*190),"white")
    draw=ImageDraw.Draw(out)
    for i,(name,value) in enumerate(items):
        panel=value if isinstance(value,Image.Image) else Image.fromarray(palette(value))
        x=(i%columns)*width;y=(i//columns)*190
        out.paste(panel.convert("RGB").resize((width,160),Image.Resampling.BILINEAR),(x,y+24))
        draw.text((x+4,y+5),name,fill="black")
    path.parent.mkdir(parents=True,exist_ok=True)
    out.save(path,optimize=True)


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
        raise ValueError("SG-combined-v1 locked cells/thresholds changed")
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
    if writer.completed-set(side):raise ValueError("metric shard ahead of diagnostics")
    visual_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {"score":None}
    for row in rows:
        image_id=row["image_id"]
        if image_id in writer.completed:continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth=np.asarray(sample.target_mask,bool)
            image=np.asarray(sample.generated_image,np.uint8)
            raw,transient=sg_combined_v1.aggregate(sa,seed)
            maps={name:cv.normalize_batch(raw[name][None],"minmax")[0] for name in RAW}
            if args.verify_controls:
                assert_control_close(maps["C0"],previous["S0"],"combined C0/FINE-P2")
                if args.max_samples:
                    sg31_maps,_,_=sg31_seediff.aggregate(sa,seed)
                    assert_control_close(maps["C1"],sg31_maps["S2"],"combined C1/SG31 hierarchy")
            maps["C0"]=previous["S0"]
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
            if args.dataset=="voc" and not args.max_samples:
                at=spec["thresholds"].index(spec["diagnostic_threshold"])
                a=counts[2*8,at];b=counts[2*6,at]
                score=float(a[0]/max(a[:3].sum(),1)-b[0]/max(b[:3].sum(),1))
                better=visual_meta["score"] is None or (score>visual_meta["score"] if args.class_shard<10
                      else score<visual_meta["score"])
                if better:
                    sheet(meta_path.with_suffix(".png"),image,truth,seed,transient,maps,classname,image_id)
                    visual_meta={"image_id":image_id,"class":classname,
                                 "selection":"best C8-C6" if args.class_shard<10 else "worst C8-C6",
                                 "score":score,"diagnostic_only":True}
                    meta_path.write_text(json.dumps(visual_meta,indent=2)+"\n")
            writer.add(image_id,classname,counts,ap)
            print(f"[sg-combined:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(side)!=writer.completed:
            raise ValueError("incomplete SG-combined-v1 class shard")
        writer.finish()


if __name__=="__main__":main()
