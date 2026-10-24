#!/usr/bin/env python3
"""Resumable SG26 class shard: one live image, seven SA cells, reduced diagnostics."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from scripts import run_coco_cap_transfer as transfer
from scripts.run_sg24 import atomic_npz
from src.attention_capture import live
from src.segmentation import controlled_cv as cv
from src.segmentation import sg24_ca, sg25_sa, sg26_sa
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg26"
SPEC = ROOT / "configs/sg26.json"
NAMES = tuple(name + suffix for name in sg26_sa.NAMES for suffix in ("", "+R2"))
MECHANISM_FIELDS = ("tp_gained", "tp_lost", "fp_gained", "fp_removed",
                    "fp_gt16_gained", "fp_gt16_removed", "fp_gt64_gained", "fp_gt64_removed",
                    "separate_fp_gained", "separate_fp_removed")


def _separate_fp(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    components, _ = ndimage.label(prediction)
    touching = np.unique(components[prediction & truth])
    return prediction & ~truth & ~np.isin(components, touching[touching > 0])


def mechanism_counts(maps: dict[str,torch.Tensor], truth: np.ndarray, threshold: float) -> np.ndarray:
    batch = torch.stack([maps[name] for name in sg26_sa.NAMES])
    normalized = cv.normalize_batch(batch,"minmax")
    full = F.interpolate(normalized[:,None],size=truth.shape,mode="bilinear",align_corners=False)[:,0]
    masks = (full >= threshold).cpu().numpy()
    reference = masks[0]
    reference_separate = _separate_fp(reference,truth)
    distance = ndimage.distance_transform_edt(~truth)
    rows=[]
    for candidate in masks[3:7]:
        gained = candidate & ~reference
        removed = reference & ~candidate
        candidate_separate = _separate_fp(candidate,truth)
        rows.append((int((gained & truth).sum()),int((removed & truth).sum()),
                     int((gained & ~truth).sum()),int((removed & ~truth).sum()),
                     int((gained & ~truth & (distance > 16)).sum()),
                     int((removed & ~truth & (distance > 16)).sum()),
                     int((gained & ~truth & (distance > 64)).sum()),
                     int((removed & ~truth & (distance > 64)).sum()),
                     int((gained & candidate_separate).sum()),
                     int((removed & reference_separate).sum())))
    return np.asarray(rows,np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("voc","coco"),required=True)
    p.add_argument("--class-shard",type=int,choices=range(20),required=True)
    p.add_argument("--max-samples",type=int)
    p.add_argument("--verify-controls",action="store_true",help="smoke only: compare S0/S1/S2 to SG25 on the same captured observations")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--model",default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site",type=Path,default=ROOT/"outputs/dependencies/simplecrf_py310")
    args=p.parse_args()
    spec=json.loads(SPEC.read_text())
    sg25_spec=json.loads((ROOT/"configs/sg25.json").read_text())
    frozen=json.loads((ROOT/"configs/controlled_sa_seed_frozen.json").read_text())
    sg24_spec=json.loads((ROOT/"configs/sg24.json").read_text())
    if (not frozen["frozen"] or frozen["modules"]!=sg24_spec["selected_ca_modules"]
            or spec["sa_modules"]!=sg25_spec["sa_modules"]):
        raise ValueError("SG26 frozen CA or SA modules changed")
    if spec["thresholds"]!=list(transfer.THRESHOLDS):
        raise ValueError("threshold grid changed")
    classname,all_rows=(transfer.voc_rows if args.dataset=="voc" else transfer.coco_rows)(args.class_shard)
    rows=all_rows[:args.max_samples] if args.max_samples else all_rows
    pipe,ca_modules,selected_sa,_=sg18.make_pipe(args.model,"cuda",json.loads((ROOT/"configs/sg12_crf_sg14_sg15.json").read_text()),frozen)
    if selected_sa!=spec["sa_modules"]["64"]:
        raise ValueError("SG26 fine module changed")
    modules={int(side):name for side,name in spec["sa_modules"].items()}
    available={n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith(".attn1.processor")}
    if set(modules.values())-available:
        raise ValueError("SG26 coarse/fine SA module missing")
    sg25_sa.SG25Recorder.pipe=pipe
    sg25_sa.SG25Recorder.ca_modules=tuple(ca_modules[i] for i in frozen["modules"])
    sg25_sa.SG25Recorder.selected_modules=sg25_sa.SG25Recorder.ca_modules
    sg25_sa.SG25Recorder.selected_sa=selected_sa
    sg25_sa.SG25Recorder.extra_sa=(modules[16],modules[32])
    sg25_sa.SG25Recorder.experiment="a"
    sg25_sa.SG25Recorder.enabled=("A3",)
    recorders=[]
    original_init=sg25_sa.SG25Recorder.__init__
    def tracked_init(self,*a,**k):
        original_init(self,*a,**k)
        recorders.append(self)
    sg25_sa.SG25Recorder.__init__=tracked_init
    live.CompactAttentionRecorder=sg25_sa.SG25Recorder
    _import_densecrf(args.crf_site)
    folds=cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset=="voc" else {r["image_id"]:0 for r in all_rows}
    out=args.output_dir or BASE
    stem=f'class_{args.class_shard:02d}_{classname.replace(" ","_")}'
    path=out/args.dataset/"shards"/(stem+".npz")
    side=path.with_suffix(".diagnostics.npz")
    settings={"spec":spec,"dataset":args.dataset,"model":args.model,
              "ca_modules":list(sg25_sa.SG25Recorder.ca_modules),"sa_modules":modules,
              "crf":list(OVAM_CRF_PARAMETERS)}
    writer=cv.ShardWriter(path,NAMES,folds,settings,spec["thresholds"])
    diagnostics={}
    if side.exists():
        with np.load(side,allow_pickle=False) as data:
            diagnostics={str(image_id):(mechanism,edge,groups)
                         for image_id,mechanism,edge,groups in zip(data["image_ids"],data["mechanism"],
                                                                   data["edge"],data["edge_groups"])}
    if writer.completed-set(diagnostics):
        raise ValueError("metric shard ahead of SG26 diagnostic sidecar")
    manifest=transfer.EVAL/("ovam/voc_sim" if args.dataset=="voc" else "ovam/coco_captions")
    from src.segmentation.attention_aggregation import find_token_subsequence
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        prompt=row["attribution_prompt"]
        ids=pipe.tokenizer(prompt,padding="max_length",max_length=77,truncation=True)["input_ids"]
        class_ids=pipe.tokenizer(row["classname"],add_special_tokens=False)["input_ids"]
        target=find_token_subsequence(ids,class_ids)
        sg25_sa.SG25Recorder.content=sg24_ca.content_positions(pipe.tokenizer,prompt,tuple(target))
        sg25_sa.SG25Recorder.attribution_prompt=prompt
        recorders.clear()
        sample=live.generate_attention_sample(pipe,row,manifest,"float16",cross_map_kind="conditional")
        recorder=recorders[-1]
        try:
            seed=recorder.seeds()["selected/A3"]
            sa={16:recorder.multiscale_qk[modules[16]],32:recorder.multiscale_qk[modules[32]],64:recorder.sa_qk}
            truth=np.asarray(sample.target_mask,bool)
            truth64=cv.target_at_size(truth).cpu().numpy()
            maps,edge,edge_groups=sg26_sa.aggregate(sa,seed,truth64,row["image_id"])
            if args.verify_controls:
                old, _=sg25_sa.aggregate(sa,seed)
                for new,prior in (("S0","FINE-P2"),("S1","16-to-64"),("S2","32-to-64")):
                    torch.testing.assert_close(maps[new],old[prior],rtol=1e-5,atol=2e-6)
            image=np.asarray(sample.generated_image,np.uint8)
            counts=np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap=np.full(len(NAMES),np.nan)
            for j,name in enumerate(sg26_sa.NAMES):
                raw_counts,raw_ap=cv.score_maps(maps[name][None].cpu(),truth,"minmax",spec["thresholds"])
                counts[2*j]=raw_counts[0];ap[2*j]=raw_ap[0]
                counts[2*j+1]=transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            mechanism=mechanism_counts(maps,truth,spec["diagnostic_threshold"])
            diagnostics[row["image_id"]]=(mechanism,edge,edge_groups)
            atomic_npz(side,image_ids=np.asarray(list(diagnostics)),
                       mechanism=np.stack([x[0] for x in diagnostics.values()]),
                       edge=np.stack([x[1] for x in diagnostics.values()]),
                       edge_groups=np.stack([x[2] for x in diagnostics.values()]))
            writer.add(row["image_id"],classname,counts,ap)
            print(f"[sg26:{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}",flush=True)
        finally:
            recorder.clear();sample.clear();recorders.clear()
            del recorder,sample
            gc.collect();torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed!={r["image_id"] for r in all_rows} or set(diagnostics)!=writer.completed:
            raise ValueError("incomplete SG26 class shard")
        writer.finish()


if __name__=="__main__":
    main()
