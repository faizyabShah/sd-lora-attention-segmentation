#!/usr/bin/env python3
"""NERVE half of the seed x propagation swap (runs/17_seed_walk_swap/PREREG.md). Runs in the NERVE environment.

Uses NERVE's released code (kunal-mahatha/nerve @ 1ddc5f9) with side N's patch (comparisons/nerve/repo_patch.diff),
copied to <BRACE_CACHE_DIR>/jc_swap/nerve_repo. Settings follow side N's variant A driver
(comparisons/nerve/nerve_classword.py): vocabulary {class word} + NERVE's VOC background words, ViT-L/14, size 336,
logit scale 50, fused-attn, SFSA refinement, time step index 45, SD 1.5 with its true head dim (40), noise seed 0.

seed  NERVE's CLIP map for the class on the stored benchmark image:
      <CACHE>/jc_swap/nerve_seed/<ds>/<id>.npz  seg (native CLIP grid), seed64 (as NERVE's refinement resizes it)
walk  NERVE's refinement (Pipeline.refinement, unchanged) on the stored image, applied to
      (a) NERVE's own seed and (b) BRACE's cross-attention seed saved by scripts/jc_swap.py capture:
      <CACHE>/jc_swap/walk/<ds>/<id>.npz  walk_clip, walk_ca [64,64]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
SWAP = CACHE / "jc_swap"
REPO = SWAP / "nerve_repo"
MANIFEST = {"voc": ROOT / "comparisons/manifests/manifest_full_voc_sim.csv",
            "coco": ROOT / "comparisons/manifests/manifest_full_coco_cap.csv"}
N_PREFIX = "/home/sidna/TRDP/sd-lora-attention-segmentation/"   # side N's checkout; same archive under ROOT
BENCH20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
           "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]
NOISE_SEED = 0


def rows_for(ds, shard, num_shards, limit):
    rows = list(csv.DictReader(open(MANIFEST[ds])))[shard::num_shards]
    for r in rows:
        if not r["image"].startswith(N_PREFIX):
            raise ValueError("unexpected manifest image path: " + r["image"])
        r["image_local"] = str(ROOT / r["image"][len(N_PREFIX):])
        r["text"] = BENCH20[BENCH20.index(r["classname"].replace("-", " "))]
    return rows[:limit] if limit else rows


def build(refine):
    sys.path.insert(0, str(REPO))
    from util.tools import load_yaml
    from data.voc.voc_datasets import VOC2012_CAT_NAME
    from ovs.pipeline_clean_mean_final_run_correct import Pipeline
    cfg = load_yaml(str(REPO / "scripts/config/vit-l-14/ovs_voc21.yaml"))
    for k, v in dict(fuse_feature=True, refinement="SFSA", attention_idx=1, attn_type="fused-attn",
                     time_step=45, sd_version="v1.5", size=336, log_path="log").items():   # util/args.py defaults
        setattr(cfg, k, v)
    if not refine:
        cfg.refinement = "none"          # CLIP only: Pipeline then loads no diffusion model
    cfg.semantic_templates = [l.strip() for l in open(REPO / "template/car_template.txt")]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pipe = Pipeline(cfg, device=torch.device("cuda:0"))
    if refine:
        sd = pipe.attn_refine.model
        from diffusion_model.Processor import DIFFUSION_LAYERS
        dims = set()
        for i in cfg.attention_layers_to_use:
            m = eval("unet." + DIFFUSION_LAYERS[i], {"unet": sd.unet})
            dims.add(m.to_q.out_features // m.heads)
        assert len(dims) == 1
        cfg.sfsa_head_dim = dims.pop()   # 40 on SD 1.5 (side N's deviation D2)
        sd.scheduler.set_timesteps(50, device=pipe.device)
        print("SD", cfg.sd_version, "layers", cfg.attention_layers_to_use, "head dim", cfg.sfsa_head_dim,
              "t", int(sd.scheduler.timesteps[cfg.time_step]), flush=True)
    bg_words = re.split(r"[\s,]+", VOC2012_CAT_NAME[0])   # voc_datasets.py:69
    return pipe, cfg, bg_words


def seed(a):
    pipe, cfg, bg_words = build(refine=False)
    dev = pipe.device
    bg_text = pipe.nerve.classifier(bg_words, cfg.semantic_templates)
    fg_cache = {}
    out = SWAP / ("nerve_seed" + a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for r in rows_for(a.dataset, a.shard, a.num_shards, a.max_images):
        o = out / (r["key"] + ".npz")
        if o.exists():
            continue
        src = Path(a.image_dir) / a.dataset / (r["key"] + ".png") if a.image_dir else r["image_local"]
        if a.image_dir and not Path(src).exists():
            continue                                            # --image-dir: only images generated so far
        pil = Image.open(src).convert("RGB")
        if r["text"] not in fg_cache:
            fg_cache[r["text"]] = pipe.nerve.classifier([r["text"]], cfg.semantic_templates)
        img = pipe.nerve.preprocess(pil)[None].to(dev)
        with torch.no_grad():
            res = pipe.nerve(img, fg_cache[r["text"]].to(dev), bg_text.to(dev))     # Pipeline.forward, first half
            seg = res["seg"]
            base_score = seg.amax(dim=0)
            seg = seg.transpose(0, 1).reshape(-1, pipe.nerve.img_h, pipe.nerve.img_w)   # [1, h, w]
            seed64 = F.interpolate(seg[None], size=(64, 64), mode="bilinear", align_corners=False)[0]  # as refinement()
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, seg=seg.float().cpu().numpy(), seed64=seed64.float().cpu().numpy(),
                 base_score=base_score.float().cpu().numpy(), cls=r["classname"], text=r["text"])
        os.replace(tmp, o)
        print("[seed:%s] %s %s grid %dx%d score %.3f" % (a.dataset, r["key"], r["text"], seg.shape[1], seg.shape[2],
                                                         float(base_score[0])), flush=True)


def walk(a):
    pipe, cfg, _ = build(refine=True)
    dev = pipe.device
    out = SWAP / ("walk" + a.tag + a.walk_tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    capdir = SWAP / ("cap" + a.tag) / a.dataset
    rows = rows_for(a.dataset, a.shard, a.num_shards, 0)
    if a.max_images:                      # smoke runs: only the images the capture smoke produced
        rows = [r for r in rows if (capdir / (r["key"] + ".npz")).exists()][: a.max_images]
    for r in rows:
        o = out / (r["key"] + ".npz")
        if o.exists():
            continue
        zs = np.load(SWAP / ("nerve_seed" + a.tag) / a.dataset / (r["key"] + ".npz"))
        zc = np.load(SWAP / ("cap" + a.tag) / a.dataset / (r["key"] + ".npz"))
        pil = Image.open(r["image_local"]).convert("RGB")
        ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]
        seeds = {"clip": torch.from_numpy(zs["seg"]).to(dev),                     # native grid, exactly as forward() passes it
                 "ca": torch.from_numpy(zc["seed_ca"].astype(np.float32))[None].to(dev)}  # BRACE seed, [1, 64, 64]
        res = {}
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        with torch.no_grad():
            for name, s in seeds.items():
                torch.manual_seed(NOISE_SEED)      # same noise, hence the same transition matrix, for both seeds
                res["walk_" + name] = pipe.refinement(ori, s)[0].float().cpu().numpy()
        if not all(np.isfinite(v).all() for v in res.values()):
            raise ValueError("non-finite walk output for " + r["key"])
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, cls=r["classname"], **res); os.replace(tmp, o)
        print("[walk:%s] %s %.2fs peak %.0f MiB" % (a.dataset, r["key"], time.time() - t0,
                                                    torch.cuda.max_memory_allocated() / 2**20), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("seed", "walk"):
        p = sub.add_parser(nm); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
        p.add_argument("--walk-tag", default="", help="extra suffix for the walk output folder (device checks)")
        p.add_argument("--image-dir", default="", help="seed only: read <image-dir>/<dataset>/<key>.png instead of the "
                       "manifest image (runs/20 new reruns: our corrected-prompt generations)")
    a = ap.parse_args(); {"seed": seed, "walk": walk}[a.cmd](a)
