#!/usr/bin/env python3
"""Resumable full-benchmark driver for the canonical GoCA class-word adapter.

Imports the prompt/token code and helpers from scripts/goca_classword.py unchanged
and repeats its per-item pipeline (run_one + saving) exactly; the only additions are
(1) resumability: an item is done when records/<key>__<class>.json exists (written
last, after all maps and masks), (2) an atomic per-item record instead of one
per_image.json at the end, (3) a --shard i/n option.

Outputs under --out (same names as the smoke): maps/<key>__<class>__{post,pre}{,_bg}.npy
(float16, mask resolution), masks/<key>__<class>__{post,pre}.png (0.5 on min-max map),
records/<key>__<class>.json (seconds, peak memory, prompt bookkeeping, sanity IoU counts).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import goca_classword as gc  # noqa: E402  canonical adapter (imports GoCA + diffusion_feature)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=gc.GOCA / "configs/config-model/config_model_15.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--model-path", default=None,
                   help="load the diffusion weights from this HF repo id or local dir instead of the id hard-coded "
                        "in generic-diffusion-feature (models.py); passed as FeatureExtractor(offline_lora=...) "
                        "without a filename, which models.py uses as model_id and which loads no LoRA. "
                        "Used for SD 2.1-base: the official repo was removed (PREREG_backbones.md)")
    args = p.parse_args()

    rows = []
    for m in args.manifest:
        rows += list(csv.DictReader(open(m, newline="")))
    si, sn = map(int, args.shard.split("/"))
    rows = rows[si::sn]
    for d in ("maps", "masks", "records"):
        (args.out / d).mkdir(parents=True, exist_ok=True)

    def rec_path(r):
        return args.out / "records" / f"{r['key']}__{r['classname']}.json"

    todo = [r for r in rows if not rec_path(r).exists()]
    print(f"shard {args.shard}: {len(rows)} items, {len(rows) - len(todo)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    cfg = gc.load_model_config(args.config)
    # --- identical to goca_classword.main (main.py:38-64 logic) ---
    cross_setting = copy.deepcopy(cfg.cross_attn_setting)
    cross_setting.setdefault("keep_all_objects", None)
    space_setting = copy.deepcopy(cfg.space_attn_setting)
    post_setting = copy.deepcopy(cfg.postprocess_setting)
    layer, attention = set(), set()
    gc.cross_attn_prepare(cross_setting["task"], attention, layer, cross_setting)
    gc.space_attn_prepare(space_setting["task"], attention, layer, space_setting)
    for task in post_setting:
        if "dense_feat_id" in task:
            layer.add(task["dense_feat_id"])
    t_load = time.time()
    fe = gc.FeatureExtractor(layer={l: True for l in layer}, version=cfg.version, device="cuda", img_size=cfg.img_size,
                             offline_lora=args.model_path)
    load_s = time.time() - t_load
    sch = copy.deepcopy(fe.scheduler_backup)
    sch.set_timesteps(1000, device="cuda")
    fe.pipe.scheduler = sch
    ts, _ = fe.pipe.get_timesteps(1000, cfg.t / 1000, "cuda")
    actual_t = int(ts[0])

    def run_one(img, classname):
        caption, objects, resc, bg_objs, tokens = gc.classword_prompt(fe.pipe.tokenizer, classname)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cs = dict(cross_setting)
        cs.update({"feature_extractor": fe, "img": img, "size": cfg.feat_size, "t": cfg.t, "prompt": caption,
                   "rescaling_token_id": resc, "objects": objects, "background_objects": bg_objs})
        cross_feat, features = gc.cross_attn(cs["task"], cs)
        ss = dict(space_setting)
        ss.update({"features": features, "size": cfg.feat_size})
        space_feat = gc.space_attn(ss["task"], ss)
        pre = cross_feat.clone()
        for step in post_setting:
            st = dict(step)
            st.update({"cross_feat": cross_feat, "space_feat": space_feat, "image_path": None, "features": features,
                       "img_size": 512, "label": None, "class_count": 2, "image": img})
            cross_feat = gc.postprocess(st["task"], st)
        del features, space_feat
        return pre, cross_feat, dict(caption=caption, objects=objects, rescaling_token_id=resc,
                                     background_objects=bg_objs, tokens=tokens)

    meta = {"config": str(args.config), "version": cfg.version, "img_size": cfg.img_size, "feat_size": cfg.feat_size,
            "t_config": cfg.t, "t_actual_scheduler": actual_t, "scheduler": type(fe.pipe.scheduler).__name__,
            "dtype": str(fe.pipe.unet.dtype), "noise_seed": args.seed, "model_load_seconds": round(load_s, 1),
            "gpu": torch.cuda.get_device_name(0), "node": os.uname().nodename, "torch": torch.__version__,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "shard": args.shard,
            "model_path": args.model_path or "generic-diffusion-feature default"}
    (args.out / f"meta_shard{si}of{sn}_job{meta['slurm_job']}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)

    # warm-up (not timed, not saved), as in the smoke; it also checks that the extractor returns every
    # requested layer: GoCA silently falls back to plain head averaging when a special-* key is missing.
    seen = {}
    extract = fe.extract

    def extract_check(*a, **k):
        out = extract(*a, **k)
        seen.setdefault("keys", set(out))
        return out
    fe.extract = extract_check
    with torch.no_grad():
        run_one(Image.open(todo[0]["image"]), todo[0]["classname"])
    fe.extract = extract
    missing = sorted(set(layer) - seen["keys"])
    counts = {"cross-map": sum("cross-map" in l for l in layer), "special": sum("special" in l for l in layer),
              "self-map": sum("self-map" in l for l in layer), "requested": len(layer), "returned": len(seen["keys"])}
    print(json.dumps({"layer_check": counts, "missing": missing}), flush=True)
    if missing:
        raise SystemExit(f"extractor did not return {len(missing)} requested layers, e.g. {missing[:5]}")
    torch.cuda.synchronize()

    for n, r in enumerate(todo):
        try:
            img = Image.open(r["image"])
            target = gc.binary(r["annotation"])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pre, post, info = run_one(img, r["classname"])
            torch.cuda.synchronize()
            sec = time.time() - t0
            rec = {"key": r["key"], "dataset": r["dataset"], "classname": r["classname"], "image_source": r["image_source"],
                   "seconds": round(sec, 3), "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                   "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20), **info,
                   "objects": {str(k): v for k, v in info["objects"].items()},
                   "gpu": meta["gpu"], "node": meta["node"], "slurm_job": meta["slurm_job"]}
            for name, feat in (("post", post), ("pre", pre)):
                up = F.interpolate(feat.float().unsqueeze(0), target.shape, mode="bilinear").squeeze(0).cpu().numpy()
                cls_map, bg_map = up[0], up[-1]
                rec[f"nan_{name}"] = bool(np.isnan(cls_map).any())
                rec[f"constant_{name}"] = bool(np.nanmax(cls_map) - np.nanmin(cls_map) < 1e-6) if not np.isnan(cls_map).all() else True
                np.save(args.out / "maps" / f"{r['key']}__{r['classname']}__{name}.npy", cls_map.astype(np.float16))
                np.save(args.out / "maps" / f"{r['key']}__{r['classname']}__{name}_bg.npy", bg_map.astype(np.float16))
                pred = gc.minmax(np.nan_to_num(cls_map)) > 0.5
                Image.fromarray((pred * 255).astype(np.uint8)).save(args.out / "masks" / f"{r['key']}__{r['classname']}__{name}.png")
                i, u = int((pred & target).sum()), int((pred | target).sum())
                rec[f"inter_{name}"], rec[f"union_{name}"], rec[f"iou_{name}"] = i, u, i / u if u else 1.0
            tmp = rec_path(r).with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(rec_path(r))
            if n % 25 == 0:
                print(json.dumps({k: rec[k] for k in ("key", "classname", "seconds", "peak_alloc_mib", "iou_post")}), flush=True)
        except Exception as exc:  # record and continue; a rerun retries it
            print(f"FAILED {r['key']} {r['classname']}: {type(exc).__name__}: {exc}", flush=True)
    print("DONE", args.shard, flush=True)


if __name__ == "__main__":
    main()
