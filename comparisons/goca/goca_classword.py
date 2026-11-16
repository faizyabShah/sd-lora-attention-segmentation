#!/usr/bin/env python3
"""GoCA (Meng et al., CVPR 2026; github.com/Darkbblue/goca @475892f) under the class-word protocol.

Input per item: a finished benchmark image and the queried class word. Nothing
else (no caption, no GPT label, no ground-truth class list).

What is GoCA's own code, imported unchanged from goca/src-main:
  * components/cross_attn_dispatch.py  goca_cross_prepare / goca_cross_run
    (head aggregation, layer aggregation, sum-1 rescaling + per-token renorm+)
  * components/space_attn_dispatch.py  self_attn_prepare / self_attn_run
  * components/postprocess_dispatch.py postprocess('affinity', order 2)
  * configs/config-model/config_model_15.py ModelConfig (t, methods, layers)
  * feature extraction: generic-diffusion-feature + goca/install/components patch
    (patched copy in n_side/gdf), FeatureExtractor built exactly as main.py:58-64.
What is adapted (this file):
  * Prompt and token bookkeeping. GoCA's captioner.py builds the prompt from a
    GPT entity list ("e1, e2, ...") plus ", background" (captioner.py:164-170,
    add_background_token=True in config_dataset_voc.py:27) and aligns tokens to
    entities (captioner.py:252-348). Here the entity list is the class word
    alone, so the prompt is "<class>, background"; the token alignment below is a
    line-for-line port of captioner.py:252-348 with the queried class as the only
    candidate class. The class word is the manifest class name (e.g.
    "dining-table"); "-" tokens are dropped by GoCA's own stop-token rule.
  * Output. GoCA's multi-class argmax (downstream_dispatch.py:71-121) is not run.
    We keep the per-class soft map for the queried class, bilinearly resized to the
    mask resolution as downstream_dispatch.py:90 does, both before and after the
    configured affinity postprocess, plus GoCA's merged background channel.
  * Noise: torch.manual_seed(seed) immediately before each extract() call; the
    VAE posterior sample and the added noise both draw from the global CUDA RNG.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GOCA = Path("/home/sidna/TRDP/external/goca/src-main")
sys.path.insert(0, str(GOCA))
from components.cross_attn_dispatch import cross_attn, cross_attn_prepare  # noqa: E402
from components.space_attn_dispatch import space_attn, space_attn_prepare  # noqa: E402
from components.postprocess_dispatch import postprocess  # noqa: E402
from diffusion_feature import FeatureExtractor  # noqa: E402  (n_side/gdf/feature, pip -e)


def load_model_config(path: Path):
    spec = importlib.util.spec_from_file_location("goca_model_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelConfig()


def sd15_tokenization(tokenizer, s):
    """captioner.py:57-68 + 83-96."""
    out = tokenizer(s, padding="max_length", max_length=tokenizer.model_max_length, truncation=True)
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    refined = []
    for token in [vocab[i] for i in out["input_ids"]]:
        if token == "<|startoftext|>":
            token = "<sos>"
        elif token == "<|endoftext|>":
            token = "<pad>"
        else:
            token = token.replace("</w>", "")
        refined.append(token)
    return refined, ["<sos>", "<pad>"], ["<sos>"]


def classword_prompt(tokenizer, classname: str):
    """Port of captioner.py:164-348 with entity list = [classname] and one candidate class (id 1)."""
    caption = classname
    if caption[-1] == ".":
        caption = caption[:-1]
    caption += ", background"            # add_background_token (captioner.py:167-168)
    caption = caption.lower()            # captioner.py:170
    name_to_id = {1: [classname.lower()]}
    existent_objects_gt = [1]
    entities = list({e.strip(): None for e in caption.split(",")}.keys())  # no parentheses here
    entities = [{"prompt": e, "aligned class": None, "is_closure": False} for e in entities]
    caption = ", ".join(e["prompt"] for e in entities)
    existent_objects = {}
    for e in entities:
        for idx, name in name_to_id.items():
            if e["prompt"] in name and idx in existent_objects_gt:
                e["aligned class"] = idx
                existent_objects[idx] = []
                break
    caption_tokens, special_tokens, global_tokens = sd15_tokenization(tokenizer, caption)
    caption_tokens = [{"token": t} for t in caption_tokens]
    for e in entities:
        start = caption.find(e["prompt"])
        e["char_ids"] = (start, start + len(e["prompt"]))
    idx = 0
    for t in caption_tokens:
        if t["token"] in special_tokens:
            t["char_ids"] = None
        else:
            start = idx + caption[idx:].find(t["token"])
            t["char_ids"] = (start, start + len(t["token"]))
            idx = start + len(t["token"])
        if idx == len(caption):
            idx = 0
    for t in caption_tokens:
        if t["char_ids"]:
            if t["token"].lower() in ["a", "an", "the", "and", ",", ".", "-"]:
                continue
            for i, e in enumerate(entities):
                if t["char_ids"][0] >= e["char_ids"][0] and t["char_ids"][1] <= e["char_ids"][1]:
                    t["entity"] = i
                    break
    rescaling_token_id, background_objects = [], {}
    for token_id, t in enumerate(caption_tokens):
        if "entity" in t:
            rescaling_token_id.append(token_id)
            class_id = entities[t["entity"]]["aligned class"]
            if class_id:
                existent_objects[class_id].append(token_id)
            else:
                background_objects.setdefault(t["entity"], []).append(token_id)
    tokens = [t["token"] for t in caption_tokens[:12]]
    if not existent_objects.get(1):
        raise RuntimeError(f"class word {classname!r} got no tokens: {tokens}")
    return caption, existent_objects, rescaling_token_id, list(background_objects.values()), tokens


def binary(path):
    a = np.asarray(Image.open(path))
    return (a.sum(axis=-1) if a.ndim == 3 else a) != 0


def minmax(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=GOCA / "configs/config-model/config_model_15.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    cfg = load_model_config(args.config)
    rows = list(csv.DictReader(open(args.manifest, newline="")))
    if args.limit:
        rows = rows[: args.limit]
    (args.out / "maps").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)

    # --- main.py:38-64, unchanged logic ---
    cross_setting = copy.deepcopy(cfg.cross_attn_setting)
    cross_setting.setdefault("keep_all_objects", None)
    space_setting = copy.deepcopy(cfg.space_attn_setting)
    post_setting = copy.deepcopy(cfg.postprocess_setting)
    layer, attention = set(), set()
    cross_attn_prepare(cross_setting["task"], attention, layer, cross_setting)
    space_attn_prepare(space_setting["task"], attention, layer, space_setting)
    for task in post_setting:
        if "dense_feat_id" in task:
            layer.add(task["dense_feat_id"])
    t_load = time.time()
    fe = FeatureExtractor(layer={l: True for l in layer}, version=cfg.version, device="cuda", img_size=cfg.img_size)
    load_s = time.time() - t_load

    # actual scheduler timestep used by extract(t=cfg.t) (diffusion_feature.py:288-295)
    sch = copy.deepcopy(fe.scheduler_backup)
    sch.set_timesteps(1000, device="cuda")
    fe.pipe.scheduler = sch
    ts, _ = fe.pipe.get_timesteps(1000, cfg.t / 1000, "cuda")
    actual_t = int(ts[0])

    def run_one(img, classname):
        caption, objects, resc, bg_objs, tokens = classword_prompt(fe.pipe.tokenizer, classname)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cs = dict(cross_setting)
        cs.update({"feature_extractor": fe, "img": img, "size": cfg.feat_size, "t": cfg.t, "prompt": caption,
                   "rescaling_token_id": resc, "objects": objects, "background_objects": bg_objs})
        cross_feat, features = cross_attn(cs["task"], cs)  # [class, merged background] x 64 x 64
        ss = dict(space_setting)
        ss.update({"features": features, "size": cfg.feat_size})
        space_feat = space_attn(ss["task"], ss)
        pre = cross_feat.clone()
        for step in post_setting:  # main.py:94-100
            st = dict(step)
            st.update({"cross_feat": cross_feat, "space_feat": space_feat, "image_path": None, "features": features,
                       "img_size": 512, "label": None, "class_count": 2, "image": img})
            cross_feat = postprocess(st["task"], st)
        del features, space_feat
        return pre, cross_feat, dict(caption=caption, objects=objects, rescaling_token_id=resc,
                                     background_objects=bg_objs, tokens=tokens)

    # warm-up (not timed, not saved)
    with torch.no_grad():
        run_one(Image.open(rows[0]["image"]), rows[0]["classname"])
    torch.cuda.synchronize()

    records = []
    for r in rows:
        img = Image.open(r["image"])
        target = binary(r["annotation"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            pre, post, info = run_one(img, r["classname"])
        torch.cuda.synchronize()
        sec = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_res = torch.cuda.max_memory_reserved() / 2**20
        rec = {"key": r["key"], "dataset": r["dataset"], "classname": r["classname"], "image_source": r["image_source"],
               "seconds": round(sec, 3), "peak_alloc_mib": round(peak_alloc), "peak_reserved_mib": round(peak_res), **info,
               "objects": {str(k): v for k, v in info["objects"].items()}}
        for name, feat in (("post", post), ("pre", pre)):
            up = F.interpolate(feat.float().unsqueeze(0), target.shape, mode="bilinear").squeeze(0).cpu().numpy()
            cls_map, bg_map = up[0], up[-1]
            rec[f"nan_{name}"] = bool(np.isnan(cls_map).any())
            np.save(args.out / "maps" / f"{r['key']}__{r['classname']}__{name}.npy", cls_map.astype(np.float16))
            np.save(args.out / "maps" / f"{r['key']}__{r['classname']}__{name}_bg.npy", bg_map.astype(np.float16))
            pred = minmax(np.nan_to_num(cls_map)) > 0.5
            Image.fromarray((pred * 255).astype(np.uint8)).save(args.out / "masks" / f"{r['key']}__{r['classname']}__{name}.png")
            i, u = int((pred & target).sum()), int((pred | target).sum())
            rec[f"inter_{name}"], rec[f"union_{name}"], rec[f"iou_{name}"] = i, u, i / u if u else 1.0
            # GoCA's own binarisation reduced to one class: argmax(class, background + offset), offset 0
            g = cls_map > bg_map
            gi, gu = int((g & target).sum()), int((g | target).sum())
            rec[f"iou_{name}_goca_argmax_offset0"] = gi / gu if gu else 1.0
        records.append(rec)
        print(json.dumps({k: rec[k] for k in ("key", "classname", "seconds", "peak_alloc_mib", "iou_post", "iou_pre",
                                              "iou_post_goca_argmax_offset0", "caption")}), flush=True)

    def macro(recs, name):
        by = {}
        for x in recs:
            a = by.setdefault(x["classname"], [0, 0])
            a[0] += x[f"inter_{name}"]; a[1] += x[f"union_{name}"]
        ious = {c: i / u for c, (i, u) in by.items() if u}
        return {"classes": len(ious), "images": len(recs), "miou": float(np.mean(list(ious.values()))), "class_iou": ious}

    summary = {"label": "smoke, fixed threshold 0.5 on min-max-normalised map, not a result",
               "goca_commit": "475892fdb6a1d26a794ec5f9be86bc5928ed6439",
               "gdf_commit": "adf80e1781127ad4af519ccea608a45f3d5e4e0e (copy patched with goca/install/components)",
               "config": str(args.config), "version": cfg.version, "img_size": cfg.img_size, "feat_size": cfg.feat_size,
               "t_config": cfg.t, "t_actual_scheduler": actual_t, "scheduler": type(fe.pipe.scheduler).__name__,
               "dtype": str(fe.pipe.unet.dtype), "noise_seed": args.seed,
               "cross_attn_setting": {k: v for k, v in cross_setting.items() if k != "layer"},
               "n_cross_layers": len(cross_setting["layer"]), "space_attn_setting": space_setting,
               "postprocess_setting": post_setting, "model_load_seconds": round(load_s, 1),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "per_dataset": {}}
    for ds in sorted({x["dataset"] for x in records}):
        recs = [x for x in records if x["dataset"] == ds]
        summary["per_dataset"][ds] = {"post_affinity": macro(recs, "post"), "pre_affinity": macro(recs, "pre"),
                                      "mean_seconds": float(np.mean([x["seconds"] for x in recs])),
                                      "max_peak_alloc_mib": max(x["peak_alloc_mib"] for x in recs)}
    json.dump(records, open(args.out / "per_image.json", "w"), indent=1)
    json.dump(summary, open(args.out / "summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary["per_dataset"]), flush=True)


if __name__ == "__main__":
    main()
