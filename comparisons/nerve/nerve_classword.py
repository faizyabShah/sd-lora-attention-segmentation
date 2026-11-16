"""NERVE (WACV 2026, github.com/kunal-mahatha/nerve) class-word binary driver, side N.

Drives NERVE's own Pipeline (repo/ovs/pipeline_clean_mean_final_run_correct.py, the one main_ovs.py imports)
on stored benchmark images, without mmseg and without NERVE's dataset classes. For each manifest row:

  vocabulary  = {queried class}  (fg)  +  NERVE's VOC21 background word list (cls_voc21.txt line 0, split as
                voc_datasets.py:69 does), with NERVE's own extra "mean of fg" competitor (model.py forward).
  pipeline    = NERVE forward + SFSA refinement + closed-form random walk, exactly as main_ovs.py:57-63.
  outputs     = soft map (refined pred_mask for the class, bilinear to mask res, float16 .npy),
                binary at 0.5 on the min-max map,
                NERVE's own binarisation (cam_to_label, main_ovs.py:62, with the config's score gate),
                NERVE's argmax-vs-background without the class-score gate (class presence is given here),
                runtime and peak GPU memory per image.

Config = repo/scripts/config/vit-l-14/ovs_voc21.yaml + util/args.py defaults (refinement SFSA, fused-attn,
fuse_feature, time_step 45, size 336), with sd_version overridden (v1.5 by default).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo"
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from util.tools import load_yaml  # noqa: E402
from util.miou import cam_to_label  # noqa: E402
from data.voc.voc_datasets import VOC2012_CAT_NAME  # noqa: E402

# Text forms. "classword": the manifest class word, hyphen -> space (VOC-Sim "dining-table" and COCO-Cap
# "dining table" both -> "dining table"; "tv" stays "tv"). "nerve": NERVE's own cls_voc21.txt line for that class,
# verbatim (e.g. "diningtable", "bird avian", "tvmonitor screen"), as voc_datasets.py:70 passes it to classifier().
BENCH20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
           "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]
NERVE20 = VOC2012_CAT_NAME[1:]  # same order as BENCH20 (VOC order)
assert len(NERVE20) == 20 and all(n.split()[0].startswith(b.replace(" ", "")[:4]) for b, n in zip(BENCH20, NERVE20))


def class_index(classname: str) -> int:
    return BENCH20.index(classname.replace("-", " "))


def text_for(classname: str, form: str) -> str:
    i = class_index(classname)
    return BENCH20[i] if form == "classword" else NERVE20[i]


def binary_gt(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return (a.sum(axis=-1) if a.ndim == 3 else a) != 0


def minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def overlay(img: Image.Image, soft: np.ndarray, pred: np.ndarray, gt: np.ndarray, path: Path) -> None:
    rgb = np.asarray(img.convert("RGB").resize(gt.shape[::-1]), dtype=np.float32)
    heat = 0.5 * rgb + 0.5 * np.stack([soft, np.zeros_like(soft), 1 - soft], -1) * 255
    def edge(m):
        e = np.zeros_like(m)
        e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
        return e
    panel = rgb.copy()
    panel[pred] = 0.5 * panel[pred] + 0.5 * np.array([255, 0, 0])
    panel[edge(gt)] = [0, 255, 0]
    Image.fromarray(np.concatenate([rgb, heat, panel], 1).clip(0, 255).astype(np.uint8)).save(path)


def build_cfg(args):
    cfg = load_yaml(str(REPO / "scripts/config/vit-l-14/ovs_voc21.yaml"))
    # util/args.py defaults that override the yaml in NERVE's own entry point (args.py:12-35)
    defaults = dict(fuse_feature=True, refinement="SFSA", attention_idx=1, attn_type="fused-attn",
                    time_step=45, sd_version="v2.1", size=336, log_path="log")
    for k, v in defaults.items():
        setattr(cfg, k, v)
    cfg.sd_version = args.sd_version
    if args.model_name:
        cfg.model_name = args.model_name
    cfg.semantic_templates = [l.strip() for l in open(REPO / "template/car_template.txt")]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/sidna/TRDP/external/n_side/outputs/smoke_manifest.csv")
    ap.add_argument("--out", default=str(HERE / "outputs/smoke_sd15"))
    ap.add_argument("--sd-version", default="v1.5")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--variants", default="A", help="comma list of A (vocab={class}, classword, d=head dim), "
                    "V (vocab=20 VOC classes, classword, d=head dim), "
                    "W (vocab=NERVE's 20 cls_voc21 names, d=64 literal: NERVE as released)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="smoke, fixed threshold, not a result")
    ap.add_argument("--summary-name", default="smoke_summary.json")
    args = ap.parse_args()

    from ovs.pipeline_clean_mean_final_run_correct import Pipeline  # import after sys.path

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = build_cfg(args)
    dev = torch.device("cuda:0")
    t0 = time.time()
    pipe = Pipeline(cfg, device=dev)
    load_s = time.time() - t0
    sd = pipe.attn_refine.model
    sd.scheduler.set_timesteps(50, device=dev)
    t_value = int(sd.scheduler.timesteps[cfg.time_step])
    layer_names = {i: __import__("diffusion_model.Processor", fromlist=["x"]).DIFFUSION_LAYERS[i]
                   for i in cfg.attention_layers_to_use}
    mods = {i: eval(f"unet.{n}", {"unet": sd.unet}) for i, n in layer_names.items()}
    heads = {i: m.heads for i, m in mods.items()}
    head_dim = {i: m.to_q.out_features // m.heads for i, m in mods.items()}
    bg_words = re.split(r"[\s,]+", VOC2012_CAT_NAME[0])  # voc_datasets.py:69
    bg_text = pipe.nerve.classifier(bg_words, cfg.semantic_templates)

    rows = [r for m in args.manifest.split(",") for r in csv.DictReader(open(m))]  # comma list of manifests
    assert len({r["key"] for r in rows}) == len(rows), "duplicate keys across manifests"
    if args.limit:
        rows = rows[: args.limit]
    base_meta = {
        "method": "NERVE (WACV 2026) public code, kunal-mahatha/nerve @ 1ddc5f9, patched copy in n_side/nerve/repo",
        "entry": "ovs/pipeline_clean_mean_final_run_correct.py Pipeline (as imported by ovs/main_ovs.py)",
        "config": "scripts/config/vit-l-14/ovs_voc21.yaml + util/args.py defaults",
        "clip": cfg.model_name, "clip_size_short_side": cfg.size, "logit_scale": cfg.logit_scale,
        "refinement": cfg.refinement, "attn_type": cfg.attn_type, "fuse_feature": cfg.fuse_feature,
        "sd_version": cfg.sd_version, "sd_repo": sd.name_or_path if hasattr(sd, "name_or_path") else None,
        "sd_prompt": "", "time_step_index": cfg.time_step, "scheduler": type(sd.scheduler).__name__,
        "num_inference_steps": 50, "timestep_value": t_value, "noise_seed": args.seed,
        "noise_seed_how": "torch.manual_seed(seed) immediately before each image's forward (global CUDA RNG used "
                          "by randn_tensor in stable_diffusion.py one_step)",
        "attention_layers_to_use": cfg.attention_layers_to_use, "layer_names": layer_names,
        "heads_per_layer": heads, "head_dim_per_layer": head_dim,
        "random_walk": "closed form (1-0.9)(I-0.9F)^-1 then F @ (.), DELTA=0.9, BETA=0.8, ALPHA=3.0",
        "bg_words": bg_words, "templates": "template/car_template.txt (15 templates)",
        "nerve_own_binarisation": f"cam_to_label(bkg_thre={cfg.bkg_thre}, cls_thre={cfg.score_threshold}, "
                                  f"is_normalize={cfg.is_normalize}) on pred/amax*final_score (main_ovs.py:61-63)",
        "model_load_s": load_s, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    import diffusers, clip  # noqa: E401
    base_meta["diffusers"] = diffusers.__version__

    variants = {"A": ("single", "classword", "head"), "V": ("voc20", "classword", "head"),
                "W": ("voc20", "nerve", 64)}
    for vname in args.variants.split(","):
        vocab, form, dmode = variants[vname]
        if vocab == "voc20":
            names20 = BENCH20 if form == "classword" else NERVE20
            fg20 = pipe.nerve.classifier(names20, cfg.semantic_templates)
        dvals = set(head_dim.values())
        assert len(dvals) == 1
        cfg.sfsa_head_dim = dvals.pop() if dmode == "head" else 64
        out = Path(args.out) / f"variant_{vname}"
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(base_meta, variant=vname, vocab=vocab, name_form=form, sfsa_d=cfg.sfsa_head_dim,
                    vocab_texts=(names20 if vocab == "voc20" else "{queried class}"))
        # resumable: records.jsonl holds one line per finished item; items with a line and a soft map are skipped
        jl = out / "records.jsonl"
        done = {}
        if jl.exists():
            for line in jl.read_text().splitlines():
                if line.strip():
                    x = json.loads(line)
                    if (out / f"{x['key']}__soft.npy").exists():
                        done[x["key"]] = x
        print(f"variant {vname}: {len(done)} items already done, {len(rows) - len(done)} to run", flush=True)
        jf = open(jl, "a")
        fg_cache = {}
        records, n_overlay = [], {}
        for r in rows:
            if r["key"] in done:
                records.append(done[r["key"]])
                n_overlay[r["dataset"]] = n_overlay.get(r["dataset"], 0) + 1
                continue
            gt = binary_gt(r["annotation"])
            H, W = gt.shape
            pil = Image.open(r["image"]).convert("RGB")
            text = text_for(r["classname"], form)
            if vocab == "single":
                if text not in fg_cache:
                    fg_cache[text] = pipe.nerve.classifier([text], cfg.semantic_templates)
                fg_text, c = fg_cache[text], 0
            else:
                fg_text, c = fg20, class_index(r["classname"])
            ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]
            img = pipe.nerve.preprocess(pil)[None].to(dev)

            torch.manual_seed(args.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            ts = time.time()
            with torch.no_grad():
                pred_mask, final_score = pipe(ori, img, fg_text.to(dev), bg_text.to(dev))
                # main_ovs.py:61-63, verbatim
                pred_mask = F.interpolate(pred_mask[None], size=(H, W), mode="bilinear")[0]
                soft_t = pred_mask[c].clone()
                pm = pred_mask / (pred_mask.amax((-1, -2), keepdim=True) + 1e-5) * final_score[..., None, None]
                _, arg_own = cam_to_label(pm[None].clone(), cls_label=final_score[None], bkg_thre=cfg.bkg_thre,
                                          cls_thre=cfg.score_threshold, is_normalize=cfg.is_normalize)
                # class given (score gate disabled): NERVE's rule on the queried class alone,
                # argmax of {(1-cam)^2 background, cam} with cam min-max normalised (i.e. cam > 0.382)
                _, arg_nogate = cam_to_label(pm[c:c + 1][None].clone(), cls_label=torch.ones(1, 1, device=dev),
                                             bkg_thre=cfg.bkg_thre, cls_thre=0.0, is_normalize=cfg.is_normalize)
            torch.cuda.synchronize()
            runtime = time.time() - ts
            peak = torch.cuda.max_memory_allocated() / 2**20
            peak_res = torch.cuda.max_memory_reserved() / 2**20

            soft = soft_t.float().cpu().numpy()
            is_nan = bool(~np.isfinite(soft).all())
            is_const = bool(np.nanmax(soft) - np.nanmin(soft) < 1e-6) if not np.isnan(soft).all() else True
            stem = r["key"]
            np.save(out / f"{stem}__soft.npy", soft.astype(np.float16))
            pred05 = minmax(soft) > 0.5
            own = arg_own[0].cpu().numpy() == c + 1
            nog = arg_nogate[0].cpu().numpy() == 1
            for tag, m in (("mask_t0.5", pred05), ("mask_nerve_own", own), ("mask_nerve_nogate", nog)):
                Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{stem}__{tag}.png")
            k = n_overlay.get(r["dataset"], 0)
            if (r["dataset"] == "voc_sim" and k < args.overlays - 1) or (r["dataset"] == "coco_cap" and k < 1):
                overlay(pil, minmax(soft), pred05, gt, out / f"{stem}__overlay.png")
            n_overlay[r["dataset"]] = k + 1
            iu = lambda m: [int((m & gt).sum()), int((m | gt).sum())]  # noqa: E731
            rec = {"key": stem, "dataset": r["dataset"], "classname": r["classname"], "query_text": text,
                   "final_score": float(final_score[c]), "passes_score_gate": bool(final_score[c] > cfg.score_threshold),
                   "own_argmax_classes": sorted(int(v) for v in arg_own[0].unique().cpu()), "soft_min": float(soft.min()), "soft_max": float(soft.max()),
                   "nonfinite": is_nan, "constant": is_const, "runtime_s": runtime, "peak_alloc_MiB": peak, "peak_reserved_MiB": peak_res,
                   "iu_t0.5": iu(pred05), "iu_nerve_own": iu(own), "iu_nerve_nogate": iu(nog),
                   "image": r["image"], "annotation": r["annotation"]}
            records.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            print(vname, stem, text, f"score={rec['final_score']:.3f}", f"t={runtime:.2f}s", f"peak={peak:.0f}MiB",
                  "IoU0.5=%.3f own=%.3f nogate=%.3f" % tuple(a / max(b, 1) for a, b in
                                                            (rec["iu_t0.5"], rec["iu_nerve_own"],
                                                             rec["iu_nerve_nogate"])), flush=True)

        jf.close()
        summary = {}
        for ds in sorted({x["dataset"] for x in records}):  # per benchmark only (no pooled row)
            sub = [x for x in records if x["dataset"] == ds]
            summary[ds] = {"n_items": len(sub), "n_nonfinite": sum(bool(x.get("nonfinite")) for x in sub),
                           "n_constant": sum(bool(x.get("constant")) for x in sub),
                           "n_fail_score_gate": sum(not x["passes_score_gate"] for x in sub)}
            for key in ("iu_t0.5", "iu_nerve_own", "iu_nerve_nogate"):
                by = {}
                for x in records:
                    if ds in ("all", x["dataset"]):
                        a = by.setdefault(x["classname"], [0, 0])
                        a[0] += x[key][0]
                        a[1] += x[key][1]
                ious = {c: i / u for c, (i, u) in by.items() if u}
                summary[ds][key] = {"class_macro_iou": float(np.mean(list(ious.values()))),
                                    "classes": len(ious), "per_class": ious}
        rt = [x["runtime_s"] for x in records]  # includes items from earlier (resumed) jobs
        timing = {"runtime_s_mean": float(np.mean(rt)), "runtime_s_mean_excl_first": float(np.mean(rt[1:])),
                  "runtime_s_max": float(np.max(rt)),
                  "peak_alloc_MiB_max": float(np.max([x["peak_alloc_MiB"] for x in records])),
                  "peak_reserved_MiB_max": float(np.max([x["peak_reserved_MiB"] for x in records]))}
        res = {"label": args.label, "meta": meta, "summary": summary,
               "timing": timing, "records": records}
        (out / args.summary_name).write_text(json.dumps(res, indent=1))
        print("SUMMARY", vname, json.dumps({ds: {k: (round(v["class_macro_iou"], 4) if isinstance(v, dict) else v)
                                                 for k, v in s.items()}
                                            for ds, s in summary.items()}), json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
