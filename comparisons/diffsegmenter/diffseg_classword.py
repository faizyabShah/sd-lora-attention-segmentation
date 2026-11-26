#!/usr/bin/env python3
"""DiffSegmenter (Wang et al., TIP 2025; github.com/vcg-team/diffsegmenter) under the
class-word binary protocol (JC review workplan #3, side N).

Input: a finished benchmark image + the queried class word.  Output per (image, class):
DiffSegmenter's per-class soft map at mask resolution (float16 .npy, before the per-class
min-max x255 and before the background-score argmax), a binary mask at 0.5 of the min-max
map, DiffSegmenter's own binarisation (background-score rule), runtime and peak GPU memory.

Used unchanged (imported from the repo copy n_side/diffseg/repo, not copied):
  * open_vocabulary/voc12/ptp_stable_best.py: AttentionStore, aggregate_all_attention,
    run_and_display, encode_imgs, constants (GUIDANCE_SCALE, LOW_RESOURCE, MAX_NUM_WORDS)
  * ptp_utils.py: text2image_ldm_stable (single UNet pass, Compel text embedding),
    register_attention_control (attention capture)
Transcribed from ptp_stable_best.py:467-551 (the __main__ loop body) with these settings
unchanged: t = [100] (:451), resolution weights [0.3, 0.5, 0.1, 0.1] for 8/16/32/64 (:452),
per-resolution min-max of the cross map then weighted sum (:514-524), refinement by the
64x64 self-attention (mean over all 64x64 layers and heads) matmul the cross map (:525-528),
bilinear resize to the mask size (:540), per-class min-max (:543).  fp16, DDIM scheduler.
Adapted (this file):
  * model id: "runwayml/stable-diffusion-v1-5" -> cached "stable-diffusion-v1-5/stable-diffusion-v1-5"
  * input: our manifest image instead of the VOC12 dataloader (ToTensor, clamp, bilinear 512
    as in mydatasets.build_transform + :471-478)
  * label source: the queried class word (manifest name) instead of the BLIP/CLIP-predicted
    label json (:459, :489)
  * BLIP caption tail: the original prompt is f"a photograph of {label}" + "++" + the tail of
    a BLIP-large conditional caption (:495-502).  --caption-mode selects:
      none  -> tail = "" (BLIP-large weights not available offline), prompt "a photograph of {c}++"
      plain -> prompt "a photograph of {c}" (no Compel up-weighting, no tail)
      blip  -> faithful, needs --blip-path (Salesforce/blip-image-captioning-large)
  * token selection: original uses CLIP token index 4 (the first token after "a photograph of"),
    or [4, 5] if the 5th space-separated word ends in "ing" (:515-518).  For multi-token class
    words (dining-table, potted-plant) index 4 is only the first sub-token; we average all
    non-punctuation tokens of the class word ("cls_all", primary) and also save the
    original index-4 rule ("tok4") for comparison.  For single-token classes both are identical.
  * space-separated labels (COCO-Cap "dining table", "potted plant"): Compel's "word++" would
    weight only the last word, so the label is wrapped as "(dining table)++" to weight the whole
    label, as "plane++" does for the original one-word labels.
  * resumable: a per-item <stem>.json record is written last; items with a record are skipped.
  * seed: the original never seeds (torch.randn noise :482 and VAE posterior.sample() :315 use
    the global RNG); we call torch.manual_seed(seed) before each image.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo"
MODEL_KEY = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TIMES = [100]                    # ptp_stable_best.py:451
WEIGHT = [0.3, 0.5, 0.1, 0.1]    # ptp_stable_best.py:452
BG_SWEEP = [i / 100 for i in range(30, 75)]  # evaluation_voc12.py:199-201 (start=30, end=75)


def load_ds_module():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("ptp_stable_best", REPO / "open_vocabulary/voc12/ptp_stable_best.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top level only defines classes/functions/constants
    return mod


def binary(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return (a.sum(axis=-1) if a.ndim == 3 else a) != 0


def minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def class_token_positions(tokenizer, prefix: str, classname: str) -> list[int]:
    """Positions (with BOS at 0) of the class word's tokens in prefix + ' ' + classname."""
    n_prefix = len(tokenizer(prefix)["input_ids"]) - 2
    cls_ids = tokenizer(classname)["input_ids"][1:-1]
    pos = []
    for j, tid in enumerate(cls_ids):
        tok = tokenizer.decode([tid]).strip()
        if tok and any(ch.isalnum() for ch in tok):
            pos.append(1 + n_prefix + j)
    return pos


def overlay(image: Image.Image, soft: np.ndarray, pred: np.ndarray, gt: np.ndarray, path: Path, title: str) -> None:
    img = np.asarray(image.convert("RGB").resize(gt.shape[::-1]), dtype=np.float32)
    heat = np.stack([soft * 255, soft * 80, (1 - soft) * 120], -1)
    a = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
    err = img.copy() * 0.4
    err[pred & gt] = [0, 200, 0]       # TP green
    err[pred & ~gt] = [220, 0, 0]      # FP red
    err[~pred & gt] = [0, 90, 255]     # FN blue
    canvas = np.concatenate([img.astype(np.uint8), a, err.astype(np.uint8)], 1)
    im = Image.fromarray(canvas)
    try:
        from PIL import ImageDraw
        ImageDraw.Draw(im).text((6, 6), title, fill=(255, 255, 0))
    except Exception:
        pass
    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", default=["/home/sidna/TRDP/external/n_side/outputs/smoke_manifest.csv"])
    ap.add_argument("--label", default="smoke, fixed threshold, not a result")
    ap.add_argument("--out", default=str(HERE / "outputs/smoke_sd15"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caption-mode", default="none", choices=["none", "plain", "blip"])
    ap.add_argument("--blip-path", default="")
    ap.add_argument("--overlays", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import csv
    rows = [r for m in args.manifest for r in csv.DictReader(open(m))]
    if args.limit:
        rows = rows[: args.limit]

    ds = load_ds_module()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    import diffusers, transformers, compel

    device = torch.device("cuda")
    dtype = torch.float16
    ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL_KEY, local_files_only=True, torch_dtype=dtype).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_pretrained(MODEL_KEY, subfolder="scheduler", local_files_only=True)
    tokenizer = ldm_stable.tokenizer
    vae = ldm_stable.vae
    # module globals the imported functions/classes read
    ds.ldm_stable, ds.tokenizer, ds.device = ldm_stable, tokenizer, device
    controller = ds.AttentionStore()
    g_cpu = torch.Generator(4307)  # ptp_stable_best.py:450 (passed through, unused for the single pass)

    blip = None
    if args.caption_mode == "blip":
        from transformers import BlipProcessor, BlipForConditionalGeneration
        blip = (BlipProcessor.from_pretrained(args.blip_path),
                BlipForConditionalGeneration.from_pretrained(args.blip_path).to(device))

    out = Path(args.out) / args.caption_mode
    out.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    records = []
    with torch.no_grad():
        for r in rows:
            cls = r["classname"]
            stem = f"{r['dataset']}__{r['key']}__{cls.replace(' ', '_')}"
            rec_path = out / r["dataset"] / f"{stem}.json"
            if rec_path.exists():  # resumable: the per-item record is written last
                records.append(json.loads(rec_path.read_text()))
                continue
            gt = binary(r["annotation"])
            h, w = gt.shape
            image = Image.open(r["image"]).convert("RGB")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # :467-480
            orig = to_tensor(image).unsqueeze(0).clamp(0, 1)
            rgb_512 = F.interpolate(orig, (512, 512), mode="bilinear", align_corners=False).to(device).half()
            input_latent = ds.encode_imgs(rgb_512, vae).to(device)
            noise = torch.randn([1, 4, 64, 64]).to(device).half()  # :482
            # :495-502
            base = f"a photograph of {cls}"
            if args.caption_mode == "plain":
                prompt = [base]
            else:
                tail = ""
                if blip is not None:
                    inputs = blip[0](image, base, return_tensors="pt").to(device)
                    o = blip[1].generate(**inputs)
                    tail = blip[0].decode(o[0], skip_special_tokens=True)[len(base):]
                # "word++" weights only the preceding word; for space-separated labels
                # ("dining table") wrap the label so the whole label is weighted, as for one-word labels
                lab = f"({cls})" if " " in cls else cls
                prompt = [f"a photograph of {lab}++" + tail]
            # token rule :515-518 (original) and all class tokens (adapted, primary)
            words = prompt[0].split(" ")
            tok4 = [4, 5] if (len(words) > 4 and words[4].endswith("ing")) else [4]
            cls_pos = class_token_positions(tokenizer, "a photograph of", cls)
            if len(words) > 4 and words[4].endswith("ing"):
                cls_all = cls_pos + [cls_pos[-1] + 1]
            else:
                cls_all = cls_pos
            maps = {}
            for t in TIMES:  # :506-534
                controller.reset()
                latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=device)).half()
                ds.run_and_display(prompt, controller, latent=latents_noisy, run_baseline=False, generator=g_cpu, t=t, noise_sample_num=1)
                cross = ds.aggregate_all_attention(prompt, controller, ("up", "mid", "down"), True, 0)
                selfm = ds.aggregate_all_attention(prompt, controller, ("up", "mid", "down"), False, 0)
                self_att = selfm[3].view(64 * 64, 64 * 64).float()
                for name, toks in (("cls_all", cls_all), ("tok4", tok4)):
                    out_atts = []
                    for idx, res in enumerate([8, 16, 32, 64]):
                        ca = cross[idx][:, :, toks].mean(2).view(res, res).float()
                        if res != 64:
                            ca = F.interpolate(ca[None, None], size=(64, 64), mode="bilinear", align_corners=False).squeeze()
                        ca = (ca - ca.min()) / (ca.max() - ca.min())
                        out_atts.append(ca * WEIGHT[idx])
                    cmap = torch.stack(out_atts).sum(0).view(64 * 64, 1)
                    att_map = torch.matmul(self_att, cmap).view(64, 64)
                    maps.setdefault(name, []).append(att_map)
            soft = {k: F.interpolate(torch.stack(v).mean(0)[None, None], size=(h, w), mode="bilinear",
                                     align_corners=False)[0, 0].cpu().numpy() for k, v in maps.items()}  # :537-540
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20
            peak_res = torch.cuda.max_memory_reserved() / 2**20

            d = out / r["dataset"]
            d.mkdir(exist_ok=True)
            rec = {k: r[k] for k in ("key", "dataset", "classname", "image_source")}
            rec.update({"prompt": prompt[0], "tok4_positions": tok4, "cls_all_positions": cls_all,
                        "tokens": [tokenizer.decode([i]) for i in tokenizer(prompt[0].replace("++", "").replace("(", "").replace(")", ""))["input_ids"]],
                        "runtime_s": round(dt, 3), "peak_alloc_MiB": round(peak, 1), "peak_reserved_MiB": round(peak_res, 1)})
            for name, s in soft.items():
                if name == "tok4" and cls_all == tok4:
                    continue
                sfx = "" if name == "cls_all" else "__tok4"
                np.save(d / f"{stem}{sfx}.npy", s.astype(np.float16))
                mm = minmax(s)  # == DiffSegmenter's .mat value / 255 (:543)
                pred = mm > 0.5
                Image.fromarray(pred.astype(np.uint8) * 255).save(d / f"{stem}{sfx}__mask_t0.5.png")
                rec[f"{name}_soft_min"], rec[f"{name}_soft_max"] = float(s.min()), float(s.max())
                rec[f"{name}_finite"] = bool(np.isfinite(s).all())
                rec[f"{name}_constant"] = bool(not float(s.max()) > float(s.min()))
                rec[f"{name}_iu_t0.5"] = [int((pred & gt).sum()), int((pred | gt).sum())]
                # DiffSegmenter's own rule: argmax([bg=thr, map/255]) -> fg iff map/255 > thr (ties go to bg)
                rec[f"{name}_iu_bg_sweep"] = [[int(((mm > t) & gt).sum()), int(((mm > t) | gt).sum())] for t in BG_SWEEP]
            records.append(rec)
            rec_path.write_text(json.dumps(rec))
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("bg_sweep")}), flush=True)
            n_ds = sum(1 for x in records if x["dataset"] == r["dataset"])
            if (r["dataset"] == "voc_sim" and n_ds <= args.overlays - 1) or (r["dataset"] == "coco_cap" and n_ds == 1):
                mm = minmax(soft["cls_all"])
                overlay(image, mm, mm > 0.5, gt, d / f"{stem}__overlay.png", f"{cls} | {prompt[0]} | t=100 | min-max>0.5")

    # smoke class-macro IoU (pooled per class, macro over classes), per dataset
    def macro(key, ds_name, sweep_idx=None):
        by = {}
        for x in records:
            if x["dataset"] != ds_name:
                continue
            k = key if key in x else "cls_all" + key[len("tok4"):]
            v = x[k] if sweep_idx is None else x[k][sweep_idx]
            a = by.setdefault(x["classname"], [0, 0])
            a[0] += v[0]
            a[1] += v[1]
        ious = {c: i / u for c, (i, u) in by.items() if u}
        return {"class_macro_iou": float(np.mean(list(ious.values()))), "classes": len(ious), "per_class": ious}

    summary, own = {}, {}
    for ds_name in sorted({x["dataset"] for x in records}):
        summary[ds_name] = {k: macro(k, ds_name) for k in ("cls_all_iu_t0.5", "tok4_iu_t0.5")}
    # DiffSegmenter's own binarisation: background score chosen by its sweep on VOC-Sim (dev), applied to COCO-Cap
    for name in ("cls_all", "tok4"):
        curve = [macro(f"{name}_iu_bg_sweep", "voc_sim", i)["class_macro_iou"] for i in range(len(BG_SWEEP))]
        bi = int(np.argmax(curve))
        own[name] = {"bg_threshold_from_voc_sim_smoke": BG_SWEEP[bi],
                     "voc_sim": macro(f"{name}_iu_bg_sweep", "voc_sim", bi),
                     "coco_cap": macro(f"{name}_iu_bg_sweep", "coco_cap", bi) if any(x["dataset"] == "coco_cap" for x in records) else None,
                     "voc_sim_curve": dict(zip(BG_SWEEP, [round(c, 4) for c in curve]))}
    meta = {"method": "DiffSegmenter", "repo": "github.com/vcg-team/diffsegmenter @ b3a8686",
            "script": "open_vocabulary/voc12/ptp_stable_best.py", "model": MODEL_KEY, "dtype": "float16",
            "scheduler": "DDIMScheduler (add_noise only; one UNet pass)", "timesteps": TIMES,
            "res_weights_8_16_32_64": WEIGHT, "self_attention_refine": "64x64 SA (mean over layers/heads) @ cross",
            "crf": "none in repo (no CRF in ptp_stable_best.py or evaluation_voc12.py)",
            "own_binarisation": "argmax over [bg score, minmax map] ; bg score swept 0.30..0.74 on the eval set (evaluation_voc12.py:199-206)",
            "caption_mode": args.caption_mode, "noise_seed": args.seed,
            "gpu": torch.cuda.get_device_name(0), "node": platform.node(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "compel": getattr(compel, "__version__", "2.0.2"),
            "threshold_rule": "0.5 on min-max-normalised soft map (smoke only, not a result)"}
    timing = {"runtime_s_mean": float(np.mean([x["runtime_s"] for x in records])),
              "runtime_s_mean_excl_first": float(np.mean([x["runtime_s"] for x in records[1:]])) if len(records) > 1 else None,
              "runtime_s_max": float(np.max([x["runtime_s"] for x in records])),
              "peak_alloc_MiB_max": float(np.max([x["peak_alloc_MiB"] for x in records])),
              "peak_reserved_MiB_max": float(np.max([x["peak_reserved_MiB"] for x in records]))}
    qc = {d: {"items": sum(x["dataset"] == d for x in records),
              "nonfinite_maps": sum(x["dataset"] == d and not x["cls_all_finite"] for x in records),
              "constant_maps": sum(x["dataset"] == d and x.get("cls_all_constant", False) for x in records)}
          for d in sorted({x["dataset"] for x in records})}
    result = {"label": args.label, "qc": qc, "meta": meta, "summary_t0.5": summary,
              "own_binarisation_smoke": own, "timing": timing, "records": records}
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print("SUMMARY", json.dumps({d: {k: round(v["class_macro_iou"], 4) for k, v in s.items()} for d, s in summary.items()}),
          json.dumps({n: {"thr": o["bg_threshold_from_voc_sim_smoke"], "voc": round(o["voc_sim"]["class_macro_iou"], 4),
                          "coco": round(o["coco_cap"]["class_macro_iou"], 4) if o["coco_cap"] else None} for n, o in own.items()}),
          json.dumps(timing), json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
