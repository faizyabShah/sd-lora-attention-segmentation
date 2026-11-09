#!/usr/bin/env python3
"""Florence-2 captions as SD prompts on real photos (D, exploratory, 2026-11-09).

Question: is the real-photo failure of BRACE (and the quality of the SD-side maps of the CLIP pipeline) a prompt
problem? Two places take a prompt: the DDIM inversion (currently empty) and the attention-capture run
(currently "a photograph of a {cls}"). Four conditions, all on the same photo/class pairs:
  base   inversion ""        capture "a photograph of a {cls}"     (= the stored D pipeline; replay-checked)
  inv    inversion caption   capture "a photograph of a {cls}"     (inversion effect only)
  cap    inversion ""        capture caption-prompt               (capture-prompt effect only)
  both   inversion caption   capture caption-prompt
caption = Florence-2-large "<CAPTION>" of the photo (label-free). caption-prompt = the caption if it already
contains the class word, else "{caption}, a {cls}" (the class token must exist for the A3 seed, as in BRACE).
Per pair, two seeds propagated through the condition's own SA with the frozen functions (seed, P1, V0, V3, C16):
  base  the frozen A3 seed (BRACE),   clip  the SCLIP VOC21 class probability (mm), as in the CLIP pipeline.
Output per condition: <out>/<cond>/<iid>__<k>.npz in the sg_class_comp format, so `sg_class_comp.py score` scores it.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402

CAPS = ROOT / "outputs/segmentation/sg_real_atlas/analysis/florence_captions_D.json"
CONDS = ("base", "inv", "cap", "both")


def caption(a):
    from unittest.mock import patch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports

    def no_flash(path):                       # Florence-2 remote code imports flash_attn unconditionally
        return [i for i in get_imports(path) if i != "flash_attn"]
    with patch("transformers.dynamic_module_utils.get_imports", no_flash):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True,
                                                     attn_implementation="eager", torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    photos = sorted({e["image"] for e in json.loads(st.MANIFEST.read_text())[a.set]})
    out = {}; t0 = time.perf_counter()
    for iid in photos:
        img = Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"); out[iid] = {}
        for task in ("<CAPTION>", "<DETAILED_CAPTION>"):
            x = proc(text=task, images=img, return_tensors="pt").to("cuda", torch.float16)
            ids = model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=128,
                                 num_beams=3, do_sample=False)
            txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
            out[iid][task] = proc.post_process_generation(txt, task=task, image_size=img.size)[task].strip()
    CAPS.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} photos {(time.perf_counter() - t0) / len(out):.2f}s/photo; e.g. {next(iter(out.items()))}")


def cap_prompt(capt, cls, tok):
    # the frozen CA recorder's slow-tokenizer alignment rejects possessives ("person's" -> one BPE piece "'s"),
    # so the CAPTURE prompt drops apostrophes; the inversion keeps the raw caption
    import re
    from src.segmentation import sg24_ca
    capt = capt.replace("'", "").replace("\u2019", "")
    try:
        sg24_ca.content_positions(tok, capt, ())
    except ValueError:
        capt = re.sub(r"[^\w\s,.-]", "", capt)
    ids = tok(capt, add_special_tokens=False)["input_ids"]; cid = tok(cls, add_special_tokens=False)["input_ids"]
    has = any(ids[i:i + len(cid)] == cid for i in range(len(ids) - len(cid) + 1))
    return (capt if has else f"{capt.rstrip('.').rstrip()}, a {cls}"), has


def capture(a):
    import run_sg_real_atlas as ra
    from scripts import sg27_runtime
    caps = json.loads(CAPS.read_text())
    entries = json.loads(st.MANIFEST.read_text())[a.set][a.shard::a.num_shards]
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    pipe.scheduler.set_timesteps(30, device=pipe.device)
    ts_unique = sorted(set(pipe.scheduler.timesteps.tolist()))
    enc = lambda s: pipe.text_encoder(pipe.tokenizer([s], padding="max_length", max_length=77, truncation=True,
                                                     return_tensors="pt").input_ids.to(pipe.device))[0].to(pipe.unet.dtype)
    for c in CONDS:
        (a.out / c).mkdir(parents=True, exist_ok=True)
    scratch = a.out / f".gt_{a.set}_{a.shard}.png"; stored = ROOT.parent / "sg_real_atlas" / a.set
    meta = {}
    for entry in entries:
        iid, ks = entry["image"], entry["classes"]
        if all((a.out / c / f"{iid}__{k}.npz").exists() for c in CONDS for k in ks):
            continue
        t0 = time.perf_counter()
        img, lab, names = ra.load_pair_source(entry)
        Image.fromarray((lab == ks[0]).astype(np.uint8) * 255).save(scratch)
        z0 = ra.encode(pipe, ra.to512(img)); capt = caps[iid]["<CAPTION>"]
        tables = {"": ra.ddim_invert(pipe, z0, ts_unique, enc(""))[0], "caption": ra.ddim_invert(pipe, z0, ts_unique, enc(capt))[0]}
        P21 = np.load(f"/home/shafa/sg_comp/clip_D/{iid}__B16.npz")["prob21_64"]
        for k in ks:
            cls = names[k - 1]; cp, has = cap_prompt(capt, cls, pipe.tokenizer)
            meta[f"{iid}__{k}"] = {"caption": capt, "cap_prompt": cp, "class_in_caption": has}
            for c, (inv, prompt) in {"base": ("", f"a photograph of a {cls}"), "inv": ("caption", f"a photograph of a {cls}"),
                                     "cap": ("", cp), "both": ("caption", cp)}.items():
                out = a.out / c / f"{iid}__{k}.npz"
                if out.exists():
                    continue
                row = {"image_id": f"{iid}/{cls}", "classname": cls, "prompt": prompt, "attribution_prompt": prompt,
                       "seed": "0", "num_inferece_steps": "30", "annotation": str(scratch)}
                with ra.trajectory(ra.table_fn(tables[inv])), sg27_runtime.live_maps(runtime, row, "voc") as (_s, sa, seed, _b, _p):
                    seed = seed.float().clone()
                    clip = cc.norm(torch.from_numpy(P21[k]).float().to(seed.device), "mm")
                    maps = {"base": cc.propagate(sa, seed), "clip": cc.propagate(sa, clip)}
                check = float("nan")
                if c == "base" and (stored / iid / str(k) / "maps.npz").exists():
                    sm = np.load(stored / iid / str(k) / "maps.npz")
                    check = max(float(np.abs(st.mm(maps["base"][i]) - st.mm(sm[f"ddim/{m}"])).max())
                                for i, m in ((0, "A3"), (2, "V0"), (3, "V3"), (4, "C16")))
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez(tmp, **{f"map/{n}": v for n, v in maps.items()}, cls=cls, k=k, check=check, set=a.set,
                         prompt=prompt, inversion=inv, class_in_caption=has)
                os.replace(tmp, out)
            print(f"[cap] {iid}/{k} {cls} in-caption={has} prompt='{cp}'", flush=True)
        del tables; torch.cuda.empty_cache()
        print(f"[cap] {iid} {len(ks)} pairs {time.perf_counter() - t0:.0f}s", flush=True)
    (a.out / f"meta_{a.shard}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("caption"); p.add_argument("--set", default="D")
    p = sub.add_parser("capture"); p.add_argument("--set", default="D")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--out", type=Path, default=Path("/home/shafa/sg_comp/cap_D"))
    a = ap.parse_args()
    {"caption": caption, "capture": capture}[a.cmd](a)
