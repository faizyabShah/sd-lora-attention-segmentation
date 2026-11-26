#!/usr/bin/env python3
"""JC #8: per-image runtime and peak GPU memory of BRACE (frozen) and the fine-only reference, by stage.

Uses side F's committed frozen runtime unchanged (`scripts/sg27_runtime.py`: SD 1.5 fp32, the manifest's 30 steps,
CFG 7.5, 512 px; `src/segmentation/sg_combined_v2.py`; DenseCRF R2 = `src/segmentation/ovam_full.apply_ovam_densecrf`),
on runs/20's corrected prompts (`scripts/jc_newrerun.rows`), safety checker off. Fixed subset: `rows(ds, max_images=50)`
(every k-th image of the sorted manifest), 50 VOC-Sim + 50 COCO-Cap. One warm-up image is run and discarded.

Stages per image (CUDA-synchronised wall clock; peak = torch.cuda.max_memory_allocated within the stage):
  gen_plain    generation alone: same pipe, same seed, prompt embeddings and latents, no recorder (run first,
               before any recorder is installed)
  gen_capture  the frozen live generation with attention recording and seed extraction (`live_maps` entry),
               minus the runtime's extra diagnostic maps (sg27_fusion.base_paths / proposed_maps, timed separately
               as `runtime_extras`; BRACE does not use them)
  fine_map     the fine-only reference map: V0 lines of sg_combined_v2.aggregate (two fine SA passes, 11 calls)
  hier_map     BRACE's map: V3 lines of aggregate (16->32->64 with the one-sided cap at 16->32, 11 calls)
  aggregate    F's aggregate() as committed (computes all seven variants; reported for reference)
  crf_*        CPU: min-max, bilinear upsampling to 512, threshold (BRACE 0.55, fine-only 0.425), DenseCRF R2
The V0 / V3 re-implementations are checked against aggregate()'s output on every image (max |diff| reported).
"""
from __future__ import annotations

import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "comparisons/official"):
    sys.path.insert(0, str(p))
EXT = Path.home() / "TRDP/external/n_side"
os.environ.setdefault("BRACE_CACHE_DIR", str(EXT / "cost/cache"))
import sg_steps_test as st  # noqa: E402
import jc_newrerun as jn  # noqa: E402
from common import crf_site  # noqa: E402


def _official_patch_rows():
    """comparisons/official/brace_repro.patch_rows, loaded by path (scripts/ has an unrelated brace_repro.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("n_official_brace_repro", ROOT / "comparisons/official/brace_repro.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.patch_rows

_official_patch_rows()(EXT / "brace_repro/ovam_coco_cap.csv")


def sync():
    torch.cuda.synchronize()


class Stage:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __enter__(self):
        sync(); torch.cuda.reset_peak_memory_stats(); self.t = time.perf_counter(); return self

    def __exit__(self, *exc):
        sync()
        self.rec[self.name] = time.perf_counter() - self.t
        self.rec[self.name + "_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20


def fine_only(sa, seed):
    from src.segmentation.sg25_sa import _apply, _attention
    c64 = seed.float().reshape(-1); s = torch.zeros(4096, device=seed.device)
    for call in range(11):
        h = _attention(sa[64][call], 64); u = _apply(h, c64); s += _apply(h, u).mean(0) / 11
        del h, u
    return s.reshape(64, 64)


def hierarchy(sa, seed):
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
    c64 = seed.float().reshape(-1); c16 = _resize(c64, 64, 16).reshape(-1); s = torch.zeros(4096, device=seed.device)
    for call in range(11):
        h16 = _attention(sa[16][call], 16); h32 = _attention(sa[32][call], 32); h64 = _attention(sa[64][call], 64)
        r16 = _apply(h16, c16).mean(0); r32 = _apply(h32, _resize(r16, 16, 32).reshape(-1)).mean(0)
        z32, _ = cap(r16, r32, 16)
        s += _apply(h64, _resize(z32, 32, 64).reshape(-1)).mean(0) / 11
        del h16, h32, h64
    return s.reshape(64, 64)


@torch.no_grad()
def main(a):
    from scripts import sg27_runtime
    from src.segmentation import sg27_fusion, sg_combined_v2
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    from src.attention_capture.live import _seed_everything
    _import_densecrf(crf_site())
    rows = [(ds, r) for ds in ("voc", "coco") for r in jn.rows(ds, max_images=a.n)]
    rows = [rows[0]] + rows                                   # first entry = warm-up, discarded
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    runtime["recorder_class"].enabled = ("A0", "A3"); pipe.safety_checker = None
    recs = [dict(ds=ds, image_id=r["image_id"]) for ds, r in rows]
    plain_img = []
    for rec, (ds, r) in zip(recs, rows):                      # phase 1: generation alone (no recorder installed yet)
        with Stage(rec, "gen_plain"):
            _seed_everything(int(r["seed"]))
            pe, ne = pipe.encode_prompt(prompt=r["prompt"], device=pipe.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True, negative_prompt=None)
            lat = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=pipe.device, dtype=pe.dtype)
            out = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=lat, height=512, width=512,
                       num_inference_steps=int(r["num_inferece_steps"]), guidance_scale=7.5, output_type="pil")
        plain_img.append(np.asarray(out.images[0].convert("RGB")).copy()); del out, pe, ne, lat
    extras = {"t": 0.0}
    for fn in ("base_paths", "proposed_maps"):               # time the runtime's diagnostic maps (unused by BRACE)
        orig = getattr(sg27_fusion, fn)

        def timed(*x, _o=orig, **k):
            sync(); t = time.perf_counter(); y = _o(*x, **k); sync(); extras["t"] += time.perf_counter() - t; return y
        setattr(sg27_fusion, fn, timed)
    for i, (rec, (ds, r)) in enumerate(zip(recs, rows)):     # phase 2: frozen live pipeline
        extras["t"] = 0.0; cm = sg27_runtime.live_maps(runtime, dict(r), ds)
        with Stage(rec, "gen_capture_total"):
            sample, sa, seed, _b, _p = cm.__enter__()
        try:
            rec["runtime_extras"] = extras["t"]; rec["gen_capture"] = rec["gen_capture_total"] - extras["t"]
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            rec["image_equal_plain"] = bool((img == plain_img[i]).all())
            with Stage(rec, "fine_map"):
                v0 = fine_only(sa, seed)
            with Stage(rec, "hier_map"):
                v3 = hierarchy(sa, seed)
            with Stage(rec, "aggregate"):
                ref, _ = sg_combined_v2.aggregate(sa, seed.float().clone())
            rec["v0_maxdiff"] = float((ref["V0"].float() - v0).abs().max() / ref["V0"].float().abs().max())
            rec["v3_maxdiff"] = float((ref["V3"].float() - v3).abs().max() / ref["V3"].float().abs().max())
            for name, m, thr in (("crf_fine", v0, 0.425), ("crf_brace", v3, 0.55)):
                t = time.perf_counter()
                x = st.up(st.mm(m.cpu().numpy().astype(np.float32)), truth.shape)
                apply_ovam_densecrf(img, x >= thr, crf_site())
                rec[name] = time.perf_counter() - t
            del ref, v0, v3
        finally:
            cm.__exit__(None, None, None)
        print(json.dumps(rec), flush=True)
    import diffusers, transformers
    meta = dict(gpu=torch.cuda.get_device_name(0), node=platform.node(), cpu=platform.processor() or platform.machine(),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                diffusers=diffusers.__version__, transformers=transformers.__version__, python=platform.python_version(),
                driver=os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip(),
                unet_dtype=str(pipe.unet.dtype), n_per_set=a.n, warmup_discarded=1)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "images": recs[1:]}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    main(ap.parse_args())
