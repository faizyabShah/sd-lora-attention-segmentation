#!/usr/bin/env python3
"""Idea 2: dose-response segmentation on VOC-Sim / COCO-Cap (pre-registered in runs/02_dose_response/PREREG.md).

Per image (nothing but per-image score counts is written; no maps are stored):
  1. frozen live capture -> BRACE (A3 seed -> V3 -> R2) counts, A3 seed, image, truth
  2. base regeneration with a reweighting cross-attention processor at w = 1 (records 16x16 token attention)
  3. dose branches w in {0.5, 2.0}, from denoising step s in {0, 10}
  4. response maps in latent and pixel space -> counts on the tau grid, raw and + R2

run     (GPU)  python scripts/idea2_dose.py run --dataset voc|coco [--shard i --num-shards n] [--max-images k]
report  (CPU)  python scripts/idea2_dose.py report
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/02_dose_response"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DOSES = (0.5, 2.0)
STARTS = (0, 10)
CANDS = ("lat_s0", "lat_s10", "pix_s0", "pix_s10")
TAU = 0.55
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


# ------------------------------------------------------------------ reweighting cross-attention processor
class Control:
    def __init__(self):
        self.w, self.start, self.step, self.tokens, self.record = 1.0, 0, 0, (), False
        self.tok16 = None; self.n16 = 0


class DoseCrossProc:
    """diffusers AttnProcessor (explicit probabilities) with P2P-style token reweighting on the conditional half."""

    def __init__(self, ctl):
        self.ctl = ctl

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query); key = attn.head_to_batch_dim(key); value = attn.head_to_batch_dim(value)
        probs = attn.get_attention_scores(query, key, attention_mask)
        ctl, H = self.ctl, attn.heads
        if probs.shape[0] == 2 * H:                       # CFG batch: [uncond, cond]
            if ctl.record and probs.shape[1] == 256:
                m = probs[H:].mean(0).float()             # [256, 77]
                ctl.tok16 = m if ctl.tok16 is None else ctl.tok16 + m; ctl.n16 += 1
            if ctl.w != 1.0 and ctl.step >= ctl.start and ctl.tokens:
                probs = probs.clone(); probs[H:, :, list(ctl.tokens)] *= ctl.w
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states); hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install(pipe, ctl):
    saved = dict(pipe.unet.attn_processors); new = {}
    for name, p in saved.items():
        orig = getattr(p, "original", p)                  # unwrap the recording processor of the live capture
        new[name] = DoseCrossProc(ctl) if name.endswith("attn2.processor") else orig
    pipe.unet.set_attn_processor(new)
    return saved


def generate(pipe, row, ctl):
    """Same settings as src/attention_capture/live.py: seed, CFG 7.5, 512x512, fp32 UNet. Returns final latents."""
    seed = int(row["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    pe, ne = pipe.encode_prompt(prompt=row["prompt"], device=pipe.device, num_images_per_prompt=1,
                                do_classifier_free_guidance=True, negative_prompt=None)
    noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=pipe.device, dtype=pe.dtype)
    ctl.step = 0

    def on_end(_p, step, _t, kw):
        ctl.step = step + 1
        return kw
    out = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=noise, height=512, width=512,
               num_inference_steps=int(row["num_inferece_steps"]), guidance_scale=7.5, output_type="latent",
               callback_on_step_end=on_end, callback_on_step_end_tensor_inputs=["latents"])
    return out.images.float()                             # [1,4,64,64]


@torch.no_grad()
def decode(pipe, z):
    x = pipe.vae.decode(z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor).sample
    return ((x.float().clamp(-1, 1) + 1) / 2)[0]          # [3,512,512] in [0,1]


def choose_tokens(pipe, row, dataset, seed64, tok16):
    ids = pipe.tokenizer(row["prompt"], padding="max_length", max_length=77, truncation=True)["input_ids"]
    cls_ids = pipe.tokenizer(row["classname"], add_special_tokens=False)["input_ids"]
    for s in range(77 - len(cls_ids) + 1):
        if ids[s:s + len(cls_ids)] == cls_ids:
            return tuple(range(s, s + len(cls_ids))), "class_word"
    if dataset == "voc":
        raise ValueError(f"class word absent from VOC-Sim prompt: {row['prompt']}")
    from src.segmentation import sg24_ca
    n = len(pipe.tokenizer(row["prompt"])["input_ids"])
    try:
        cand = [p for p in sg24_ca.content_positions(pipe.tokenizer, row["prompt"], ()) if 0 < p < n - 1]
    except ValueError:
        cand = []
    if not cand:
        cand = list(range(1, n - 1))
    s16 = F.adaptive_avg_pool2d(seed64[None, None].float(), 16).reshape(-1)
    s16 = s16 / (s16.norm() + 1e-12)
    T = tok16.reshape(256, 77)[:, cand]; T = T / (T.norm(dim=0, keepdim=True) + 1e-12)
    j = int(torch.argmax(s16 @ T))
    return (cand[j],), "fallback"


# ------------------------------------------------------------------ run (GPU)
def run(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:
        rows = rows[: a.max_images]
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]
    out = CACHE / f"idea2_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
    crf = lambda img, x: np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in cc.THR_CRF]).astype(bool)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        # 1. frozen live capture -> BRACE
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed64 = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, seed64)[0]["V3"].float().cpu().numpy()
        valid = np.ones_like(truth); other = np.zeros_like(truth)
        res = {"cls": cls, "area": float(truth.mean())}
        xb = st.up(st.mm(v3), truth.shape)
        res["brace|raw"] = cc.counts(xb[None] >= cc.THR[:, None, None], truth, valid, other)
        res["brace+R2|raw"] = cc.counts(crf(img, xb), truth, valid, other)
        t1 = time.perf_counter()
        # 2-3. reweighting branches (processors restored afterwards)
        ctl = Control(); saved = install(pipe, ctl)
        try:
            ctl.record = True; z1 = generate(pipe, row, ctl); ctl.record = False
            tokens, how = choose_tokens(pipe, row, a.dataset, seed64, ctl.tok16 / max(ctl.n16, 1))
            ctl.tokens = tokens
            x1 = decode(pipe, z1); zs, xs = {}, {}
            for s in STARTS:
                for w in DOSES:
                    ctl.w, ctl.start = w, s
                    zs[(s, w)] = generate(pipe, row, ctl); xs[(s, w)] = decode(pipe, zs[(s, w)])
            ctl.w = 1.0
        finally:
            pipe.unet.set_attn_processor(saved)
        base_img = (x1.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        img_diff = float(np.abs(base_img.astype(int) - img.astype(int)).mean())
        t2 = time.perf_counter()
        # 4. response maps -> counts
        for s in STARTS:
            lat = torch.stack([(zs[(s, w)] - z1)[0].norm(dim=0) / abs(math.log(w)) for w in DOSES]).mean(0)
            pix = torch.stack([(xs[(s, w)] - x1).norm(dim=0) / abs(math.log(w)) for w in DOSES]).mean(0)
            maps = {f"lat_s{s}": st.up(st.mm(lat.cpu().numpy()), truth.shape),
                    f"pix_s{s}": st.mm(F.interpolate(pix[None, None], size=truth.shape, mode="bilinear",
                                                     align_corners=False)[0, 0].cpu().numpy())}
            for k, x in maps.items():
                res[f"{k}|raw"] = cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other)
                res[f"{k}+R2|raw"] = cc.counts(crf(base_img, x), truth, valid, other)
        res.update(tokens=np.array(tokens), token_rule=how, img_meandiff=img_diff,
                   sec_live=t1 - t0, sec_branches=t2 - t1, sec_total=time.perf_counter() - t0)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[dose:{a.dataset}] {iid} {cls} tokens {tokens} ({how}) base-vs-live |Δpix| {img_diff:.2f} "
              f"live {t1 - t0:.1f}s branches {t2 - t1:.1f}s total {time.perf_counter() - t0:.1f}s", flush=True)
        del zs, xs, z1, x1; torch.cuda.empty_cache()


# ------------------------------------------------------------------ report (CPU)
def report(a):
    j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    S = {}
    for ds in ("voc", "coco"):
        d = CACHE / f"idea2_{ds}"
        if d.exists() and any(d.glob("*.npz")):
            S[ds] = cc.load_gen(d)
            Z = [np.load(f) for f in sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp"))]
            S[ds]["rule"] = np.array([str(z["token_rule"]) for z in Z])
            S[ds]["sec"] = np.array([float(z["sec_total"]) for z in Z])
            S[ds]["imgdiff"] = np.array([float(z["img_meandiff"]) for z in Z])
    one = lambda ds: np.ones(len(S[ds]["grp"]), bool)
    oof = lambda ds, k: cc.oof(S[ds], S[ds][f"{k}+R2|raw"])
    L = ["# Idea 2: dose–response segmentation (pre-registered in `PREREG.md`)\n",
         "Response of each pixel to turning the target token's cross-attention weight to 0.5× and 2× during "
         "generation (same seed), in final-latent or decoded-pixel space, from step 0 or 10. BRACE = frozen V3 + R2 "
         "from the same process. Class-macro mIoU; OOF = τ chosen on 4 folds (crc32 % 5); Δ with image-bootstrap 95% CI.\n"]
    for ds in S:
        n = len(S[ds]["grp"]); ref = oof(ds, "brace")
        L += [f"## {NAME[ds]} ({n} images; base-vs-live mean |Δpixel| {S[ds]['imgdiff'].mean():.2f}/255; "
              f"{S[ds]['sec'].mean():.1f} s/image)\n",
              "| map | OOF + R2 | Δ vs BRACE OOF + R2 | prec / rec | OOF raw | @0.55 + R2 |", "|---|---:|---|---|---:|---:|"]
        for k in ("brace",) + CANDS:
            m, P, _ = oof(ds, k)
            if k == "brace":
                d = "—"
            else:
                lo, hi = cc.boot(S[ds], P, ref[1]); d = f"{m - ref[0]:+.3f} ({lo:+.3f}…{hi:+.3f})"
            pr = cc.prf(P); raw = cc.oof(S[ds], S[ds][f"{k}|raw"])[0]
            f55 = cc.miou(S[ds][f"{k}+R2|raw"][:, j55][:, None], S[ds]["cls"], one(ds))[0]
            L.append(f"| {'BRACE (frozen)' if k == 'brace' else k} | {m:.3f} | {d} | {pr[0]:.3f} / {pr[1]:.3f} | {raw:.3f} | {f55:.3f} |")
        if ds == "coco":
            L.append("")
            for r in ("class_word", "fallback"):
                sel = S[ds]["rule"] == r
                L.append(f"- token rule **{r}**: {sel.sum()} images ({sel.mean() * 100:.0f}%); OOF + R2 on this subset: " + ", ".join(
                    f"{k} {cc.miou(oof(ds, k)[1][:, None], S[ds]['cls'], sel)[0]:.3f}" for k in ("brace",) + CANDS))
        L.append("\nPer class, OOF + R2 (BRACE / " + " / ".join(CANDS) + "): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(oof(ds, k)[1][:, None], S[ds]['cls'], S[ds]['cls'] == c)[0]:.2f}" for k in ("brace",) + CANDS)
            for c in np.unique(S[ds]["cls"])) + "\n")
    res = {}
    if "voc" in S and "coco" in S and len(S["voc"]["grp"]) == 490 and len(S["coco"]["grp"]) == 707:
        ch = max(CANDS, key=lambda k: oof("voc", k)[0])
        m, P, _ = oof("coco", ch); rb = oof("coco", "brace"); lo, hi = cc.boot(S["coco"], P, rb[1])
        verdict = "dose–response beats BRACE" if lo > 0 else "no gain over BRACE"
        L.append(f"## Registered test\n\nChosen on VOC-Sim: **{ch}** (OOF + R2 {oof('voc', ch)[0]:.3f} vs BRACE "
                 f"{oof('voc', 'brace')[0]:.3f}). COCO-Cap: {ch} {m:.3f} vs BRACE {rb[0]:.3f}, Δ {m - rb[0]:+.3f} "
                 f"(CI {lo:+.3f}…{hi:+.3f}) → **{verdict}**.")
        res = {"chosen": ch, "coco_delta": m - rb[0], "ci": [lo, hi], "verdict": verdict}
    else:
        L.append("## Registered test\n\nNot evaluated: both full sets (490 / 707) are required.")
    for ds in S:
        res[ds] = {k: oof(ds, k)[0] for k in ("brace",) + CANDS} | {"n": len(S[ds]["grp"])}
    txt = "\n".join(L) + "\n"; print(txt)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "result.md").write_text(txt); (RUN / "result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int)
    sub.add_parser("report")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)
