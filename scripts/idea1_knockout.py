#!/usr/bin/env python3
"""Idea 1: counterfactual knockout segmentation on VOC-Sim / COCO-Cap (pre-registered in runs/01_knockout/PREREG.md).

Per image (nothing but per-image score counts is written; no maps are stored):
  1. frozen live capture -> BRACE (A3 seed -> V3 -> R2) counts, A3 seed, image, truth
  2. base trajectory with a manual PNDM loop equivalent to pipe() (keeps z_i, scheduler state at k, 16x16 token attention)
  3. target tokens (class word, or label-free fallback on COCO-Cap)
  4. knockout from step k in {5, 10}: 'attn' (target-token cross-attention x 0, conditional half) or
     'swap' (target span replaced by "object" in the conditional embedding)
       fin = |z_KO,final - z_base,final|          (branch from z_k)
       ins = sum_{i>=k} |eps_cond,KO(z_i) - eps_cond(z_i)|   (along the base trajectory, same UNet batch)
  5. Gaussian sigma 1 (64 grid) -> bilinear 512 -> robust p1/p99 normalisation -> counts on the tau grid, raw and + R2

selftest (GPU)  python scripts/idea1_knockout.py selftest            manual loop == pipe(); re-encoded ids == encode_prompt
run      (GPU)  python scripts/idea1_knockout.py run --dataset voc|coco [--shard i --num-shards n] [--max-images k]
report   (CPU)  python scripts/idea1_knockout.py report
"""
from __future__ import annotations

import argparse, copy, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
from idea2_dose import choose_tokens  # noqa: E402  (same label-free token rule as idea 2)

RUN = ROOT / "runs/01_knockout"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
KS = (5, 10)
MODES = ("attn", "swap")
MEASURES = ("fin", "ins")
CANDS = tuple(f"{m}_{e}_k{k}" for m in MODES for e in MEASURES for k in KS)
SWAP_WORD = "object"
SIGMA = 1.0
TAU = 0.55
CFG = 7.5
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


# ------------------------------------------------------------------ knockout cross-attention processor
class Control:
    def __init__(self):
        self.tokens, self.ko_rows, self.record = (), (), False
        self.tok16 = None; self.n16 = 0


class KOCrossProc:
    """diffusers AttnProcessor with explicit probabilities. Batch rows in ctl.ko_rows get the target tokens'
    attention probability set to 0 (no renormalisation). Records 16x16 conditional token attention when asked."""

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
        if ctl.record and probs.shape[0] == 2 * H and probs.shape[1] == 256:
            m = probs[H:].mean(0).float()                 # conditional half, [256, 77]
            ctl.tok16 = m if ctl.tok16 is None else ctl.tok16 + m; ctl.n16 += 1
        if ctl.ko_rows and ctl.tokens:
            probs = probs.clone()
            for r in ctl.ko_rows:
                probs[r * H:(r + 1) * H, :, list(ctl.tokens)] = 0.0
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
        new[name] = KOCrossProc(ctl) if name.endswith("attn2.processor") else orig
    pipe.unet.set_attn_processor(new)
    return saved


# ------------------------------------------------------------------ manual generation (== pipe(), no safety checker)
@torch.no_grad()
def embeds(pipe, row):
    seed = int(row["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    pe, ne = pipe.encode_prompt(prompt=row["prompt"], device=pipe.device, num_images_per_prompt=1,
                                do_classifier_free_guidance=True, negative_prompt=None)
    noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=pipe.device, dtype=pe.dtype)
    return pe, ne, noise


@torch.no_grad()
def encode_ids(pipe, ids):
    return pipe.text_encoder(torch.tensor([ids], device=pipe.device))[0].to(pipe.text_encoder.dtype)


def prompt_ids(pipe, row):
    return pipe.tokenizer(row["prompt"], padding="max_length", max_length=77, truncation=True)["input_ids"]


def swap_embeds(pipe, row, tokens):
    ids = prompt_ids(pipe, row); obj = pipe.tokenizer(SWAP_WORD, add_special_tokens=False)["input_ids"]
    assert len(obj) == 1, obj
    new = (ids[:tokens[0]] + obj + ids[tokens[-1] + 1:])[:77]
    new = new + [pipe.tokenizer.pad_token_id] * (77 - len(new))
    return encode_ids(pipe, new)


def new_sched(pipe, steps):
    s = copy.deepcopy(pipe.scheduler); s.set_timesteps(steps, device=pipe.device)
    return s


@torch.no_grad()
def denoise(pipe, ctl, sched, z, i0, ne, pe, ko_rows=(), keep=None):
    """Steps i0..end with CFG, exactly as StableDiffusionPipeline.__call__. keep(i, z, sched) is called before step i."""
    z = z * (sched.init_noise_sigma if i0 == 0 else 1.0)
    enc = torch.cat([ne, pe])
    for i in range(i0, len(sched.timesteps)):
        t = sched.timesteps[i]
        if keep is not None:
            keep(i, z, sched)
        ctl.ko_rows = ko_rows
        eps = pipe.unet(sched.scale_model_input(torch.cat([z] * 2), t), t, encoder_hidden_states=enc, return_dict=False)[0]
        ctl.ko_rows = ()
        eu, ec = eps.chunk(2)
        z = sched.step(eu + CFG * (ec - eu), t, z, return_dict=False)[0]
    return z


@torch.no_grad()
def decode(pipe, z):
    x = pipe.vae.decode(z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor).sample
    return ((x.float().clamp(-1, 1) + 1) / 2)[0]          # [3,512,512] in [0,1]


def to_uint8(x):
    return (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def post(m64, hw):
    """sigma-1 Gaussian on the 64 grid, bilinear up, robust p1/p99 normalisation (pre-registered)."""
    x = st.up(gaussian_filter(np.asarray(m64, np.float32), SIGMA), hw)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ selftest (GPU)
def selftest(a):
    from scripts import sg27_runtime
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]
    row = cc.gen_rows("voc")[0]; steps = int(row["num_inferece_steps"])
    ctl = Control(); saved = install(pipe, ctl)
    try:
        pe, ne, noise = embeds(pipe, row)
        ref = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=noise.clone(), height=512, width=512,
                   num_inference_steps=steps, guidance_scale=CFG, output_type="latent").images.float()
        z = denoise(pipe, ctl, new_sched(pipe, steps), noise.clone(), 0, ne, pe).float()
        d1 = float((z - ref).abs().max())
        d2 = float((encode_ids(pipe, prompt_ids(pipe, row)) - pe).abs().max())
        tokens, how = choose_tokens(pipe, row, "voc", torch.ones(64, 64, device=pipe.device), torch.ones(256, 77, device=pipe.device))
        d3 = float((swap_embeds(pipe, row, tokens) - pe).abs().max())
    finally:
        pipe.unet.set_attn_processor(saved)
    print(f"manual loop vs pipe(): max |Δlatent| {d1:.3g} (expect < 1e-4)")
    print(f"re-encoded prompt ids vs encode_prompt: max |Δ| {d2:.3g} (expect < 1e-5)")
    print(f"swap embedding differs from base: max |Δ| {d3:.3g} (expect > 0.1); tokens {tokens} ({how})")
    ok = d1 < 1e-4 and d2 < 1e-5 and d3 > 0.1
    print("SELFTEST PASS" if ok else "SELFTEST FAIL"); sys.exit(0 if ok else 1)


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
    out = CACHE / f"idea1_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
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
        # 2-4. base trajectory, tokens, knockouts (processors restored afterwards)
        steps = int(row["num_inferece_steps"]); maps64 = {}   # PNDM: len(timesteps) == steps + 1
        ctl = Control(); saved = install(pipe, ctl)
        try:
            pe, ne, noise = embeds(pipe, row)
            zs, snaps = {}, {}

            def keep(i, z, sched):
                zs[i] = z.clone()
                if i in KS:
                    snaps[i] = copy.deepcopy(sched)
            sched = new_sched(pipe, steps)
            ctl.record = True; zb = denoise(pipe, ctl, sched, noise, 0, ne, pe, keep=keep); ctl.record = False
            tokens, how = choose_tokens(pipe, row, a.dataset, seed64, ctl.tok16 / max(ctl.n16, 1))
            ctl.tokens = tokens; ps = swap_embeds(pipe, row, tokens)
            # instantaneous effects along the base trajectory: batch [base, attn-KO, swap], conditional only
            ins = {m: [] for m in MODES}; ts = sched.timesteps
            with torch.no_grad():
                for i in range(min(KS), len(ts)):
                    zi = sched.scale_model_input(zs[i].repeat(3, 1, 1, 1), ts[i])
                    ctl.ko_rows = (1,)
                    e = pipe.unet(zi, ts[i], encoder_hidden_states=torch.cat([pe, pe, ps]), return_dict=False)[0].float()
                    ctl.ko_rows = ()
                    ins["attn"].append((e[1] - e[0]).norm(dim=0)); ins["swap"].append((e[2] - e[0]).norm(dim=0))
            for m in MODES:
                E = torch.stack(ins[m])                   # [steps - min(KS), 64, 64]
                for k in KS:
                    maps64[f"{m}_ins_k{k}"] = E[k - min(KS):].sum(0).cpu().numpy()
            # final-latent effects: branch from z_k with a copy of the scheduler state
            for k in KS:
                za = denoise(pipe, ctl, copy.deepcopy(snaps[k]), zs[k], k, ne, pe, ko_rows=(1,))
                zw = denoise(pipe, ctl, copy.deepcopy(snaps[k]), zs[k], k, ne, ps)
                maps64[f"attn_fin_k{k}"] = (za - zb)[0].float().norm(dim=0).cpu().numpy()
                maps64[f"swap_fin_k{k}"] = (zw - zb)[0].float().norm(dim=0).cpu().numpy()
            base_img = to_uint8(decode(pipe, zb))
        finally:
            pipe.unet.set_attn_processor(saved)
        img_diff = float(np.abs(base_img.astype(int) - img.astype(int)).mean())
        t2 = time.perf_counter()
        # 5. maps -> counts
        for k in CANDS:
            x = post(maps64[k], truth.shape)
            res[f"{k}|raw"] = cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other)
            res[f"{k}+R2|raw"] = cc.counts(crf(base_img, x), truth, valid, other)
        res.update(tokens=np.array(tokens), token_rule=how, img_meandiff=img_diff,
                   sec_live=t1 - t0, sec_branches=t2 - t1, sec_total=time.perf_counter() - t0)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[knockout:{a.dataset}] {iid} {cls} tokens {tokens} ({how}) base-vs-live |Δpix| {img_diff:.2f} "
              f"live {t1 - t0:.1f}s branches {t2 - t1:.1f}s total {time.perf_counter() - t0:.1f}s", flush=True)
        del zs, snaps, maps64; torch.cuda.empty_cache()


# ------------------------------------------------------------------ report (CPU)
def report(a):
    j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    S = {}
    for ds in ("voc", "coco"):
        d = CACHE / f"idea1_{ds}"
        if d.exists() and any(d.glob("*.npz")):
            S[ds] = cc.load_gen(d)
            Z = [np.load(f) for f in sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp"))]
            S[ds]["rule"] = np.array([str(z["token_rule"]) for z in Z])
            S[ds]["sec"] = np.array([float(z["sec_total"]) for z in Z])
            S[ds]["imgdiff"] = np.array([float(z["img_meandiff"]) for z in Z])
    one = lambda ds: np.ones(len(S[ds]["grp"]), bool)
    oof = lambda ds, k: cc.oof(S[ds], S[ds][f"{k}+R2|raw"])
    L = ["# Idea 1: counterfactual knockout segmentation (pre-registered in `PREREG.md`)\n",
         "Effect on each pixel of knocking the target word out from denoising step k (5 or 10), same seed. "
         "attn = target-token cross-attention × 0; swap = target replaced by \"object\" in the text embedding. "
         "fin = final-latent change of a branch; ins = summed instantaneous noise-prediction change along the base "
         "trajectory. Maps: Gaussian σ 1 (64 grid), robust p1/p99 normalisation. BRACE = frozen V3 + R2 from the same "
         "process. Class-macro mIoU; OOF = τ chosen on 4 folds (crc32 % 5); Δ with image-bootstrap 95% CI.\n"]
    for ds in S:
        n = len(S[ds]["grp"]); ref = oof(ds, "brace"); dd = S[ds]["imgdiff"]
        L += [f"## {NAME[ds]} ({n} images; base-loop vs live mean |Δpixel| {dd.mean():.2f}/255, "
              f"{(dd > 2).sum()} images > 2/255, max {dd.max():.2f}; {S[ds]['sec'].mean():.1f} s/image)\n",
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
        verdict = "knockout beats BRACE" if lo > 0 else "no gain over BRACE"
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
    sub.add_parser("report"); sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "report": report, "selftest": selftest}[a.cmd](a)
