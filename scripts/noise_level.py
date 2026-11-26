#!/usr/bin/env python3
"""Exploratory follow-up to runs/09_equiv_noise (see runs/10_noise_level/NOTE.md): noise variability of the class-token
cross-attention map per noise level (single UNet call per level), COCO-Cap screening images, BRACE frozen.

python scripts/noise_level.py run [--shard i --nshard n] [--limit k] [--tag t]   GPU -> CACHE/noise_level[_t]/coco/<id>.npz
python scripts/noise_level.py score [--tag t] [--workers 16]                     CPU -> runs/10_noise_level/result.md
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import equiv_noise as en  # noqa: E402
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/10_noise_level"; DS = "coco"
LEVELS = (101, 201, 331, 430, 529, 661, 761, 861, 941); SEEDS = range(6)
BRACE_TS = set(range(331, 662, 33))


def work(tag):
    return CACHE / ("noise_level" + (f"_{tag}" if tag else ""))


def run(a):
    import torch
    from scripts import sg27_runtime
    import idea1_knockout as k1
    from ideas_grouped import screen_ids
    import sg_class_comp as cc
    ids = sorted(screen_ids(DS))[: a.limit or None][a.shard::a.nshard]
    rows = {r["image_id"]: r for r in cc.gen_rows(DS) if r["image_id"] in set(ids)}
    runtime = sg27_runtime.make_runtime(k1.MODEL); pipe = runtime["pipe"]; dev = pipe.device
    ac = pipe.scheduler.alphas_cumprod.to(dev)
    out = work(a.tag) / DS; out.mkdir(parents=True, exist_ok=True)
    saved = dict(pipe.unet.attn_processors)
    for iid in ids:
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(en.DUMP / DS / f"{iid}.npz", allow_pickle=True)
        tok = [int(t) for t in np.atleast_1d(z["tokens"])]
        pe, _ = pipe.encode_prompt(prompt=rows[iid]["prompt"], device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        store = {"sa": {}, "ca": []}
        new = {n: (en.Cap(None, tok, store) if n.removesuffix(".processor").endswith("attn2") else getattr(p, "original", p))
               for n, p in saved.items()}
        pipe.unet.set_attn_processor(new); res = {"levels": np.array(LEVELS)}
        try:
            x = torch.from_numpy(z["image"]).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
            with torch.no_grad():
                z0 = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mean * pipe.vae.config.scaling_factor
            for s in SEEDS:
                eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(s)).to(dev, z0.dtype)
                for t in LEVELS:
                    store["ca"].clear(); zt = ac[t].sqrt() * z0 + (1 - ac[t]).sqrt() * eps
                    with torch.no_grad():
                        pipe.unet(zt, torch.tensor(t, device=dev), encoder_hidden_states=pe, return_dict=False)
                    res[f"ca/t{t}/s{s}"] = torch.stack(store["ca"]).mean(0).cpu().numpy().astype(np.float32)
        finally:
            pipe.unet.set_attn_processor(dict(saved))
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[nl] {iid} tokens {tok} {res['sec']:.1f}s", flush=True)


def one(job):
    iid, W = job
    import sa_boundary_diag as sbd
    r = en.one((DS, iid, str(en.work(""))))
    if not r["ok"] or not len(r["y"]):
        return {"id": iid, "ok": False}
    c = np.load(Path(W) / DS / f"{iid}.npz"); Q = r["Q"]
    gt = sbd.pool(np.load(en.DUMP / DS / f"{iid}.npz", allow_pickle=True)["truth"].astype(np.float64), 64)
    lv = {}
    for t in LEVELS:
        m = np.stack([en.mm_valid(c[f"ca/t{t}/s{s}"]) for s in SEEDS])
        mean, sd = m.mean(0), m.std(0)
        lv[t] = dict(mean=mean.ravel()[Q], v=sd.ravel()[Q], auc_obj=sbd.auc((gt >= 0.5).ravel(), mean.ravel()))
    return {"id": iid, "ok": True, "y": r["y"], "X": r["X"], "lv": lv, "sec": float(c["sec"])}


def score(a):
    import sa_boundary_diag as sbd
    sbd.NBOOT = 2000
    W = str(work(a.tag))
    BASE = ["dLab_F", "dLab_B", "contour", "dist_F", "conf", "depth", "area", "s_rgb", "mean_ca", "mean_v3"]
    jobs = [(f.stem, W) for f in sorted((Path(W) / DS).glob("*.npz")) if not f.stem.endswith(".tmp")]
    with Pool(a.workers) as p:
        R = p.map(one, jobs, chunksize=1)
    ok = [r for r in R if r["ok"]]
    y = np.concatenate([r["y"] for r in ok]); grp = np.concatenate([np.full(len(r["y"]), r["id"]) for r in ok])
    X = {k: np.concatenate([r["X"][k] for r in ok]) for k in BASE}
    B0 = np.stack([X[k] for k in BASE], 1)
    res = {"images": len(ok), "excluded": len(R) - len(ok), "cells": int(len(y)), "bleed": float(y.mean()), "levels": {}}
    ac = None
    try:
        import torch
        b = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2; ac = torch.cumprod(1 - b, 0).numpy()
    except Exception:
        pass
    L = ["# Exploratory: noise variability of SD1.5 cross-attention at BRACE's bleed, per noise level (`NOTE.md`)\n",
         f"COCO-Cap screening images ({len(ok)} scored, {len(R) - len(ok)} excluded); 6 noise seeds; one conditional UNet call per level. "
         "**Exploratory, in-sample (same images as run 09), 9 levels: not confirmatory.**\n",
         "Baseline per level = run 09 baseline + this level's mean CA; test = + this level's v_CA. Bar for 'promising': Δ ≥ +0.05, CI > 0.\n",
         "| t | √ᾱ (image share) | in BRACE window | mean CA obj/bg cell AUROC | v_CA single AUROC (bleed) | baseline | + v_CA Δ (95% CI) | promising |",
         "|---:|---:|:-:|---:|---:|---:|---|:-:|"]
    for t in LEVELS:
        mean = np.concatenate([r["lv"][t]["mean"] for r in ok]); v = np.concatenate([r["lv"][t]["v"] for r in ok])
        g = sbd.compare(np.c_[B0, mean], np.c_[B0, mean, v], y, grp)
        prom = bool(g["delta"] >= 0.05 and g["lo"] > 0)
        d = dict(gate=g, single=float(sbd.auc(y, v)), auc_obj=float(np.nanmean([r["lv"][t]["auc_obj"] for r in ok])), promising=prom)
        res["levels"][t] = d
        sa = f"{np.sqrt(ac[t]):.2f}" if ac is not None else "?"
        L.append(f"| {t} | {sa} | {'yes' if t in BRACE_TS else ''} | {d['auc_obj']:.3f} | {d['single']:.3f} | {g['rgb']:.3f} | "
                 f"{g['delta']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] | {'**yes**' if prom else 'no'} |")
    L += [f"\nCells {res['cells']:,} ({res['bleed']:.1%} bleed); {np.mean([r['sec'] for r in ok]):.0f} s/image capture. "
          "Run 09 reference (11 BRACE timesteps averaged, COCO-Cap): +0.0252 [+0.0047, +0.0444].",
          "\nA level marked promising licenses only a new pre-registered single-level test on images not used here."]
    RUN.mkdir(parents=True, exist_ok=True); out = RUN / (f"result_{a.tag}" if a.tag else "result")
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
    p = sub.add_parser("score"); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)
