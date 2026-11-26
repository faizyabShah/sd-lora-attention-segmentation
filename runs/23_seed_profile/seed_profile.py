#!/usr/bin/env python3
"""Exploratory, descriptive (stored maps only, no new capture): what kind of seed gives the best BRACE hierarchy (V3)?

Two captures, analysed separately (never mixed): runs/20 (23 seeds: CA uniform/GOCA, NERVE CLIP, SCLIP, CA+CLIP
hybrid, templates, SD-only remaps) and runs/21 (21 seeds: BRACE, uniform, pixel/region/grid gates and shuffles,
temporal windows/contrasts). Raw maps at 64x64 against GT64 (area >= 0.5); one threshold per (capture, set, seed,
stage), best on that set (audit-style). Seed properties are label-based descriptors used only for analysis.

Question: across seeds, which seed property predicts hierarchy quality, and is it the seed's own mIoU?
"""
from __future__ import annotations

import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

BASE = ("CAu", "CAg", "NV", "SC")
CAL = ("hA", "TG", "TC") + tuple(f"PW{g:g}" for g in (0.5, 1.5, 2.0, 3.0)) + tuple(f"FL{p}" for p in (50, 70, 80, 90)) \
    + tuple(f"TK{k}" for k in (5, 10, 20, 40)) + ("FO",)
CAP = {"r20": (CACHE / "jc_newrerun", BASE + CAL),
       "r21": (CACHE / "jc_layergate", ("BR", "UN") + tuple(x for p in ("px", "s16", "s64", "s256", "g16", "g64", "g256")
                                                            for x in (f"G{p}", f"G{p}x")) + ("TE", "TM", "TL", "TD+", "TD-"))}
CELLS = ("seed", "V0", "V3", "W")
THR = np.round(np.arange(0.05, 0.951, 0.05), 2)
PROPS = ("auc64", "auc16", "mass64", "mass16", "interior", "near", "far", "contrast", "cover", "top10", "grad_in", "seed_iou")


def mm(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def auc(s, g):
    pos, neg = s[g], s[~g]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    rk = rankdata(np.concatenate([pos, neg]))
    return (rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def blk(x, k=16):
    s = x.shape[0] // k
    return x.reshape(k, s, k, s).mean((1, 3))


def one(job):
    f, seeds = job
    z = np.load(f)
    tr = z["truth"].astype(np.float32); H = tr.shape[0] // 64
    gf = tr.reshape(64, H, 64, H).mean((1, 3)); g = gf >= 0.5
    if g.sum() < 4 or (~g).sum() < 4:
        return None
    g16f = blk(gf); g16 = g16f >= 0.5
    inner = ndi.binary_erosion(g, iterations=2); inner = inner if inner.any() else g
    near = ndi.binary_dilation(g, iterations=4) & ~g; far = ~ndi.binary_dilation(g, iterations=4)
    if not far.any():
        far = ~g
    M = z["maps"].astype(np.float32)
    out = {"cls": str(z["cls"]), "area": float(g.mean())}
    for si, s in enumerate(seeds):
        x = mm(M[si, 0]); x16 = blk(x)
        p = {"auc64": auc(x.ravel(), g.ravel()),
             "auc16": auc(x16.ravel(), g16.ravel()) if g16.any() and (~g16).any() else np.nan,
             "mass64": x[g].sum() / max(x.sum(), 1e-12), "mass16": (x16 * g16f).sum() / max(x16.sum(), 1e-12),
             "interior": x[inner].mean(), "near": x[near].mean(), "far": x[far].mean()}
        p["contrast"] = p["interior"] - p["far"]
        p["cover"] = float((x[g] >= 0.5).mean())                                  # share of object above mid-level
        p["top10"] = float(np.sort(x.ravel())[-410:].sum() / max(x.sum(), 1e-12))  # peakiness
        p["grad_in"] = float(x[inner].std())                                      # unevenness inside the object
        for k, v in p.items():
            out[f"{s}|{k}"] = float(v)
        for ci, c in enumerate(CELLS):
            y = mm(M[si, ci])
            out[f"{s}|{c}|I"] = np.array([((y >= t) & g).sum() for t in THR])
            out[f"{s}|{c}|U"] = np.array([((y >= t) | g).sum() for t in THR])
            out[f"{s}|{c}|auc"] = float(auc(y.ravel(), g.ravel()))
    return out


def miou_best(rows, s, c):
    cls = np.array([r["cls"] for r in rows]); I = np.stack([r[f"{s}|{c}|I"] for r in rows]); U = np.stack([r[f"{s}|{c}|U"] for r in rows])
    m = np.array([np.mean([I[cls == k, j].sum() / U[cls == k, j].sum() for k in np.unique(cls)]) for j in range(len(THR))])
    j = int(m.argmax()); return float(m[j]), float(THR[j]), I[:, j] / np.maximum(U[:, j], 1)


def main():
    res = {}
    for cap, (d, seeds) in CAP.items():
        for ds in ("voc", "coco"):
            fs = sorted(f for f in (d / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
            with Pool(16) as p:
                rows = [r for r in p.map(one, [(f, seeds) for f in fs], chunksize=4) if r is not None]
            tab = {}; per_img_iou = {}
            for s in seeds:
                t = {k: float(np.nanmean([r[f"{s}|{k}"] for r in rows])) for k in PROPS if k != "seed_iou"}
                for c in CELLS:
                    t[f"{c}_miou"], t[f"{c}_thr"], iou = miou_best(rows, s, c); per_img_iou[(s, c)] = iou
                    t[f"{c}_auc"] = float(np.nanmean([r[f"{s}|{c}|auc"] for r in rows]))
                t["seed_iou"] = t["seed_miou"]
                tab[s] = t
            # across seeds: which property ranks seeds like their hierarchy mIoU does?
            y = np.array([tab[s]["V3_miou"] for s in seeds])
            across = {k: float(spearmanr([tab[s][k] for s in seeds], y)[0]) for k in PROPS}
            # within image: across seeds, which property ranks seeds like their per-image V3 IoU?
            within = {}
            for k in PROPS:
                rhos = []
                for i, r in enumerate(rows):
                    a = np.array([r[f"{s}|{k}"] if k != "seed_iou" else per_img_iou[(s, "seed")][i] for s in seeds])
                    b = np.array([per_img_iou[(s, "V3")][i] for s in seeds])
                    if np.isfinite(a).all() and a.std() > 0 and b.std() > 0:
                        rhos.append(spearmanr(a, b)[0])
                within[k] = float(np.nanmean(rhos))
            res[f"{cap}|{ds}"] = {"n": len(rows), "seeds": tab, "across_rho_vs_V3": across, "within_image_rho_vs_V3": within}
            print(cap, ds, len(rows), json.dumps(across), flush=True)
    o = Path(__file__).with_name("seed_profile.json"); o.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
