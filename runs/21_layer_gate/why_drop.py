#!/usr/bin/env python3
"""Post-hoc, exploratory (not pre-registered): why does the per-location gate improve the seed but not the hierarchy?

Reads the stored runs/21 capture (no regeneration, no new scoring of registered cells). Raw maps at 64x64 against
GT64 (area fraction >= 0.5). Thresholds: one per (set, seed, stage), best on that set (audit-style), so this is
descriptive only.
"""
from __future__ import annotations

import sys, json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_layergate"
SEEDS_ALL = ("BR", "UN") + tuple(x for p in ("px", "s16", "s64", "s256", "g16", "g64", "g256") for x in (f"G{p}", f"G{p}x")) \
    + ("TE", "TM", "TL", "TD+", "TD-")
SHOW = ("BR", "UN", "Gpx", "Gpxx", "Gs256", "Gg16")
CELLS = ("seed", "V0", "V3", "W")
THR = np.round(np.arange(0.10, 0.951, 0.05), 2)
# selected CA modules [0,1,4,5,6,7,8,9,13,14,15] -> SD1.5 resolution of each
RES = np.array([64, 64, 16, 16, 8, 16, 16, 16, 64, 64, 64])


def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)


def down_up(x, k=16):
    s = 64 // k
    return np.kron(x.reshape(k, s, k, s).mean((1, 3)), np.ones((s, s)))


def auc(s, g):
    pos, neg = s[g], s[~g]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    r = ndi.rankdata if False else None  # noqa
    from scipy.stats import rankdata
    rk = rankdata(np.concatenate([pos, neg]))
    return (rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def one(f):
    z = np.load(f)
    tr = z["truth"].astype(np.float32)
    H = tr.shape[0] // 64
    g = tr.reshape(64, H, 64, H).mean((1, 3)) >= 0.5
    if g.sum() == 0 or (~g).sum() == 0:
        return None
    M = z["maps"].astype(np.float32)
    out = {"cls": str(z["cls"]), "area": float(g.mean())}
    stages = {}
    for s in SHOW:
        si = SEEDS_ALL.index(s)
        st = {c: mm(M[si, CELLS.index(c)]) for c in CELLS}
        st["seed16"] = mm(down_up(st["seed"]))
        stages[s] = st
        for c, x in st.items():
            out[f"{s}|{c}|inter"] = np.array([(x >= t)[g].sum() for t in THR])
            out[f"{s}|{c}|union"] = np.array([((x >= t) | g).sum() for t in THR])
            out[f"{s}|{c}|auc"] = auc(x.ravel(), g.ravel())
            out[f"{s}|{c}|mass"] = float(x[g].sum() / max(x.sum(), 1e-12))           # share of map mass on GT
            top = x >= np.quantile(x, 1 - max(g.mean(), 0.01))                         # top-|GT| pixels
            out[f"{s}|{c}|topprec"] = float(top[g].mean() if False else (top & g).sum() / max(top.sum(), 1))
    # where does the gate change the seed? (Gpx - BR, both min-max)
    d = stages["Gpx"]["seed"] - stages["BR"]["seed"]
    inner = ndi.binary_erosion(g, iterations=2)
    band_in = g & ~inner
    near = ndi.binary_dilation(g, iterations=4) & ~g
    far = ~ndi.binary_dilation(g, iterations=4)
    for nm, m in (("fg_interior", inner), ("fg_boundary", band_in), ("bg_near", near), ("bg_far", far)):
        out[f"d|{nm}"] = float(d[m].mean()) if m.any() else np.nan
        out[f"BRval|{nm}"] = float(stages["BR"]["seed"][m].mean()) if m.any() else np.nan
        out[f"Gval|{nm}"] = float(stages["Gpx"]["seed"][m].mean()) if m.any() else np.nan
    # frequency split of the change: coarse (16x16 block means) vs fine residual
    dc = down_up(d)
    out["d|energy_coarse"] = float((dc ** 2).sum()); out["d|energy_fine"] = float(((d - dc) ** 2).sum())
    # similarity of each seed to BRACE's own propagated maps (is the gate doing what propagation does?)
    for s in ("BR", "Gpx"):
        for c in ("V0", "V3"):
            out[f"corr|{s}seed~BR{c}"] = float(np.corrcoef(stages[s]["seed"].ravel(), stages["BR"][c].ravel())[0, 1])
    # layer content: which resolutions carry the class map, call-averaged layers vs gate change
    L = z["layers"].astype(np.float32).mean(0)                                         # [11,64,64]
    W = z["weights"].mean(0)                                                           # [11]
    for r in (64, 16, 8):
        k = RES == r
        Lr = (W[k, None, None] * L[k]).sum(0) / max(W[k].sum(), 1e-12)
        out[f"layer{r}|auc"] = auc(Lr.ravel(), g.ravel())
        out[f"layer{r}|w"] = float(W[k].sum() / W.sum())
        out[f"layer{r}|corr_d"] = float(np.corrcoef(d.ravel(), (mm(Lr) - stages["BR"]["seed"]).ravel())[0, 1])
    return out


def summarize(rows):
    rep = {}
    for s in SHOW:
        for c in ("seed", "seed16", "V0", "V3", "W"):
            I = np.stack([r[f"{s}|{c}|inter"] for r in rows]); U = np.stack([r[f"{s}|{c}|union"] for r in rows])
            cls = np.array([r["cls"] for r in rows])
            miou = np.array([np.mean([I[cls == k, j].sum() / U[cls == k, j].sum() for k in np.unique(cls)])
                             for j in range(len(THR))])
            j = int(miou.argmax())
            rep[f"{s}|{c}"] = {"best_mIoU": float(miou[j]), "thr": float(THR[j]),
                               "auc": float(np.nanmean([r[f"{s}|{c}|auc"] for r in rows])),
                               "mass_on_gt": float(np.mean([r[f"{s}|{c}|mass"] for r in rows])),
                               "top_prec": float(np.mean([r[f"{s}|{c}|topprec"] for r in rows]))}
    for k in rows[0]:
        if k.startswith(("d|", "BRval|", "Gval|", "corr|", "layer")):
            rep[k] = float(np.nanmean([r[k] for r in rows]))
    # per-image: does the seed gain predict the hierarchy gain?
    dA = np.array([r["Gpx|seed|auc"] - r["BR|seed|auc"] for r in rows])
    dV = np.array([r["Gpx|V3|auc"] - r["BR|V3|auc"] for r in rows])
    dA16 = np.array([r["Gpx|seed16|auc"] - r["BR|seed16|auc"] for r in rows])
    ar = np.array([r["area"] for r in rows])
    rep["perimg|corr(dSeedAUC,dV3AUC)"] = float(np.corrcoef(dA, dV)[0, 1])
    rep["perimg|corr(dSeed16AUC,dV3AUC)"] = float(np.corrcoef(dA16, dV)[0, 1])
    rep["perimg|share seed AUC up"] = float((dA > 0).mean()); rep["perimg|share V3 AUC up"] = float((dV > 0).mean())
    for nm, m in (("small", ar < np.quantile(ar, 1 / 3)), ("large", ar >= np.quantile(ar, 2 / 3))):
        rep[f"perimg|{nm} dSeedAUC"] = float(dA[m].mean()); rep[f"perimg|{nm} dSeed16AUC"] = float(dA16[m].mean())
        rep[f"perimg|{nm} dV3AUC"] = float(dV[m].mean())
    return rep


if __name__ == "__main__":
    res = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (WORK / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        with Pool(16) as p:
            rows = [r for r in p.map(one, fs, chunksize=4) if r is not None]
        res[ds] = summarize(rows); res[ds]["n"] = len(rows)
        print(ds, len(rows), flush=True)
    o = Path(__file__).with_name("why_drop.json"); o.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
