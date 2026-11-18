#!/usr/bin/env python3
"""Recoverability audit (diagnostic only; stored runs/21 capture, corrected prompts; GT used only for scoring).

Does any individual CA layer carry foreground evidence that BRACE's GOCA-style aggregation leaves weak, and can it be
told apart from background without labels?

Per image, per selected CA layer l: call-aggregated map L_l = sum_t w_tl A_tl / sum_t w_tl. Three per-layer
normalisations (controls): mm (per-layer min-max), q (per-layer percentile rank in the image), raw (no normalisation,
divided by the image max over layers). "Fires" = >= 0.5 (mm, raw) or >= 0.9 (q).
S = BRACE seed (min-max). Missed foreground M = GT & (S < 0.5). Ambiguous band = 0.2 <= S < 0.5.

Q1 share of M where >= 1 layer fires (per resolution group too). Q2 the same at 16x16 (block means; GT16 >= 0.5).
Q3/Q4 pooled maps max_l, Q75_l and blend 0.5 S + 0.5 Q75: share of M recovered (>= 0.5) and new false positives
(background with S < 0.5 now >= 0.5, near vs far). Magnitude control: per image, lower S's threshold until the same
number of new pixels is added; compare the true-foreground share of added pixels.
Complementarity: AUC (fg vs bg) inside the ambiguous band of S vs of each pooled score.
Q5 among pixels added by the blend: univariate AUC (true fg vs bg) of label-free features: S, Q75, #layers firing,
16-res firing, 64-res firing, temporal persistence (share of calls the strongest layer fires), distance to the
S >= 0.5 core, connected to the core.
"""
from __future__ import annotations

import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_layergate"
RES = np.array([64, 64, 16, 16, 8, 16, 16, 16, 64, 64, 64])
NORMS = ("mm", "q", "raw")
FEATS = ("S", "Q75", "nfire", "fire16", "fire64", "persist", "dist_core", "conn_core")
rng = np.random.default_rng(0)


def mm(x, ax=None):
    lo = x.min(axis=ax, keepdims=True); hi = x.max(axis=ax, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def blk(x, k=16):
    s = x.shape[-1] // k
    return x.reshape(*x.shape[:-2], k, s, k, s).mean((-3, -1))


def auc(s, g):
    pos, neg = s[g], s[~g]
    if pos.size < 5 or neg.size < 5:
        return np.nan
    rk = rankdata(np.concatenate([pos, neg]))
    return (rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def norm(L, how):
    if how == "mm":
        return mm(L.reshape(11, -1), 1).reshape(L.shape)
    if how == "q":
        return (np.argsort(np.argsort(L.reshape(11, -1), 1), 1) / (L[0].size - 1)).reshape(L.shape)
    return L / max(L.max(), 1e-12)


def one(f):
    z = np.load(f)
    tr = z["truth"].astype(np.float32); H = tr.shape[0] // 64
    gf = tr.reshape(64, H, 64, H).mean((1, 3)); g = gf >= 0.5
    if g.sum() < 10 or (~g).sum() < 10:
        return None
    A = z["layers"].astype(np.float32); W = z["weights"].astype(np.float32)            # [31,11,64,64], [31,11]
    L = (W[:, :, None, None] * A).sum(0) / np.maximum(W.sum(0), 1e-12)[:, None, None]   # [11,64,64]
    S = mm(z["maps"][0, 0].astype(np.float32))                                         # BR seed (runtime A3)
    near = ndi.binary_dilation(g, iterations=4) & ~g; far = ~g & ~near
    M = g & (S < 0.5); band = (S >= 0.2) & (S < 0.5); core = S >= 0.5
    out = {"cls": str(z["cls"]), "nM": int(M.sum()), "nG": int(g.sum()), "nband": int(band.sum())}
    g16 = blk(gf) >= 0.5; S16 = mm(blk(S)); M16 = g16 & (S16 < 0.5)
    out["nM16"] = int(M16.sum())
    for how in NORMS:
        thr = 0.9 if how == "q" else 0.5
        N = norm(L, how); fire = N >= thr
        out[f"{how}|Q1_any"] = float(fire[:, M].any(0).mean()) if M.any() else np.nan
        out[f"{how}|Q1_bgfire"] = float(fire[:, ~g & (S < 0.5)].any(0).mean())       # same rule on missed background
        for gr in (64, 16, 8):
            out[f"{how}|Q1_any{gr}"] = float(fire[RES == gr][:, M].any(0).mean()) if M.any() else np.nan
        N16 = norm(blk(L), how); f16 = N16 >= thr
        out[f"{how}|Q2_any16"] = float(f16[:, M16].any(0).mean()) if M16.any() else np.nan
        out[f"{how}|Q2_bgfire16"] = float(f16[:, ~g16 & (S16 < 0.5)].any(0).mean()) if (~g16 & (S16 < 0.5)).any() else np.nan
        pools = {"max": mm(N.max(0)), "q75": mm(np.quantile(N, 0.75, axis=0))}
        pools["blend"] = mm(0.5 * S + 0.5 * pools["q75"])
        for pn, X in pools.items():
            add = (X >= 0.5) & (S < 0.5)
            out[f"{how}|{pn}|rec"] = float((X[M] >= 0.5).mean()) if M.any() else np.nan
            out[f"{how}|{pn}|lost"] = float((X[g & core] < 0.5).mean()) if (g & core).any() else np.nan
            out[f"{how}|{pn}|newTP"] = int((add & g).sum()); out[f"{how}|{pn}|newFPnear"] = int((add & near).sum())
            out[f"{how}|{pn}|newFPfar"] = int((add & far).sum())
            # magnitude control: same number of added pixels by lowering S's threshold
            k = int(add.sum()); cand = np.flatnonzero((S < 0.5).ravel())
            if k > 0 and cand.size:
                top = cand[np.argsort(-S.ravel()[cand])[:k]]
                out[f"{how}|{pn}|ctrlTP"] = int(g.ravel()[top].sum())
            else:
                out[f"{how}|{pn}|ctrlTP"] = 0
            out[f"{how}|{pn}|auc16"] = auc(mm(blk(X)).ravel(), g16.ravel()) if g16.any() and (~g16).any() else np.nan
            out[f"{how}|{pn}|bandAUC"] = auc(X[band], g[band]) if band.any() else np.nan
            if pn == "blend" and how == "mm":
                # Q5 features on added pixels (subsample up to 300)
                idx = np.flatnonzero(add.ravel())
                if idx.size >= 10:
                    idx = rng.choice(idx, min(300, idx.size), replace=False)
                    Af = A.reshape(31, 11, -1); best = N.reshape(11, -1)[:, idx].argmax(0)
                    pa = mm(Af.reshape(31 * 11, -1), 1).reshape(31, 11, -1)
                    dist = ndi.distance_transform_edt(~core) if core.any() else np.full(S.shape, 64.)
                    lab, _ = ndi.label(X >= 0.5); corel = np.unique(lab[core & (lab > 0)])
                    F = {"S": S.ravel()[idx], "Q75": pools["q75"].ravel()[idx], "nfire": fire.reshape(11, -1)[:, idx].sum(0),
                         "fire16": fire[(RES == 16) | (RES == 8)].reshape(-1, 4096)[:, idx].sum(0),
                         "fire64": fire[RES == 64].reshape(-1, 4096)[:, idx].sum(0),
                         "persist": np.array([(pa[:, b, i] >= 0.5).mean() for b, i in zip(best, idx)]),
                         "dist_core": -dist.ravel()[idx], "conn_core": np.isin(lab.ravel()[idx], corel).astype(float)}
                    out["Q5"] = {k: v.astype(np.float32) for k, v in F.items()}; out["Q5y"] = g.ravel()[idx]
        out[f"{how}|S|bandAUC"] = auc(S[band], g[band]) if band.any() else np.nan
        out[f"{how}|S|auc16"] = auc(S16.ravel(), g16.ravel()) if g16.any() and (~g16).any() else np.nan
        out[f"{how}|perlayer_bandAUC"] = [auc(N[l][band], g[band]) if band.any() else np.nan for l in range(11)]
    return out


def summarize(rows):
    r = {"n": len(rows), "missed_fg_share": float(np.sum([x["nM"] for x in rows]) / np.sum([x["nG"] for x in rows]))}
    mean = lambda k: float(np.nanmean([x[k] for x in rows if k in x]))
    tot = lambda k: float(np.sum([x[k] for x in rows if k in x]))
    for how in NORMS:
        for k in ("Q1_any", "Q1_bgfire", "Q1_any64", "Q1_any16", "Q1_any8", "Q2_any16", "Q2_bgfire16"):
            r[f"{how}|{k}"] = mean(f"{how}|{k}")
        r[f"{how}|S|bandAUC"] = mean(f"{how}|S|bandAUC"); r[f"{how}|S|auc16"] = mean(f"{how}|S|auc16")
        r[f"{how}|perlayer_bandAUC"] = list(np.nanmean(np.array([x[f"{how}|perlayer_bandAUC"] for x in rows], float), 0))
        for pn in ("max", "q75", "blend"):
            p = f"{how}|{pn}"
            add = tot(f"{p}|newTP") + tot(f"{p}|newFPnear") + tot(f"{p}|newFPfar")
            r[p] = {"recovered_missed": mean(f"{p}|rec"), "lost_detected": mean(f"{p}|lost"),
                    "added_px_per_img": add / len(rows), "added_true_share": tot(f"{p}|newTP") / max(add, 1),
                    "added_far_share": tot(f"{p}|newFPfar") / max(add, 1),
                    "magnitude_ctrl_true_share": tot(f"{p}|ctrlTP") / max(add, 1),
                    "auc16": mean(f"{p}|auc16"), "bandAUC": mean(f"{p}|bandAUC")}
    q = [x for x in rows if "Q5" in x]
    y = np.concatenate([x["Q5y"] for x in q])
    r["Q5_n_added_px"] = int(y.size); r["Q5_true_share"] = float(y.mean())
    r["Q5_auc"] = {k: float(auc(np.concatenate([x["Q5"][k] for x in q]), y)) for k in FEATS}
    return r


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    res = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (WORK / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        fs = fs[:lim] if lim else fs
        with Pool(16) as p:
            rows = [x for x in p.map(one, fs, chunksize=2) if x is not None]
        res[ds] = summarize(rows); print(ds, len(rows), flush=True)
    o = Path(__file__).with_name(f"audit{tag}.json"); o.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:3000])
