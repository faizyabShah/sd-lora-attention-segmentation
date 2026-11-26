#!/usr/bin/env python3
"""Diagnostic (pre-registered in runs/05_sa_boundary/PREREG.md): does SD self-attention carry boundary information
beyond RGB, in particular on BRACE's bleed cells? CPU only, stored rich capture (CACHE/ana_dump), native 16/32 grids.
Ground truth is used only to measure.

python scripts/sa_boundary_diag.py [--workers 16] [--max-images k]   -> runs/05_sa_boundary/result.md, result.json
"""
from __future__ import annotations

import argparse, json, sys, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import ana_ops_common as oc  # noqa: E402

RUN = ROOT / "runs/05_sa_boundary"
RES = (16, 32); BLOCKS = (0, 1, 2); PRIMARY = (32, 1)
BANDS = (1, 2, 3); NSAMP = 16; MAX_EDGES = 3000; NBOOT = 1000
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


# ------------------------------------------------------------------ image features
def lab(img):
    x = img.astype(np.float64) / 255
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def pool(x, s):
    h = x.shape[0] // s
    return x.reshape(s, h, s, h, *x.shape[2:]).mean((1, 3))


def offsets():
    o = []
    for dy in range(0, 4):
        for dx in range(-3, 4):
            if (dy == 0 and dx <= 0):
                continue
            b = int(round(np.hypot(dy, dx)))
            if b in BANDS:
                o.append((dy, dx, b))
    return o


OFFS = offsets()


def contour_max(G, p0, p1, cell):
    """max of G along segments between cell centres p0 [n,2] and p1 [n,2] (grid units)."""
    t = np.linspace(0, 1, NSAMP)[None, :, None]
    P = ((p0[:, None] + (p1 - p0)[:, None] * t) + 0.5) * cell
    yy = np.clip(P[..., 0].round().astype(int), 0, G.shape[0] - 1); xx = np.clip(P[..., 1].round().astype(int), 0, G.shape[1] - 1)
    return G[yy, xx].max(1)


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ per image
def one(f):
    oc.crf_init()
    z = np.load(f, allow_pickle=True); iid = Path(f).stem
    img = z["image"]; truth = z["truth"].astype(np.float64)
    L = lab(img); Ls = gaussian_filter(L, (1, 1, 0))
    G = np.sqrt(sum(sobel(Ls[..., c], 0) ** 2 + sobel(Ls[..., c], 1) ** 2 for c in range(3)))
    x = st.up(st.mm(z["v3"]), truth.shape); mask = oc.crf(img, x >= 0.55).astype(np.float64)
    v3m = st.mm(z["v3"])
    out = {"id": iid}
    for s in RES:
        cell = 512 // s; gt = pool(truth, s); col = pool(L, s); mk = pool(mask, s)
        vv = pool(st.up(v3m, (512, 512)), s); core = vv >= 0.9
        if not core.any():
            core = vv >= np.quantile(vv, 0.95)
        obj = gt >= 0.75; bg = gt <= 0.25
        yy, xx = np.mgrid[:s, :s]
        for blk in BLOCKS:
            key = f"sa_full/{s}_b{blk}_c"
            if key not in z.files:
                continue
            A = z[key].astype(np.float64); W = (A + A.T) / 2; lw = np.log(W + 1e-8)
            # ---- edges (Q1, Q2)
            E = {k: [] for k in ("cross", "band", "logw", "colour", "contour")}
            q1 = {"sa": [], "colour": [], "contour": []}
            rng = np.random.default_rng(zlib.crc32(f"{iid}{s}{blk}".encode()))
            for b in BANDS:
                Eb = {k: [] for k in E}
                for dy, dx, bb in OFFS:
                    if bb != b:
                        continue
                    y0, x0 = yy.ravel(), xx.ravel(); y1, x1 = y0 + dy, x0 + dx
                    ok = (y1 < s) & (x1 >= 0) & (x1 < s)
                    y0, x0, y1, x1 = y0[ok], x0[ok], y1[ok], x1[ok]
                    lab0 = np.where(obj[y0, x0], 1, np.where(bg[y0, x0], 0, -1)); lab1 = np.where(obj[y1, x1], 1, np.where(bg[y1, x1], 0, -1))
                    keep = (lab0 >= 0) & (lab1 >= 0)
                    y0, x0, y1, x1, lab0, lab1 = y0[keep], x0[keep], y1[keep], x1[keep], lab0[keep], lab1[keep]
                    i0, i1 = y0 * s + x0, y1 * s + x1
                    Eb["cross"].append(lab0 != lab1); Eb["band"].append(np.full(len(i0), b)); Eb["logw"].append(lw[i0, i1])
                    Eb["colour"].append(np.linalg.norm(col[y0, x0] - col[y1, x1], axis=1))
                    Eb["contour"].append(contour_max(G, np.c_[y0, x0].astype(float), np.c_[y1, x1].astype(float), cell))
                Eb = {k: np.concatenate(v) for k, v in Eb.items()}
                if len(Eb["cross"]) == 0:
                    continue
                if Eb["cross"].any() and (~Eb["cross"]).any():
                    q1["sa"].append(auc(Eb["cross"], -Eb["logw"])); q1["colour"].append(auc(Eb["cross"], Eb["colour"]))
                    q1["contour"].append(auc(Eb["cross"], Eb["contour"]))
                sel = rng.permutation(len(Eb["cross"]))[:MAX_EDGES]
                for k in E:
                    E[k].append(Eb[k][sel])
            E = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in E.items()}
            # ---- Q3 cells inside BRACE's mask
            inside = mk >= 0.5; bleed = inside & bg; good = inside & obj
            Q3 = {}
            for rname, R in (("core", core), ("gt", obj)):
                if not R.any() or not (bleed.any() and good.any()):
                    Q3[rname] = None; continue
                d, (ny, nx) = distance_transform_edt(~R, return_indices=True)
                cells = np.flatnonzero((bleed | good).ravel()); cy, cx = cells // s, cells % s
                Rf = R.ravel(); nR = Rf.sum()
                Wr = W[:, Rf].sum(1) - np.where(Rf, np.diag(W), 0); den = nR - Rf.astype(int)
                saconn = np.log(np.where(den > 0, Wr / np.maximum(den, 1), np.nan) + 1e-8)
                rcol = col[R].mean(0)
                Q3[rname] = dict(y=bleed.ravel()[cells].astype(np.int8), dist=d[cy, cx], colour=np.linalg.norm(col[cy, cx] - rcol, axis=1),
                                 contour=contour_max(G, np.c_[cy, cx].astype(float), np.c_[ny[cy, cx], nx[cy, cx]].astype(float), cell),
                                 sa=saconn[cells])
            out[(s, blk)] = dict(E=E, q1={k: float(np.nanmean(v)) if v else np.nan for k, v in q1.items()}, Q3=Q3)
    return out


# ------------------------------------------------------------------ models
def logit_fit(X, y, lam=1e-3, it=50):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30))); g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def oof_scores(X, y, grp):
    folds = np.array([zlib.crc32(g.encode()) % 5 for g in grp]); s = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xt = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]; Xe = np.c_[np.ones(te.sum()), (X[te] - mu) / sd]
        s[te] = Xe @ logit_fit(Xt, y[tr].astype(float))
    return s


def boot_delta(y, s0, s1, grp, seed=0):
    ids, inv = np.unique(grp, return_inverse=True); rows = [np.flatnonzero(inv == i) for i in range(len(ids))]
    rng = np.random.default_rng(seed); d = []
    for _ in range(NBOOT):
        r = np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))])
        d.append(auc(y[r], s1[r]) - auc(y[r], s0[r]))
    return np.nanpercentile(d, [2.5, 97.5])


def compare(feats0, feats1, y, grp):
    keep = np.isfinite(feats1).all(1) & np.isfinite(feats0).all(1)
    X0, X1, y, grp = feats0[keep], feats1[keep], y[keep], grp[keep]
    s0, s1 = oof_scores(X0, y, grp), oof_scores(X1, y, grp)
    a0, a1 = auc(y, s0), auc(y, s1); lo, hi = boot_delta(y, s0, s1, grp)
    return dict(n=int(len(y)), pos=int(y.sum()), images=int(len(np.unique(grp))), rgb=a0, rgb_sa=a1, delta=a1 - a0, lo=float(lo), hi=float(hi))


def mean_ci(v, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)]); rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NBOOT)]
    return float(v.mean()), *np.percentile(b, [2.5, 97.5])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    R = {}; out = {}
    L = ["# Does SD self-attention carry boundary information beyond RGB? (diagnostic; pre-registered in `PREREG.md`)\n",
         "Stored rich capture of the 150/150 screening images; native grids; ground truth only measures. "
         "Primary setting: 32 px, block 1 (steps 10–19). Decisive: Q3(a) on COCO-Cap.\n"]
    for ds in ("voc", "coco"):
        fs = sorted(str(f) for f in (CACHE / "ana_dump" / ds).glob("*.npz"))[: a.max_images]
        with Pool(a.workers) as p:
            R[ds] = p.map(one, fs, chunksize=2)
        L.append(f"## {NAME[ds]} ({len(R[ds])} images)\n")
        L += ["| res | block | Q1 AUROC SA / colour / contour (95% CI for SA) | Q2 AUROC RGB → RGB+SA, Δ (95% CI) | "
              "Q3(a) core: RGB → RGB+SA, Δ (95% CI) [bleed/correct cells] | Q3 single-feature AUROC, core ref: dist / colour / contour / −SA | Q3(b) gt ref, single −SA |",
              "|---:|---:|---|---|---|---|---|"]
        for s in RES:
            for blk in BLOCKS:
                rs = [r[(s, blk)] for r in R[ds] if (s, blk) in r]
                if not rs:
                    continue
                q1 = {k: mean_ci([r["q1"][k] for r in rs]) for k in ("sa", "colour", "contour")}
                grp = np.concatenate([[r["id"]] * len(r[(s, blk)]["E"]["cross"]) for r in R[ds] if (s, blk) in r])
                E = {k: np.concatenate([r["E"][k] for r in rs]) for k in ("cross", "band", "logw", "colour", "contour")}
                bandX = np.stack([E["band"] == b for b in BANDS[1:]], 1).astype(float)
                X0 = np.c_[E["colour"], E["contour"], bandX]; X1 = np.c_[X0, E["logw"]]
                q2 = compare(X0, X1, E["cross"].astype(int), grp)
                q3 = {}
                for rn in ("core", "gt"):
                    parts = [(r["id"], r[(s, blk)]["Q3"][rn]) for r in R[ds] if (s, blk) in r and r[(s, blk)]["Q3"][rn] is not None]
                    g3 = np.concatenate([[i] * len(c["y"]) for i, c in parts]); C = {k: np.concatenate([c[k] for _, c in parts]) for k in ("y", "dist", "colour", "contour", "sa")}
                    X0 = np.c_[C["dist"], C["colour"], C["contour"]]
                    q3[rn] = compare(X0, np.c_[X0, C["sa"]], C["y"].astype(int), g3) if rn == "core" else {}
                    q3[rn]["single"] = {k: auc(C["y"], C[k] if k != "sa" else -C[k]) for k in ("dist", "colour", "contour", "sa")}
                out[f"{ds}/{s}/b{blk}"] = dict(q1=q1, q2=q2, q3=q3)
                fmt = lambda q: f"{q['rgb']:.3f} → {q['rgb_sa']:.3f}, {q['delta']:+.3f} ({q['lo']:+.3f}…{q['hi']:+.3f})"
                tag = " **(primary)**" if (s, blk) == PRIMARY else ""
                L.append(f"| {s}{tag} | {blk} | {q1['sa'][0]:.3f} ({q1['sa'][1]:.3f}…{q1['sa'][2]:.3f}) / {q1['colour'][0]:.3f} / {q1['contour'][0]:.3f} | "
                         f"{fmt(q2)} | {fmt(q3['core'])} [{q3['core']['pos']}/{q3['core']['n'] - q3['core']['pos']} in {q3['core']['images']} img] | "
                         + " / ".join(f"{v:.3f}" for v in q3['core']['single'].values()) + f" | {q3['gt']['single']['sa']:.3f} |")
        L.append("")
    k = f"coco/{PRIMARY[0]}/b{PRIMARY[1]}"
    if k in out:
        q = out[k]["q3"]["core"]; ok = q["delta"] >= 0.05 and q["lo"] > 0
        L.append(f"## Decision (fixed in PREREG.md)\n\nQ3(a) on COCO-Cap, 32 px, block 1: ΔAUROC {q['delta']:+.3f} (95% CI {q['lo']:+.3f}…{q['hi']:+.3f}); "
                 f"rule: Δ ≥ +0.05 and CI lower bound > 0 → **{'CONTINUE: SA carries bleed information beyond RGB' if ok else 'CLOSE: no boundary mechanism is built'}**"
                 + (" (fewer images than registered: provisional)" if a.max_images else "") + "\n")
        out["decision"] = dict(delta=q["delta"], lo=q["lo"], hi=q["hi"], pass_=bool(ok))
    L.append("Q1: per-image AUROC of −log affinity (or colour distance / contour) for boundary-crossing vs within edges, averaged over "
             "distance bands 1–3. Q2/Q3: pooled out-of-fold AUROC of a logistic model (5 folds by image), image-bootstrap CI of the "
             "difference. Q3 RGB baseline = distance to the nearest reference cell, colour distance to the reference mean, intervening "
             "contour to the nearest reference cell; + SA = log mean affinity to the reference set.")
    txt = "\n".join(L) + "\n"; RUN.mkdir(parents=True, exist_ok=True)
    if not a.max_images:
        (RUN / "result.md").write_text(txt); (RUN / "result.json").write_text(json.dumps(out, indent=1, default=float))
    print(txt)


if __name__ == "__main__":
    main()
