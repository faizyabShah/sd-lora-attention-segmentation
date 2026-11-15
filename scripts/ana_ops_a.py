#!/usr/bin/env python3
"""Operator batch A (CPU, exploratory): label-free threshold / region operators on the stored BRACE maps
(CACHE/vocsim_maps 490, cococap_maps 707). Targets the two largest losses found in part 1: the fixed threshold
(oracle per-image tau +7 / +10 points) and far false positives ("bleed") on COCO-Cap.

  A1 stability threshold (SAM stability score / MSER): per image, the tau in the R2 grid whose mask changes least
     between its neighbours (IoU(M[tau-0.05], M[tau+0.05])); variant on the raw masks
  A2 area-matched threshold: the tau whose R2 mask area is closest to c * a_hat (a_hat = area of the A3 seed >= 0.5);
     c = 1, and c fitted on VOC-Sim only (median true area / a_hat), then applied unchanged to COCO-Cap
  A3 connected-component filtering of the BRACE mask: keep components with >= 5% of the mask's V3 mass / only the
     largest-mass component / components that contain a seed peak (A3 >= 0.9)
  A4 geodesic gating: V3 x exp(-d / d0), d = image-edge geodesic distance from the V3 core (>= 0.9), d0 = 90th
     percentile of d inside V3 >= 0.55
  A5 closed-form matting (Levin et al. 2008) at 256 px instead of DenseCRF; trimap fg V3 >= 0.75, bg V3 <= 0.30

python scripts/ana_ops_a.py run [--workers 16] [--max-images k]   -> CACHE/ops_a/<ds>/<id>.npz
python scripts/ana_ops_a.py report                                -> runs/analysis_cluster/ops_a.md
"""
from __future__ import annotations

import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import ana_ops_common as oc  # noqa: E402

STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
WORK = CACHE / "ops_a"; OUT = ROOT / "runs/analysis_cluster"


def geodesic(img, x, side=128):
    I = np.asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR), np.float32) / 255
    I = gaussian_filter(I, (1, 1, 0)); xs = st.up(x, (side, side)) if x.shape[0] != side else x
    idx = np.arange(side * side).reshape(side, side); rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[: side - dy, : side - dx]; b = idx[dy:, dx:]
        g = np.linalg.norm(I[: side - dy, : side - dx] - I[dy:, dx:], axis=2)
        rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; w += [g.ravel(), g.ravel()]
    w = np.concatenate(w); w = w / (w.mean() + 1e-8) + 0.05
    G = coo_matrix((w, (np.concatenate(rows), np.concatenate(cols))), shape=(side * side,) * 2).tocsr()
    src = np.flatnonzero(xs.ravel() >= 0.9)
    if len(src) == 0:
        src = np.argsort(-xs.ravel())[:20]
    d = dijkstra(G, indices=src, min_only=True).reshape(side, side)
    return st.up(d.astype(np.float32), x.shape)


def matting(img, x, side=256, eps=1e-7, lam=100.0):
    from numpy.lib.stride_tricks import sliding_window_view
    I = np.asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR), np.float64) / 255
    xs = st.up(x.astype(np.float32), (side, side)); h = w = side; win = 9
    idx = np.arange(h * w).reshape(h, w)
    Iw = sliding_window_view(I, (3, 3), axis=(0, 1)).reshape(h - 2, w - 2, 3, win).transpose(0, 1, 3, 2).reshape(-1, win, 3)
    iw = sliding_window_view(idx, (3, 3)).reshape(-1, win)
    mu = Iw.mean(1); X = Iw - mu[:, None]
    cov = X.transpose(0, 2, 1) @ X / win
    inv = np.linalg.inv(cov + eps / win * np.eye(3))
    vals = np.eye(win)[None] - (1 + X @ inv @ X.transpose(0, 2, 1)) / win
    L = coo_matrix((vals.ravel(), (np.repeat(iw, win, 1).ravel(), np.tile(iw, (1, win)).ravel())), shape=(h * w,) * 2).tocsr()
    fg = (xs >= 0.75).ravel(); bg = (xs <= 0.30).ravel(); known = (fg | bg).astype(np.float64)
    alpha = spsolve((L + lam * diags(known)).tocsc(), lam * fg.astype(np.float64))
    return np.clip(st.up(alpha.reshape(h, w).astype(np.float32), x.shape), 0, 1)


def one(job):
    ds, f = job; o = WORK / ds / f.name
    if o.exists():
        return
    oc.crf_init()
    z = np.load(f); img, t = z["image"], z["truth"].astype(bool); hw = t.shape; M = z["map/base"]
    x = st.up(st.mm(M[3]), hw); s = st.up(st.mm(M[0]), hw)
    res = {"cls": str(z["cls"]), "area": float(t.mean()), "a_hat": float((st.mm(M[0]) >= 0.5).mean())}
    Mk = oc.crf_grid((img, x))                                        # [13, H, W]
    res["brace|raw"] = oc.grid(x, t); res["brace+R2|raw"] = oc.cnt(Mk, t)
    # A1 stability scores (chosen at report time), A2 areas
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    res["stab_crf"] = np.array([iou(Mk[j - 1], Mk[j + 1]) if 0 < j < len(Mk) - 1 else -1 for j in range(len(Mk))])
    Rk = np.stack([x >= tt for tt in cc.THR_CRF])
    lo, hi = np.stack([x >= tt - 0.05 for tt in cc.THR_CRF]), np.stack([x >= tt + 0.05 for tt in cc.THR_CRF])
    res["stab_raw"] = np.array([iou(lo[j], hi[j]) for j in range(len(Rk))])
    res["crf_area"] = Mk.mean((1, 2))
    # A3 components of the BRACE mask
    m = Mk[oc.J55]; lab, n = label(m)
    if n:
        mass = np.bincount(lab.ravel(), weights=x.ravel(), minlength=n + 1)[1:]
        peak = np.array([(s[lab == i + 1] >= 0.9).any() for i in range(n)])
        keep = {"cc_mass05": mass >= 0.05 * mass.sum(), "cc_top1": np.arange(n) == mass.argmax(),
                "cc_seedpk": peak if peak.any() else np.arange(n) == mass.argmax()}
        for k, kk in keep.items():
            res[f"{k}|mask"] = oc.cnt(np.isin(lab, np.flatnonzero(kk) + 1), t)
    else:
        for k in ("cc_mass05", "cc_top1", "cc_seedpk"):
            res[f"{k}|mask"] = oc.cnt(m, t)
    res["n_cc"] = n
    # A4 geodesic gating
    d = geodesic(img, x); core = x >= 0.55
    d0 = np.percentile(d[core], 90) if core.any() else np.percentile(d, 10)
    xg = st.mm(x * np.exp(-d / max(d0, 1e-6)))
    res["geo|raw"] = oc.grid(xg, t); res["geo+R2|raw"] = oc.cnt(oc.crf_grid((img, xg)), t)
    # A5 matting
    al = matting(img, x)
    res["matte|raw"] = oc.grid(al, t); res["matte05|mask"] = oc.cnt(al >= 0.5, t)
    res["matte05+R2|mask"] = oc.cnt(oc.crf(img, al >= 0.5), t)
    o.parent.mkdir(parents=True, exist_ok=True)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def run(a):
    jobs = [(ds, f) for ds in ("voc", "coco") for f in sorted(STORED[ds].glob("*.npz"))[: a.max_images] if not f.stem.endswith(".tmp")]
    with Pool(a.workers) as p:
        for n, _ in enumerate(p.imap_unordered(one, jobs, chunksize=2), 1):
            if n % 100 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def select(S, key, rule):
    """Per-image R2 counts at the tau chosen by a label-free per-image rule."""
    j = np.array([rule(z) for z in S["_Z"]]); return S["brace+R2|raw"][np.arange(len(j)), j][:, None]


def report(a):
    S = {ds: oc.load(WORK / ds) for ds in ("voc", "coco")}
    S = {k: v for k, v in S.items() if v is not None}
    # A2 fitted constant on VOC-Sim only
    c_fit = float(np.median([float(z["area"]) / max(float(z["a_hat"]), 1e-4) for z in S["voc"]["_Z"]])) if "voc" in S else 1.0
    rules = {"stab_crf": lambda z: int(np.argmax(z["stab_crf"])), "stab_raw": lambda z: int(np.argmax(z["stab_raw"])),
             "area_c1": lambda z: int(np.argmin(np.abs(np.log(z["crf_area"] + 1e-6) - np.log(z["a_hat"] + 1e-6)))),
             "area_cfit": lambda z: int(np.argmin(np.abs(np.log(z["crf_area"] + 1e-6) - np.log(c_fit * z["a_hat"] + 1e-6))))}
    for ds in S:
        for k, r in rules.items():
            S[ds][f"{k}|mask"] = select(S[ds], k, r)
    groups = {"A1 stability τ": ("stab_crf", "stab_raw"), "A2 area-matched τ": ("area_c1", "area_cfit"),
              "A3 components": ("cc_mass05", "cc_top1", "cc_seedpk"), "A4 geodesic": ("geo",),
              "A5 matting": ("matte", "matte05", "matte05+R2")}
    L = ["# Operator batch A: label-free threshold and region operators on the stored BRACE maps (exploratory)\n",
         "`scripts/ana_ops_a.py`. Stored paper capture; ground truth only scores. A2's fitted constant c = "
         f"{c_fit:.3f} comes from VOC-Sim only and is applied unchanged to COCO-Cap. Nothing here is a registered test.\n"]
    for ds in S:
        n = len(S[ds]["grp"]); ncc = np.array([int(z["n_cc"]) for z in S[ds]["_Z"]])
        L += [f"## {oc.NAME[ds]} ({n} images; BRACE mask components: mean {ncc.mean():.1f}, median {np.median(ncc):.0f})\n"]
        L += oc.table(S[ds], None, groups=groups) + [""]
        jt = {k: np.array([r(z) for z in S[ds]["_Z"]]) for k, r in rules.items()}
        L.append("Chosen τ (mean ± sd): " + ", ".join(f"{k} {cc.THR_CRF[v].mean():.3f} ± {cc.THR_CRF[v].std():.3f}" for k, v in jt.items()) + "\n")
    txt = "\n".join(L) + "\n"; OUT.mkdir(parents=True, exist_ok=True); (OUT / "ops_a.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--workers", type=int, default=16); p.add_argument("--max-images", type=int)
    sub.add_parser("report")
    a = ap.parse_args(); {"run": run, "report": report}[a.cmd](a)
