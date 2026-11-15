#!/usr/bin/env python3
"""Map analysis, part 1 (CPU, exploratory diagnostics): where does frozen BRACE lose on the generated sets?

Uses the stored paper capture (CACHE/vocsim_maps, cococap_maps: A3 seed, P1, V0, V3, C16 at 64x64, background-word
A3 maps, image, truth) and its score counts (CACHE/vocsim_scores, cococap_scores). Ground truth is used only to
measure. Nothing here is a method or a registered test.

  1a  oracle bounds: fixed tau 0.55 / OOF / per-image best tau (raw, +R2) / perfect ranking of each map's pixels
  1b  error anatomy of the BRACE mask: FN / FP split into boundary band (<= 5 px), interior, far background;
      by object size and class; per-image failure type
  1c  worst 10% contact sheets
  1d  calibration: foreground fraction per map-value bin; per-image best tau vs label-free statistics; Otsu threshold
  1e  background-word maps at BRACE's false positives

python scripts/ana_part1.py run [--workers 16]      -> CACHE/ana/part1_<ds>.npz, runs/analysis_cluster/part1.md, fig/
"""
from __future__ import annotations

import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

OUT = ROOT / "runs/analysis_cluster"; FIG = OUT / "fig"; WORK = CACHE / "ana"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
SCORES = {"voc": CACHE / "vocsim_scores", "coco": CACHE / "cococap_scores"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
MAPS = ("seed", "P1", "V0", "V3", "C16")
BG = ("wall", "floor", "ground", "sky", "grass", "tree", "building", "road")
TAU = 0.55; BAND = 5


def best_iou(x, t):
    """Best IoU over all thresholds of map x against mask t (perfect use of the map's ranking), and AUROC."""
    v = x.reshape(-1); g = t.reshape(-1).astype(np.int64); o = np.argsort(-v, kind="stable")
    tp = np.cumsum(g[o]); k = np.arange(1, len(v) + 1); G = g.sum()
    iou = tp / (k + G - tp); r = np.empty(len(v)); r[o] = np.arange(len(v), 0, -1)
    n1 = G; n0 = len(v) - G
    auc = (r[g == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    return float(iou.max()), float(auc)


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins, range=(0, 1)); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    mt = m[-1]; w1 = w[-1] - w; ok = (w > 0) & (w1 > 0)
    s = np.where(ok, (mt * w / w[-1] - m) ** 2 / np.maximum(w * w1, 1), 0)
    return float(c[int(np.argmax(s))])


def one(job):
    ds, f = job
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, t = z["image"], z["truth"].astype(bool); hw = t.shape; M = z["map/base"]
    X = {k: st.up(st.mm(M[i]), hw) for i, k in enumerate(MAPS)}; x = X["V3"]
    mask = apply_ovam_densecrf(img, x >= TAU, st.CRF_SITE).astype(bool)
    r = {"id": f.stem, "cls": str(z["cls"]), "area": float(t.mean())}
    tp = (mask & t).sum(); fp = (mask & ~t).sum(); fn = (~mask & t).sum(); G = max(t.sum(), 1)
    r.update(iou=tp / max(tp + fp + fn, 1), prec=tp / max(tp + fp, 1), rec=tp / G, pred_area=float(mask.mean()))
    din = distance_transform_edt(t); dout = distance_transform_edt(~t)
    band_in = t & (din <= BAND); band_out = ~t & (dout <= BAND)
    r.update(fn_band=(~mask & band_in).sum() / G, fn_int=(~mask & t & ~band_in).sum() / G,
             fp_band=(mask & band_out).sum() / G, fp_far=(mask & ~t & ~band_out).sum() / G)
    r["n_inst"] = int(label(t)[1]); r["n_pred_cc"] = int(label(mask)[1])
    for k, v in X.items():
        r[f"best_{k}"], r[f"auc_{k}"] = best_iou(v, t)
    # label-free statistics of the V3 map and seed
    r.update(a_hat=float((st.mm(M[0]) >= 0.5).mean()), v3_mean=float(x.mean()), v3_frac55=float((x >= TAU).mean()),
             v3_entropy=float(-(np.histogram(x, 20, (0, 1))[0] / x.size * np.log(np.histogram(x, 20, (0, 1))[0] / x.size + 1e-12)).sum()))
    th = otsu(x); r["otsu"] = th
    mo = apply_ovam_densecrf(img, x >= th, st.CRF_SITE).astype(bool)
    r["iou_otsu"] = (mo & t).sum() / max((mo | t).sum(), 1)
    # calibration histogram: per bin of x, (#pixels, #foreground)
    b = np.minimum((x * 20).astype(int), 19)
    r["cal_n"] = np.bincount(b.ravel(), minlength=20); r["cal_fg"] = np.bincount(b.ravel(), weights=t.ravel(), minlength=20)
    # background words at false positives vs true positives of the BRACE mask
    W = np.stack([st.up(st.mm(z[f"word/{w}"]), hw) for w in BG if f"word/{w}" in z.files])
    wmax = W.max(0); fpm = mask & ~t; tpm = mask & t
    if fpm.sum() > 50 and tpm.sum() > 50:
        y = np.r_[np.ones(fpm.sum()), np.zeros(tpm.sum())]; s = np.r_[wmax[fpm], wmax[tpm]]
        o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
        n1, n0 = fpm.sum(), tpm.sum(); r["bg_auc_fp_vs_tp"] = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        r["bg_fp_argmax"] = np.bincount(W[:, fpm].argmax(0), minlength=len(BG))
    else:
        r["bg_auc_fp_vs_tp"] = np.nan; r["bg_fp_argmax"] = np.zeros(len(BG), int)
    # failure type
    if r["iou"] >= 0.5:
        ft = "ok"
    elif tp / G < 0.2:
        ft = "miss (wrong region)"
    elif r["prec"] >= 0.85 and r["rec"] < 0.6:
        ft = "part only (under-segment)"
    elif r["rec"] >= 0.8 and r["prec"] < 0.6:
        ft = "bleed (over-segment)"
    else:
        ft = "mixed"
    r["ftype"] = ft
    # thumbnail for the contact sheet: image | truth | V3 | mask, 128 px each
    th128 = lambda a: np.asarray(Image.fromarray(a).resize((128, 128), Image.BILINEAR))
    gray = lambda m: np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2)
    r["thumb"] = np.concatenate([th128(img), th128(gray(t.astype(float))), th128(gray(x)), th128(gray(mask.astype(float)))], 1)
    return ds, r


def macro(vals, cls):
    return float(np.mean([np.mean(vals[cls == c]) for c in np.unique(cls)]))


def run(a):
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, f) for ds in ("voc", "coco") for f in sorted(STORED[ds].glob("*.npz")) if not f.stem.endswith(".tmp")]
    if a.max_images:
        jobs = [j for ds in ("voc", "coco") for j in [x for x in jobs if x[0] == ds][: a.max_images]]
    R = {"voc": [], "coco": []}
    with Pool(a.workers) as p:
        for n, (ds, r) in enumerate(p.imap_unordered(one, jobs, chunksize=4), 1):
            R[ds].append(r)
            if n % 100 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    L = ["# Map analysis part 1: where does frozen BRACE lose? (exploratory diagnostics)\n",
         "Stored paper capture (VOC-Sim 490, COCO-Cap 707). Ground truth is used only to measure. BRACE mask = V3 → "
         "τ 0.55 → DenseCRF R2. Per-image IoU means are class-macro (mean over classes of the per-image mean) unless "
         "stated; the paper's pooled class-macro mIoU is given for reference.\n"]
    for ds in ("voc", "coco"):
        rs = sorted(R[ds], key=lambda r: r["id"])
        if not rs:
            continue
        cls = np.array([r["cls"] for r in rs]); col = lambda k: np.array([r[k] for r in rs], float)
        S = cc.load_gen(SCORES[ds]); idx = {g: i for i, g in enumerate(S["grp"])}; ii = np.array([idx[r["id"]] for r in rs])
        C = S["base|V3+R2|raw"][ii]; Craw = S["base|V3|raw"][ii]
        per = lambda C: C[..., 0] / np.maximum(C[..., :3].sum(-1), 1)
        j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
        pooled55 = cc.miou(S["base|V3+R2|raw"][:, j55][:, None], S["cls"], np.ones(len(S["grp"]), bool))[0]
        oofm = cc.oof(S, S["base|V3+R2|raw"])[0]
        Cb = C[np.arange(len(C)), per(C).argmax(1)]
        pooled_or = cc.miou(Cb[:, None], cls, np.ones(len(cls), bool))[0]
        L += [f"## {NAME[ds]} ({len(rs)} images)\n", "### 1a. Bounds\n",
              "| what | per-image IoU (class-macro) | pooled class-macro mIoU |", "|---|---:|---:|",
              f"| BRACE @ τ 0.55 + R2 | {macro(per(C)[:, j55], cls):.3f} | {pooled55:.3f} |",
              f"| BRACE, OOF τ + R2 | — | {oofm:.3f} |",
              f"| oracle per-image τ + R2 (grid 0.30–0.90) | {macro(per(C).max(1), cls):.3f} | {pooled_or:.3f} |",
              f"| oracle per-image τ, raw V3 (grid) | {macro(per(Craw).max(1), cls):.3f} | — |"]
        for k in MAPS:
            L.append(f"| perfect ranking of {k} (best of all thresholds, raw) | {macro(col('best_' + k), cls):.3f} | — |")
        L += ["", "AUROC (pixel ranking quality), class-macro: " + ", ".join(f"{k} {macro(col('auc_' + k), cls):.3f}" for k in MAPS), "",
              "### 1b. Error anatomy of the BRACE mask (fractions of the object area, mean over images)\n",
              "| subset | n | IoU | FN boundary | FN interior | FP boundary | FP far | instances (mean) | predicted components (mean) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        area = col("area"); q = np.quantile(area, [0, .2, .4, .6, .8, 1])
        subsets = [("all", np.ones(len(rs), bool))] + [(f"area quintile {i + 1} ({q[i]:.2f}–{q[i + 1]:.2f})",
                                                         (area >= q[i]) & (area <= q[i + 1])) for i in range(5)]
        subsets += [("multi-instance (≥2 components)", col("n_inst") >= 2)]
        for nm, s in subsets:
            L.append(f"| {nm} | {s.sum()} | {col('iou')[s].mean():.3f} | {col('fn_band')[s].mean():.3f} | {col('fn_int')[s].mean():.3f} | "
                     f"{col('fp_band')[s].mean():.3f} | {col('fp_far')[s].mean():.3f} | {col('n_inst')[s].mean():.2f} | {col('n_pred_cc')[s].mean():.2f} |")
        L += ["", "Per class (mean IoU, prec, rec, FN interior, FP far):", "",
              "| class | n | IoU | prec | rec | FN interior | FP far | worst failure type |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        ft = np.array([r["ftype"] for r in rs])
        for c in np.unique(cls):
            s = cls == c; bad = [x for x in ft[s] if x != "ok"]
            w = max(set(bad), key=bad.count) if bad else "—"
            L.append(f"| {c} | {s.sum()} | {col('iou')[s].mean():.3f} | {col('prec')[s].mean():.3f} | {col('rec')[s].mean():.3f} | "
                     f"{col('fn_int')[s].mean():.3f} | {col('fp_far')[s].mean():.3f} | {w} ({bad.count(w) if bad else 0}) |")
        L += ["", "Failure types (IoU < 0.5): " + ", ".join(f"{t} {int((ft == t).sum())}" for t in sorted(set(ft))), ""]
        # 1c contact sheet of the worst 10%
        worst = np.argsort(col("iou"))[: max(1, len(rs) // 10)]
        rows = [np.concatenate([rs[i]["thumb"], np.full((128, 8, 3), 255, np.uint8)], 1) for i in worst]
        cols = 2; pad = (-len(rows)) % cols; rows += [np.full_like(rows[0], 255)] * pad
        sheet = np.concatenate([np.concatenate(rows[i:i + cols], 1) for i in range(0, len(rows), cols)], 0)
        Image.fromarray(sheet).save(FIG / f"part1_worst_{ds}.jpg", quality=85)
        L += [f"### 1c. Worst 10% ({len(worst)} images)\n",
              f"Contact sheet `fig/part1_worst_{ds}.jpg` (per image: image | truth | V3 | BRACE mask; in IoU order, "
              "row-major, two per row).\n", "| rank | image | class | IoU | prec | rec | type |", "|---:|---|---|---:|---:|---:|---|"]
        for n, i in enumerate(worst, 1):
            L.append(f"| {n} | {rs[i]['id']} | {rs[i]['cls']} | {rs[i]['iou']:.2f} | {rs[i]['prec']:.2f} | {rs[i]['rec']:.2f} | {rs[i]['ftype']} |")
        # 1d calibration and threshold predictability
        cn = sum(r["cal_n"] for r in rs); cf = sum(r["cal_fg"] for r in rs)
        btau = cc.THR_CRF[per(C).argmax(1)]
        L += ["", "### 1d. Calibration and threshold\n", "Foreground fraction per V3 value bin (all pixels pooled):", "",
              "| V3 bin | " + " | ".join(f"{i / 20:.2f}" for i in range(0, 20, 2)) + " |", "|---|" + "---:|" * 10,
              "| fg fraction | " + " | ".join(f"{(cf[i] + cf[i + 1]) / max(cn[i] + cn[i + 1], 1):.2f}" for i in range(0, 20, 2)) + " |",
              "| share of pixels | " + " | ".join(f"{(cn[i] + cn[i + 1]) / cn.sum():.3f}" for i in range(0, 20, 2)) + " |", "",
              f"Per-image best τ (+R2): mean {btau.mean():.3f}, sd {btau.std():.3f}; share at the grid edges "
              f"(0.30 / 0.90): {(btau == cc.THR_CRF[0]).mean():.2f} / {(btau == cc.THR_CRF[-1]).mean():.2f}.", "",
              "| label-free statistic | Spearman ρ with best τ | Spearman ρ with true area |", "|---|---:|---:|"]
        for k in ("a_hat", "v3_mean", "v3_frac55", "v3_entropy", "otsu"):
            L.append(f"| {k} | {spearmanr(col(k), btau)[0]:+.3f} | {spearmanr(col(k), area)[0]:+.3f} |")
        L += ["", f"Otsu threshold on V3 then R2 (label-free, exploratory): per-image IoU (class-macro) {macro(col('iou_otsu'), cls):.3f} "
                  f"vs fixed 0.55 {macro(col('iou'), cls):.3f}.", "",
              "### 1e. Background-word maps at BRACE's false positives\n"]
        au = col("bg_auc_fp_vs_tp"); ok = np.isfinite(au); arg = sum(r["bg_fp_argmax"] for r in rs)
        L += [f"Images with ≥ 50 FP and ≥ 50 TP pixels: {ok.sum()}. AUROC of max background-word map, FP vs TP pixels inside "
              f"the mask: mean {np.nanmean(au):.3f} (median {np.nanmedian(au):.3f}; share > 0.7: {(au[ok] > 0.7).mean():.2f}).", "",
              "Dominant background word at FP pixels: " + ", ".join(f"{w} {arg[i] / max(arg.sum(), 1):.2f}" for i, w in enumerate(BG)), ""]
        np.savez(WORK / f"part1_{ds}.npz", **{k: np.array([r[k] for r in rs]) for k in rs[0] if k not in ("thumb", "cal_n", "cal_fg", "bg_fp_argmax")})
    txt = "\n".join(L) + "\n"; (OUT / "part1.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--workers", type=int, default=16); p.add_argument("--max-images", type=int)
    a = ap.parse_args(); run(a)
