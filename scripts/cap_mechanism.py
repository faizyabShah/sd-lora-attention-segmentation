#!/usr/bin/env python3
"""Mechanism diagnostic (pre-registered in runs/06_cap_mechanism/PREREG.md): does BRACE's cap (V3 vs V0) help most where
SCLIP sees competing objects? Stored paper captures (one capture: image, truth, V0, V3); nothing is regenerated.
Ground truth only measures (and the diagnostic area control).

python scripts/cap_mechanism.py sclip  [--limit k] [--tag t]      GPU (CPU fallback): SCLIP VOC21 21-class probabilities
python scripts/cap_mechanism.py all    [--limit k] [--tag t] [--workers 32]   CPU: masks, gate, statistics, report
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/06_cap_mechanism"
MAPS = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
SCORES = {"voc": CACHE / "vocsim_scores", "coco": CACHE / "cococap_scores"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
TAU3 = 0.55; REG = ("0.5", "0.3", "am"); FALLBACK_ZERO = 0.5
BANDS = (0, 8, 16, 32, 64, np.inf)
NBOOT = 2000; GATE = 1e-3
I0, I3 = cc.MAPS.index("V0"), cc.MAPS.index("V3")


def files(ds, limit=None):
    fs = sorted(f for f in MAPS[ds].glob("*.npz") if not f.stem.endswith(".tmp"))
    return fs[:limit] if limit else fs


def work(tag):
    return CACHE / ("cap_mech" + (f"_{tag}" if tag else ""))


# ------------------------------------------------------------------ SCLIP (GPU)
def sclip(a):
    import torch, torch.nn.functional as F
    import sg_clip_dense as cd
    from sg_gen_clip import voc_k
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device", dev, flush=True)
    M = cd.DenseCLIP("B16", dev); n21, i21 = cd.split([cd.VOC21_BG] + cd.VOC20_NAMES); T21 = M.text(n21)
    for ds in ("voc", "coco"):
        out = work(a.tag) / f"sclip21_{ds}"; out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter(); n = 0
        for f in files(ds, a.limit):
            o = out / f.name
            if o.exists():
                continue
            z = np.load(f); img = z["image"]; cls = str(z["cls"])
            _, P = cd.postprocess(M.logits(img, T21), i21, 65, 0.0, None)       # [21,H,W], no prob/area thresholds
            P128 = F.adaptive_avg_pool2d(P[None].float(), (128, 128))[0].cpu().numpy().astype(np.float16)
            tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, P=P128, k=voc_k(cls), cls=cls); os.replace(tmp, o); n += 1
        print(f"[sclip21:{ds}] {n} images {(time.perf_counter() - t0) / max(n, 1):.2f}s/img", flush=True)


# ------------------------------------------------------------------ per image (CPU)
def iou(c):
    return c[0] / max(c[0] + c[1] + c[2], 1)


def pass1(job):
    """V0+R2 counts on the CRF grid, V3+R2 at 0.55 (gate against the stored counts)."""
    f, ds, o = Path(job[0]), job[1], Path(job[2])
    if o.exists():
        return
    import ana_ops_common as oc
    oc.crf_init()
    z = np.load(f); img = z["image"]; truth = z["truth"].astype(bool); v = np.ones_like(truth); ot = np.zeros_like(truth)
    x0 = st.up(st.mm(z["map/base"][I0]), truth.shape); x3 = st.up(st.mm(z["map/base"][I3]), truth.shape)
    g0 = cc.counts(np.stack([oc.crf(img, x0 >= t) for t in cc.THR_CRF]), truth, v, ot)
    c3 = cc.counts(oc.crf(img, x3 >= TAU3)[None], truth, v, ot)[0]
    stored = np.load(SCORES[ds] / f.name)["base|V3+R2|raw"][list(np.round(cc.THR_CRF, 3)).index(TAU3)]
    gate = float(np.abs(c3[:3] - stored[:3]).max() / truth.size)
    np.savez(o, g0=g0, c3=c3, gate=gate)


def bandtab(fp, rem, onr, dist):
    T = np.zeros((len(BANDS) - 1, 4), np.int64)                   # a=onR&rem, b=onR&ret, c=offR&rem, d=offR&ret
    for j in range(len(BANDS) - 1):
        m = fp & (dist > BANDS[j]) & (dist <= BANDS[j + 1])
        T[j] = [(m & onr & rem).sum(), (m & onr & ~rem).sum(), (m & ~onr & rem).sum(), (m & ~onr & ~rem).sum()]
    return T


def pass2(job):
    f, ds, o, tau0, sdir = Path(job[0]), job[1], Path(job[2]), job[3], Path(job[4])
    if o.exists():
        return
    import torch, torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt
    import ana_ops_common as oc
    torch.set_num_threads(1); oc.crf_init()
    z = np.load(f); img = z["image"]; truth = z["truth"].astype(bool); v = np.ones_like(truth); ot = np.zeros_like(truth)
    x0 = st.up(st.mm(z["map/base"][I0]), truth.shape); x3 = st.up(st.mm(z["map/base"][I3]), truth.shape)
    M0 = oc.crf(img, x0 >= tau0); M3 = oc.crf(img, x3 >= TAU3)
    s = np.load(sdir / f.name); k = int(s["k"])
    P = F.interpolate(torch.from_numpy(s["P"].astype(np.float32))[None], size=truth.shape, mode="bilinear",
                      align_corners=False)[0].numpy()
    comp = [j for j in range(1, 21) if j != k]; q = P[comp].max(0)
    am = P.argmax(0); K = int(sum((am == j).mean() > 0.01 for j in comp))
    dist = distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
    fp0 = M0 & ~truth; rem = ~M3
    res = {"cls": str(z["cls"]), "area": float(truth.mean()), "K": K,
           "c0": cc.counts(M0[None], truth, v, ot)[0], "c3": cc.counts(M3[None], truth, v, ot)[0]}
    regions = {"0.5": q > 0.5, "0.3": q > 0.3, "am": np.isin(am, comp)}
    for e, R in regions.items():
        res[f"C{e}"] = float(R.mean())
        res[f"fpR0_{e}"] = int((fp0 & R).sum()); res[f"fpR3_{e}"] = int((M3 & ~truth & R).sum())
        res[f"tab_{e}"] = bandtab(fp0, rem, R, dist)
    np.savez(o, **res)


# ------------------------------------------------------------------ statistics
def ols_beta(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


def design(zc, la, cls, with_area=True):
    U = np.unique(cls); D = (cls[:, None] == U[None]).astype(float)
    cols = [zc[:, None]] + ([la[:, None]] if with_area else []) + [D]
    return np.hstack(cols)


def boot_strat(n_items, cls, fn, seed=0):
    rng = np.random.default_rng(seed); idx_by = [np.flatnonzero(cls == c) for c in np.unique(cls)]; out = []
    for _ in range(NBOOT):
        ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in idx_by]); out.append(fn(ix))
    out = np.array(out, float); out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5]) if len(out) else np.array([np.nan, np.nan])


def boot_cluster(cls, fn, seed=1):
    rng = np.random.default_rng(seed); U = np.unique(cls); idx_by = [np.flatnonzero(cls == c) for c in U]; out = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(U), len(U))
        ix = np.concatenate([idx_by[p] for p in pick]); lab = np.concatenate([np.full(len(idx_by[p]), i) for i, p in enumerate(pick)])
        out.append(fn(ix, lab))
    out = np.array(out, float); out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5]) if len(out) else np.array([np.nan, np.nan])


def h1(y, C, area, cls, with_area=True):
    if np.std(C) < 1e-12:
        return {"beta": np.nan, "ci": [np.nan, np.nan], "ci_cluster": [np.nan, np.nan], "note": "C constant"}
    zc = (C - C.mean()) / C.std(); la = np.log(np.maximum(area, 1e-6))
    beta = ols_beta(y, design(zc, la, cls, with_area))
    ci = boot_strat(len(y), cls, lambda ix: ols_beta(y[ix], design(zc[ix], la[ix], cls[ix], with_area)) if np.std(zc[ix]) > 0 else np.nan)
    cc_ = boot_cluster(cls, lambda ix, lab: ols_beta(y[ix], design(zc[ix], la[ix], lab, with_area)) if np.std(zc[ix]) > 0 else np.nan)
    return {"beta": float(beta), "ci": [float(x) for x in ci], "ci_cluster": [float(x) for x in cc_]}


def spearman_within(y, C, cls):
    from scipy.stats import spearmanr
    r = []
    for c in np.unique(cls):
        m = cls == c
        if m.sum() >= 5 and np.std(C[m]) > 0 and np.std(y[m]) > 0:
            r.append(spearmanr(y[m], C[m])[0])
    return float(np.mean(r)) if r else np.nan, len(r)


def mh_logor(T):
    """T: [n_strata, 4] = a, b, c, d. Mantel–Haenszel common OR, returned as log."""
    n = T.sum(1); m = n > 0; a, b, c, d = (T[m, i].astype(float) for i in range(4)); n = n[m]
    num = (a * d / n).sum(); den = (b * c / n).sum()
    return float(np.log(num / den)) if num > 0 and den > 0 else np.nan


def raw_or(T):
    a, b, c, d = T.reshape(-1, 4).sum(0).astype(float)
    return float(np.log(a * d / (b * c))) if min(a, b, c, d) > 0 else np.nan


def cmiou(C, cls):
    return float(np.mean([iou(C[cls == c].sum(0)) for c in np.unique(cls)]))


def prec_rec(C):
    s = C.sum(0); return float(s[0] / max(s[0] + s[1], 1)), float(s[0] / max(s[0] + s[2], 1))


def macro_pr(C, cls):
    P = [prec_rec(C[cls == c]) for c in np.unique(cls)]
    return float(np.mean([p for p, _ in P])), float(np.mean([r for _, r in P]))


# ------------------------------------------------------------------ driver
def all_(a):
    W = work(a.tag); out = {}; RUN.mkdir(parents=True, exist_ok=True)
    rd = RUN / (f"test_{a.tag}" if a.tag else ""); rd.mkdir(parents=True, exist_ok=True)
    # pass 1
    P1 = {}
    for ds in ("voc", "coco"):
        d = W / f"p1_{ds}"; d.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), ds, str(d / f.name)) for f in files(ds, a.limit)]
        with Pool(a.workers) as p:
            p.map(pass1, jobs, chunksize=1)
        Z = [np.load(j[2]) for j in jobs]
        P1[ds] = {"g0": np.stack([z["g0"] for z in Z]), "c3": np.stack([z["c3"] for z in Z]),
                  "gate": np.array([float(z["gate"]) for z in Z]),
                  "cls": np.array([str(np.load(j[0])["cls"]) for j in jobs])}
        gmax = float(P1[ds]["gate"].max()); out[f"gate_{ds}"] = gmax
        print(f"[gate:{ds}] max |Δcount|/pixels {gmax:.2e} over {len(jobs)} images", flush=True)
        if gmax > GATE:
            (rd / "result.md").write_text(f"# GATE FAILED on {NAME[ds]}: max {gmax:.3e} > {GATE}\n"); sys.exit(1)
    # tau0 from full VOC-Sim (ties -> lower); V3 with the same rule as a check
    S = cc.load_gen(SCORES["voc"]); vocc = P1["voc"]["cls"]
    m0 = np.array([cmiou(P1["voc"]["g0"][:, i], vocc) for i in range(len(cc.THR_CRF))])
    tau0 = float(cc.THR_CRF[int(np.argmax(m0))])
    keep = np.isin(S["grp"], [f.stem for f in files("voc", a.limit)])
    m3 = np.array([cmiou(S["base|V3+R2|raw"][keep][:, i], S["cls"][keep]) for i in range(len(cc.THR_CRF))])
    out["tau0"] = tau0; out["tau3_rule"] = float(cc.THR_CRF[int(np.argmax(m3))])
    print(f"tau0 {tau0}  (V3 by the same rule: {out['tau3_rule']})", flush=True)
    # raw (no CRF) taus from stored counts
    Sv = cc.load_gen(SCORES["voc"]); kv = np.isin(Sv["grp"], [f.stem for f in files("voc", a.limit)])
    traw = {m: int(np.argmax([cmiou(Sv[f"base|{m}|raw"][kv][:, i], Sv["cls"][kv]) for i in range(len(cc.THR))])) for m in ("V0", "V3")}
    out["tau_raw"] = {m: float(cc.THR[i]) for m, i in traw.items()}
    # pass 2
    L = ["# Does BRACE's cap help most where competing objects are present? (mechanism diagnostic; `PREREG.md`)\n",
         f"Stored paper captures, one capture per image. V0 + R2 at τ₀ = {tau0:.2f} (full-VOC rule; the same rule gives "
         f"V3 {out['tau3_rule']:.2f}); V3 + R2 at 0.55. Gate: max count difference / pixels {out['gate_voc']:.1e} (VOC), "
         f"{out['gate_coco']:.1e} (COCO). Raw secondary: V0 τ {out['tau_raw']['V0']:.3f}, V3 τ {out['tau_raw']['V3']:.3f}.\n"]
    for ds in ("voc", "coco"):
        d = W / f"p2_{ds}"; d.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), ds, str(d / f.name), tau0, str(W / f"sclip21_{ds}")) for f in files(ds, a.limit)]
        with Pool(a.workers) as p:
            p.map(pass2, jobs, chunksize=1)
        Z = [np.load(j[2]) for j in jobs]; ids = [Path(j[0]).stem for j in jobs]
        cls = np.array([str(z["cls"]) for z in Z]); area = np.array([float(z["area"]) for z in Z])
        c0 = np.stack([z["c0"] for z in Z]); c3 = np.stack([z["c3"] for z in Z])
        assert np.array_equal(c3[:, :3], P1[ds]["c3"][:, :3])
        dI = np.array([iou(x) for x in c3]) - np.array([iou(x) for x in c0])
        Sd = cc.load_gen(SCORES[ds]); pos = {g: i for i, g in enumerate(Sd["grp"])}; ix = np.array([pos[i] for i in ids])
        dIraw = (np.array([iou(x) for x in Sd["base|V3|raw"][ix][:, traw["V3"]]])
                 - np.array([iou(x) for x in Sd["base|V0|raw"][ix][:, traw["V0"]]]))
        R = {"n": len(ids)}
        for e in REG:
            C = np.array([float(z[f"C{e}"]) for z in Z])
            R[f"C{e}_dist"] = {"mean": float(C.mean()), "q10_50_90": [float(x) for x in np.percentile(C, [10, 50, 90])],
                               "frac_zero": float((C == 0).mean())}
            R[f"H1_{e}"] = h1(dI, C, area, cls)
            R[f"H1_{e}_noarea"] = h1(dI, C, area, cls, with_area=False)
            R[f"H1_{e}_raw"] = h1(dIraw, C, area, cls)
            R[f"spearman_{e}"] = spearman_within(dI, C, cls)
            T = np.stack([z[f"tab_{e}"] for z in Z])                       # [n, bands, 4]
            lo = mh_logor(T.reshape(-1, 4))
            R[f"H2_{e}"] = {"logOR_MH": lo,
                            "ci": [float(x) for x in boot_strat(len(T), cls, lambda ii: mh_logor(T[ii].reshape(-1, 4)))],
                            "logOR_unstrat": raw_or(T),
                            "logOR_band": [mh_logor(T[:, j]) for j in range(T.shape[1])],
                            "n_fp0_band": [int(T[:, j].sum()) for j in range(T.shape[1])],
                            "frac_fp0_onR": float(T[..., :2].sum() / max(T.sum(), 1))}
            fr0 = sum(int(z[f"fpR0_{e}"]) for z in Z); fr3 = sum(int(z[f"fpR3_{e}"]) for z in Z)
            R[f"fp_split_{e}"] = {"onR_V0": fr0, "onR_V3": fr3, "offR_V0": int(c0[:, 1].sum() - fr0), "offR_V3": int(c3[:, 1].sum() - fr3)}
        Kc = np.array([int(z["K"]) for z in Z]).astype(float)
        R["H1_K"] = h1(dI, Kc, area, cls); R["K_dist"] = {"mean": float(Kc.mean()), "frac_zero": float((Kc == 0).mean())}
        gt = c0[:, 0] + c0[:, 2]
        hybA = np.stack([c0[:, 0], c3[:, 1], c0[:, 2], c0[:, 3]], 1)          # only FP changes
        hybB = np.stack([c3[:, 0], c0[:, 1], c3[:, 2], c0[:, 3]], 1)          # only TP/FN changes
        R["H3"] = {"mIoU_V0": cmiou(c0, cls), "mIoU_V3": cmiou(c3, cls), "mIoU_onlyFP": cmiou(hybA, cls),
                   "mIoU_onlyTP": cmiou(hybB, cls), "pooled_PR_V0": prec_rec(c0), "pooled_PR_V3": prec_rec(c3),
                   "macro_PR_V0": macro_pr(c0, cls), "macro_PR_V3": macro_pr(c3, cls),
                   "dFP": int(c3[:, 1].sum() - c0[:, 1].sum()), "dTP": int(c3[:, 0].sum() - c0[:, 0].sum()),
                   "GT": int(gt.sum()), "frac_up_down_same": [float((dI > 1e-3).mean()), float((dI < -1e-3).mean()),
                                                              float((np.abs(dI) <= 1e-3).mean())],
                   "mean_dIoU": float(dI.mean())}
        out[ds] = R
    # verdict (COCO-Cap, eta 0.5, R2)
    prim = "am" if out["coco"]["C0.5_dist"]["frac_zero"] > FALLBACK_ZERO else "0.5"; out["primary_region"] = prim
    hc, h2c = out["coco"][f"H1_{prim}"], out["coco"][f"H2_{prim}"]
    a1 = np.isfinite(hc["beta"]) and hc["beta"] > 0 and hc["ci"][0] > 0
    a2 = np.isfinite(h2c["logOR_MH"]) and h2c["logOR_MH"] > 0 and h2c["ci"][0] > 0
    out["verdict"] = "Supported" if a1 and a2 else "Partial" if a1 or a2 else "Not supported"
    out["H1_pass"], out["H2_pass"] = bool(a1), bool(a2)
    L.append(f"## Verdict (COCO-Cap, region {prim}, R2): **{out['verdict']}** (H1 {'pass' if a1 else 'fail'}, H2 {'pass' if a2 else 'fail'})\n")
    f3 = lambda x: f"{x:+.4f}" if np.isfinite(x) else "nan"
    ci = lambda c: f"[{f3(c[0])}, {f3(c[1])}]"
    L.append("## H1: Δ IoU(V3+R2 − V0+R2) ~ z(C) + log area + class\n")
    L.append("| set | score | β per SD | image-strat CI | class-cluster CI | no-area β (CI) | raw-mask β (CI) | within-class ρ (classes) |")
    L.append("|---|---|---:|---|---|---|---|---|")
    for ds in ("voc", "coco"):
        R = out[ds]
        for e in REG:
            h, hn, hr, sp = R[f"H1_{e}"], R[f"H1_{e}_noarea"], R[f"H1_{e}_raw"], R[f"spearman_{e}"]
            L.append(f"| {NAME[ds]} | C, η {e} | {f3(h['beta'])} | {ci(h['ci'])} | {ci(h['ci_cluster'])} | "
                     f"{f3(hn['beta'])} {ci(hn['ci'])} | {f3(hr['beta'])} {ci(hr['ci'])} | {sp[0]:+.3f} ({sp[1]}) |")
        h = R["H1_K"]
        L.append(f"| {NAME[ds]} | K (count) | {f3(h['beta'])} | {ci(h['ci'])} | {ci(h['ci_cluster'])} | | | |")
    L.append("\n**Score distributions:**\n")
    for ds in ("voc", "coco"):
        R = out[ds]
        for e in REG:
            dd = R[f"C{e}_dist"]
            L.append(f"- {NAME[ds]}, C at η {e}: mean {dd['mean']:.3f}; 10/50/90% {', '.join(f'{x:.3f}' for x in dd['q10_50_90'])}; "
                     f"zero on {dd['frac_zero']:.0%} of images.")
        L.append(f"- {NAME[ds]}, K: mean {R['K_dist']['mean']:.2f}; zero on {R['K_dist']['frac_zero']:.0%} of images.")
    L.append("\n## H2: is a V0 false-positive pixel more likely to be removed by V3 when it lies on a competitor?\n")
    L.append("Mantel–Haenszel log odds ratio, strata = image × distance band from the ground-truth object.\n")
    L.append("| set | η | log OR (MH) | CI | unstratified | per band (0–8, 8–16, 16–32, 32–64, >64 px) | FP₀ on R |")
    L.append("|---|---|---:|---|---:|---|---:|")
    for ds in ("voc", "coco"):
        for e in REG:
            h = out[ds][f"H2_{e}"]
            L.append(f"| {NAME[ds]} | {e} | {f3(h['logOR_MH'])} | {ci(h['ci'])} | {f3(h['logOR_unstrat'])} | "
                     f"{', '.join(f3(x) for x in h['logOR_band'])} | {h['frac_fp0_onR']:.1%} |")
    L.append("\n## H3: what does V3 change relative to V0 (both + R2)?\n")
    L.append("| set | mIoU V0 → V3 | only FP change | only TP change | pooled P, R: V0 → V3 | class-macro P, R: V0 → V3 | ΔFP / GT px | ΔTP / GT px | images up / down / same |")
    L.append("|---|---|---:|---:|---|---|---:|---:|---|")
    for ds in ("voc", "coco"):
        h = out[ds]["H3"]; pr = lambda t: f"{t[0]:.3f}, {t[1]:.3f}"
        L.append(f"| {NAME[ds]} | {h['mIoU_V0']:.4f} → {h['mIoU_V3']:.4f} | {h['mIoU_onlyFP']:.4f} | {h['mIoU_onlyTP']:.4f} | "
                 f"{pr(h['pooled_PR_V0'])} → {pr(h['pooled_PR_V3'])} | {pr(h['macro_PR_V0'])} → {pr(h['macro_PR_V3'])} | "
                 f"{h['dFP'] / h['GT']:+.3f} | {h['dTP'] / h['GT']:+.3f} | "
                 f"{' / '.join(f'{x:.0%}' for x in h['frac_up_down_same'])} |")
    L.append("\n**False-positive pixels on and off the competitor region (V0 → V3):**\n")
    for ds in ("voc", "coco"):
        for e in REG:
            s = out[ds][f"fp_split_{e}"]
            L.append(f"- {NAME[ds]}, η {e}: on R {s['onR_V0']:,} → {s['onR_V3']:,}; off R {s['offR_V0']:,} → {s['offR_V3']:,}.")
    L.append("\nVOC-Sim does not decide (little variation in competition by design). Exploratory mechanism analysis; "
             "COCO-Cap is used post hoc to explain an existing gain, not to select anything.")
    (rd / "result.md").write_text("\n".join(L) + "\n")
    (rd / "result.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("sclip", "all"):
        p = sub.add_parser(n); p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
        p.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(); {"sclip": sclip, "all": all_}[a.cmd](a)
