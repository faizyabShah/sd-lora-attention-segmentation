#!/usr/bin/env python3
"""Diagnostic (pre-registered in runs/08_feat_bleed/PREREG.md): does frozen UNet feature content separate BRACE's bleed
cells from true object cells beyond RGB, BRACE confidence and spatial cues, and can a label-free correction use it?
CPU only, stored rich capture (CACHE/ana_dump). Ground truth only scores.

python scripts/feat_bleed.py [--workers 16] [--max-images k] [--tag t]   -> runs/08_feat_bleed/result.md, result.json
"""
from __future__ import annotations

import argparse, json, sys, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import sa_boundary_diag as sbd  # noqa: E402

sbd.NBOOT = 2000
RUN = ROOT / "runs/08_feat_bleed"
FEAT = "feat/up2/s15"; STEPS = (5, 15, 25); NPC = 32; LAM = 1.0; MINCORE = 5
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
BASE = ["dLab_F", "dLab_B", "contour", "dist_F", "conf", "depth", "area", "s_rgb"]
TEST = ["s_proto", "s_feat"]; SEC = ["rowdiv", "consist"]


def wlogit(X, y, lam=LAM, it=60):
    """Balanced-weight L2 logistic (intercept unpenalised) on standardised X. Returns (mu, sd, w)."""
    mu, sd = X.mean(0), X.std(0) + 1e-8; Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    wt = np.where(y > 0, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    R = lam * np.eye(Z.shape[1]); R[0, 0] = 0; w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (wt * (p - y)) + R @ w; H = (Z * (wt * p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def apply(m, X):
    mu, sd, w = m; return np.c_[np.ones(len(X)), (X - mu) / sd] @ w


def proto(f, F, B):
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    pF, pB = fn[F].mean(0), fn[B].mean(0)
    pF, pB = pF / (np.linalg.norm(pF) + 1e-8), pB / (np.linalg.norm(pB) + 1e-8)
    return fn @ pF - fn @ pB


def one(path):
    import ana_ops_common as oc
    oc.crf_init()
    z = np.load(path, allow_pickle=True); iid = Path(path).stem
    img, truth = z["image"], z["truth"].astype(np.float64); v = st.mm(z["v3"])
    M = oc.crf(img, st.up(v, truth.shape) >= 0.55); Mc = sbd.pool(M.astype(float), 64) >= 0.5; gt = sbd.pool(truth, 64)
    F = v >= 0.9
    if not F.any():
        F = v >= np.quantile(v, 0.95)
    B = (v <= 0.1) & ~Mc
    out = {"id": iid, "ok": bool(F.sum() >= MINCORE and B.sum() >= MINCORE)}
    if not out["ok"]:
        return out
    L = sbd.lab(img); Lc = sbd.pool(L, 64); Ls = np.sqrt(np.maximum(sbd.pool(L ** 2, 64) - Lc ** 2, 0))
    Lg = gaussian_filter(L, (1, 1, 0)); G = np.sqrt(sum(sobel(Lg[..., c], 0) ** 2 + sobel(Lg[..., c], 1) ** 2 for c in range(3)))
    dF, (iy, ix) = distance_transform_edt(~F, return_indices=True); depth = distance_transform_edt(Mc)
    cells = np.argwhere(np.ones((64, 64), bool)); near = np.c_[iy.ravel(), ix.ravel()]
    contour = sbd.contour_max(G, cells.astype(float), near.astype(float), 8).reshape(64, 64)
    mF, mB = Lc[F].mean(0), Lc[B].mean(0)
    Fv, Bv = F.ravel(), B.ravel(); tr = Fv | Bv; ytr = Fv[tr].astype(float)
    X6 = np.c_[Lc.reshape(-1, 3), Ls.reshape(-1, 3)]; s_rgb = apply(wlogit(X6[tr], ytr), X6)
    f = z[FEAT].astype(np.float32).reshape(640, -1).T
    fs = (f - f.mean(0)) / (f.std(0) + 1e-6); U, S, Vt = np.linalg.svd(fs, full_matrices=False); P = fs @ Vt[:NPC].T
    s_feat = apply(wlogit(P[tr], ytr), P); s_proto = proto(f, Fv, Bv)
    # secondary: TRACE-style row divergence to F at 32 px block 1, three-step consistency of s_proto
    A = z["sa_full/32_b1_c"].astype(np.float64); A = A / A.sum(1, keepdims=True)
    par = (cells[:, 0] // 2) * 32 + cells[:, 1] // 2; rF = A[np.unique(par[Fv])].mean(0)
    Ap = A[par] + 1e-8; q = rF + 1e-8
    rowdiv = 0.5 * ((Ap * np.log(Ap / q)).sum(1) + (q * np.log(q / Ap)).sum(1))
    consist = np.std([proto(z[f"feat/up2/s{s}"].astype(np.float32).reshape(640, -1).T, Fv, Bv) for s in STEPS], 0)
    D = {"dLab_F": np.linalg.norm(Lc - mF, axis=-1).ravel(), "dLab_B": np.linalg.norm(Lc - mB, axis=-1).ravel(),
         "contour": contour.ravel(), "dist_F": dF.ravel(), "conf": v.ravel(), "depth": depth.ravel(),
         "area": np.full(4096, Mc.mean()), "s_rgb": s_rgb, "s_proto": s_proto, "s_feat": s_feat, "rowdiv": rowdiv, "consist": consist}
    Q = (Mc & ~F & ((gt >= 0.75) | (gt <= 0.25))).ravel(); y = (gt <= 0.25).ravel()[Q]
    out.update(X={k: d[Q].astype(np.float32) for k, d in D.items()}, y=y, nF=int(F.sum()), nB=int(B.sum()))
    # gate 2: count-matched corrections
    cand = (Mc & ~F).ravel(); pf = 1 / (1 + np.exp(-s_feat)); rem = cand & (pf < 0.5); k = int(rem.sum())
    def lowest(s):
        r = np.zeros(4096, bool); ix_ = np.flatnonzero(cand)
        if k:
            r[ix_[np.argsort(s[ix_], kind="stable")[:k]]] = True
        return r
    tb = truth.astype(bool)
    iou = lambda m: float((m & tb).sum() / max((m | tb).sum(), 1))
    cut = lambda r: M & ~np.kron(r.reshape(64, 64), np.ones((8, 8), bool))
    out.update(k=k, iou_M=iou(M), iou_feat=iou(cut(rem)), iou_rgb=iou(cut(lowest(s_rgb))), iou_conf=iou(cut(lowest(v.ravel()))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max-images", type=int)
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    res = {}; L = ["# Does UNet feature content separate BRACE's bleed from the object? (diagnostic; `PREREG.md`)\n",
                   f"Stored rich capture, 150/150 screening images. Features `{FEAT}`; cells inside BRACE + R2 not in the core F. "
                   "Decisive: COCO-Cap, gates 1 and 2.\n"]
    for ds in ("voc", "coco"):
        fs = sorted(str(f) for f in (CACHE / "ana_dump" / ds).glob("*.npz") if not f.stem.endswith(".tmp"))[: a.max_images]
        with Pool(a.workers) as p:
            R = p.map(one, fs, chunksize=1)
        ok = [r for r in R if r["ok"] and len(r["y"])]
        grp = np.concatenate([np.full(len(r["y"]), r["id"]) for r in ok]); y = np.concatenate([r["y"] for r in ok])
        X = {k: np.concatenate([r["X"][k] for r in ok]) for k in BASE + TEST + SEC}
        S = lambda ks: np.stack([X[k] for k in ks], 1)
        g1 = sbd.compare(S(BASE), S(BASE + TEST), y, grp)
        rep = {"proto_only": sbd.compare(S(BASE), S(BASE + ["s_proto"]), y, grp),
               "feat_only": sbd.compare(S(BASE), S(BASE + ["s_feat"]), y, grp),
               "rowdiv": sbd.compare(S(BASE), S(BASE + ["rowdiv"]), y, grp),
               "consist": sbd.compare(S(BASE), S(BASE + ["consist"]), y, grp)}
        sign = {"dLab_F": 1, "dLab_B": -1, "contour": 1, "dist_F": 1, "conf": -1, "depth": -1, "area": 1, "s_rgb": -1,
                "s_proto": -1, "s_feat": -1, "rowdiv": 1, "consist": 1}           # direction: larger = more bleed-like
        single = {k: sbd.auc(y, sign[k] * X[k]) for k in X}
        G = {c: np.array([r[f"iou_{c}"] - r["iou_M"] for r in ok]) for c in ("feat", "rgb", "conf")}
        g2 = {c: sbd.mean_ci(G[c]) for c in G}
        res[ds] = dict(images=len(ok), excluded=len(R) - len(ok), cells=int(len(y)), bleed_share=float(y.mean()),
                       gate1=g1, report=rep, single=single, gate2=g2, k_mean=float(np.mean([r["k"] for r in ok])),
                       wins_feat=int((G["feat"] > 0).sum()))
    c, v = res["coco"], res["voc"]
    p1 = bool(c["gate1"]["delta"] >= 0.05 and c["gate1"]["lo"] > 0)
    gf = c["gate2"]["feat"]
    p2 = bool(gf[0] > 0 and gf[1] > 0 and gf[0] > c["gate2"]["rgb"][0] and gf[0] > c["gate2"]["conf"][0] and v["gate2"]["feat"][0] >= -0.005)
    verdict = "CONTINUE (new pre-registered screen)" if p1 and p2 else "CLOSE"
    res.update(verdict=verdict, gate1_pass=p1, gate2_pass=p2)
    f4 = lambda x: f"{x:+.4f}"
    L += [f"## Verdict: **{verdict}** (gate 1 {'pass' if p1 else 'fail'}, gate 2 {'pass' if p2 else 'fail'})\n",
          "## Gate 1: OOF AUROC for bleed vs correct cells inside BRACE (baseline = RGB, contour, distance, BRACE confidence, depth, area, matched RGB classifier)\n",
          "| set | images (excl.) | cells (bleed share) | baseline | + features | Δ | CI | + s_proto only Δ | + s_feat only Δ |",
          "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for ds in ("voc", "coco"):
        r = res[ds]; g = r["gate1"]
        L.append(f"| {NAME[ds]} | {r['images']} ({r['excluded']}) | {r['cells']:,} ({r['bleed_share']:.1%}) | {g['rgb']:.3f} | {g['rgb_sa']:.3f} | "
                 f"{f4(g['delta'])} | [{f4(g['lo'])}, {f4(g['hi'])}] | {f4(r['report']['proto_only']['delta'])} | {f4(r['report']['feat_only']['delta'])} |")
    L += ["\n## Gate 2: count-matched label-free corrections, mean per-image IoU gain over BRACE + R2 (95% CI)\n",
          "| set | features (s_feat < 0.5) | RGB classifier | BRACE confidence | cells removed / image | feature wins |", "|---|---|---|---|---:|---:|"]
    for ds in ("voc", "coco"):
        r = res[ds]; m = lambda t: f"{f4(t[0])} [{f4(t[1])}, {f4(t[2])}]"
        L.append(f"| {NAME[ds]} | {m(r['gate2']['feat'])} | {m(r['gate2']['rgb'])} | {m(r['gate2']['conf'])} | {r['k_mean']:.1f} | {r['wins_feat']}/{r['images']} |")
    L += ["\n## Reported, not deciding\n", "**Single-feature AUROC (oriented so that higher = more bleed-like):**\n",
          "| feature | " + " | ".join(NAME[d] for d in ("voc", "coco")) + " |", "|---|---:|---:|"]
    for k in BASE + TEST + SEC:
        L.append(f"| {k} | {res['voc']['single'][k]:.3f} | {res['coco']['single'][k]:.3f} |")
    L += ["\n**Secondary signals added to the baseline (Δ OOF AUROC, CI):**\n"]
    for k in ("rowdiv", "consist"):
        for ds in ("voc", "coco"):
            g = res[ds]["report"][k]
            L.append(f"- {k}, {NAME[ds]}: {f4(g['delta'])} [{f4(g['lo'])}, {f4(g['hi'])}].")
    L.append("\nExploratory mechanism diagnostic. GT only scores; cores, classifiers and the correction are label-free.")
    out = RUN / (f"result_{a.tag}" if a.tag else "result"); RUN.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
