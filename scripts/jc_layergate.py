#!/usr/bin/env python3
"""runs/21_layer_gate/PREREG.md: per-location CA layer reliability gate (Exp. 1) and signed temporal CA contrast (Exp. 2).

capture  (GPU, trdp)  one live generation per image (frozen runtime, runs/20 corrected prompts, safety checker off);
                      per-(call, layer) CA maps + GOCA-style weights; all seeds x {seed, V0, V3, walk}; diagnostic features
score    (CPU)        counts raw and + DenseCRF at 17 thresholds for every cell
diag     (CPU)        pixel-level error-prediction diagnostics -> <WORK>/diag{tag}.json
report   (CPU)        runs/21_layer_gate/result{tag}.md
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import nerve_walk_port as port  # noqa: E402
import jc_srcrule as sr  # noqa: E402
import jc_newrerun as nr  # noqa: E402
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_layergate"
NR = CACHE / "jc_newrerun"
OUT = ROOT / "runs/21_layer_gate"
NAME = nr.NAME
THR = nr.THR
KS = (16, 64, 256)
PARTS = ("px",) + tuple(f"s{k}" for k in KS) + tuple(f"g{k}" for k in KS)
GATES = tuple(x for p in PARTS for x in (f"G{p}", f"G{p}x"))
TEMP = ("TE", "TM", "TL", "TD+", "TD-")
SEEDS = ("BR", "UN") + GATES + TEMP
CELLS = nr.CELLS
LAB = nr.LAB
DESC = {"BR": "BRACE seed (global GOCA-style layer weights)", "UN": "uniform layer weights",
        "TE": "early calls 0-9", "TM": "middle calls 10-20", "TL": "late calls 21-30",
        "TD+": "seed + early - late", "TD-": "seed + late - early"}
for p in PARTS:
    nm = "pixels" if p == "px" else (f"SA k-means k={p[1:]}" if p[0] == "s" else f"grid {int(p[1:]) ** .5:.0f}x{int(p[1:]) ** .5:.0f}")
    DESC[f"G{p}"] = f"gate, {nm}"; DESC[f"G{p}x"] = f"gate, {nm}, shuffled"
FEAT = ("S", "V3", "Rbar", "RA", "E", "L", "fire", "gt")


def mm(x):
    return cc.norm(x, "mm")


# ------------------------------------------------------------------ capture (GPU)
def recorder_class():
    from src.segmentation import sg25_sa

    class LayerRec(sg25_sa.SG25Recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lmaps, self.lw = [], []

        def flush_step(self, step, timestep, latent):
            ups = [F.interpolate(self.pending_ca[m]["A3"], size=(64, 64), mode="bilinear", align_corners=False)[0, 0].clone()
                   for m in self.selected_modules]
            super().flush_step(step, timestep, latent)
            w = [x for x in self.weights if x[1] == "selected"][-1][2]
            self.lmaps.append(torch.stack(ups)); self.lw.append(torch.from_numpy(np.asarray(w, np.float32)))
    return LayerRec


def kmeans(X: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    C = X[torch.linspace(0, X.shape[0] - 1, k, device=X.device).round().long()].clone()
    xx = (X * X).sum(1, keepdim=True)
    for _ in range(iters):
        lab = torch.argmin(xx - 2 * X @ C.T + (C * C).sum(1)[None], 1)
        cnt = torch.bincount(lab, minlength=k).float()
        S = torch.zeros_like(C).index_add_(0, lab, X)
        ok = cnt > 0; C[ok] = S[ok] / cnt[ok, None]
    return torch.argmin(xx - 2 * X @ C.T + (C * C).sum(1)[None], 1)


def grid(k: int, dev) -> torch.Tensor:
    g = int(round(k ** .5)); s = 64 // g; i = torch.arange(64, device=dev) // s
    return (i[:, None] * g + i[None, :]).reshape(-1)


def pool(e: torch.Tensor, lab: torch.Tensor, n: int) -> torch.Tensor:
    """e [M, 4096] -> per-unit mean [M, n]."""
    cnt = torch.bincount(lab, minlength=n).float().clamp_min(1)
    return torch.zeros(e.shape[0], n, device=e.device).index_add_(1, lab, e) / cnt


@torch.no_grad()
def build(A: torch.Tensor, W: torch.Tensor, P: torch.Tensor, iid: str) -> tuple[dict, dict]:
    """A [31, 11, 4096] per-(call, layer) CA maps, W [31, 11] layer weights, P [4096, 4096] row-stochastic SA."""
    dev = A.device; T, Lm, N = A.shape; Af = A.reshape(T * Lm, N); w = W.reshape(-1, 1)
    br = torch.zeros(N, device=dev)
    for t in range(T):
        for i in range(Lm):
            br += A[t, i] * W[t, i] / 31
    num = (Af - Af @ P.T).abs()
    e = num / (num.median(1, keepdim=True).values + 1e-12)
    g = torch.Generator(device="cpu").manual_seed(zlib.crc32(iid.encode()))
    labs = {"px": (torch.arange(N, device=dev), N)}
    for k in KS:
        labs[f"s{k}"] = (kmeans(P, k), k); labs[f"g{k}"] = (grid(k, dev), k)
    out = {"UN": Af.sum(0) / (T * Lm)}
    rpx = None
    for p, (lab, n) in labs.items():
        ru = torch.exp(-pool(e, lab, n))                         # [M, n] unit reliability
        perm = torch.randperm(n, generator=g).to(dev)
        for sfx, rr in (("", ru), ("x", ru[:, perm])):
            r = rr[:, lab]
            out[f"G{p}{sfx}"] = (w * r * Af).sum(0) / (w * r).sum(0).clamp_min(1e-20)
            if p == "px" and sfx == "":
                rpx = r
    win = {"TE": range(0, 10), "TM": range(10, 21), "TL": range(21, 31)}
    for nm, rg in win.items():
        idx = torch.tensor(list(rg), device=dev)
        out[nm] = (A[idx] * W[idx][..., None]).sum((0, 1))
    S, E, L = mm(br), mm(out["TE"]), mm(out["TL"])
    out["TD+"] = (S + E - L).clamp_min(0); out["TD-"] = (S + L - E).clamp_min(0)
    out = {k: mm(v).reshape(64, 64) for k, v in out.items()}
    fire = (mm_rows(A.mean(0)) >= 0.5).sum(0).float()
    feats = {"Rbar": (w * rpx).sum(0) / w.sum(), "RA": (w * rpx * Af).sum(0) / (w * Af).sum(0).clamp_min(1e-20),
             "E": E, "L": L, "fire": fire, "br_raw": br}
    return out, feats


def mm_rows(x):
    lo = x.min(1, keepdim=True).values; hi = x.max(1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp_min(1e-12)


@torch.no_grad()
def capture(a):
    from PIL import Image
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2, sg25_sa
    from src.segmentation import controlled_cv as cv
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=recorder_class()); pipe = runtime["pipe"]
    pipe.safety_checker = None; dev = pipe.device
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for r in nr.rows(a.dataset, a.shard, a.num_shards, a.max_images, a.ids):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(r), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]
            A = torch.stack(rec.lmaps).float().reshape(31, 11, 4096); W = torch.stack(rec.lw).float().to(dev)
            if A.shape[0] != 31 or W.shape != (31, 11):
                raise ValueError(f"unexpected layer capture {tuple(A.shape)} {tuple(W.shape)}")
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            P = torch.zeros(4096, 4096, device=dev)
            for c in range(11):
                P += sg25_sa._attention(sa[64][c], 64).mean(0) / 11
            sd, feats = build(A, W, P, iid); del P
            rchk = float((cv.normalize_batch(feats["br_raw"].reshape(1, 64, 64), "minmax")[0] - seed.float()).abs().max())
            if rchk > 1e-5:
                raise ValueError(f"{iid}: per-layer reconstruction differs from the runtime A3 seed by {rchk:.2e}")
            sd["BR"] = seed.float().clone()
            res = {s: {"seed": sd[s]} for s in SEEDS}
            for s in SEEDS:
                v, _ = sg_combined_v2.aggregate(sa, mm(sd[s]).reshape(64, 64)); res[s]["V0"], res[s]["V3"] = v["V0"], v["V3"]; del v
        nstore = sr.noised_pass(pipe, Image.fromarray(img), sr.NERVE_LAYERS)
        d = int(nstore[sr.NERVE_LAYERS[0]][0]["query"].shape[-1])
        op = port.walk_operator(port.chained_attention_product([nstore[n][0] for n in sr.NERVE_LAYERS],
                                                                torch.ones(4096, 1, device=dev), d=d))
        for s in SEEDS:
            res[s]["W"] = port.apply_walk(op, mm(sd[s]).reshape(1, 64, 64))[0]
        del op, nstore
        maps = np.stack([np.stack([res[s][c].float().reshape(64, 64).cpu().numpy() for c in CELLS]) for s in SEEDS])
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        gt64 = F.adaptive_avg_pool2d(torch.from_numpy(truth).float()[None, None], (64, 64))[0, 0].reshape(-1)
        fe = np.stack([mm(sd["BR"]).reshape(-1).cpu().numpy(), mm(res["BR"]["V3"]).reshape(-1).cpu().numpy()]
                      + [feats[k].reshape(-1).cpu().numpy() for k in ("Rbar", "RA", "E", "L", "fire")] + [gt64.numpy()])
        prev = NR / a.dataset / f"{iid}.npz"; idiff, v3d = float("nan"), float("nan")
        if prev.exists():
            z = np.load(prev); idiff = int(np.abs(img.astype(int) - z["image"].astype(int)).max())
            v3d = float(np.abs(st.mm(maps[SEEDS.index("BR"), CELLS.index("V3")]) - st.mm(z["maps"][nr.SEEDS.index("CAg"), CELLS.index("V3")].astype(np.float32))).max())
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), layers=A.reshape(31, 11, 64, 64).cpu().numpy().astype(np.float16),
                 weights=W.cpu().numpy(), feats=fe.astype(np.float32), cls=r["label"], changed=r["changed"],
                 off_diff=float(np.abs(img.astype(int) - nr.official(a.dataset, r).astype(int)).mean()),
                 recon_check=rchk, img_vs_run20=idiff, v3_vs_run20=v3d, truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[lg:{a.dataset}] {iid} {r['label']} changed={r['changed']} recon {rchk:.1e} img-vs-r20 {idiff} "
              f"V3-vs-r20 {v3d:.1e} {time.perf_counter() - t0:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ score (CPU)
def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"]), "off_diff": float(z["off_diff"]),
           "recon_check": float(z["recon_check"]), "img_vs_run20": float(z["img_vs_run20"]), "v3_vs_run20": float(z["v3_vs_run20"])}
    for si, s in enumerate(SEEDS):
        for ci, c in enumerate(CELLS):
            x = st.up(st.mm(z["maps"][si, ci].astype(np.float32)), truth.shape)
            res[f"{s}|{c}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{s}|{c}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool_:
            for n, _ in enumerate(pool_.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 25 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------ diag (CPU)
def logit_fit(X, y, lam=1e-4, iters=50):
    if len(X) < 10 or y.min() == y.max():                      # smoke only: too few pixels to fit
        return lambda Xn: np.zeros(len(Xn))
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu) / sd]; b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30))); g = Z.T @ (p - y) + lam * b
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(b)); step = np.linalg.solve(H, g); b -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b


def wauc(score_, y, w):
    """Weighted Mann-Whitney AUC (ties count 1/2); score_ pre-sorted groups handled by unique."""
    o = np.argsort(score_, kind="stable"); s, yy, ww = score_[o], y[o], w[o]
    _, start = np.unique(s, return_index=True); grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(s)]))
    pos = np.bincount(grp, ww * yy); neg = np.bincount(grp, ww * (1 - yy))
    below = np.cumsum(neg) - neg
    tot = pos.sum() * neg.sum()
    return float(((pos * (below + 0.5 * neg)).sum()) / tot) if tot > 0 else float("nan")


def diag(a):
    rng = np.random.default_rng(0); D = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (WORK / (ds + a.tag)).glob("*.npz") if not f.stem.endswith(".tmp"))
        X, img, fold = [], [], []
        for n, f in enumerate(fs):
            fe = np.load(f)["feats"]; idx = rng.choice(4096, 1024, replace=False)
            X.append(fe[:, idx].T); img.append(np.full(1024, n)); fold.append(np.full(1024, zlib.crc32(f.stem.encode()) % 5))
        D[ds] = (np.concatenate(X), np.concatenate(img), np.concatenate(fold), len(fs))
    fi = {k: i for i, k in enumerate(FEAT)}
    sets = {"base": ["S", "V3"], "exp1": ["S", "V3", "Rbar", "RA"], "exp2": ["S", "V3", "E", "L"]}
    targets = {"FP_vs_TP": lambda X: X[:, fi["V3"]] >= 0.55, "FN_vs_TN": lambda X: X[:, fi["V3"]] < 0.55}
    res = {}
    for tname, sub in targets.items():
        pred = {}
        for ds in ("voc", "coco"):
            X, im, fd, nimg = D[ds]; m = sub(X); gt = X[:, fi["gt"]] >= 0.5
            y = (~gt[m]).astype(float) if tname == "FP_vs_TP" else gt[m].astype(float)   # 1 = error
            pred[ds] = {"y": y, "img": im[m], "n": nimg}
            for nm, cols in sets.items():
                ci = [fi[c] for c in cols]
                if ds == "voc":
                    p = np.zeros(m.sum()); Xm, fm = X[m][:, ci], fd[m]
                    for k in range(5):
                        p[fm == k] = logit_fit(Xm[fm != k], y[fm != k])(Xm[fm == k])
                else:
                    Xv, _, _, _ = D["voc"]; mv = sub(Xv); gv = Xv[:, fi["gt"]] >= 0.5
                    yv = (~gv[mv]).astype(float) if tname == "FP_vs_TP" else gv[mv].astype(float)
                    p = logit_fit(Xv[mv][:, ci], yv)(X[m][:, ci])
                pred[ds][nm] = p
            Dd = X[m][:, fi["E"]] - X[m][:, fi["L"]]
            pred[ds]["D"], pred[ds]["-D"] = Dd, -Dd
        for ds in ("voc", "coco"):
            P_ = pred[ds]; y, im, n = P_["y"], P_["img"], P_["n"]; one = np.ones(len(y))
            r = {"n_pix": int(len(y)), "err_rate": float(y.mean())}
            for nm in ("base", "exp1", "exp2", "D", "-D"):
                r[f"auc_{nm}"] = wauc(P_[nm], y, one)
            bs = {"exp1": [], "exp2": []}
            for _ in range(a.boot):
                cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float); w = cnt[im]
                b0 = wauc(P_["base"], y, w)
                for e in bs:
                    bs[e].append(wauc(P_[e], y, w) - b0)
            for e in bs:
                lo, hi = np.nanpercentile(bs[e], [2.5, 97.5]); r[f"d_{e}"] = [r[f"auc_{e}"] - r["auc_base"], float(lo), float(hi)]
            res[f"{ds}|{tname}"] = r
    # agreeing-wrong check (Exp. 1)
    for ds in ("voc", "coco"):
        X = D[ds][0]; fp = (X[:, fi["V3"]] >= 0.55) & (X[:, fi["gt"]] < 0.5); tp = (X[:, fi["V3"]] >= 0.55) & (X[:, fi["gt"]] >= 0.5)
        res[f"{ds}|agree"] = {"fp_majority_fire": float((X[fp, fi["fire"]] >= 6).mean()) if fp.any() else float("nan"),
                              "tp_majority_fire": float((X[tp, fi["fire"]] >= 6).mean()) if tp.any() else float("nan"),
                              "rbar_fp": float(X[fp, fi["Rbar"]].mean()) if fp.any() else float("nan"),
                              "rbar_tp": float(X[tp, fi["Rbar"]].mean()) if tp.any() else float("nan")}
    for e in ("exp1", "exp2"):
        res[f"pass_{e}"] = any(res[f"voc|{t}"][f"d_{e}"][0] >= 0.01 and res[f"voc|{t}"][f"d_{e}"][1] > 0 for t in targets)
    WORK.mkdir(parents=True, exist_ok=True); (WORK / f"diag{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ report (CPU)
def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    Z = {ds: [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")] for ds in S}
    G = json.loads((WORK / f"diag{a.tag}.json").read_text())
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    t55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
    k = lambda s, c, crf=True: f"{s}|{c}{'+R2' if crf else ''}"
    mi = lambda ds, P, sub=None: cc.miou(P[:, None], S[ds]["cls"], one[ds] if sub is None else sub)[0]
    tbest = lambda key: int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))
    sel = lambda ds, key: cc.oof(S["voc"], S["voc"][f"{key}|raw"])[1] if ds == "voc" else S["coco"][f"{key}|raw"][:, tbest(key)]
    fx = lambda ds, key: S[ds][f"{key}|raw"][:, t55]
    best = lambda ds, key: float(cc.miou(S[ds][f"{key}|raw"], S[ds]["cls"], one[ds]).max())

    def dci(ds, Pa, Pb):
        pt = mi(ds, Pa) - mi(ds, Pb); rng = np.random.default_rng(0); d = []; o1 = np.ones(n[ds], bool)
        for _ in range(a.boot):
            s = rng.integers(0, n[ds], n[ds])
            d.append(cc.miou(Pa[s][:, None], S[ds]["cls"][s], o1)[0] - cc.miou(Pb[s][:, None], S[ds]["cls"][s], o1)[0])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", (pt, lo, hi)

    L = ["# Per-location CA layer reliability gate and signed temporal CA contrast (`PREREG.md`)\n",
         "`scripts/jc_layergate.py`. One live capture per image (runs/20 corrected prompts, safety checker off). Class-macro "
         "mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold applied unchanged on COCO-Cap. Δ with "
         f"image-bootstrap 95% interval ({a.boot}). Exploratory.\n", "## Data checks\n"]
    for ds in S:
        rc = np.array([float(z["recon_check"]) for z in Z[ds]]); im = np.array([float(z["img_vs_run20"]) for z in Z[ds]])
        v3 = np.array([float(z["v3_vs_run20"]) for z in Z[ds]])
        L.append(f"- {NAME[ds]}: {n[ds]} images. Per-layer reconstruction vs runtime seed: max |Δ| {rc.max():.1e}. Image vs "
                 f"runs/20 capture: identical on {(im == 0).sum()}/{n[ds]}, median max |Δ| {np.nanmedian(im):.0f}. BRACE hierarchy map "
                 f"vs runs/20: median max |Δ| {np.nanmedian(v3):.1e}.")
    L.append("\n## Diagnostics (pixel level; ΔAUC = augmented − base, VOC-Sim 5-fold OOF, COCO-Cap fit on VOC-Sim)\n")
    L.append("| set | target | pixels | error rate | AUC base | AUC +reliability (Exp. 1) | ΔAUC Exp. 1 | AUC +early/late (Exp. 2) | ΔAUC Exp. 2 | AUC D=E−L | AUC −D |")
    L.append("|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|")
    for ds in S:
        for t in ("FP_vs_TP", "FN_vs_TN"):
            r = G[f"{ds}|{t}"]; f3 = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
            L.append(f"| {NAME[ds]} | {t.replace('_', ' ')} | {r['n_pix']} | {r['err_rate']:.3f} | {r['auc_base']:.4f} | {r['auc_exp1']:.4f} | "
                     f"{f3(r['d_exp1'])} | {r['auc_exp2']:.4f} | {f3(r['d_exp2'])} | {r['auc_D']:.4f} | {r['auc_-D']:.4f} |")
    L.append(f"\n**Diagnostic pass (VOC-Sim ΔAUC ≥ 0.01, lower bound > 0, either target):** Exp. 1 "
             f"**{'pass' if G['pass_exp1'] else 'fail'}**, Exp. 2 **{'pass' if G['pass_exp2'] else 'fail'}**.\n")
    L.append("Agreeing-wrong check (≥ 6/11 layers fire at ≥ 0.5): " + "; ".join(
        f"{NAME[ds]} FP {G[f'{ds}|agree']['fp_majority_fire']:.3f} vs TP {G[f'{ds}|agree']['tp_majority_fire']:.3f}, "
        f"mean R̄ FP {G[f'{ds}|agree']['rbar_fp']:.3f} vs TP {G[f'{ds}|agree']['rbar_tp']:.3f}" for ds in S) + ".\n")
    convs = (("VOC-sel. threshold, with DenseCRF (primary)", sel, True), ("fixed 0.55, with DenseCRF", fx, True),
             ("VOC-sel. threshold, raw", sel, False), ("fixed 0.55, raw", fx, False))
    for ds in S:
        L.append(f"## Components, {NAME[ds]} ({n[ds]} images)\n")
        for title, f, crf in convs:
            L.append(f"**{title}**\n"); L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---:|" * len(CELLS))
            for s in SEEDS:
                L.append(f"| {s}: {DESC[s]} | " + " | ".join(f"{mi(ds, f(ds, k(s, c, crf))):.3f}" for c in CELLS) + " |")
            L.append("")
        L.append("**Best threshold on this set, with DenseCRF (audit only)**\n")
        L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---:|" * len(CELLS))
        for s in SEEDS:
            L.append(f"| {s} | " + " | ".join(f"{best(ds, k(s, c)):.3f}" for c in CELLS) + " |")
        L.append("")
    L.append("## Selection and contrasts (hierarchy + DenseCRF)\n")
    chosen = {}
    for fam in ("s", "g"):
        v = {f"G{fam}{kk}": mi("voc", sel("voc", k(f"G{fam}{kk}", "V3"))) for kk in KS}; chosen[fam] = max(v, key=v.get)
        L.append(f"- {'SA regions' if fam == 's' else 'grid'}: VOC-Sim OOF " + ", ".join(f"{m} {x:.3f}" for m, x in v.items()) + f" → **{chosen[fam]}**")
    cand = ["Gpx", chosen["s"], chosen["g"], "TD+", "TD-"]
    L.append("\n| Contrast | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |"); L.append("|---|---|---|---|---|")
    con = [("BR − UN (GOCA weighting)", "BR", "UN")] + [(f"{c} − BR", c, "BR") for c in cand] + \
          [(f"{c} − {c}x (shuffled)", c, f"{c}x") for c in cand[:3]] + [(f"{chosen['s']} − {chosen['g']} (regions vs grid)", chosen["s"], chosen["g"])]
    CI = {}
    for lab_, A_, B_ in con:
        cells = []
        for f in (sel, fx):
            for ds in S:
                txt, v = dci(ds, f(ds, k(A_, "V3")), f(ds, k(B_, "V3"))); cells.append(txt); CI[(lab_, ds, f is sel)] = v
        L.append(f"| {lab_} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    L.append("\nThe same contrasts at the other propagations (VOC-sel. + DenseCRF; VOC-Sim / COCO-Cap):\n")
    L.append("| Contrast | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---|" * len(CELLS))
    for lab_, A_, B_ in con:
        L.append(f"| {lab_} | " + " | ".join(" / ".join(f"{mi(ds, sel(ds, k(A_, c))) - mi(ds, sel(ds, k(B_, c))):+.3f}" for ds in S) for c in CELLS) + " |")
    L.append("\n**Reading** (helps = COCO Δ vs BR interval > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0; gates also need Δ vs shuffled interval > 0 on COCO):\n")
    for c in cand:
        pc, lo, _ = CI[(f"{c} − BR", "coco", True)]; pv = CI[(f"{c} − BR", "voc", True)][0]
        bd = best("coco", k(c, "V3")) - best("coco", k("BR", "V3"))
        ok = lo > 0 and pv >= 0 and bd > 0
        extra = ""
        if c.startswith("G"):
            slo = CI[(f"{c} − {c}x (shuffled)", "coco", True)][1]
            if ok and slo <= 0:
                ok = False; extra = " (not better than shuffled: generic reweighting)"
        gate = G["pass_exp1"] if c.startswith("G") else G["pass_exp2"]
        verdict = ("helps" if ok else ("calibration only" if lo > 0 and pv >= 0 else "does not help")) + extra
        L.append(f"- {c} ({DESC[c]}): COCO {pc:+.3f} (lower {lo:+.3f}), VOC {pv:+.3f}, COCO best-thr. {bd:+.3f} → **{verdict}**; "
                 f"diagnostic {'passed' if gate else 'failed'}")
    L.append("\n**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**\n")
    L.append("| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |"); L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    area = S["coco"]["area"]; q1, q2 = np.quantile(area, [1 / 3, 2 / 3]); cls_ = S["coco"]["cls"]; U = np.unique(cls_)
    for s in ["BR", "UN"] + cand:
        for c in ("V3", "W"):
            P = sel("coco", k(s, c)); iou = P[:, 0] / np.maximum(P[:, :3].sum(1), 1)
            prec = np.mean([P[cls_ == cl, 0].sum() / max(P[cls_ == cl][:, [0, 1]].sum(), 1) for cl in U])
            rec = np.mean([P[cls_ == cl, 0].sum() / max(P[cls_ == cl][:, [0, 2]].sum(), 1) for cl in U])
            L.append(f"| {s} | {LAB[c]} | {mi('coco', P):.3f} | {iou[area <= q1].mean():.3f} | {iou[(area > q1) & (area <= q2)].mean():.3f} | "
                     f"{iou[area > q2].mean():.3f} | {(iou < 0.3).mean():.3f} | {P[:, 1].mean() / (512 * 512):.4f} | {prec:.3f} | {rec:.3f} |")
    L.append("\n## Per class, VOC-sel. + CRF, hierarchy (BR / Gpx / " + chosen["s"] + ")\n")
    for ds in S:
        cl = S[ds]["cls"]
        L.append(f"{NAME[ds]}: " + ", ".join(f"{c} " + " / ".join(f"{mi(ds, sel(ds, k(s, 'V3')), cl == c):.2f}" for s in ("BR", "Gpx", chosen["s"])) for c in np.unique(cl)) + "\n")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5"); p.add_argument("--ids", default="")
    for nm in ("score", "diag", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=16); p.add_argument("--tag", default="")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"capture": capture, "score": score, "diag": diag, "report": report}[a.cmd](a)
