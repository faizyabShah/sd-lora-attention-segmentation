#!/usr/bin/env python3
"""runs/31_multineg_gate/PREREG.md: query-profile gate diagnostics and a multi-negative-prototype gate (exploratory).

split    write runs/31_multineg_gate/split.json (class-stratified dev/test halves, dev folds dA/dB)
diag30   Part A, CPU, on run 30 captures                          -> runs/31_multineg_gate/diag30.md
dump     Part B, GPU, dev images: per-pixel CA queries + class-word attention -> <CACHE>/jc_mneg/dump_{set}{tag}/
fit      Part C0 (GPU if available): profiles for regimes X (VOC+COCO dev) and V (VOC dev); fits all/dA/dB
diag     Part C1-C4 on the dump                                   -> runs/31_multineg_gate/diag{tag}.md
capture  Part D, GPU, all images, 17 seed arms x 3 maps           -> <CACHE>/jc_mneg/{set}{tag}/
score    Part D, CPU                                              report  -> runs/31_multineg_gate/result{tag}.md
"""
from __future__ import annotations

import argparse, itertools, json, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_qgate as QG  # noqa: E402

AB, T, J, cc, st, nr = QG.AB, QG.T, QG.J, QG.cc, QG.st, QG.nr
WORK = J.CACHE / "jc_mneg"
OUT = ROOT / "runs/31_multineg_gate"
SETS = ("voc", "coco")
CLASSES = T.CLASSES
THR, T55 = J.THR, J.T55
KS = (1, 2, 4, 8)
TAU = 10.0
REGIMES = ("X", "V")
GATED = ("NC", "NO", "NH", "MK2", "MK4", "MK8", "MG4", "MS4")
ARMS = ("A",) + tuple(f"{r}:{g}" for r in REGIMES for g in GATED)
GMAP_ARMS = ("X:NC", "X:MK4", "X:MG4", "X:MS4")
MAPS = ("seed", "V0", "V3")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def h(s):
    return zlib.crc32(s.encode())


# ---------------------------------------------------------------- split
def split(a):
    out = {}
    for ds in SETS:
        by = {}
        for r in nr.rows(ds):
            by.setdefault(AB.label(r), []).append(r["image_id"])
        for c, ids in by.items():
            for rank, i in enumerate(sorted(ids, key=h)):
                dev = rank % 2 == 0
                out[i] = {"ds": ds, "cls": c, "part": "dev" if dev else "test",
                          "fold": ("dA" if (rank // 2) % 2 == 0 else "dB") if dev else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps(out, indent=0, sort_keys=True))
    for ds in SETS:
        v = [x for x in out.values() if x["ds"] == ds]
        print(ds, {p: sum(x["part"] == p for x in v) for p in ("dev", "test")},
              {f: sum(x["fold"] == f for x in v) for f in ("dA", "dB")})


def load_split():
    return json.loads((OUT / "split.json").read_text())


def rows_for(a, part):
    S = load_split(); R = nr.rows(a.set)
    if part == "dev":
        R = [r for r in R if S[r["image_id"]]["part"] == "dev"]
    R = R[a.shard::a.num_shards]
    if a.max_images:
        R = R[:: max(1, len(R) // a.max_images)][: a.max_images]
    return R


# ---------------------------------------------------------------- Part A (run 30 captures)
def auroc(p, n):
    if len(p) == 0 or len(n) == 0:
        return np.nan
    x = np.concatenate([p, n]); r = x.argsort(kind="mergesort").argsort(kind="mergesort") + 1.0
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def _a_one(f):
    torch.set_num_threads(1)
    z = np.load(f); truth = z["truth"].astype(bool); fg, near, far = QG.regions(truth)
    g = z["gates"].astype(np.float32); res = {"cls": str(z["cls"]), "grp": Path(f).stem}
    rng = np.random.default_rng(h(Path(f).stem))
    for gi, k in ((0, "NC"), (1, "NO")):
        for rn, r in (("fg", fg), ("near", near), ("far", far)):
            v = g[gi][r]; res[f"{k}_{rn}"] = v[rng.permutation(len(v))[:400]]     # sample for pooled stats
            res[f"{k}_{rn}_n"] = len(v)
            for t in (0.3, 0.5, 0.7):
                res[f"{k}_{rn}_gt{t}"] = int((v > t).sum())
        res[f"{k}_auc_far"] = auroc(g[gi][fg], g[gi][far]); res[f"{k}_auc_near"] = auroc(g[gi][fg], g[gi][near])
    seed = z["maps"][0, 0].astype(np.float32); valid = np.ones_like(truth); other = np.zeros_like(truth)
    for gm in GAMMAS:
        s = st.mm(seed * (g[0] ** gm if gm else 1.0)); tot = max(float(s.sum()), 1e-8)
        res[f"far_mass_{gm}"] = float(s[far].sum()) / tot
        x = st.up(s, truth.shape); res[f"cnt_{gm}"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
    return res


def diag30(a):
    L = ["# Run 31, Part A: gate diagnostics on run 30 captures (`PREREG.md`)\n",
         "Module-mean gate maps saved by run 30 (NC = absent-class gate, NO = all-pixel gate); regions from GT at 64 px "
         "(object; near background within 6 px; far background). Ground truth only defines regions and scores.\n"]
    for ds in SETS:
        fs = sorted(f for f in (J.CACHE / "jc_qgate" / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        with Pool(a.workers) as p:
            R = p.map(_a_one, [str(f) for f in fs], chunksize=4)
        L.append(f"\n## {'VOC-Sim' if ds == 'voc' else 'COCO-Cap'} ({len(R)} images)\n")
        L.append("### A1 gate distributions\n")
        L.append("| Gate | Region | q05 | q25 | q50 | q75 | q95 | share g>0.3 | g>0.5 | g>0.7 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in ("NC", "NO"):
            for rn in ("fg", "near", "far"):
                v = np.concatenate([r[f"{k}_{rn}"] for r in R]); n = sum(r[f"{k}_{rn}_n"] for r in R)
                q = np.quantile(v, [.05, .25, .5, .75, .95])
                sh = [sum(r[f"{k}_{rn}_gt{t}"] for r in R) / max(n, 1) for t in (0.3, 0.5, 0.7)]
                L.append(f"| {k} | {rn} | " + " | ".join(f"{x:.2f}" for x in q) + " | " + " | ".join(f"{x:.2f}" for x in sh) + " |")
        L.append("\nObject retention = share of object pixels with g > t; far rejection = 1 − share of far pixels with g > t.\n")
        L.append("| Gate | AUROC object vs far (per-image mean) | pooled | AUROC object vs near (per-image mean) | pooled |")
        L.append("|---|---:|---:|---:|---:|")
        for k in ("NC", "NO"):
            fg = np.concatenate([r[f"{k}_fg"] for r in R]); far = np.concatenate([r[f"{k}_far"] for r in R])
            near = np.concatenate([r[f"{k}_near"] for r in R])
            L.append(f"| {k} | {np.nanmean([r[f'{k}_auc_far'] for r in R]):.3f} | {auroc(fg, far):.3f} | "
                     f"{np.nanmean([r[f'{k}_auc_near'] for r in R]):.3f} | {auroc(fg, near):.3f} |")
        L.append("\nPer class (NC): median per-image AUROC object vs far; object retention and far rejection at 0.5.\n")
        L.append("| Class | AUROC | retention | rejection |"); L.append("|---|---:|---:|---:|")
        for c in sorted({r["cls"] for r in R}):
            rc = [r for r in R if r["cls"] == c]
            ret = sum(r["NC_fg_gt0.5"] for r in rc) / max(sum(r["NC_fg_n"] for r in rc), 1)
            rej = 1 - sum(r["NC_far_gt0.5"] for r in rc) / max(sum(r["NC_far_n"] for r in rc), 1)
            L.append(f"| {c} | {np.nanmedian([r['NC_auc_far'] for r in rc]):.3f} | {ret:.2f} | {rej:.2f} |")
        A2 = {ds: R}
        L.append("\n### A2 seed-level sharpening (approximation; no self-attention rerun)\n")
        L.append("| Seed | far-background share of seed mass | raw seed mIoU (" + ("OOF" if ds == "voc" else "VOC-Sim-selected") + ") |")
        L.append("|---|---:|---:|")
        globals().setdefault("_A2", {})[ds] = R
        for gm in GAMMAS:
            L.append(f"| {'BRACE seed' if gm == 0 else f'seed × NC gate^{gm:g}'} | {np.mean([r[f'far_mass_{gm}'] for r in R]):.3f} | {{A2_{ds}_{gm}}} |")
    # A2 mIoU: VOC OOF; COCO at VOC-best threshold
    txt = "\n".join(L) + "\n"
    Sv = _a2_struct(globals()["_A2"]["voc"]); Sc = _a2_struct(globals()["_A2"]["coco"])
    for gm in GAMMAS:
        k = f"cnt_{gm}"; one = np.ones(len(Sv["cls"]), bool)
        vo = cc.oof(Sv, Sv[k])[0]; tb = int(np.argmax(cc.miou(Sv[k], Sv["cls"], one)))
        co = cc.miou(Sc[k][:, tb][:, None], Sc["cls"], np.ones(len(Sc["cls"]), bool))[0]
        txt = txt.replace(f"{{A2_voc_{gm}}}", f"{vo:.3f}").replace(f"{{A2_coco_{gm}}}", f"{co:.3f} (thr {THR[tb]:.3f})")
    txt += ("\nA2 is at the seed only. After propagation the ceiling for a perfectly clean background is run 24's oracle "
            "(about +0.05 / +0.06 BRACE mIoU).\n")
    print(txt); (OUT / "diag30.md").write_text(txt)


def _a2_struct(R):
    S = {"cls": np.array([r["cls"] for r in R]), "grp": np.array([r["grp"] for r in R]),
         "fold": np.array([h(r["grp"]) % 5 for r in R])}
    for gm in GAMMAS:
        S[f"cnt_{gm}"] = np.stack([r[f"cnt_{gm}"] for r in R])
    return S


# ---------------------------------------------------------------- Part B dump
@torch.no_grad()
def dump(a):
    rt = J.make_runtime(a.model); K = T.class_keys(rt, CLASSES)
    out = WORK / f"dump_{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows_for(a, "dev"):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        Q = T.qbar(rec); mods = list(Q)
        S, Tm = T.ca_seeds(rec, {**K, "__T": (None, list(rec.target))}, tbar=True)
        seed = rec.seeds()["selected/A3"].float(); fid = float((S["__T"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"seed fidelity {fid:.2e} for {iid}")
        fg, _, _ = QG.regions(np.asarray(sample.target_mask, bool))
        d = {"modules": np.array(mods), "seed": seed.cpu().numpy().astype(np.float32), "fg64": fg, "label": lab,
             "ds": a.set, "fid": fid, "mention": np.array([T.mentions(row["prompt"], c) for c in CLASSES])}
        for i, m in enumerate(mods):
            d[f"q_{i:02d}"] = Q[m].half().cpu().numpy()
            d[f"t_{i:02d}"] = torch.stack([Tm[(c, m)] for c in CLASSES]).half().cpu().numpy()
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[mneg:dump:{a.set}] {iid} fid {fid:.1e} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


def dump_files(tag):
    fs = []
    for ds in SETS:
        fs += sorted(f for f in (WORK / f"dump_{ds}{tag}").glob("*.npz") if not f.stem.endswith(".tmp"))
    return fs


# ---------------------------------------------------------------- clustering helpers
def skmeans(X, K, seed=0, restarts=10, iters=100):
    """Spherical k-means on unit rows X; k-means++ from seed; returns labels of the lowest-inertia restart."""
    n = X.shape[0]; g = torch.Generator(device="cpu").manual_seed(seed); best = None
    for _ in range(restarts):
        idx = [int(torch.randint(n, (1,), generator=g))]; D = 1 - X @ X[idx[0]]
        for _k in range(1, K):
            p = D.clamp(min=0) ** 2; p = p / p.sum() if float(p.sum()) > 0 else torch.full_like(p, 1.0 / n)
            j = int(torch.multinomial(p.cpu(), 1, generator=g)); idx.append(j); D = torch.minimum(D, 1 - X @ X[j])
        C = F.normalize(X[idx], dim=1)
        for _it in range(iters):
            a = (X @ C.T).argmax(1)
            M = torch.zeros(K, X.shape[1], device=X.device).index_add_(0, a, X); cnt = torch.bincount(a, minlength=K)
            if (cnt == 0).any():
                worst = (X @ C.T).max(1).values.argsort()[: int((cnt == 0).sum())]; M[cnt == 0] = X[worst]
            Cn = F.normalize(M, dim=1)
            done = bool(torch.allclose(Cn, C, atol=1e-6)); C = Cn
            if done:
                break
        a = (X @ C.T).argmax(1); inertia = float((1 - (X * C[a]).sum(1)).sum())
        if best is None or inertia < best[0]:
            best = (inertia, a.clone())
    return best[1], best[0]


def cluster(X, K):
    """Plain-mean centroids (rows of unit vectors averaged, not renormalised) after dropping small clusters."""
    n = X.shape[0]
    if K == 1 or n < 20 * K:
        return X.mean(0, keepdim=True), torch.zeros(n, dtype=torch.long, device=X.device), float((1 - X @ F.normalize(X.mean(0), dim=0)).sum())
    a, inertia = skmeans(X, K); minsz = max(20, math.ceil(0.02 * n)); cnt = torch.bincount(a, minlength=K)
    keep = [k for k in range(K) if int(cnt[k]) >= minsz]
    if not keep:
        return X.mean(0, keepdim=True), torch.zeros(n, dtype=torch.long, device=X.device), inertia
    C = torch.stack([X[a == k].mean(0) for k in keep])
    if len(keep) < K:
        a = (X @ F.normalize(C, dim=1).T).argmax(1); C = torch.stack([X[a == k].mean(0) for k in range(len(keep))])
    else:
        a = torch.tensor([keep.index(int(x)) for x in a.tolist()], device=X.device) if keep != list(range(K)) else a
    return C, a, inertia


def silhouette(X, a, nmax=2000, seed=0):
    if int(a.max()) == 0:
        return np.nan
    g = torch.Generator(device="cpu").manual_seed(seed); idx = torch.randperm(X.shape[0], generator=g)[:nmax].to(X.device)
    Xs, as_ = X[idx], a[idx]; D = 1 - Xs @ Xs.T; K = int(as_.max()) + 1
    M = torch.stack([(D * (as_ == k)[None]).sum(1) / (as_ == k).sum().clamp(min=1) for k in range(K)], 1)
    own = as_[:, None]; sz = torch.bincount(as_, minlength=K)[as_].float()
    aa = M.gather(1, own)[:, 0] * sz / (sz - 1).clamp(min=1)
    M2 = M.clone(); M2.scatter_(1, own, float("inf")); bb = M2.min(1).values
    s = (bb - aa) / torch.maximum(aa, bb).clamp(min=1e-8); s[sz <= 1] = 0
    return float(s.mean())


def match_cos(A, B):
    """Best mean cosine over one-to-one matchings of the rows of A to rows of B (normalised)."""
    A, B = F.normalize(A, dim=1), F.normalize(B, dim=1)
    if A.shape[0] > B.shape[0]:
        A, B = B, A
    S = (A @ B.T).cpu().numpy(); best = -1
    for perm in itertools.permutations(range(B.shape[0]), A.shape[0]):
        best = max(best, float(np.mean([S[i, j] for i, j in enumerate(perm)])))
    return best


def ari(x, y):
    from math import comb
    x, y = np.asarray(x), np.asarray(y); n = len(x)
    if n < 2:
        return np.nan
    tab = {}
    for i, j in zip(x, y):
        tab[(i, j)] = tab.get((i, j), 0) + 1
    sij = sum(comb(v, 2) for v in tab.values())
    sa = sum(comb(int(v), 2) for v in np.bincount(x)); sb = sum(comb(int(v), 2) for v in np.bincount(y)); tot = comb(n, 2)
    exp = sa * sb / tot; mx = (sa + sb) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else np.nan


# ---------------------------------------------------------------- Part C0 fit
def _meta(fs):
    M = []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        M.append({"id": f.stem, "label": str(z["label"]), "ds": str(z["ds"]), "mention": z["mention"].astype(bool),
                  "modules": [str(x) for x in z["modules"]]})
    return M


def subsets(M, Sp):
    """{regime: {subset: [indices]}} over dump images."""
    dev = {"X": [i for i, m in enumerate(M)], "V": [i for i, m in enumerate(M) if m["ds"] == "voc"]}
    return {r: {"all": dev[r], "dA": [i for i in dev[r] if Sp[M[i]["id"]]["fold"] == "dA"],
                "dB": [i for i in dev[r] if Sp[M[i]["id"]]["fold"] == "dB"]} for r in REGIMES}


def fit(a):
    fs = dump_files(a.tag); M = _meta(fs); Sp = load_split(); SUB = subsets(M, Sp); mods = M[0]["modules"]
    FIT = {r: {s: {"mo": {}, "neg": {}, "gall": {}, "g4": {}} for s in SUB[r]} for r in REGIMES}
    STAT = {r: {s: {} for s in SUB[r]} for r in REGIMES}
    print(f"fit: {len(fs)} dump images; subsets " + str({r: {s: len(v) for s, v in SUB[r].items()} for r in REGIMES}), flush=True)
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); Q, Tt = [], []
        for f in fs:
            z = np.load(f); Q.append(F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1)); Tt.append(torch.from_numpy(z[f"t_{mi:02d}"]).float())
        N = Q[0].shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
        obj_sum = {}; neg_vec = {}; neg_src = {}; gsamp = []; gsum = []
        for i, (q, t) in enumerate(zip(Q, Tt)):
            lab = M[i]["label"]; ci = CLASSES.index(lab)
            obj_sum[i] = (q[t[ci].topk(ko).indices].sum(0), ko)
            for cj, c in enumerate(CLASSES):
                if c == lab or M[i]["mention"][cj]:
                    continue
                neg_vec.setdefault((c, i), q[t[cj].topk(kn).indices])
            rng = torch.Generator().manual_seed(h(M[i]["id"]))
            gsamp.append(q[torch.randperm(N, generator=rng)[: max(1, round(0.02 * N))]]); gsum.append((q.sum(0), N))
        for r in REGIMES:
            for s, ids in SUB[r].items():
                P = FIT[r][s]; ids_set = set(ids)
                gs = torch.cat([gsamp[i] for i in ids]).to(DEV)
                P["gall"][m] = (sum(gsum[i][0] for i in ids) / sum(gsum[i][1] for i in ids))
                C4, a4, _ = cluster(gs, 4); P["g4"][m] = C4.cpu()
                STAT[r][s][("__generic", m, 4)] = {"silh": silhouette(gs, a4), "sizes": torch.bincount(a4).tolist()}
                for c in CLASSES:
                    oi = [i for i in ids if M[i]["label"] == c]
                    if oi:
                        P["mo"][(c, m)] = sum(obj_sum[i][0] for i in oi) / sum(obj_sum[i][1] for i in oi)
                    keys = [(c, i) for i in ids if (c, i) in neg_vec]
                    if not keys:
                        continue
                    X = torch.cat([neg_vec[k] for k in keys]).to(DEV)
                    src = np.concatenate([[M[k[1]]["label"]] * neg_vec[k].shape[0] for k in keys])
                    P["neg"][(c, m)] = {}; base = None
                    for K in KS:
                        C, lab_, inertia = cluster(X, K); P["neg"][(c, m)][K] = C.cpu()
                        if K == 1:
                            base = inertia
                        al = lab_.cpu().numpy(); srcs = []
                        for k in range(C.shape[0]):
                            u, n_ = np.unique(src[al == k], return_counts=True); o = np.argsort(-n_)[:3]
                            srcs.append([(str(u[j]), round(float(n_[j] / max(n_.sum(), 1)), 2)) for j in o])
                        STAT[r][s][(c, m, K)] = {"n": int(X.shape[0]), "k": int(C.shape[0]), "silh": silhouette(X, lab_),
                                                 "drop": (1 - inertia / base) if base and K > 1 else 0.0,
                                                 "sizes": torch.bincount(lab_, minlength=C.shape[0]).tolist(), "src": srcs,
                                                 "cos_mean": (F.normalize(C, dim=1) @ F.normalize(X.mean(0), dim=0)).cpu().tolist()}
        print(f"fit module {mi} ({m}) N={N}: {time.perf_counter() - t0:.0f}s", flush=True)
        del Q, Tt
    torch.save({"fit": FIT, "stat": STAT, "modules": mods}, WORK / f"fit{a.tag}.pt")


# ---------------------------------------------------------------- gates
def g_single(q, mo, n):
    w = F.normalize(mo - n, dim=0); pp, pm = float(w @ mo), float(w @ n)
    return torch.sigmoid(T.GAIN * (q @ w - (pp + pm) / 2) / max(pp - pm, 1e-6))


def g_multi(q, mo, C, soft=False):
    W = F.normalize(mo[None] - C, dim=1); pp = W @ mo; pm = (W * C).sum(1)
    G = torch.sigmoid(T.GAIN * (q @ W.T - (pp + pm) / 2) / (pp - pm).clamp(min=1e-6))
    sim = q @ F.normalize(C, dim=1).T
    if soft:
        return (torch.softmax(TAU * sim, 1) * G).sum(1)
    return G.gather(1, sim.argmax(1, keepdim=True))[:, 0]


def arm_gate(P, arm, c, m, q):
    mo = P["mo"].get((c, m)); neg = P["neg"].get((c, m))
    if mo is None or (neg is None and arm not in ("NO", "MG4")):
        return None
    if arm == "NC":
        return g_single(q, mo, neg[1][0])
    if arm == "NH":
        return g_single(q, mo, neg[1][0]).sqrt()
    if arm == "NO":
        return g_single(q, mo, P["gall"][m])
    if arm == "MG4":
        return g_multi(q, mo, P["g4"][m])
    if arm == "MS4":
        return g_multi(q, mo, neg[4], soft=True)
    return g_multi(q, mo, neg[int(arm[2:])])


def to_dev(P):
    return {k: ({kk: (({K: v.to(DEV) for K, v in vv.items()} if isinstance(vv, dict) else vv.to(DEV))) for kk, vv in d.items()})
            for k, d in P.items()}


# ---------------------------------------------------------------- Part C1-C4 diagnostics
def _regions_at(fg64, res):
    t = torch.from_numpy(fg64.astype(np.float32))[None, None]
    fg = F.adaptive_avg_pool2d(t, res)[0, 0] > 0.5; rr = max(1, round(6 * res / 64))
    near = (F.max_pool2d(fg.float()[None, None], 2 * rr + 1, 1, rr)[0, 0] > 0) & ~fg
    return fg.reshape(-1), near.reshape(-1), (~fg & ~near).reshape(-1)


def diag(a):
    fs = dump_files(a.tag); M = _meta(fs); Sp = load_split(); B = torch.load(WORK / f"fit{a.tag}.pt")
    FIT, STAT, mods = B["fit"], B["stat"], B["modules"]
    PX = {s: to_dev(FIT["X"][s]) for s in ("all", "dA", "dB")}
    VAR = ("MK2", "MK4", "MK8", "MG4", "MS4")
    acc = {}       # (c, m) -> dict
    for mi, m in enumerate(mods):
        for i, f in enumerate(fs):
            z = np.load(f); q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float().to(DEV), dim=1); N = q.shape[0]; res = int(round(N ** 0.5))
            c = M[i]["label"]; other = "dB" if Sp[M[i]["id"]]["fold"] == "dA" else "dA"; P = PX[other]
            gnc = arm_gate(P, "NC", c, m, q)
            if gnc is None:
                continue
            fg, near, far = (x.to(DEV) for x in _regions_at(z["fg64"], res)); bg = near | far
            A = acc.setdefault((c, m), {"fp": [], "tp": [], "fp_img": [], "fpg": {v: [0, 0] for v in VAR}, "obj": {v: [0, 0] for v in VAR},
                                        "fg_g": [], "cos_n": [], "cos_o": [], "fg_near_neg": [0, 0], "fg_negk": []})
            hgfp, hgtp = bg & (gnc > 0.5), fg & (gnc > 0.5)
            A["fp"].append(q[hgfp].cpu()); A["tp"].append(q[hgtp].cpu()); A["fp_img"] += [h(M[i]["id"]) & 1] * int(hgfp.sum())
            for v in VAR:
                gv = arm_gate(P, v, c, m, q)
                A["fpg"][v][0] += int((hgfp & (gv < 0.5)).sum()); A["fpg"][v][1] += int(hgfp.sum())
                A["obj"][v][0] += int((hgtp & (gv < 0.5)).sum()); A["obj"][v][1] += int(hgtp.sum())
            mo = P["mo"][(c, m)]; nn = P["neg"][(c, m)][1][0]
            Pall = PX["all"]; mo_a = Pall["mo"].get((c, m), mo); C4 = Pall["neg"].get((c, m), P["neg"][(c, m)])[4]
            qf = q[fg]
            if qf.shape[0]:
                A["fg_g"].append(gnc[fg].cpu()); A["cos_n"].append((qf @ F.normalize(nn, dim=0)).cpu()); A["cos_o"].append((qf @ F.normalize(mo, dim=0)).cpu())
                sim = qf @ F.normalize(C4, dim=1).T; k = sim.argmax(1); closer = sim.max(1).values > (qf @ F.normalize(mo_a, dim=0))
                A["fg_near_neg"][0] += int(closer.sum()); A["fg_near_neg"][1] += int(qf.shape[0]); A["fg_negk"].append(k[closer].cpu())
        print(f"diag module {mi} done", flush=True)
    # summarise per (class, module)
    rows = []; ari_rows = []
    for (c, m), A in acc.items():
        fp = torch.cat(A["fp"]) if A["fp"] else torch.zeros(0); tp = torch.cat(A["tp"]) if A["tp"] else torch.zeros(0)
        g = torch.Generator().manual_seed(h(c + m)); r = {"c": c, "m": m, "n_fp": int(fp.shape[0])}
        if fp.shape[0] >= 80:
            fp_s = fp[torch.randperm(fp.shape[0], generator=g)[:5000]].to(DEV)
            tp_s = tp[torch.randperm(tp.shape[0], generator=g)[: fp_s.shape[0]]].to(DEV) if tp.shape[0] >= 80 else None
            for K in (2, 4, 8):
                C, al, inert = cluster(fp_s, K); _, _, i1 = cluster(fp_s, 1)
                r[f"fp_silh{K}"] = silhouette(fp_s, al); r[f"fp_drop{K}"] = 1 - inert / i1 if i1 else np.nan
                if tp_s is not None:
                    Ct, at, it = cluster(tp_s, K); _, _, j1 = cluster(tp_s, 1)
                    r[f"tp_silh{K}"] = silhouette(tp_s, at); r[f"tp_drop{K}"] = 1 - it / j1 if j1 else np.nan
                if K == 4:
                    C4 = C
                    half = torch.tensor(A["fp_img"]).bool()
                    sel = torch.randperm(fp.shape[0], generator=g)
                    h0, h1 = fp[half][:5000].to(DEV), fp[~half][:5000].to(DEV)
                    if h0.shape[0] >= 80 and h1.shape[0] >= 80:
                        r["fp_half_cos"] = match_cos(cluster(h0, 4)[0], cluster(h1, 4)[0])
                    neg_all = FIT["X"]["all"]["neg"].get((c, m), {}).get(4)
                    if neg_all is not None:
                        Sn = F.normalize(C4, dim=1) @ F.normalize(neg_all.to(DEV), dim=1).T
                        r["cov_mk"] = float(Sn.max(1).values.mean())
                        mo = FIT["X"]["all"]["mo"][(c, m)].to(DEV)
                        simn = (fp_s @ F.normalize(neg_all.to(DEV), dim=1).T).max(1).values
                        r["fp_closer_neg"] = float((simn > fp_s @ F.normalize(mo, dim=0)).float().mean())
                    Sg = F.normalize(C4, dim=1) @ F.normalize(FIT["X"]["all"]["g4"][m].to(DEV), dim=1).T
                    r["cov_mg"] = float(Sg.max(1).values.mean())
        for v in VAR:
            r[f"fp_rm_{v}"] = A["fpg"][v]; r[f"obj_rm_{v}"] = A["obj"][v]
        if A["fg_g"]:
            fgg = torch.cat(A["fg_g"]); r["fg_q"] = np.quantile(fgg.numpy(), [.05, .25, .5, .75, .95]).tolist()
            r["fg_lt05"] = float((fgg < 0.5).float().mean()); r["cos_n"] = float(torch.cat(A["cos_n"]).mean()); r["cos_o"] = float(torch.cat(A["cos_o"]).mean())
            r["fg_closer_neg"] = A["fg_near_neg"][0] / max(A["fg_near_neg"][1], 1)
            ks = torch.cat(A["fg_negk"]) if A["fg_negk"] else torch.zeros(0, dtype=torch.long)
            r["fg_negk"] = torch.bincount(ks, minlength=4).tolist() if ks.numel() else []
        sx = STAT["X"]["all"].get((c, m, 4))
        if sx:
            r["neg_silh4"] = sx["silh"]; r["neg_drop4"] = sx["drop"]; r["neg_sizes4"] = sx["sizes"]; r["neg_src4"] = sx["src"]; r["neg_cosmean4"] = sx["cos_mean"]
            for K in (2, 8):
                s2 = STAT["X"]["all"].get((c, m, K)); r[f"neg_silh{K}"] = s2["silh"] if s2 else np.nan
        nA, nB = FIT["X"]["dA"]["neg"].get((c, m), {}).get(4), FIT["X"]["dB"]["neg"].get((c, m), {}).get(4)
        if nA is not None and nB is not None and nA.shape[0] > 1 and nB.shape[0] > 1:
            r["neg_ab_cos"] = match_cos(nA, nB)
        rows.append(r)
    # cross-module ARI of nearest-MK4-prototype assignments on shared HGFP pixels (64-px modules)
    m64 = [m for mi, m in enumerate(mods) if np.load(fs[0])[f"q_{mi:02d}"].shape[0] == 4096]
    for c in CLASSES:
        idx = [i for i in range(len(fs)) if M[i]["label"] == c][:40]
        if not idx:
            continue
        per = {m: [] for m in m64}; masks = {m: [] for m in m64}
        for i in idx:
            z = np.load(fs[i]); other = "dB" if Sp[M[i]["id"]]["fold"] == "dA" else "dA"; P = PX[other]
            fg, near, far = (x.to(DEV) for x in _regions_at(z["fg64"], 64))
            for m in m64:
                mi = mods.index(m); q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float().to(DEV), dim=1)
                gnc = arm_gate(P, "NC", c, m, q); neg = P["neg"].get((c, m), {}).get(4)
                if gnc is None or neg is None:
                    per[m].append(None); masks[m].append(None); continue
                masks[m].append(((near | far) & (gnc > 0.5)).cpu()); per[m].append((q @ F.normalize(neg, dim=1).T).argmax(1).cpu())
        for m1, m2 in itertools.combinations(m64, 2):
            xs, ys = [], []
            for j in range(len(idx)):
                if per[m1][j] is None or per[m2][j] is None:
                    continue
                both = masks[m1][j] & masks[m2][j]; xs.append(per[m1][j][both]); ys.append(per[m2][j][both])
            if xs and sum(len(x) for x in xs) >= 50:
                ari_rows.append(ari(torch.cat(xs).numpy(), torch.cat(ys).numpy()))
    report_diag(a, rows, ari_rows, STAT)


def _med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return (np.median(v), np.quantile(v, .25), np.quantile(v, .75), len(v)) if v else (np.nan, np.nan, np.nan, 0)


def report_diag(a, rows, ari_rows, STAT):
    f3 = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}] (n={t[3]})"
    L = ["# Run 31, Part C: query-space diagnostics on the dev dump (`PREREG.md`)\n",
         "Regime X (VOC-Sim + COCO-Cap dev), cross-fitted profiles. HGFP = background pixels with NC gate > 0.5; "
         "medians over (class, module) cells with [IQR]. Ground truth only labels regions.\n"]
    neg4 = _med([r.get("neg_silh4") for r in rows]); ab = _med([r.get("neg_ab_cos") for r in rows])
    go = (neg4[0] >= 0.10) and (ab[0] >= 0.90)
    L.append("## Go / no-go (fixed in PREREG)\n")
    L.append(f"- Median negative-pool silhouette at K = 4: {f3(neg4)} (rule ≥ 0.10)")
    L.append(f"- Median dA vs dB matched-centroid cosine at K = 4: {f3(ab)} (rule ≥ 0.90)")
    L.append(f"- **Structure present: {'yes' if go else 'no'}**\n")
    L.append("## C1 are the remaining false positives clustered (vs the true-positive null)?\n")
    L.append("| K | HGFP silhouette | high-gated object (null) silhouette | HGFP inertia drop | null inertia drop |"); L.append("|---|---|---|---|---|")
    for K in (2, 4, 8):
        L.append(f"| {K} | {f3(_med([r.get(f'fp_silh{K}') for r in rows]))} | {f3(_med([r.get(f'tp_silh{K}') for r in rows]))} | "
                 f"{f3(_med([r.get(f'fp_drop{K}') for r in rows]))} | {f3(_med([r.get(f'tp_drop{K}') for r in rows]))} |")
    L.append("\n## C2 reproducibility\n")
    L.append(f"- HGFP clusters, split-half matched cosine (K = 4): {f3(_med([r.get('fp_half_cos') for r in rows]))}")
    L.append(f"- Negative pools, dA vs dB matched cosine (K = 4): {f3(ab)}")
    L.append(f"- Cross-module ARI of nearest-prototype assignments on shared HGFP pixels (64-px module pairs): {f3(_med(ari_rows))}\n")
    L.append("## C3 does one negative mean blend several modes?\n")
    L.append("| K | negative-pool silhouette |"); L.append("|---|---|")
    for K in (2, 4, 8):
        L.append(f"| {K} | {f3(_med([r.get(f'neg_silh{K}') for r in rows]))} |")
    L.append(f"\nGeneric all-pixel pools (MG4), silhouette K = 4: {f3(_med([v['silh'] for k, v in STAT['X']['all'].items() if k[0] == '__generic']))}\n")
    L.append("Effect of each gate relative to NC (pooled over all cells): share of HGFP pixels pushed below 0.5 (background removed) "
             "and share of object pixels with NC ≥ 0.5 pushed below 0.5 (object wrongly removed).\n")
    L.append("| Gate | background removed | object wrongly removed |"); L.append("|---|---:|---:|")
    for v in ("MK2", "MK4", "MK8", "MG4", "MS4"):
        fp = np.sum([r[f"fp_rm_{v}"] for r in rows], 0); ob = np.sum([r[f"obj_rm_{v}"] for r in rows], 0)
        L.append(f"| {v} | {fp[0] / max(fp[1], 1):.3f} | {ob[0] / max(ob[1], 1):.3f} |")
    L.append("\n## C1 vs C0: do the gate's prototypes cover the actual errors?\n")
    L.append(f"- Max cosine of each HGFP centroid to the absent-class centroids (MK4): {f3(_med([r.get('cov_mk') for r in rows]))}")
    L.append(f"- ... to the generic centroids (MG4): {f3(_med([r.get('cov_mg') for r in rows]))}")
    L.append(f"- Share of HGFP pixels closer to some MK4 centroid than to the object profile: {f3(_med([r.get('fp_closer_neg') for r in rows]))}\n")
    L.append("## C4 classes harmed vs helped by the gate (pooled over modules)\n")
    L.append("| Class | object gate median | object g<0.5 | cos(object, negative mean) | cos(object, object profile) | object closer to a negative centroid | background removed by MK4 | object wrongly removed by MK4 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in ("tv", "train", "potted plant", "person", "dining table", "aeroplane"):
        rc = [r for r in rows if r["c"] == c and "fg_q" in r]
        if not rc:
            continue
        fp = np.sum([r["fp_rm_MK4"] for r in rc], 0); ob = np.sum([r["obj_rm_MK4"] for r in rc], 0)
        L.append(f"| {c} | {np.median([r['fg_q'][2] for r in rc]):.2f} | {np.mean([r['fg_lt05'] for r in rc]):.2f} | {np.mean([r['cos_n'] for r in rc]):.3f} | "
                 f"{np.mean([r['cos_o'] for r in rc]):.3f} | {np.mean([r['fg_closer_neg'] for r in rc]):.2f} | {fp[0] / max(fp[1], 1):.3f} | {ob[0] / max(ob[1], 1):.3f} |")
    L.append("\nTop source classes of the absent-class clusters (K = 4) that absorb object pixels, for the harmed classes (64-px module 0 shown):\n")
    for c in ("tv", "train", "potted plant"):
        rc = [r for r in rows if r["c"] == c and r.get("neg_src4") and r.get("fg_negk")]
        if rc:
            r = rc[0]; L.append(f"- {c} ({r['m']}): object pixels per cluster {r['fg_negk']}; cluster sources {r['neg_src4']}")
    L.append("\nCaveat: 'absent' uses the caption, so an unmentioned class may still be visible (likely for person and car).\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"diag{a.tag}.md").write_text(txt)
    torch.save(rows, WORK / f"diag_rows{a.tag}.pt")


# ---------------------------------------------------------------- Part D capture / score / report
@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(a.model); B = torch.load(WORK / f"fit{a.tag}.pt"); FIT = B["fit"]; Sp = load_split(); cache = {}
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows_for(a, "all"):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); sp = Sp[iid]; rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target)); G = {}; miss = 0
        for r in REGIMES:
            if sp["part"] == "test" or (r == "V" and sp["ds"] == "coco"):
                sub = "all"
            else:
                sub = "dB" if sp["fold"] == "dA" else "dA"
            if (r, sub) not in cache:
                cache[(r, sub)] = to_dev(FIT[r][sub])
            P = cache[(r, sub)]
            for gname in GATED:
                gd = {}
                for m, q in Q.items():
                    g = arm_gate(P, gname, lab, m, q)
                    if g is None:
                        g = torch.ones(q.shape[0], device=q.device); miss += 1
                    gd[m] = g
                G[f"{r}:{gname}"] = gd
        if a.tag:   # smoke: K = 1 must equal NC
            for m, q in Q.items():
                P = cache[("X", "all" if sp["part"] == "test" else ("dB" if sp["fold"] == "dA" else "dA"))]
                if (lab, m) in P["neg"] and (lab, m) in P["mo"]:
                    d = float((g_multi(q, P["mo"][(lab, m)], P["neg"][(lab, m)][1]) - g_single(q, P["mo"][(lab, m)], P["neg"][(lab, m)][1][0])).abs().max())
                    assert d < 1e-5, f"K=1 != NC ({d:.2e})"
        S = T.ca_seeds(rec, {arm: tgt for arm in ARMS}, gate=G)
        fid = float((S["A"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"seed fidelity {fid:.2e} for {iid}")
        S["A"] = seed; maps = []
        for arm in ARMS:
            v = sg_combined_v2.aggregate(sa, S[arm])[0]
            maps.append(np.stack([x.float().reshape(64, 64).cpu().numpy() for x in (S[arm], v["V0"], v["V3"])]))
        maps = np.stack(maps).astype(np.float32); gm = np.stack([QG.gate_map(G[k]).cpu().numpy() for k in GMAP_ARMS]).astype(np.float32)
        if not (np.isfinite(maps).all() and np.isfinite(gm).all()):
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), gates=gm.astype(np.float16), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid, part=sp["part"], fold=str(sp["fold"]), gate_miss=miss)
        os.replace(tmp, o)
        print(f"[mneg:{a.set}] {iid} {sp['part']} fid {fid:.1e} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"cls": str(z["cls"]), "part": str(z["part"]), "fold": str(z["fold"]), "fid_seed": float(z["fid_seed"]), "area": float(truth.mean())}
    for ai, arm in enumerate(ARMS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][ai, mi].astype(np.float32)), truth.shape)
            res[f"{arm}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            if m == "V3":
                ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
                res[f"{arm}|V3+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    jobs = []
    for s in SETS:
        sc = WORK / f"{s}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name)) for f in sorted((WORK / f"{s}{a.tag}").glob("*.npz")) if not f.stem.endswith(".tmp")]
    jobs = jobs[a.shard::a.num_shards]
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[1]).exists()]; print(f"attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 50 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)


def _load_scores(ds, tag):
    fs = sorted(f for f in (WORK / f"{ds}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([f.stem for f in fs]), "part": np.array([str(z["part"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k:
            S[k] = np.stack([z[k] for z in Z])
    return S


def report(a):
    S = {ds: _load_scores(ds, a.tag) for ds in SETS}
    sub = lambda s, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in s.items()}
    dev = {ds: sub(S[ds], S[ds]["part"] == "dev") for ds in SETS}; tst = {ds: sub(S[ds], S[ds]["part"] == "test") for ds in SETS}
    mi = lambda D, P: cc.miou(P[:, None], D["cls"], np.ones(len(D["cls"]), bool))[0] if len(D["cls"]) else np.nan
    curve = lambda D, k: cc.miou(D[k], D["cls"], np.ones(len(D["cls"]), bool)) if len(D["cls"]) else np.zeros(len(THR))

    def thr(reg, k):
        cv = curve(dev["voc"], k)
        return int(np.argmax(cv if reg == "V" else (cv + curve(dev["coco"], k)) / 2))

    def pred(reg, ds, k, rule="dev"):
        D = tst[ds]
        if rule == "55":
            return D[k][:, T55]
        if rule == "best":
            return D[k][:, int(np.argmax(curve(D, k)))]
        return D[k][:, thr(reg, k)]

    def cag(reg, ds, m="V3+R2"):
        cand = ["A", f"{reg}:NH", f"{reg}:NC"]; ch = {}; P = np.zeros_like(tst[ds][f"A|{m}|raw"][:, 0])
        for c in CLASSES:
            best = None
            for arm in cand:
                k = f"{arm}|{m}|raw"; t = thr(reg, k); cnt = []
                for d in (["voc", "coco"] if reg == "X" else ["voc"]):
                    sel = dev[d]["cls"] == c
                    if sel.any():
                        cnt.append(dev[d][k][sel, t].sum(0))
                if not cnt:
                    continue
                x = np.sum(cnt, 0); iou = x[0] / max(x[:3].sum(), 1)
                if best is None or iou > best[0] + 1e-9:
                    best = (iou, arm)
            ch[c] = best[1] if best else "A"
        for c in np.unique(tst[ds]["cls"]):
            arm = ch.get(c, "A"); k = f"{arm}|{m}|raw"; sel = tst[ds]["cls"] == c; P[sel] = tst[ds][k][sel, thr(reg, k)]
        return P, ch

    def dci(ds, Pa, Pb):
        if len(tst[ds]["cls"]) < 2:
            return "n/a", np.nan, np.nan, np.nan
        lo, hi = cc.boot(tst[ds], Pa, Pb, n=a.boot); d = mi(tst[ds], Pa) - mi(tst[ds], Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo, hi

    L = ["# Run 31, Part D: multi-negative gate evaluation (`PREREG.md`)\n",
         "Test halves only; thresholds and per-class strengths chosen on dev. **Exploratory**: both benchmarks were seen in "
         "runs 28–30. Regime V = VOC-Sim-only development → transfer; regime X = mixed VOC-Sim + COCO-Cap development.\n",
         f"Test images: VOC-Sim {len(tst['voc']['cls'])}, COCO-Cap {len(tst['coco']['cls'])}.\n"]
    verdict = {}
    for reg in REGIMES:
        L.append(f"\n## Regime {reg} ({'VOC-Sim-only development' if reg == 'V' else 'mixed development'})\n")
        L.append("| Seed arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | VOC-Sim, test-best | COCO-Cap, test-best |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in ("A",) + tuple(f"{reg}:{g}" for g in GATED):
            k = f"{arm}|V3+R2|raw"
            L.append(f"| {arm} | {mi(tst['voc'], pred(reg, 'voc', k)):.3f} | {mi(tst['coco'], pred(reg, 'coco', k)):.3f} | "
                     f"{mi(tst['voc'], pred(reg, 'voc', k, '55')):.3f} | {mi(tst['coco'], pred(reg, 'coco', k, '55')):.3f} | "
                     f"{mi(tst['voc'], pred(reg, 'voc', k, 'best')):.3f} | {mi(tst['coco'], pred(reg, 'coco', k, 'best')):.3f} |")
        Pc = {ds: cag(reg, ds) for ds in SETS}
        L.append(f"| CAg | {mi(tst['voc'], Pc['voc'][0]):.3f} | {mi(tst['coco'], Pc['coco'][0]):.3f} | — | — | — | — |")
        L.append("\nHierarchy + DenseCRF, dev-selected thresholds.\n")
        L.append("| ID | Comparison | VOC-Sim test | COCO-Cap test |"); L.append("|---|---|---|---|")
        K = lambda arm, m="V3+R2": f"{arm}|{m}|raw"
        tests = [("T1", "MK4 − NC", f"{reg}:MK4", f"{reg}:NC"), ("T2", "MK4 − A", f"{reg}:MK4", "A"),
                 ("T3", "CAg − A", "CAg", "A"), ("T4", "MK4 − MG4", f"{reg}:MK4", f"{reg}:MG4"),
                 ("—", "MS4 − MK4", f"{reg}:MS4", f"{reg}:MK4"), ("—", "MK2 − NC", f"{reg}:MK2", f"{reg}:NC"),
                 ("—", "MK8 − NC", f"{reg}:MK8", f"{reg}:NC"), ("—", "NO − NC", f"{reg}:NO", f"{reg}:NC"),
                 ("—", "NH − NC", f"{reg}:NH", f"{reg}:NC"), ("—", "NC − A", f"{reg}:NC", "A"), ("—", "MG4 − A", f"{reg}:MG4", "A")]
        res = {}
        for tid, lb, x, y in tests:
            cells = []
            for ds in SETS:
                Px = Pc[ds][0] if x == "CAg" else pred(reg, ds, K(x)); Py = pred(reg, ds, K(y)); cells.append(dci(ds, Px, Py))
            res[tid if tid != "—" else lb] = cells
            L.append(f"| {tid} | {lb} | {cells[0][0]} | {cells[1][0]} |")
        prom = lambda c: c[1][2] > 0 and c[0][2] >= -0.005
        verdict[reg] = {t: prom(res[t]) for t in ("T1", "T2", "T3")}; verdict[reg]["T4"] = res["T4"][1][2] > 0
        L.append(f"\nReading (rule fixed in PREREG): T1 {'promising' if verdict[reg]['T1'] else 'not promising'}; "
                 f"T2 {'promising' if verdict[reg]['T2'] else 'not promising'}; T3 {'promising' if verdict[reg]['T3'] else 'not promising'}; "
                 f"spurious-mode claim (T4 above zero on COCO-Cap) {'supported' if verdict[reg]['T4'] else 'not supported'}.\n")
        L.append("Seed and fine-only (raw, dev-selected thresholds), arm − A:\n")
        L.append("| Arm | seed VOC-Sim | seed COCO-Cap | fine-only VOC-Sim | fine-only COCO-Cap |"); L.append("|---|---|---|---|---|")
        for g in GATED:
            arm = f"{reg}:{g}"; cells = []
            for m in ("seed", "V0"):
                for ds in SETS:
                    cells.append(dci(ds, pred(reg, ds, K(arm, m)), pred(reg, ds, K("A", m)))[0])
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L.append("\nCAg choices (non-A): " + ", ".join(f"{c} {v.split(':')[-1]}" for c, v in Pc["coco"][1].items() if v != "A") + "\n")
        cl = tst["coco"]["cls"]; arms = ["A", f"{reg}:NC", f"{reg}:MK4", f"{reg}:MG4"]
        L.append(f"Per class, COCO-Cap test (A / NC / MK4 / MG4 / CAg): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(pred(reg, 'coco', K(x))[:, None], cl, cl == c)[0]:.2f}" for x in arms)
            + f" / {cc.miou(Pc['coco'][0][:, None], cl, cl == c)[0]:.2f}" for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    p = sub.add_parser("diag30"); p.add_argument("--workers", type=int, default=8)
    for nm in ("dump", "capture"):
        p = sub.add_parser(nm); p.add_argument("--set", choices=SETS, required=True)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
        p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("fit", "diag"):
        p = sub.add_parser(nm); p.add_argument("--tag", default="")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"split": split, "diag30": diag30, "dump": dump, "fit": fit, "diag": diag, "capture": capture, "score": score, "report": report}[a.cmd](a)
