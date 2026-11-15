#!/usr/bin/env python3
"""Map analysis, parts 2-5 (CPU, exploratory diagnostics) on the rich capture of scripts/ana_capture.py.

  2  self-attention spectra: ground-truth energy in the top-k eigenvectors, best single eigenvector, eigengap vs
     instances, spectral clustering / Fiedler split labelled by the A3 seed, seed diffusion depth, per-head
     object/background mass retention; oracle seed through V3 (is the seed or the propagation the bottleneck?)
  3  cross-attention: per layer / head / step-block target maps, PCA across maps, SOT / EOT / padding tokens,
     competition with other caption nouns
  4  UNet decoder features: per-image PCA, within-image and cross-image linear probes (upper bounds), k-means purity
     of features vs x0 trajectories vs SA spectral embedding
  5  dynamics: per-block quality, x0 settling time inside vs outside the object, conditional vs unconditional SA
Ground truth is used only to measure.

python scripts/ana_dump.py run [--workers 16] [--max-images k]    -> runs/analysis_cluster/part2_5.md
"""
from __future__ import annotations

import argparse, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
from brace_paths import CACHE  # noqa: E402
from ana_part1 import best_iou  # noqa: E402
from ideas_grouped import kmeans  # noqa: E402

DUMP = CACHE / "ana_dump"; OUT = ROOT / "runs/analysis_cluster"
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
KE = (1, 2, 4, 8, 16, 32, 64)
DEPTHS = (1, 2, 4, 8, 16)
TAU = 0.55


def pool(x, side):
    t = torch.from_numpy(np.asarray(x, np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, side)[0, 0].numpy()


def bi(x, t):
    """best IoU and AUROC with the sign that is better (for PCs / eigenvectors)."""
    a = best_iou(x, t); b = best_iou(-x, t)
    return a if a[0] >= b[0] else b


def cluster_maps(E, seed_r, gt_r, Ks=(2, 4, 8)):
    out = {}
    X = torch.from_numpy(E).double(); X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    for K in Ks:
        lab = kmeans(X, K, restarts=2, iters=30).numpy(); m = np.zeros(len(lab)); pur = np.zeros(len(lab), bool)
        for j in range(K):
            s = lab == j
            if s.any():
                m[s] = seed_r[s].mean(); pur[s] = gt_r[s].mean() > 0.5
        rule = m >= 0.5 * m.max()
        out[K] = (best_iou(m, gt_r)[0], iou(rule, gt_r), iou(pur, gt_r))
    return out


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def ridge(X, y, lam=1e-2):
    A = X.T @ X; A += lam * np.trace(A) / len(A) * np.eye(len(A))
    return np.linalg.solve(A, X.T @ y)


def zs(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def one(f):
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    import sg_steps_test as stt
    _import_densecrf(stt.CRF_SITE)
    z = np.load(f); truth = z["truth"].astype(bool); img = z["image"]; hw = truth.shape
    tokens = list(z["tokens"]); eot = int(z["eot"]); content = list(z["content"])
    gt = {s: pool(truth, s) >= 0.5 for s in (16, 32, 64)}
    seed = st.mm(z["seed"]); seedr = {s: pool(seed, s) for s in (16, 32, 64)}
    r = {"id": f.stem, "cls": str(z["cls"]), "area": float(truth.mean()), "n_inst": int(label(gt[64])[1]),
         "rule": str(z["token_rule"]), "imgdiff": float(z["img_meandiff"])}
    # ---------------- oracle seed (part 2a)
    for k, m in (("v3", z["v3"]), ("v3_gt", z["v3_gt"]), ("seed", z["seed"])):
        x = st.up(st.mm(m), hw); r[f"best_{k}"] = best_iou(x, truth)[0]
        if k != "seed":
            msk = apply_ovam_densecrf(img, x >= TAU, stt.CRF_SITE).astype(bool); r[f"iou55_{k}"] = iou(msk, truth)
    # ---------------- self-attention spectra (part 2)
    for key in z.files:
        if not key.startswith("sa_evec/"):
            continue
        tag = key.split("/")[1]; side = int(tag.split("_")[0]); V = z[key].astype(np.float32); ev = z[f"sa_eval/{tag}"]
        g = gt[side].reshape(-1).astype(np.float32); gn = g / (np.linalg.norm(g) + 1e-12)
        c = V @ gn
        r[f"energy/{tag}"] = np.array([float((c[:k] ** 2).sum()) for k in KE])
        j = int(np.argmax(np.abs(c[1:]))) + 1; r[f"bestvec_idx/{tag}"] = j; r[f"bestvec_cos/{tag}"] = float(abs(c[j]))
        r[f"bestvec_iou/{tag}"] = bi(V[j], g.astype(bool))[0]
        gaps = ev[1:10] - ev[2:11]; r[f"eigengap_k/{tag}"] = int(np.argmax(gaps)) + 2
        fied = V[1]; sd = seedr[side].reshape(-1)
        side_a = fied > 0; pick = side_a if sd[side_a].mean() >= sd[~side_a].mean() else ~side_a
        r[f"fiedler_iou/{tag}"] = iou(pick, g.astype(bool))
        if side == 64 or tag.endswith("_c"):
            cm = cluster_maps(V[1:9].T.copy(), sd, g.astype(bool))
            for K, (b, ru, pu) in cm.items():
                r[f"spec{K}_best/{tag}"], r[f"spec{K}_rule/{tag}"], r[f"spec{K}_pure/{tag}"] = b, ru, pu
        r[f"head_stay/{tag}"] = z[f"sa_head/{tag}"]
        if f"sa_diff/{tag}" in z.files:
            D = z[f"sa_diff/{tag}"]
            r[f"diff_best/{tag}"] = np.array([best_iou(st.up(st.mm(d), hw), truth)[0] for d in D])
    # ---------------- cross-attention (part 3)
    tmaps16 = []; per = {}
    for key in z.files:
        if not key.startswith("ca/"):
            continue
        _, layer, blk = key.split("/")
        X = z[key].astype(np.float32); H, N, T = X.shape; side = int(N ** .5)
        tgt = X[:, :, tokens].sum(2)                                  # [H, N]
        g = gt[side].reshape(-1)
        for h in range(H):
            per[(layer, blk, h if H > 1 else "mean")] = best_iou(tgt[h], g) if g.any() else (np.nan, np.nan)
            tmaps16.append(pool(tgt[h].reshape(side, side), 16).reshape(-1))
        hm = X.mean(0)
        if side in (16, 32):
            for nm, col in (("sot", 0), ("eot", eot), ("pad", T - 1)):
                r.setdefault(f"tok_auc/{nm}/{side}", []).append(best_iou(hm[:, col], g)[1] if g.any() else np.nan)
            if side == 16 and len(content) > 1:
                others = [c for c in content if c not in tokens and c < T - 1]
                if others:
                    inside = hm[g]; tgt_m = inside[:, tokens].sum(1); oth = inside[:, others]
                    r.setdefault("tgt_share_in_gt", []).append(float((tgt_m / (tgt_m + oth.sum(1) + 1e-12)).mean()))
                    r.setdefault("top_other_vs_tgt", []).append(float(oth.mean(0).max() / (tgt_m.mean() + 1e-12)))
    r["ca_per"] = per
    for k in list(r):
        if k.startswith("tok_auc/") or k in ("tgt_share_in_gt", "top_other_vs_tgt"):
            r[k] = float(np.nanmean(r[k]))
    M = np.stack(tmaps16); g16 = gt[16].reshape(-1)
    mean16 = M.mean(0); r["ca_mean16_best"], r["ca_mean16_auc"] = best_iou(mean16, g16)
    Xc = M - M.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum(); r["ca_pca_ev"] = ev[:5]
    pc1 = Vt[0] * np.sign(Vt[0] @ (mean16 - mean16.mean()) + 1e-12)
    r["ca_pc1_best"], r["ca_pc1_auc"] = best_iou(pc1, g16)
    # rank-1 structure of the raw (uncentred) stack
    S2 = np.linalg.svd(M, compute_uv=False); r["ca_rank1_share"] = float(S2[0] ** 2 / (S2 ** 2).sum())
    # ---------------- features (part 4)
    for key in z.files:
        if not key.startswith("feat/"):
            continue
        Fm = z[key].astype(np.float32); C, h, w = Fm.shape; X = zs(Fm.reshape(C, -1).T); g = gt[h].reshape(-1)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r[f"feat_pc_best/{key[5:]}"] = max(bi(U[:, i], g)[0] for i in range(3))
        cb = (np.add.outer(np.arange(h) // 8, np.arange(w) // 8) % 2 == 0).reshape(-1)   # checkerboard split
        if g[cb].any() and g[~cb].any():
            wv = ridge(np.c_[X[cb], np.ones(cb.sum())], g[cb].astype(np.float32))
            p = np.c_[X[~cb], np.ones((~cb).sum())] @ wv
            r[f"feat_probe_in/{key[5:]}"] = best_iou(p, g[~cb])[0]
        P32 = U[:, :32] * S[:32]
        lab = kmeans(torch.from_numpy(P32).double(), 8, restarts=2, iters=30).numpy()
        pur = np.zeros(len(lab), bool)
        for j in range(8):
            if (lab == j).any():
                pur[lab == j] = g[lab == j].mean() > 0.5
        r[f"feat_pure8/{key[5:]}"] = iou(pur, g.astype(bool))
    x0 = z["x0"].astype(np.float32)                                   # [31, 4, 64, 64]
    Xt = zs(x0.transpose(2, 3, 0, 1).reshape(4096, -1)); g = gt[64].reshape(-1)
    lab = kmeans(torch.from_numpy(Xt).double(), 8, restarts=2, iters=30).numpy(); pur = np.zeros(4096, bool)
    for j in range(8):
        if (lab == j).any():
            pur[lab == j] = g[lab == j].mean() > 0.5
    r["x0_pure8"] = iou(pur, g)
    # ---------------- dynamics (part 5)
    d = np.linalg.norm(x0 - x0[-1:], axis=1)                          # [31, 64, 64]
    rel = d / (d[0:1] + 1e-6); settled = rel <= 0.1
    after = np.flip(np.cumprod(np.flip(settled, 0), 0), 0).astype(bool)   # settled from this step onwards
    first = np.where(after.any(0), after.argmax(0), 31)
    r["settle_in"] = float(first[gt[64]].mean()) if gt[64].any() else np.nan
    r["settle_out"] = float(first[~gt[64]].mean())
    # cross-image probe inputs: 256 object + 256 background pixels per image per feature map
    rng = np.random.default_rng(0); samp = {}
    for key in z.files:
        if key.startswith("feat/"):
            Fm = z[key].astype(np.float32); C, h, w = Fm.shape; X = zs(Fm.reshape(C, -1).T); g = gt[h].reshape(-1)
            if g.any() and (~g).any():
                i1 = rng.choice(np.flatnonzero(g), 256); i0 = rng.choice(np.flatnonzero(~g), 256)
                samp[key[5:]] = (X[np.r_[i1, i0]].astype(np.float16), np.r_[np.ones(256), np.zeros(256)].astype(np.float32))
    r["probe_samples"] = samp
    return r


def mean_of(rs, k):
    v = [x[k] for x in rs if k in x]
    return np.nanmean(np.stack(v), 0) if v else np.nan


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (DUMP / ds).glob("*.npz") if not f.stem.endswith(".tmp"))[: a.max_images]
        if not fs:
            continue
        with Pool(a.workers) as p:
            R[ds] = list(p.imap(one, fs, chunksize=1))
        print(ds, len(R[ds]), flush=True)
    L = ["# Map analysis parts 2–5 (rich capture of a fixed development subset; exploratory diagnostics)\n",
         "`scripts/ana_capture.py` + `scripts/ana_dump.py`. Ground truth is used only to measure. Tags: `<res>_b<block>_<branch>`, "
         "block 0/1/2 = steps 0–9 / 10–19 / 20–30, branch u = unconditional, c = conditional. Best IoU = best threshold "
         "of the map (perfect use of its ranking), at the map's own resolution unless noted.\n"]
    for ds, rs in R.items():
        n = len(rs); col = lambda k: np.array([x.get(k, np.nan) for x in rs], float)
        L += [f"## {NAME[ds]} ({n} images; base-loop vs live mean |Δpixel| {col('imgdiff').mean():.2f}/255)\n",
              "### 2a. Is the seed or the propagation the bottleneck? (512 px)\n",
              "| map | best IoU (mean) | IoU @ τ 0.55 + R2 (mean) |", "|---|---:|---:|",
              f"| A3 seed (raw) | {col('best_seed').mean():.3f} | — |",
              f"| BRACE V3 (A3 seed) | {col('best_v3').mean():.3f} | {col('iou55_v3').mean():.3f} |",
              f"| V3 with oracle seed (truth pooled to 64) | {col('best_v3_gt').mean():.3f} | {col('iou55_v3_gt').mean():.3f} |", ""]
        tags = sorted({k.split("/")[1] for x in rs for k in x if k.startswith("energy/")})
        L += ["### 2b. Self-attention spectra\n",
              "Ground-truth energy in the top-k eigenvectors of D^-1/2 W D^-1/2 (fraction of ‖g‖², mean); best single "
              "non-trivial eigenvector (|cos|, best IoU, mean index); Fiedler split labelled by the seed (IoU); "
              "eigengap estimate vs ground-truth components.\n",
              "| tag | " + " | ".join(f"E{k}" for k in KE) + " | best vec |cos| | best vec IoU | best vec idx | Fiedler IoU | "
              "eigengap k (mean) | corr(eigengap k, instances) |", "|---|" + "---:|" * (len(KE) + 6)]
        ni = col("n_inst")
        for t in tags:
            E = mean_of(rs, f"energy/{t}"); gk = col(f"eigengap_k/{t}")
            cr = np.corrcoef(gk, ni)[0, 1] if np.std(gk) > 0 and np.std(ni) > 0 else np.nan
            L.append(f"| {t} | " + " | ".join(f"{e:.2f}" for e in E) + f" | {col(f'bestvec_cos/{t}').mean():.2f} | "
                     f"{col(f'bestvec_iou/{t}').mean():.3f} | {col(f'bestvec_idx/{t}').mean():.1f} | {col(f'fiedler_iou/{t}').mean():.3f} | "
                     f"{gk.mean():.1f} | {cr:+.2f} |")
        L += ["", "Spectral clustering on the top-8 non-trivial eigenvectors, clusters labelled by the A3 seed "
                  "(best IoU of the cluster-mean seed map / fixed rule: clusters with mean seed ≥ half the max / oracle: "
                  "clusters with > 50% object):", "", "| tag | K=2 | K=4 | K=8 |", "|---|---|---|---|"]
        for t in tags:
            if f"spec2_best/{t}" in rs[0]:
                L.append(f"| {t} | " + " | ".join(f"{col(f'spec{K}_best/{t}').mean():.3f} / {col(f'spec{K}_rule/{t}').mean():.3f} / "
                                                  f"{col(f'spec{K}_pure/{t}').mean():.3f}" for K in (2, 4, 8)) + " |")
        L += ["", "Seed diffusion A^d s at 64 (head-mean SA of the block), best IoU at 512 (mean), vs BRACE V3 "
                  f"{col('best_v3').mean():.3f}:", "", "| tag | " + " | ".join(f"d={d}" for d in DEPTHS) + " |", "|---|" + "---:|" * len(DEPTHS)]
        area = col("area"); small = area <= np.median(area)
        for t in tags:
            if f"diff_best/{t}" in rs[0]:
                D = np.stack([x[f"diff_best/{t}"] for x in rs])
                L.append(f"| {t} | " + " | ".join(f"{D[:, i].mean():.3f} (small {D[small, i].mean():.3f} / large {D[~small, i].mean():.3f})"
                                                  for i in range(len(DEPTHS))) + " |")
        L += ["", "Per-head mass retention (mean over object queries of the attention mass that stays on the object; same "
                  "for background), per head 0–7:", "", "| tag | object stay | background stay |", "|---|---|---|"]
        for t in tags:
            Hs = mean_of(rs, f"head_stay/{t}")
            L.append(f"| {t} | " + " ".join(f"{v:.2f}" for v in Hs[:, 0]) + " | " + " ".join(f"{v:.2f}" for v in Hs[:, 1]) + " |")
        # part 3
        agg = defaultdict(list)
        for x in rs:
            for k, v in x["ca_per"].items():
                agg[k].append(v[0])
        rows = sorted(((np.nanmean(v), k) for k, v in agg.items()), reverse=True)
        L += ["", "### 3. Cross-attention\n",
              f"Mean over all layers, heads and blocks (pooled to 16): best IoU {col('ca_mean16_best').mean():.3f}, AUROC "
              f"{col('ca_mean16_auc').mean():.3f}. PC1 of the centred map stack: best IoU {col('ca_pc1_best').mean():.3f}, AUROC "
              f"{col('ca_pc1_auc').mean():.3f}. Explained variance of the first 5 PCs: "
              + ", ".join(f"{v:.2f}" for v in mean_of(rs, "ca_pca_ev")) + f"; rank-1 share of the uncentred stack {col('ca_rank1_share').mean():.2f}.", "",
              "Top 15 and bottom 5 single maps (layer, block, head; best IoU at own resolution, mean):", "",
              "| layer | block | head | best IoU |", "|---|---|---|---:|"]
        for v, (ly, b, h) in rows[:15] + rows[-5:]:
            L.append(f"| {ly} | {b} | {h} | {v:.3f} |")
        L += ["", "Special tokens, AUROC for the object (head- and layer-mean; < 0.5 = background-like):", "",
              "| token | 16 | 32 |", "|---|---:|---:|"]
        for nm in ("sot", "eot", "pad"):
            L.append(f"| {nm} | {col(f'tok_auc/{nm}/16').mean():.3f} | {col(f'tok_auc/{nm}/32').mean():.3f} |")
        if ds == "coco":
            sh = col("tgt_share_in_gt"); ok = np.isfinite(sh)
            L += ["", f"Competition (COCO-Cap, images with other content tokens: {ok.sum()}): inside the object, the target's share of "
                      f"content-token attention {np.nanmean(sh):.2f}; strongest other token's mean / target mean {np.nanmean(col('top_other_vs_tgt')):.2f}; "
                      f"correlation of target share with BRACE best IoU {np.corrcoef(sh[ok], col('best_v3')[ok])[0, 1]:+.2f}.",
                  "", "Token rule split: " + ", ".join(f"{rl} {int((np.array([x['rule'] for x in rs]) == rl).sum())}" for rl in ("class_word", "fallback"))]
        # part 4
        fk = sorted({k.split("/", 1)[1] for x in rs for k in x if k.startswith("feat_pc_best/")})
        L += ["", "### 4. Decoder features\n", "Per-image: best of the top-3 PCs (best IoU); within-image ridge probe on a checkerboard "
              "split (best IoU on held-out pixels, an upper bound); k-means K=8 oracle cluster selection (IoU).", "",
              "| features | top-3 PC | within-image probe | k-means K=8 oracle |", "|---|---:|---:|---:|"]
        for k in fk:
            L.append(f"| {k} | {col(f'feat_pc_best/{k}').mean():.3f} | {col(f'feat_probe_in/{k}').mean():.3f} | {col(f'feat_pure8/{k}').mean():.3f} |")
        L.append(f"| x0 trajectory (31×4) | — | — | {col('x0_pure8').mean():.3f} |")
        L.append("")
        # part 5
        L += ["### 5. Dynamics\n", f"x0 settling step (first step after which the x0 prediction stays within 10% of its initial distance "
              f"to the final): inside the object {np.nanmean(col('settle_in')):.1f}, outside {np.nanmean(col('settle_out')):.1f}.", ""]
    # cross-image linear probe (generic 'prompt object' direction?)
    if "voc" in R and len(R["voc"]) >= 4:
        L += ["## Cross-image linear probe (features standardised per image)\n",
              "Ridge on 512 sampled pixels per image. Train: first half of the VOC-Sim subset; test: second half and COCO-Cap. "
              "Reported: pixel AUROC on the test images' samples. Upper bound of a single class-agnostic linear 'object of the "
              "prompt' direction.", "", "| features | VOC-Sim test AUROC | COCO-Cap AUROC |", "|---|---:|---:|"]
        keys = sorted(R["voc"][0]["probe_samples"])
        for k in keys:
            tr = R["voc"][: len(R["voc"]) // 2]; te = R["voc"][len(R["voc"]) // 2:]
            get = lambda rs: [x["probe_samples"][k] for x in rs if k in x["probe_samples"]]
            Xtr = np.concatenate([s[0] for s in get(tr)]).astype(np.float32); ytr = np.concatenate([s[1] for s in get(tr)])
            w = ridge(np.c_[Xtr, np.ones(len(Xtr))], ytr)
            def auc(rs):
                S = get(rs)
                if not S:
                    return np.nan
                X = np.concatenate([s[0] for s in S]).astype(np.float32); y = np.concatenate([s[1] for s in S])
                p = np.c_[X, np.ones(len(X))] @ w; o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
                n1 = y.sum(); n0 = len(y) - n1
                return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            L.append(f"| {k} | {auc(te):.3f} | {auc(R.get('coco', [])):.3f} |")
    txt = "\n".join(L) + "\n"; (OUT / "part2_5.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--workers", type=int, default=16); p.add_argument("--max-images", type=int)
    a = ap.parse_args(); run(a)
