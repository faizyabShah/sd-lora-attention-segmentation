#!/usr/bin/env python3
"""Report for SG-real-atlas analyses 2-4 + spectral (Section 137), exploratory on D.
Usage: report_sg_real_atlas_alignment.py ALIGN_DIR SPECTRAL_DIR CAPTURE_DIR OUT_MD"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sg_real_atlas_alignment import ROLES, SIDE, auroc, gt_at, side_of, VOC  # noqa: E402
from PIL import Image

AL, SP, CAP, OUT = (Path(x) for x in sys.argv[1:5])
MANIFEST = Path(__file__).resolve().parents[1] / "outputs/segmentation/sg_real_atlas/manifest.json"
fold_of = {e["image"]: e["fold"] for e in json.loads(MANIFEST.read_text())["D"]}
VOCN = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
W = slice(10, 21)
L = []; w = L.append
files = sorted((AL / "D").glob("*.npz"))
R = [np.load(f, allow_pickle=False) for f in files]
IMG = np.array([f.stem.split("__")[0] for f in files]); FOLD = np.array([fold_of[i] for i in IMG])
AREA = np.array([float(r["area_full"]) for r in R]); KCL = np.array([int(r["k"]) for r in R])
q1, q2 = np.quantile(AREA, [1 / 3, 2 / 3]); THIRDS = {"small": AREA < q1, "medium": (AREA >= q1) & (AREA <= q2),
                                                      "large": AREA > q2}
mods = [str(m) for m in R[0]["mods"]]; sa_mods = [str(m) for m in R[0]["sa_mods"]]
short = lambda m: m.replace(".transformer_blocks.0.attn2", "").replace(".transformer_blocks.0.attn1", "") \
    .replace("down_blocks.", "d").replace("up_blocks.", "u").replace("mid_block", "mid").replace(".attentions.", ".")
w(f"# SG-real-atlas alignment diagnostics on D ({len(R)} pairs, {len(set(IMG))} photos)\n")
w("Exploratory. The real mask is only used to measure the stored attention. AUROC: 0.5 = chance, 1 = the map "
  "ranks every object pixel above every non-object pixel; it ignores the threshold. The SA window is calls 10–20 "
  f"(t ≈ 661→331). Object-size thirds cut at {q1:.3f} / {q2:.3f} of the image.\n")

# seed AUROC at 64 for reference
seed_auc = {}
for cond in ("renoise/n0", "ddim"):
    v = []
    for f, r in zip(files, R):
        iid, k = f.stem.split("__"); lab = np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png"))
        y, keep = gt_at(lab == int(k), lab != 255, 64)
        a3 = np.load(CAP / "D" / iid / k / "maps.npz")[f"{cond}/A3"].ravel()
        v.append(float(auroc(a3[keep][None], y[keep])[0]))
    seed_auc[cond] = np.array(v)

# ------------------------------------------------------------------ 1. CA atlas
w("## 1. Cross-attention atlas (class token)\n")
SIG = ("prob", "logit", "cos", "|q|")
for cond in ("renoise_n0", "ddim"):
    A = np.stack([r[f"{cond}/ca_auc"] for r in R])              # [P,4,M,31,8]
    w(f"### {cond}\n")
    w(f"A3 seed AUROC (the pipeline's CA seed, 64 px): {np.nanmean(seed_auc[cond.replace('_', '/')]):.3f}\n")
    w("Mean AUROC over pairs, window calls, heads — by signal and module:\n")
    w("| module | side | " + " | ".join(SIG) + " | best head (prob) | worst head (prob) |")
    w("|---|---:|" + "---:|" * (len(SIG) + 2))
    for i, m in enumerate(mods):
        h = np.nanmean(A[:, 0, i, W], axis=(0, 1))
        w(f"| {short(m)} | {side_of(m)} | " + " | ".join(f"{np.nanmean(A[:, s, i, W]):.3f}" for s in range(4))
          + f" | {h.max():.3f} (h{h.argmax()}) | {h.min():.3f} (h{h.argmin()}) |")
    w("\nBy call (prob, mean over modules and heads): " + ", ".join(
        f"c{c} {np.nanmean(A[:, 0, :, c]):.3f}" for c in range(0, 31, 3)) + "\n")
    # held-out unit selection: rank (module, call, head) units by mean AUROC on 4 folds; held-out AUROC of the
    # top units vs the seed on the 5th fold
    rows = []
    for K in (1, 5, 20, 100):
        d = []
        for fo in range(5):
            tr, te = FOLD != fo, FOLD == fo
            mu = np.nanmean(A[tr, 0], 0).ravel(); top = np.argsort(-np.nan_to_num(mu, nan=-1))[:K]
            d.append(np.nanmean(A[te, 0].reshape(te.sum(), -1)[:, top]))
        rows.append((K, np.mean(d)))
    w("Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other "
      "folds: " + ", ".join(f"K={K}: {v:.3f}" for K, v in rows) + f" — versus the A3 seed {np.nanmean(seed_auc[cond.replace('_', '/')]):.3f}.\n")
    # head ranking consistency (split-half across photos)
    imgs = np.unique(IMG); rng = np.random.default_rng(0); rr = []
    for _ in range(20):
        h = set(rng.choice(imgs, len(imgs) // 2, replace=False)); a = np.isin(IMG, list(h))
        x = np.nanmean(A[a, 0][:, :, W], (0, 2)).ravel(); y = np.nanmean(A[~a, 0][:, :, W], (0, 2)).ravel()
        rr.append(spearmanr(x, y, nan_policy="omit")[0])
    VN = np.stack([r[f"{cond}/ca_vnorm"] for r in R])
    vv = np.nanmean(VN[:, :, W], (0, 2)).ravel(); aa = np.nanmean(A[:, 0][:, :, W], (0, 2)).ravel()
    w(f"Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = {np.mean(rr):.2f}. "
      f"Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = "
      f"{spearmanr(vv, aa, nan_policy='omit')[0]:+.2f}.\n")
    RP = np.stack([r[f"{cond}/ca_rprec"] for r in R])
    w(f"R-precision of prob (share of the top-|M| pixels inside the mask): mean {np.nanmean(RP[:, :, W]):.3f}; "
      + ", ".join(f"{t} {np.nanmean(RP[THIRDS[t]][:, :, W]):.3f}" for t in THIRDS) + ".\n")
    TK = np.stack([r[f"{cond}/tok_auc"] for r in R])              # [P,M,31,11]
    w("Token AUROC (head-mean probability, window, all modules): " + ", ".join(
        f"{ro} {np.nanmean(TK[:, :, W, j]):.3f}" for j, ro in enumerate(ROLES)) + "\n")
    w("By size third (prob, all modules): " + ", ".join(
        f"{t} {np.nanmean(A[THIRDS[t]][:, 0][:, :, W]):.3f} (seed {np.nanmean(seed_auc[cond.replace('_', '/')][THIRDS[t]]):.3f})"
        for t in THIRDS) + "\n")

# ------------------------------------------------------------------ 2. SA keep/leak
w("## 2. Self-attention propagation atlas (real mask, keep / leak)\n")
w("keep = share of a mask pixel's attention that lands inside the mask; leak = the same for a non-mask pixel. "
  "Chance for both is the mask area a. Reported: keep lift (keep − a)/(1 − a) (1 = perfect), leak ratio leak / a "
  "(0 = no leak, 1 = chance), for the exact, eroded (interior) and dilated (+2 px at 64) masks.\n")
AR = np.stack([r["area"][3] for r in R])                          # [P,3] (scale-invariant)
for cond in ("renoise_n0", "ddim"):
    RL = np.stack([r[f"{cond}/rl"] for r in R])                   # [P,16,31,8,3,2]
    w(f"### {cond}\n")
    w("| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    ok = AR[:, 1] > 0
    for i, m in enumerate(sa_mods):
        x = RL[:, i, W].astype(np.float64)                         # [P,11,8,3,2]
        a = AR[:, None, None, :]
        kl = (x[..., 0] - a) / (1 - a); lr = x[..., 1] / np.maximum(a, 1e-6)
        hk = np.nanmean(kl[ok][..., 0], (0, 1))
        w(f"| {short(m)} | {side_of(m)} | {np.nanmean(kl[..., 0]):.3f} | {np.nanmedian(lr[..., 0]):.3f} | "
          f"{np.nanmean(kl[ok][..., 1]):.3f} | {np.nanmedian(lr[..., 2]):.3f} | {hk.max() - hk.min():.3f} |")
    x = RL[:, :, :, :, 0].astype(np.float64)
    a = AR[:, None, None, None, 0]
    kl = (x[..., 0] - a) / (1 - a)
    w("\nKeep lift by call (exact, mean over modules and heads): " + ", ".join(
        f"c{c} {np.nanmean(kl[:, :, c]):.3f}" for c in range(0, 31, 3)) + "\n")
    for t, s in THIRDS.items():
        w(f"- {t} objects: keep lift {np.nanmean(kl[s][:, :, W]):.3f}, median leak ratio "
          f"{np.nanmedian((x[s][..., 1][:, :, W]) / np.maximum(a[s], 1e-6)):.3f}")
    w("")

# ------------------------------------------------------------------ 3. diffusion
w("## 3. How many propagation steps? (y_t = A^t seed, head-mean A, averaged over the 11 calls)\n")
for cond in ("renoise_n0", "ddim"):
    w(f"### {cond}\n")
    w("| side | metric | " + " | ".join(f"t={t}" for t in range(9)) + " |\n|---|---|" + "---:|" * 9)
    for s in (16, 32, 64):
        D = np.stack([r[f"{cond}/diff{s}"] for r in R])            # [P,2,9]
        for j, nm in enumerate(("AUROC", "AP")):
            w(f"| {s} | {nm} | " + " | ".join(f"{v:.3f}" for v in np.nanmean(D[:, j], 0)) + " |")
    C = np.stack([r[f"{cond}/c16path"] for r in R])
    w("\nC16 path A64·up(A16^t c16): AUROC " + ", ".join(f"t={t + 1} {v:.3f}" for t, v in enumerate(np.nanmean(C[:, 0], 0)))
      + "; AP " + ", ".join(f"t={t + 1} {v:.3f}" for t, v in enumerate(np.nanmean(C[:, 1], 0))) + "\n")
    for t, s in THIRDS.items():
        D16 = np.stack([r[f"{cond}/diff16"] for r in R])[s]; D64 = np.stack([r[f"{cond}/diff64"] for r in R])[s]
        w(f"- {t}: AP side 16 " + " ".join(f"{v:.3f}" for v in np.nanmean(D16[:, 1], 0)[[0, 1, 2, 4, 8]])
          + " | side 64 " + " ".join(f"{v:.3f}" for v in np.nanmean(D64[:, 1], 0)[[0, 1, 2, 4, 8]]) + " (t = 0,1,2,4,8)")
    for s in (16, 64):
        H = np.stack([r[f"{cond}/heads2_{s}"] for r in R])
        w(f"- per-head AUROC at t=2, side {s}: " + " ".join(f"h{h} {v:.3f}" for h, v in enumerate(np.nanmean(H, 0))))
    w("")

# ------------------------------------------------------------------ 4. spectral
w("## 4. Spectral structure of the SA graphs (first 50 photos, renoise/n0)\n")
sf = sorted(SP.glob("*.npz"))
if sf:
    S = [np.load(f) for f in sf]; SA_ = np.array([float(s["area"]) for s in S])
    agg = defaultdict(list)
    for s in S:
        for key in s.files:
            if "|" in key:
                mod, c, g, nm = key.split("|"); agg[(side_of(mod), "mean" if g == "mean" else "head", nm)].append(s[key])
    w(f"{len(S)} pairs. E_k = share of the mask's energy in the k smoothest non-trivial eigenvectors (rolled mask "
      "= same shape, wrong place = chance).\n")
    w("| side | graph | E_k mask (k=1,2,4,8,16,32) | E_k rolled | E_k seed | best eigvec AUROC (oracle, j≤16) | "
      "seed-picked eigvec AUROC | seed low-pass AUROC (k=4,8,16,32,64) | seed AUROC |")
    w("|---|---|---|---|---|---:|---:|---|---:|")
    for side in (16, 32, 64):
        for g in ("mean", "head"):
            if (side, g, "E") not in agg:
                continue
            E = np.nanmean(agg[(side, g, "E")], 0); Er = np.nanmean(agg[(side, g, "E_roll")], 0)
            Es = np.nanmean(agg[(side, g, "E_seed")], 0); ea = np.nanmean([np.nanmax(x) if np.isfinite(x).any() else np.nan for x in agg[(side, g, "eig_auc")]])
            ep = np.nanmean(agg[(side, g, "eig_pick_auc")]); lp = np.nanmean(agg[(side, g, "lowpass_auc")], 0)
            sa = np.nanmean(agg[(side, g, "seed_auc")])
            f3 = lambda v: " ".join(f"{x:.2f}" for x in v)
            w(f"| {side} | {g} | {f3(E)} | {f3(Er)} | {f3(Es)} | {ea:.3f} | {ep:.3f} | {f3(lp)} | {sa:.3f} |")
    # low-pass by size
    w("\nSeed low-pass AUROC on the head-mean graph by object size (k = 4 / 8 / 16, side 32 and 64) vs seed:")
    per = defaultdict(lambda: defaultdict(list))
    for s, a_ in zip(S, SA_):
        t = "small" if a_ < q1 else ("large" if a_ > q2 else "medium")
        for key in s.files:
            if key.endswith("|mean|lowpass_auc"):
                per[t][side_of(key.split("|")[0])].append(s[key])
            if key.endswith("|mean|seed_auc"):
                per[t][("seed", side_of(key.split("|")[0]))].append(float(s[key]))
    for t in ("small", "medium", "large"):
        if per[t]:
            w(f"- {t} (n={sum(1 for a_ in SA_ if (a_ < q1 if t == 'small' else a_ > q2 if t == 'large' else q1 <= a_ <= q2))}): "
              + "; ".join(f"side {sd}: " + " ".join(f"{v:.3f}" for v in np.nanmean(per[t][sd], 0)[:3])
                          + f" (seed {np.nanmean(per[t][('seed', sd)]):.3f})" for sd in (32, 64) if per[t][sd]))
    w("\nPairs where a 16 px graph had no valid in-mask cell (AUROC undefined, dropped from those means): "
      + str(sum(1 for s in S if any(k.endswith("|seed_auc") and side_of(k.split("|")[0]) == 16 and not np.isfinite(s[k]) for k in s.files))))
    w("")

# ------------------------------------------------------------------ 5. errors
w("## 5. Where the errors are\n")
w("Pixel shares at the OOF thresholds of analysis 1. FP split by what the wrong pixel actually is; FN by whether "
  "it lies within 8 px (≈ one latent cell) of the object boundary.\n")
E = [json.loads(str(r["err"])) for r in R]
w("| variant | precision | recall | FP: other object | FP: background ≤8 px | FP: background far | FN: boundary band | FN: interior |")
w("|---|---:|---:|---:|---:|---:|---:|---:|")
for v in E[0]:
    t = {k: sum(e[v][k] for e in E) for k in ("tp", "fp_other", "fp_bg_near", "fp_bg_far", "fn_band", "fn_interior")}
    fp = t["fp_other"] + t["fp_bg_near"] + t["fp_bg_far"]; fn = t["fn_band"] + t["fn_interior"]
    w(f"| {v} | {t['tp'] / (t['tp'] + fp):.3f} | {t['tp'] / (t['tp'] + fn):.3f} | {t['fp_other'] / fp:.0%} | "
      f"{t['fp_bg_near'] / fp:.0%} | {t['fp_bg_far'] / fp:.0%} | {t['fn_band'] / fn:.0%} | {t['fn_interior'] / fn:.0%} |")
w("\nInstance coverage (share of each instance's pixels predicted), by instance size as a share of the image:\n")
w("| variant | <1% | 1–5% | 5–20% | >20% | instances missed (<10% covered) |\n|---|---:|---:|---:|---:|---:|")
for v in E[0]:
    rows = [(n / max(int(np.asarray(Image.open(VOC / f"SegmentationClass/{f.stem.split('__')[0]}.png")).size), 1), c)
            for f, e in zip(files, E) for n, c in e[v]["instances"]]
    sz = np.array([x[0] for x in rows]); cv = np.array([x[1] for x in rows])
    bins = [(0, .01), (.01, .05), (.05, .2), (.2, 1.1)]
    w(f"| {v} | " + " | ".join(f"{cv[(sz >= lo) & (sz < hi)].mean():.2f} (n={((sz >= lo) & (sz < hi)).sum()})" for lo, hi in bins)
      + f" | {(cv < .1).mean():.0%} of {len(cv)} |")
w("\nMost FP pixels on another object (n0_V0R2), target → other class: ")
conf = defaultdict(int)
for e, k in zip(E, KCL):
    for c, n in e["n0_V0R2"]["fp_other_by_class"].items():
        conf[(VOCN[k - 1], VOCN[int(c) - 1])] += n
tot = sum(conf.values())
w(", ".join(f"{a}→{b} {n / tot:.0%}" for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])[:12]) + "\n")
OUT.write_text("\n".join(L) + "\n"); print("\n".join(L))
