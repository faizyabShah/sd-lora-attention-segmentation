#!/usr/bin/env python3
"""SG-real-atlas analyses 1 + 4 (Section 137), exploratory on D.

Part A (baseline report): class-macro mIoU per variant at the frozen tau 0.55, best grid tau (diagnostic),
grouped 5-fold OOF tau, folds won and a photo-bootstrap CI against the fine-only reference n0/V0+R2.
Part B (H*, label-free reliability): per pair, label-free features (seed-mode keep/leak of the SA graph,
map ranges and areas, agreement across noises / inversion / views, class-token CA mass) and the oracle
GT-mode keep/leak. Validates each feature as a predictor of per-pair IoU and of the propagation gain
(V3+R2 - V0+R2): Spearman (with permutation null), AUC for gain > 0, and the grouped-OOF held-out gain of a
one-feature switch between V0+R2 and V3+R2 (cutoff and direction picked on the 4 training folds), against
a permuted-feature null.
Usage: analyze_sg_real_atlas_reliability.py CAPTURE_DIR BASELINE_DIR OUT_MD [--set D] [--workers 16]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/segmentation/sg_real_atlas/manifest.json"
G = 10                                                    # tau 0.55
WIN = slice(10, 21)                                       # SA window calls 10..20
SIDE = {"down_blocks.0": 64, "down_blocks.1": 32, "down_blocks.2": 16, "mid_block": 8,
        "up_blocks.1": 16, "up_blocks.2": 32, "up_blocks.3": 64}
rng = np.random.default_rng(137)


_TOK = []


def tokenizer():
    if not _TOK:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import CLIPTokenizer
        _TOK.append(CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"))
    return _TOK[0]


def side_of(name):
    return next(v for k, v in SIDE.items() if name.startswith(k))


def iou_mask(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def mm(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def rl_summary(arr_by_mod, mask_idx, prefix):
    """arr [31,8,M,2] keep/leak -> window means per SA side and overall."""
    out, per = {}, {}
    for name, a in arr_by_mod.items():
        per.setdefault(side_of(name), []).append(a[WIN, :, mask_idx].astype(np.float32).mean((0, 1)))  # [2]
    allv = []
    for s, v in sorted(per.items()):
        v = np.mean(v, 0); allv.append(v)
        out[f"{prefix}keep{s}"], out[f"{prefix}leak{s}"] = float(v[0]), float(v[1])
        out[f"{prefix}contrast{s}"] = float(v[0] - v[1])
    v = np.mean(allv, 0)
    out[f"{prefix}keep"], out[f"{prefix}leak"], out[f"{prefix}contrast"] = float(v[0]), float(v[1]), float(v[0] - v[1])
    return out


def features(pdir):
    pdir = Path(pdir); meta = json.loads((pdir / "meta.json").read_text()); maps = np.load(pdir / "maps.npz")
    f = {}
    seed = np.load(pdir / "atlas_renoise_n0_seed.npz")
    sd = {k[3:]: seed[k] for k in seed.files if k.startswith("rl/")}
    f.update(rl_summary(sd, 0, "seed_soft_")); f.update(rl_summary(sd, 1, "seed_bin_"))
    full = np.load(pdir / "atlas_renoise_n0.npz")
    gt = {k[3:]: full[k] for k in full.files if k.startswith("rl/")}
    orc = rl_summary(gt, 0, "ORACLE_gt_")
    # map statistics (raw maps keep their dynamic range; A3 is stored min-max normalised)
    g = lambda c, m: maps[f"{c}/{m}"]
    for m in ("C16", "V0", "V3"):
        x = g("renoise/n0", m); f[f"range_{m}"] = float(x.max() - x.min()); f[f"mean_{m}"] = float(mm(x).mean())
    a3 = g("renoise/n0", "A3")
    f["seed_area55"] = float((a3 >= .55).mean()); f["seed_mean"] = float(a3.mean())
    p = a3.ravel() / max(a3.sum(), 1e-12); f["seed_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))
    b = lambda c, m: mm(g(c, m)) >= .55
    f["V3_area55"] = float(b("renoise/n0", "V3").mean()); f["V0_area55"] = float(b("renoise/n0", "V0").mean())
    f["agree_V0_V3"] = iou_mask(b("renoise/n0", "V0"), b("renoise/n0", "V3"))
    f["agree_seed_V3"] = iou_mask(a3 >= .55, b("renoise/n0", "V3"))
    f["agree_noise_V0"] = float(np.mean([iou_mask(b("renoise/n0", "V0"), b(f"renoise/n{j}", "V0")) for j in (1, 2)]))
    f["agree_noise_V3"] = float(np.mean([iou_mask(b("renoise/n0", "V3"), b(f"renoise/n{j}", "V3")) for j in (1, 2)]))
    f["agree_ddim_V0"] = iou_mask(b("renoise/n0", "V0"), b("ddim", "V0"))
    f["agree_prompt_V0"] = iou_mask(b("renoise/n0", "V0"), b("prompt/the", "V0"))
    # class-token CA mass over the window (prompt "a photograph of a <cls>": class tokens 5..n-2)
    toks = [full[k] for k in full.files if k.startswith("tok/")]
    sides = [int(round(t.shape[-1] ** .5)) for t in toks]
    t16 = np.mean([t[WIN].astype(np.float32).mean(0) for t, s in zip(toks, sides) if s == 16], 0)  # [n, 256]
    n_all = len(tokenizer()(meta["prompts"]["ref"], padding=False)["input_ids"])     # SOT a photograph of a <cls> EOT
    cls = t16[5:n_all - 1].sum(0); eot = t16[n_all - 1]
    f["ca_cls_mass16"] = float(cls.mean()); f["ca_cls_peak16"] = float(cls.max())
    f["ca_cls_vs_eot16"] = float(cls.mean() / max(eot.mean(), 1e-8))
    f["ca_cls_top10share16"] = float(np.sort(cls)[-26:].sum() / max(cls.sum(), 1e-12))
    f["recon_ddim_psnr"] = meta["image"]["ddim_recon"]["psnr"]; f["recon_vae_psnr"] = meta["image"]["vae"]["psnr"]
    return f, orc


def miou(C, K):
    s = np.stack([C[K == j].sum(0) for j in np.unique(K)])
    return (s[..., 0] / np.maximum(s[..., :3].sum(-1), 1)).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path); ap.add_argument("baseline", type=Path); ap.add_argument("out_md", type=Path)
    ap.add_argument("--set", default="D"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    V = json.loads((a.baseline / "variants.json").read_text())["variants"]; THR = np.array(
        json.loads((a.baseline / "variants.json").read_text())["thr"])
    fold_of = {e["image"]: e["fold"] for e in json.loads(MANIFEST.read_text())[a.set]}
    files = sorted((a.baseline / a.set).glob("*.npz"))
    keys = [f.stem.split("__") for f in files]
    C = np.stack([np.load(f)["counts"] for f in files]).astype(np.float64)        # [P, V, T, 4]
    K = np.array([int(k) for _, k in keys]); IMG = np.array([i for i, _ in keys])
    FOLD = np.array([fold_of[i] for i in IMG]); AREA = np.array([float(np.load(f)["area"]) for f in files])
    L = []
    w = L.append
    w(f"# SG-real-atlas exploratory results on {a.set} ({len(files)} pairs, {len(set(IMG))} photos, "
      f"{len(set(K))} classes)\n")
    w("Exploratory only: D is incomplete and every number here is chosen on D. Class-macro mIoU, counts "
      "pooled per class. CRF variants are scored at every other τ (0.30, 0.35, …, 0.70).\n")
    # ------------------------------------------------------------------ Part A
    ref = V.index("n0/V0+R2")
    crf_t = np.arange(0, len(THR), 2)

    def oof(vi):
        """grouped 5-fold OOF: pick tau on 4 folds (class-macro), apply to the 5th; pooled OOF counts."""
        ts = crf_t if vi >= 0 and V[vi].endswith("+R2") else np.arange(len(THR))
        pooled = np.zeros((len(C), 4)); taus = []
        for fo in range(5):
            tr, te = FOLD != fo, FOLD == fo
            b = ts[int(np.argmax(miou(C[tr, vi][:, ts], K[tr])))]; taus.append(THR[b])
            pooled[te] = C[te, vi, b]
        return pooled, taus

    def boot(pa, pb, n=1000):
        imgs = np.unique(IMG); idx = {i: np.where(IMG == i)[0] for i in imgs}; d = []
        for _ in range(n):
            s = np.concatenate([idx[i] for i in rng.choice(imgs, len(imgs))])
            d.append(miou(pa[s][:, None], K[s])[0] - miou(pb[s][:, None], K[s])[0])
        return np.percentile(d, [2.5, 97.5])

    def folds_won(pa, pb):
        return sum(miou(pa[FOLD == fo][:, None], K[FOLD == fo])[0] > miou(pb[FOLD == fo][:, None], K[FOLD == fo])[0]
                   for fo in range(5))

    ref_oof, _ = oof(ref)
    m_all = np.stack([miou(C[:, i], K) for i in range(len(V))])
    rows = []
    for i, v in enumerate(V):
        po, taus = oof(i)
        mo = miou(po[:, None], K)[0]
        rows.append((mo, v, m_all[i, G], THR[int(np.argmax(np.where(m_all[i] >= 0, m_all[i], -1)))],
                     np.nanmax(m_all[i]), taus, po))
    rows.sort(key=lambda r: -r[0])
    w("## A. Baseline: every condition and label-free fusion\n")
    w("| variant | mIoU @0.55 | best τ (diag.) | mIoU @best | OOF mIoU (τ per fold) | Δ OOF vs n0/V0+R2 | folds won | 95% photo-bootstrap CI |")
    w("|---|---:|---:|---:|---:|---:|---:|---|")
    ref_m = miou(ref_oof[:, None], K)[0]
    for mo, v, m55, bt, mb, taus, po in rows:
        if v == "n0/V0+R2":
            ci, fw = ("—", "—"), "—"
        else:
            lo, hi = boot(po, ref_oof, 400 if not v.endswith("+R2") else 1000); ci = (f"{lo:+.3f}", f"{hi:+.3f}")
            fw = f"{folds_won(po, ref_oof)}/5"
        w(f"| {v} | {m55:.3f} | {bt:.3f} | {mb:.3f} | {mo:.3f} | {mo - ref_m:+.3f} | {fw} | {ci[0]} … {ci[1]} |")
    # by size third for key variants
    w("\n### By object size (GT share of valid pixels, thirds), mIoU @0.55\n")
    q1, q2 = np.quantile(AREA, [1 / 3, 2 / 3])
    w(f"Thirds cut at {q1:.3f} and {q2:.3f} of the image.\n")
    w("| variant | small | medium | large |\n|---|---:|---:|---:|")
    for v in [r[1] for r in rows[:8]] + ["n0/V0+R2", "n0/V3+R2", "n0/A3"]:
        i = V.index(v)
        w(f"| {v} | " + " | ".join(f"{miou(C[s, i], K[s])[G]:.3f}" for s in
                                   (AREA < q1, (AREA >= q1) & (AREA <= q2), AREA > q2)) + " |")
    # ------------------------------------------------------------------ Part B
    pdirs = [str(a.capture / a.set / i / k) for i, k in keys]
    with Pool(a.workers) as pool:
        res = pool.map(features, pdirs)
    F_ = {k: np.array([r[0][k] for r in res]) for k in res[0][0]}
    O_ = {k: np.array([r[1][k] for r in res]) for k in res[0][1]}
    iou = lambda vi: C[:, vi, G, 0] / np.maximum(C[:, vi, G, :3].sum(-1), 1)
    i0, i3 = V.index("n0/V0+R2"), V.index("n0/V3+R2")
    y0, y3 = iou(i0), iou(i3); gain = y3 - y0
    w("\n## B. Label-free reliability (H*)\n")
    w(f"Per-pair targets at τ 0.55: IoU of n0/V0+R2 (mean {y0.mean():.3f}), IoU of n0/V3+R2 "
      f"(mean {y3.mean():.3f}), propagation gain V3+R2 − V0+R2 (mean {gain.mean():+.3f}; "
      f"V3 better on {(gain > 0).mean():.0%} of pairs, worse on {(gain < 0).mean():.0%}).\n")

    def auc(x, lab):
        from scipy.stats import rankdata
        r = rankdata(x); n1 = lab.sum(); n0 = len(lab) - n1
        return (r[lab].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

    def switch_oof(x):
        """use V3+R2 where sign*x > c, else V0+R2; (sign, c) picked per training folds on class-macro mIoU."""
        pooled = np.zeros((len(C), 4))
        for fo in range(5):
            tr, te = FOLD != fo, FOLD == fo
            best = (-1, None)
            for sgn in (1, -1):
                for c in np.quantile(x[tr], np.linspace(0, 1, 21)):
                    use3 = sgn * x[tr] > sgn * c if sgn == 1 else x[tr] < c
                    pc = np.where(use3[:, None], C[tr, i3, G], C[tr, i0, G])
                    s = miou(pc[:, None], K[tr])[0]
                    if s > best[0]:
                        best = (s, (sgn, c))
            sgn, c = best[1]
            use3 = x[te] > c if sgn == 1 else x[te] < c
            pooled[te] = np.where(use3[:, None], C[te, i3, G], C[te, i0, G])
        return miou(pooled[:, None], K)[0]

    base0 = miou(C[:, i0, G][:, None], K)[0]; base3 = miou(C[:, i3, G][:, None], K)[0]
    orac = miou(np.where((gain > 0)[:, None], C[:, i3, G], C[:, i0, G])[:, None], K)[0]
    w(f"Switch references (mIoU @0.55): always V0+R2 {base0:.3f}; always V3+R2 {base3:.3f}; "
      f"per-pair oracle switch {orac:.3f}.\n")
    w("| feature | ρ with IoU V0+R2 | ρ with IoU V3+R2 | ρ with gain | perm. p (gain) | AUC gain>0 | OOF switch mIoU | Δ vs always V0+R2 | null 95th pct Δ |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    allf = {**F_, **O_}
    out_rows = []
    for name, x in allf.items():
        r0 = spearmanr(x, y0)[0]; r3 = spearmanr(x, y3)[0]; rg = spearmanr(x, gain)[0]
        perm = np.array([spearmanr(rng.permutation(x), gain)[0] for _ in range(200)])
        p = (np.abs(perm) >= abs(rg)).mean()
        sw = switch_oof(x)
        null = np.array([switch_oof(rng.permutation(x)) for _ in range(20)]) - base0
        out_rows.append((sw - base0, name, r0, r3, rg, p, auc(x, gain > 0), sw, np.percentile(null, 95)))
    out_rows.sort(key=lambda r: -r[0])
    for d, name, r0, r3, rg, p, au, sw, n95 in out_rows:
        w(f"| {name} | {r0:+.2f} | {r3:+.2f} | {rg:+.2f} | {p:.3f} | {au:.2f} | {sw:.3f} | {d:+.3f} | {n95:+.3f} |")
    w("\nFeatures prefixed ORACLE_ use the real mask (diagnostic only: they show whether propagation reliability "
      "explains the gain, not a usable rule). ρ = Spearman over pairs.\n")
    # seed-mode vs oracle R/L agreement
    w("### Does the label-free seed R/L track the oracle R/L?\n")
    w("| quantity | ρ(seed_soft, oracle) | ρ(seed_bin, oracle) |\n|---|---:|---:|")
    for q in ("keep", "leak", "contrast", "contrast16", "contrast32", "contrast64"):
        if f"ORACLE_gt_{q}" in O_:
            w(f"| {q} | {spearmanr(F_[f'seed_soft_{q}'], O_[f'ORACLE_gt_{q}'])[0]:+.2f} | "
              f"{spearmanr(F_[f'seed_bin_{q}'], O_[f'ORACLE_gt_{q}'])[0]:+.2f} |")
    np.savez(a.out_md.with_suffix(".npz"), **{f"F/{k}": v for k, v in allf.items()}, y0=y0, y3=y3, K=K, IMG=IMG,
             FOLD=FOLD, AREA=AREA)
    a.out_md.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
