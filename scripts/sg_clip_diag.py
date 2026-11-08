#!/usr/bin/env python3
"""Dense CLIP (SCLIP, ViT-B/16) vs SD attention on real VOC D, and two fixed CLIP+SD pipelines (exploratory, CPU).

Pre-registered before scoring (2026-11-08):
  CLIP map   p_k   = SCLIP VOC21 softmax probability of the target class at 64x64 (its own calibrated output;
                     vocabulary = the 20 VOC names + SCLIP's 26 background names, as in the SCLIP literature)
  A3 seed    c     = min-max A3 (target word); comp = marg.5/mm/gen (generic SD competition, current best)
  Variants (stack [seed, P1, V0, V3, C16], propagated through the target run's stored DDIM SA, frozen code):
    base      c
    comp      minmax(clamp(c - 0.5 b_gen, 0))
    clip      minmax(p_k)                                    CLIP map alone + SD propagation
    clipcomp  minmax(clamp(c - 0.5 minmax(max_{j != k} p_j), 0))   CLIP as competing evidence (incl. background)
    fuse      minmax(0.5 c + 0.5 minmax(p_k))                averaging (no product, no tuned weight)
    sclip     SCLIP standalone: its own VOC21 prediction == k at full resolution (binary, no tau; +R2 shown too)
  Q1 (seed level): AUROC target-vs-other and target-vs-background, AP, of clip vs base seeds; photo-bootstrap CI.
  Q2 (hybrid):  best of {clipcomp, fuse} x {C16, V3+R2} must beat, with photo-bootstrap CI > 0,
                (a) comp C16 and (b) the best CLIP-only pipeline among {sclip, sclip+R2, clip seed, clip C16,
                clip V3+R2}. Otherwise no hybrid is claimed. D is exploration only; any survivor is frozen and
                confirmed on unused photos.
"""
from __future__ import annotations

import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

GENERIC = ["wall", "floor", "ground", "sky", "grass", "tree", "building", "road"]
STORED = ROOT.parent / "sg_real_atlas" / "D"
VARIANTS = ("base", "comp", "clip", "clipcomp", "fuse", "sclip")
CLIP_D = Path("/home/shafa/sg_comp/clip_D")


def mm(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def build_one(job):
    f, out = Path(job[0]), Path(job[1])
    if out.exists():
        return None
    torch.set_num_threads(2)
    from sg_sem_gate import paths
    z = np.load(f); cls = str(z["cls"]); iid, k = f.stem.split("__"); k = int(k)
    q = np.load(STORED / iid / str(k) / "qk_ddim.npz")
    sa = {s: [(torch.from_numpy(q[f"s{s}"][c, 0]), torch.from_numpy(q[f"s{s}"][c, 1])) for c in range(11)]
          for s in (16, 32, 64)}
    cz = np.load(CLIP_D / f"{iid}__B16.npz")
    P = torch.from_numpy(cz["prob21_64"]).float()                      # [21, 64, 64], index = VOC label
    pk = mm(P[k]); others = torch.cat([P[:k], P[k + 1:]]).max(0).values
    c = mm(torch.from_numpy(z[f"word/{cls}"]).float())
    b = torch.stack([mm(torch.from_numpy(z[f"word/{w}"]).float()) for w in GENERIC]).max(0).values
    seeds = {"base": c, "comp": mm((c - 0.5 * b).clamp_min(0)), "clip": pk,
             "clipcomp": mm((c - 0.5 * mm(others)).clamp_min(0)), "fuse": mm(0.5 * c + 0.5 * pk)}
    S = paths(sa, seeds, {n: None for n in seeds})
    chk = max(float((mm(S["base"][i]) - mm(torch.from_numpy(z["map/base"][i]))).abs().max()) for i in range(5))
    sc = (cz["pred21"] == k).astype(np.float32)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **{f"map/{n}": S[n].numpy().astype(np.float32) for n in seeds},
             **{"map/sclip": np.repeat(sc[None], 5, 0)}, cls=cls, k=k, check=chk, set="D")
    os.replace(tmp, out)
    return f.stem, chk


def build(a):
    fs = sorted(f for f in a.src.glob("*.npz") if not f.stem.endswith(".tmp"))
    if a.max_pairs:
        fs = fs[: a.max_pairs]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(a.out / f.name)) for f in fs]
    print(len(jobs), "pairs", flush=True); worst = 0.0
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(build_one, jobs), 1):
            if r:
                worst = max(worst, r[1])
            if n <= 3 or n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {r}", flush=True)
    print(f"worst replay |diff| {worst:.2e}", flush=True)
    if worst > 2e-3:                                   # known fresh-GPU vs stored-SA gap is 1.3e-3
        sys.exit("REPLAY CHECK FAILED")


def report(a):
    import sg_class_comp as cc
    S = cc.load(a.scores, "D"); L = []; w = L.append
    n = len(S["cls"]); ones = np.ones(n, bool)
    w("# Dense CLIP (SCLIP ViT-B/16) vs SD attention on real VOC D (exploratory)\n")
    w(f"{n} pairs, {len(np.unique(S['grp']))} photos. Pre-registered definitions and rules in "
      "`scripts/sg_clip_diag.py`. The SCLIP port reproduces the repo on VOC val (VOC20 81.49 vs 81.54, VOC21 59.64 "
      "vs 59.63). CLIP map = SCLIP's VOC21 class probability at 64×64. Class-given per-pair binary masks, "
      "class-macro IoU, OOF τ over 5 photo-grouped folds, photo-bootstrap 95% CIs. Max replay |Δ| "
      f"{S['check'].max():.1e}.\n")

    w("## Q1 — seed-level separation (before any SD propagation)\n")
    w("| seed | AUROC tgt vs other object | AUROC tgt vs background | AP |")
    w("|---|---:|---:|---:|")
    ho = S["has_other"]
    for v in ("base", "comp", "clip", "clipcomp", "fuse"):
        x = S[f"{v}|seed|sep"]
        w(f"| {v} | {np.nanmean(x[ho, 0]):.3f} | {np.nanmean(x[:, 1]):.3f} | {np.nanmean(x[:, 2]):.3f} |")

    def pboot(da, sel, n_=1000):
        rng = np.random.default_rng(0); g = np.unique(S["grp"][sel])
        idx = {x: np.where(sel & (S["grp"] == x))[0] for x in g}; d = []
        for _ in range(n_):
            s = np.concatenate([idx[x] for x in rng.choice(g, len(g))]); d.append(np.nanmean(da[s]))
        return np.percentile(d, [2.5, 97.5])

    for j, nm, sel in ((0, "tgt vs other", ho), (1, "tgt vs background", ones), (2, "AP", ones)):
        d = S["clip|seed|sep"][:, j] - S["base|seed|sep"][:, j]; lo, hi = pboot(d, sel)
        w(f"\nclip − A3 seed, {nm}: {np.nanmean(d[sel]):+.3f} (CI {lo:+.3f}…{hi:+.3f})")
    w("\nBy object size (seed AUROC tgt vs other / tgt vs bg):\n")
    w("| bin | n | A3 | CLIP |"); w("|---|---:|---|---|")
    for lo, hi, nm in ((0, .01, "<1%"), (.01, .05, "1–5%"), (.05, .2, "5–20%"), (.2, 1.01, ">20%")):
        sel = (S["area"] >= lo) & (S["area"] < hi)
        f_ = lambda v: f"{np.nanmean(S[f'{v}|seed|sep'][sel & ho, 0]):.3f} / {np.nanmean(S[f'{v}|seed|sep'][sel, 1]):.3f}"
        w(f"| {nm} | {sel.sum()} | {f_('base')} | {f_('clip')} |")

    w("\n## Segmentation (OOF mIoU)\n")
    maps = ("seed", "C16", "V3+R2", "P1+R2")
    P = {}
    w("| variant | " + " | ".join(maps) + " | best-map prec / rec / FP-on-other |")
    w("|---|" + "---:|" * (len(maps) + 1))
    for v in VARIANTS:
        cells, best = [], None
        for m in maps:
            mi_, Pm, taus = cc.oof(S, S[f"{v}|{m}|raw"]); P[(v, m)] = Pm
            cells.append(f"{mi_:.3f}")
            if best is None or mi_ > best[0]:
                best = (mi_, m)
        pr, rc, fo = cc.prf(P[(v, best[1])])
        w(f"| {v} | " + " | ".join(cells) + f" | {pr:.3f} / {rc:.3f} / {fo:.0%} ({best[1]}) |")
    mi = lambda v, m: cc.miou(P[(v, m)][:, None], S["cls"], ones)[0]

    w("\n## Q2 — registered hybrid comparison\n")
    hyb = max(((v, m) for v in ("clipcomp", "fuse") for m in ("C16", "V3+R2")), key=lambda x: mi(*x))
    clip_only = max([("sclip", "seed"), ("sclip", "V3+R2"), ("clip", "seed"), ("clip", "C16"), ("clip", "V3+R2")],
                    key=lambda x: mi(*x))
    w("| comparison | Δ | 95% CI | pass |"); w("|---|---:|---|---|")
    ok = []
    for vb, mb, tag in (("comp", "C16", "comp C16 (registered)"), (*clip_only, f"best CLIP-only ({clip_only[0]} "
                                                                               f"{'R2' if clip_only[1] == 'V3+R2' and clip_only[0] == 'sclip' else clip_only[1]})")):
        d = mi(*hyb) - mi(vb, mb); lo, hi = cc.boot(S, P[hyb], P[(vb, mb)]); ok.append(lo > 0)
        w(f"| best hybrid ({hyb[0]} {hyb[1]}) − {tag} | {d:+.3f} | {lo:+.3f}…{hi:+.3f} | {'yes' if lo > 0 else 'no'} |")
    for (va, ma), (vb, mb), tag in ((clip_only, ("comp", "C16"), "best CLIP-only − comp C16"),
                                    (("clip", "C16"), ("clip", "seed"), "clip C16 − clip seed (does SD SA help CLIP?)"),
                                    (("sclip", "seed"), ("comp", "C16"), "SCLIP standalone − comp C16")):
        d = mi(va, ma) - mi(vb, mb); lo, hi = cc.boot(S, P[(va, ma)], P[(vb, mb)])
        w(f"| {tag} | {d:+.3f} | {lo:+.3f}…{hi:+.3f} | — |")
    w(f"\n**Hybrid claim:** {'PASS — freeze and confirm on unused photos' if all(ok) else 'FAIL — no hybrid is claimed'}.\n")

    w("## Per-class OOF IoU (comp C16 / SCLIP standalone / best hybrid)\n")
    rows = []
    for c in np.unique(S["cls"]):
        sel = S["cls"] == c
        v = [cc.miou(P[x][:, None], S["cls"], sel)[0] for x in (("comp", "C16"), ("sclip", "seed"), hyb)]
        rows.append(f"{c} {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}")
    w(", ".join(rows) + "\n")
    w("## By object size (OOF IoU, class-macro within bin)\n")
    keys = [("base", "V3+R2"), ("comp", "C16"), ("sclip", "seed"), ("clip", "C16"), hyb]
    w("| bin | n | " + " | ".join(f"{v} {m}" for v, m in keys) + " |"); w("|---|---:|" + "---:|" * len(keys))
    for lo, hi, nm in ((0, .01, "<1%"), (.01, .05, "1–5%"), (.05, .2, "5–20%"), (.2, 1.01, ">20%")):
        sel = (S["area"] >= lo) & (S["area"] < hi)
        w(f"| {nm} | {sel.sum()} | " + " | ".join(f"{cc.miou(P[x][:, None], S['cls'], sel)[0]:.3f}" for x in keys) + " |")
    a.report.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--src", type=Path, default=Path("/home/shafa/sg_comp/D_maps"))
    p.add_argument("--out", type=Path, default=Path("/home/shafa/sg_comp/clipdiag_D_maps"))
    p.add_argument("--workers", type=int, default=8); p.add_argument("--max-pairs", type=int)
    p = sub.add_parser("report"); p.add_argument("--scores", type=Path, default=Path("/home/shafa/sg_comp/clipdiag_D_scores"))
    p.add_argument("--report", type=Path, default=ROOT / "outputs/segmentation/sg_real_atlas/analysis/clip_diagnostic.md")
    a = ap.parse_args()
    {"build": build, "report": report}[a.cmd](a)
