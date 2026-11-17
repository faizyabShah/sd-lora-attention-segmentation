"""Why is the NERVE cell 0.720 in runs/19 but 0.737 in runs/17 (COCO-Cap, VOC-sel. + CRF)? Diagnostic only."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); sys.path.insert(0, str(ROOT))
from scripts import sg_class_comp as cc

C = Path("/home/shafa/sg_comp")
THR = np.round(0.30 + 0.025 * np.arange(17), 3); t50 = int(np.flatnonzero(np.isclose(THR, 0.5))[0])


def mm(x):
    x = x.astype(np.float64); return (x - x.min()) / max(x.max() - x.min(), 1e-12)


for ds in ("coco", "voc"):
    A = cc.load_gen(C / "jc_swap/scores" / ds); B = cc.load_gen(C / f"jc_srcrule/{ds}_a1_scores")
    assert (A["grp"] == B["grp"]).all(), "image order differs"
    Zb = [np.load(C / f"jc_srcrule/{ds}_a1_scores/{g}.npz") for g in B["grp"]]
    exact = np.array([int(z["img_maxdiff"]) == 0 for z in Zb]); one = np.ones(len(exact), bool)
    print(f"\n=== {ds}: {len(exact)} images, {exact.sum()} bit-identical regenerations")
    for post in ("+R2", ""):
        a, b = A[f"clip|walk{post}|raw"], B[f"clip|N|W{post}|raw"]
        for nm, s in (("all", one), ("bit-identical", exact), ("not identical", ~exact)):
            ma, mb = cc.miou(a, A["cls"], s), cc.miou(b, B["cls"], s)
            print(f"{'CRF' if post else 'raw'} {nm:14s} @0.50 runs17 {ma[t50]:.4f} runs19 {mb[t50]:.4f} d {mb[t50]-ma[t50]:+.4f} | "
                  f"best-thr runs17 {ma.max():.4f}@{THR[ma.argmax()]} runs19 {mb.max():.4f}@{THR[mb.argmax()]}")
    # map-level comparison
    d_all, r_all = [], []
    for g in B["grp"]:
        w17 = np.load(C / f"jc_swap/walk/{ds}/{g}.npz")["walk_clip"].squeeze()
        w19 = np.load(C / f"jc_srcrule/{ds}_a1/{g}.npz")["maps"][1, 6]
        x, y = mm(w17), mm(w19)
        d_all.append(np.abs(x - y).mean()); r_all.append(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    d_all, r_all = np.array(d_all), np.array(r_all)
    for nm, s in (("bit-identical", exact), ("not identical", ~exact)):
        print(f"maps {nm:14s} mean|d| median {np.median(d_all[s]):.4f}  p90 {np.quantile(d_all[s], .9):.4f}  "
              f"corr median {np.median(r_all[s]):.4f} p10 {np.quantile(r_all[s], .1):.4f}")
    # per-image IoU at 0.5 + CRF, and per-class
    iou = lambda X: X[:, t50, 0] / np.maximum(X[:, t50, :3].sum(1), 1)
    ia, ib = iou(A["clip|walk+R2|raw"]), iou(B["clip|N|W+R2|raw"])
    dd = ib - ia
    print(f"per-image IoU d (CRF@0.5): mean {dd.mean():+.4f}; |d|>0.1 on {(np.abs(dd) > .1).sum()} imgs "
          f"({(np.abs(dd[exact]) > .1).sum()} bit-identical); worst: " +
          ", ".join(f"{B['grp'][i][:40]} {dd[i]:+.2f}{'*' if exact[i] else ''}" for i in np.argsort(dd)[:6]))
    pc = []
    for c in np.unique(A["cls"]):
        s = A["cls"] == c
        pc.append((cc.miou(B["clip|N|W+R2|raw"], B["cls"], s)[t50] - cc.miou(A["clip|walk+R2|raw"], A["cls"], s)[t50], c))
    print("per-class d (CRF@0.5): " + ", ".join(f"{c} {v:+.3f}" for v, c in sorted(pc)))
