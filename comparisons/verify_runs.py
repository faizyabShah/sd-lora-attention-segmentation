"""Independent check of the N-side comparison runs: recompute sanity IoU from saved maps,
look for broken / leaked / duplicated outputs, and compare against a trivial centre prior."""
import csv, hashlib, json, glob, os, sys
import numpy as np
from PIL import Image

N = "/home/sidna/TRDP/external/n_side"
rows = {d: list(csv.DictReader(open(f"{N}/outputs/manifest_full_{d}.csv"))) for d in ("voc_sim", "coco_cap")}

def path(method, d, r):
    k, c = r["key"], r["classname"]
    if method.startswith("goca"):
        return f"{N}/outputs/{method}_full/maps/{k}__{c}__post.npy"
    if method == "diffseg":
        ku, cu = k.replace(" ", "_"), c.replace(" ", "_")
        return f"{N}/diffseg/outputs/full_sd15/none/{d}/{d}__{ku}__{cu}.npy"
    v = method.split("_")[1]
    return f"{N}/nerve/outputs/full_sd15/variant_{v}/{k}__soft.npy"

yy, xx = np.mgrid[0:512, 0:512]
centre = np.exp(-(((yy - 255.5) / 160) ** 2 + ((xx - 255.5) / 160) ** 2))  # trivial label-free prior

def gt_of(r):
    return np.asarray(Image.open(r["annotation"]).convert("RGB")).max(-1) > 0

def mm(x):
    x = x.astype(np.float32); lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)

def auroc(s, g, rng):
    pos, neg = s[g], s[~g]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    p = rng.choice(pos, min(3000, len(pos))); n = rng.choice(neg, min(3000, len(neg)))
    return (p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()

methods = sys.argv[1:] or ["goca_sd15", "goca_sdxl", "diffseg", "nerve_A", "nerve_V", "nerve_W", "centre"]
out = {}
for m in methods:
    for d, rs in rows.items():
        rng = np.random.default_rng(0); tp = {}; fp = {}; fn = {}; aucs = []; hashes = set(); missing = 0
        perfect = 0; shape_bad = 0; nan = 0
        for r in rs:
            g = gt_of(r); c = r["classname"]
            if m == "centre":
                x = centre
            else:
                p = path(m, d, r)
                if not os.path.exists(p): missing += 1; continue
                x = np.load(p).astype(np.float32)
                if x.shape != g.shape: shape_bad += 1
                if not np.isfinite(x).all(): nan += 1; continue
                hashes.add(hashlib.md5(x.tobytes()).hexdigest())
            s = mm(x); b = s >= 0.5
            if b.shape != g.shape:
                b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize(g.shape[::-1])) > 0
            tp[c] = tp.get(c, 0) + (b & g).sum(); fp[c] = fp.get(c, 0) + (b & ~g).sum(); fn[c] = fn.get(c, 0) + (~b & g).sum()
            iou_img = (b & g).sum() / max((b | g).sum(), 1); perfect += iou_img > 0.995
            aucs.append(auroc(s.ravel(), g.ravel(), rng))
        miou = np.mean([tp[c] / (tp[c] + fp[c] + fn[c]) for c in tp])
        out[(m, d)] = miou
        print(f"{m:10s} {d:8s} n={len(rs)-missing:4d} missing={missing} bad_shape={shape_bad} nan={nan} "
              f"unique_maps={len(hashes) if m!='centre' else '-'} miou@0.5={miou:.3f} "
              f"median_AUROC={np.nanmedian(aucs):.3f} images_IoU>0.995={perfect}", flush=True)
