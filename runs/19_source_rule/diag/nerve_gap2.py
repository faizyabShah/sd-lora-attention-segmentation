"""Inspect the images where the runs/19 NERVE cell loses most against runs/17."""
from pathlib import Path
import numpy as np
C = Path("/home/shafa/sg_comp")
IDS = ["ovam-coco-556255-dog-3429633183", "ovam-coco-608007-person-1101767043", "ovam-coco-224079-chair-3063842736",
       "ovam-coco-73382-dog-3660910269", "ovam-coco-158714-dog-566976589", "ovam-coco-188034-dog-3297553683"]


def mm(x):
    x = x.astype(np.float64); return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def desc(x):
    q = np.quantile(x, [0, .01, .5, .99, 1]); top = np.sort(x.ravel())[-5:]
    return f"min {q[0]:.4g} p1 {q[1]:.4g} med {q[2]:.4g} p99 {q[3]:.4g} max {q[4]:.4g} top5 {np.round(top, 4)}"


for g in IDS:
    w17 = np.load(C / f"jc_swap/walk/coco/{g}.npz")["walk_clip"].squeeze().astype(np.float64)
    z = np.load(C / f"jc_srcrule/coco_a1/{g}.npz"); w19 = z["maps"][1, 6].astype(np.float64)
    sd = np.load(C / f"jc_swap/nerve_seed/coco/{g}.npz")
    print(f"\n{g}  img_maxdiff {int(z['img_maxdiff'])}  walk_check {float(z['walk_check']):.3g}")
    print("  seed native", sd["seg"].shape, desc(sd["seg"]))
    print("  runs17 walk", desc(w17)); print("  runs19 walk", desc(w19))
    x, y = mm(w17), mm(w19)
    print(f"  mm: corr {np.corrcoef(x.ravel(), y.ravel())[0, 1]:.4f}  mean|d| {np.abs(x - y).mean():.4f}  "
          f"frac>0.5 runs17 {(x > .5).mean():.3f} runs19 {(y > .5).mean():.3f}  argmax17 {np.unravel_index(x.argmax(), x.shape)} argmax19 {np.unravel_index(y.argmax(), y.shape)}")
    r = w19 / np.maximum(w17, 1e-12)
    print(f"  raw ratio runs19/runs17: median {np.median(r):.4f}  p1 {np.quantile(r, .01):.4f} p99 {np.quantile(r, .99):.4f}")
