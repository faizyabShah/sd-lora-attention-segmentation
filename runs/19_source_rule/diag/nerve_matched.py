"""Post hoc: runs/19 key cells on COCO-Cap images whose regeneration matches the official benchmark image."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/shafa/sd-lora-attention-segmentation")
from scripts import sg_class_comp as cc

C = Path("/home/shafa/sg_comp"); THR = np.round(0.30 + 0.025 * np.arange(17), 3)
ti = lambda t: int(np.flatnonzero(np.isclose(THR, t))[0])
TH = {"G|V3": {"ca": .55, "clip": .60, "hA": .60, "hB": .575}, "N|W": {"ca": .40, "clip": .50, "hA": .50, "hB": .40},
      "seed": {"ca": .40, "clip": .625, "hA": .60, "hB": .40}}   # full-VOC-Sim-best thresholds from result_a1.md
ds = sys.argv[1] if len(sys.argv) > 1 else "coco"
B = cc.load_gen(C / f"jc_srcrule/{ds}_a1_scores"); A17 = cc.load_gen(C / f"jc_swap/scores/{ds}")
dd = dict(np.load(f"/home/shafa/sd-lora-attention-segmentation/outputs/claude_scratch/nerve_diag/imgdiff_{ds}.npy", allow_pickle=True).tolist())
md = np.array([float(dd[g]) for g in B["grp"]]); n = len(md); cls = B["cls"]; rng = np.random.default_rng(0)


def P(cell, s):
    return B[f"{s}|{cell}+R2|raw" if cell != "seed" else f"{s}|seed|raw"][:, ti(TH[cell][s])]


def mi(X, sel):
    return cc.miou(X[:, None], cls, sel)[0]


def boot(Pa, Pb, sel):
    idx = np.flatnonzero(sel); d0 = mi(Pa, sel) - mi(Pb, sel); bs = []
    for _ in range(1000):
        m = np.bincount(rng.choice(idx, len(idx)), minlength=n)
        bs.append(mi(Pa * m[:, None], m > 0) - mi(Pb * m[:, None], m > 0))
    lo, hi = np.quantile(bs, [.025, .975]); return f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"


for nm, sel in ((f"all {n}", md >= 0), (f"matched, mean|d|<1 ({(md < 1).sum()})", md < 1), (f"differs, >=5 ({(md >= 5).sum()})", md >= 5)):
    print("==", ds, nm)
    for cell in ("seed", "G|V3", "N|W"):
        print(f"  {cell:5s} " + "  ".join(f"{s} {mi(P(cell, s), sel):.3f}" for s in ("ca", "clip", "hA", "hB")))
    for cell in ("G|V3", "N|W"):
        print(f"  {cell}: V hA-CA {boot(P(cell, 'hA'), P(cell, 'ca'), sel)}  K CLIP-hA {boot(P(cell, 'clip'), P(cell, 'hA'), sel)}  "
              f"CLIP-CA {boot(P(cell, 'clip'), P(cell, 'ca'), sel)}")
    print(f"  NERVE (runs/17, official image) - BRACE: {boot(A17['clip|walk+R2|raw'][:, ti(.5)], B['ca|G|V3+R2|raw'][:, ti(.55)], sel)}")
