"""How different are side N's PNGs (what NERVE reads, runs/17) from our stored generated images (runs/19, BRACE)?"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); C = Path("/home/shafa/sg_comp")
sys.path.insert(0, str(ROOT / "scripts"))
import jc_swap_nerve as jn

for ds, sub in (("coco", "cococap_maps"), ("voc", "vocsim_maps")):
    rows = jn.rows_for(ds, 0, 1, 0); out = []
    sc = {f.stem: f for f in (C / f"jc_srcrule/{ds}_a1_scores").glob("*.npz")}
    for r in rows:
        a = np.asarray(Image.open(r["image_local"]).convert("RGB")).astype(int)
        z = np.load(C / f"{sub}/{r['key']}.npz"); b = z["image"].astype(int)
        if a.shape != b.shape:
            out.append((r["key"], -1, -1, -1, a.shape)); continue
        d = np.abs(a - b)
        flip = np.abs(a[:, ::-1] - b).mean()
        out.append((r["key"], d.max(), d.mean(), (d.max(-1) > 32).mean(), flip))
    arr = np.array([o[2] for o in out if o[1] >= 0])
    print(f"\n== {ds}: {len(out)} images; shape mismatch {sum(o[1] < 0 for o in out)}; "
          f"identical {sum(o[1] == 0 for o in out)}; mean|d| median {np.median(arr):.2f}, p90 {np.quantile(arr, .9):.2f}, "
          f"max {arr.max():.2f}; >5 on {(arr > 5).sum()} images, >20 on {(arr > 20).sum()}")
    for o in sorted(out, key=lambda o: -o[2])[:12]:
        print(f"  {o[0][:45]:45s} max {o[1]} mean {o[2]:.1f} frac>32 {o[3]:.3f} mean|d| if h-flipped {o[4] if isinstance(o[4], tuple) else round(float(o[4]), 1)}")
    np.save(ROOT / f"outputs/claude_scratch/nerve_diag/imgdiff_{ds}.npy", np.array([(o[0], o[2]) for o in out], dtype=object))
