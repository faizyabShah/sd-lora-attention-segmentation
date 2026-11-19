#!/usr/bin/env python3
"""Fidelity of side N's runs/25 capture against side N's runs/20 reproduction (comparisons/official/brace_repro.py).

F's runs/25 compares with F's runs/20 cache, which side N cannot read. Here the class-word BRACE map (wA3, V3) and
the generated image of each runs/25 image are compared with the same image in ~/TRDP/external/n_side/brace_repro/.
Both captures ran in the same `gen` environment on V100s (runs/25 on erdos, brace_repro on wald).
"""
import json
import sys
from pathlib import Path

import numpy as np

EXT = Path.home() / "TRDP/external/n_side"
LNR = EXT / "token_ladder_nr/cache/jc_ladder_nr"
REP = EXT / "brace_repro"


def mm(x):
    lo, hi = x.min(), x.max(); return (x - lo) / max(hi - lo, 1e-20)


out = {}
for ds in ("voc", "coco"):
    img_eq, img_max, v3 = [], [], []
    for f in sorted((LNR / ds).glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        r = REP / ds / f.name
        if not r.exists():
            continue
        a, b = np.load(f), np.load(r)
        d = np.abs(a["image"].astype(int) - b["image"].astype(int))
        img_eq.append(bool(d.max() == 0)); img_max.append(float(d.mean()))
        v3.append(float(np.abs(mm(a["maps"][1, 2].astype(np.float32)) - mm(b["maps"][2].astype(np.float32))).max()))
    v3 = np.array(v3); im = np.array(img_max)
    out[ds] = {"n": len(v3), "image_identical": int(np.sum(img_eq)), "image_mean_absdiff_median": float(np.median(im)),
               "image_mean_absdiff_ge2": int((im >= 2).sum()), "v3_max_absdiff_median": float(np.median(v3)),
               "v3_max_absdiff_p95": float(np.percentile(v3, 95))}
print(json.dumps(out, indent=1))
(EXT / "token_ladder_nr/fidelity.json").write_text(json.dumps(out, indent=1))
