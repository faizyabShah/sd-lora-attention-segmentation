#!/usr/bin/env python
"""Setup check for a new machine (runs/48_vpu_setup). Mechanics only; no score is computed.

1. Files: shipped item lists and profiles are in $BRACE_CACHE_DIR; COCO annotations and photos are present.
2. GPU: SD 1.5 and SD 2.1-base load; one recorded pass of a COCO val photo per backbone; finite maps;
   the MK4-LR gate and the fine-only V0 refinement run.

    source scripts/brace_env.sh && python scripts/vpu_check_setup.py [--no-gpu]
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

COCO = ROOT / "datasets/coco2017"
NEED = ["jc_ov42/items_prof.json", "jc_ov42/items_val.json", "jc_ov42/fit.pt", "jc_ov42/thr.json",
        "jc_real44/items.json", "jc_real44/ours/items_prof44.json", "jc_real44/ours/fit44.pt", "jc_realvoc37b/thr.json"]


def files():
    ok = True
    for f in NEED:
        p = CACHE / f; print(f"{'ok ' if p.exists() else 'MISSING'} {p}"); ok &= p.exists()
    for f in ("annotations/instances_val2017.json", "annotations/instances_train2017.json",
              "annotations/captions_train2017.json"):
        p = COCO / f; print(f"{'ok ' if p.exists() else 'MISSING'} {p}"); ok &= p.exists()
    prof = {x["path"] for x in json.loads((CACHE / NEED[0]).read_text())["items"]}
    prof |= {x["path"] for x in json.loads((CACHE / NEED[5]).read_text())["jobs"]}
    val = {x["path"] for x in json.loads((CACHE / NEED[1]).read_text())}
    for name, S in (("profile photos (train2017)", prof), ("COCO-Object val photos", val)):
        have = sum((COCO / p).exists() for p in S); print(f"{name}: {have} / {len(S)} present"); ok &= have == len(S)
    return ok


def gpu():
    import numpy as np, torch
    from PIL import Image
    import jc_sd21_47 as P47
    R, O, U = P47.setup()                       # SD 2.1 path + head-generic SA check (identical on SD 1.5)
    from src.segmentation.sg25_sa import _attention, _apply
    x = next(v for v in json.loads((CACHE / "jc_ov42/items_val.json").read_text()) if v["cls"] == "dog")
    img = np.asarray(Image.open(COCO / x["path"]).convert("RGB"))
    print(f"GPU: {torch.cuda.get_device_name(0)}; photo {x['path']} word 'dog'")
    for name, model in (("SD 1.5", "stable-diffusion-v1-5/stable-diffusion-v1-5"), ("SD 2.1-base", P47.SD21)):
        R.MODEL = model; pipe, rec = R.load()
        R.one_pass(pipe, rec, img, "A photograph of a dog", "dog")
        Q = R.qbar(rec); H64 = _attention(rec.grid_qk[64], 64)
        w = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=R.DEV)
        s = R.make_seed(rec, w, None); v0 = _apply(H64, _apply(H64, s.float().reshape(-1))).mean(0)
        ok = bool(torch.isfinite(s).all() and torch.isfinite(v0).all())
        print(f"{name}: {len(Q)} gate modules, query sizes {sorted({math.isqrt(q.shape[0]) for q in Q.values()})}, "
              f"64x64 SA heads {H64.shape[0]}, seed {tuple(s.shape)}, V0 finite {ok}")
        assert ok
        del pipe, rec; torch.cuda.empty_cache()
    print("GPU check passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--no-gpu", action="store_true"); a = p.parse_args()
    good = files()
    if not a.no_gpu:
        gpu()
    print("files complete" if good else "files INCOMPLETE (see MISSING / counts above)")
    sys.exit(0 if good else 1)
