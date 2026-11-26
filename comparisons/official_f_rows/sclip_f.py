#!/usr/bin/env python3
"""SCLIP ViT-B/16 (side F's committed port `scripts/sg_clip_dense.py`, unchanged; settings of the runs/20 SCLIP seed:
VOC21 vocabulary, postprocess(65, 0.1, 0.1)) on the BRACE reproduction's stored images, so SCLIP and BRACE see
the same pixels. Runs on CPU in the `goca` env (transformers 4.49): the port does not run under transformers 5.

-> ~/TRDP/external/n_side/official_f_rows/sclip/<ds>/<image_id>.npz  (sclip_prob [512,512] f16, sclip_bin, sec)
"""
import os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_clip_dense as cd  # noqa: E402
import sg_gen_clip as gc  # noqa: E402

BR = Path.home() / "TRDP/external/n_side/brace_repro"
OUT = Path.home() / "TRDP/external/n_side/official_f_rows/sclip"

if __name__ == "__main__":
    shard, ns = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    M = cd.DenseCLIP("B16", torch.device("cpu")); n21, i21 = cd.split([cd.VOC21_BG] + cd.VOC20_NAMES); T21 = M.text(n21)
    for ds in ("voc", "coco"):
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        for f in sorted((BR / ds).glob("*.npz"))[shard::ns]:
            o = OUT / ds / f.name
            if o.exists():
                continue
            z = np.load(f); t0 = time.perf_counter()
            p21, P21 = cd.postprocess(M.logits(z["image"], T21), i21, 65, 0.1, 0.1); k = gc.voc_k(str(z["cls"]))
            tmp = o.with_name(o.stem + ".tmp.npz")
            np.savez(tmp, sclip_prob=P21[k].float().numpy().astype(np.float16), sclip_bin=(p21 == k).numpy(),
                     cls=str(z["cls"]), sec=time.perf_counter() - t0)
            os.replace(tmp, o)
        print(ds, "done", flush=True)
