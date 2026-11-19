#!/usr/bin/env python3
"""Per-item counts for side N's comparison runs under the workplan Section 3 contract.

For every (method, item) of `comparisons/manifests/` (stored OVAM benchmark images):
  soft      the method's saved soft map, min-max scaled (`sg_steps_test.mm`), thresholded on THR_EXT
  own       the method's own binarisation, where it has one:
            GoCA   argmax(class, background + offset) on its post-affinity maps, offsets = GoCA's VOC config grid
                   (`src-main/configs/config-dataset/config_dataset_voc.py`, method "offset"); ties go to background
            NERVE  the saved `__mask_nerve_own.png` (cam_to_label, bkg_thre 0, cls_thre 0.5, with its score gate)
            DiffSegmenter has no fixed rule (its evaluation script sweeps the background score on the evaluated
                   set); its rule is the threshold sweep on the soft map, handled in the report.
Each mask is counted raw and after DenseCRF R2 (`src/segmentation/ovam_full.apply_ovam_densecrf`, frozen OVAM
parameters (10, 80, 13, 3, 3, 5)), with `scripts/sg_class_comp.counts` (tp, fp, fn, tn, fp_other).
Ground truth: the OVAM annotation PNG, any non-zero channel. Inference never reads it.

-> ~/TRDP/external/n_side/official/<method>/<dataset>/<key>.npz
"""
from __future__ import annotations

import argparse, csv, os, sys, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EXT, OFFICIAL_OUT, THR_EXT, crf_site  # noqa: E402

N = EXT
MAN = ROOT / "comparisons/manifests"
METHODS = ("goca_sd15", "goca_sdxl", "goca_sd21", "diffseg", "nerve_A", "nerve_V", "nerve_W")
GOCA_OFFSETS = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
DS = {"voc": "voc_sim", "coco": "coco_cap"}


def items(ds):
    return list(csv.DictReader(open(MAN / f"manifest_full_{DS[ds]}.csv")))


def soft_path(method, d, r):                      # as comparisons/verify_runs.py
    k, c = r["key"], r["classname"]
    if method.startswith("goca"):
        return Path(f"{N}/outputs/{method}_full/maps/{k}__{c}__post.npy")
    if method == "diffseg":
        ku, cu = k.replace(" ", "_"), c.replace(" ", "_")
        return Path(f"{N}/diffseg/outputs/full_sd15/none/{d}/{d}__{ku}__{cu}.npy")
    return Path(f"{N}/nerve/outputs/full_sd15/variant_{method.split('_')[1]}/{k}__soft.npy")


def score_one(job):
    method, ds, r, o, site = job
    o = Path(o)
    if o.exists():
        return
    import torch
    torch.set_num_threads(1)
    from PIL import Image
    import sg_steps_test as st
    import sg_class_comp as cc
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(site)
    img = np.asarray(Image.open(r["image"]).convert("RGB"))
    truth = np.asarray(Image.open(r["annotation"]).convert("RGB")).max(-1) > 0
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    x = np.load(soft_path(method, DS[ds], r)).astype(np.float32)
    if x.shape != truth.shape or not np.isfinite(x).all():
        raise ValueError(f"bad map {method} {r['key']}")
    x = st.mm(x)
    res = {"cls": r["classname"], "area": float(truth.mean()), "truth_crc": zlib.crc32(np.packbits(truth).tobytes())}
    res["soft|raw"] = cc.counts(x[None] >= THR_EXT[:, None, None], truth, valid, other)
    res["soft+R2|raw"] = cc.counts(np.stack([apply_ovam_densecrf(img, x >= t, site) for t in THR_EXT]).astype(bool),
                                   truth, valid, other)
    own = None
    if method.startswith("goca"):
        k, c = r["key"], r["classname"]
        cls_map = np.load(f"{N}/outputs/{method}_full/maps/{k}__{c}__post.npy").astype(np.float32)
        bg_map = np.load(f"{N}/outputs/{method}_full/maps/{k}__{c}__post_bg.npy").astype(np.float32)
        own = cls_map[None] > bg_map[None] + GOCA_OFFSETS[:, None, None]
    elif method.startswith("nerve"):
        p = Path(str(soft_path(method, DS[ds], r)).replace("__soft.npy", "__mask_nerve_own.png"))
        own = (np.asarray(Image.open(p)) > 127)[None]
    if own is not None:
        res["own|raw"] = cc.counts(own, truth, valid, other)
        res["own+R2|raw"] = cc.counts(np.stack([apply_ovam_densecrf(img, m, site) for m in own]).astype(bool),
                                      truth, valid, other)
    o.parent.mkdir(parents=True, exist_ok=True)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def main(a):
    site = str(crf_site()); jobs = []
    for m in a.methods:
        for ds in a.sets:
            for r in items(ds):
                o = a.out / m / ds / f"{r['key']}.npz"
                if not o.exists():
                    jobs.append((m, ds, r, str(o), site))
    jobs = jobs[a.shard::a.num_shards]
    print(len(jobs), "items to score", flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
            if n % 100 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--sets", nargs="+", default=["voc", "coco"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OFFICIAL_OUT)
    main(ap.parse_args())
