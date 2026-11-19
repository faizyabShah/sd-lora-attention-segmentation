#!/usr/bin/env python3
"""Per-item counts for the JC #3 remaining rows (capture_f.py outputs), same metric code as comparisons/official/.

  raw     every item: soft map (min-max, sg_steps_test.mm) on the 27-value grid THR_EXT, raw; own rule raw and
          + DenseCRF R2
  crf     DenseCRF R2 on the soft map at the threshold indices of --taus (JSON {method: [indices]}) for one set

DenseCRF R2 costs ~4.7 s per call on one core here, so it is not run on the full grid (deviation, PREREG addendum):
VOC-Sim gets a 7-index window around each method's raw full-VOC-Sim optimum plus the 0.5 index; COCO-Cap gets
the DenseCRF full-VOC-Sim-best index of that window plus 0.5.

-> ~/TRDP/external/n_side/official_f_rows/scores/<method>/<ds>/<image_id>.npz
"""
from __future__ import annotations

import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "comparisons/official"))
from common import THR_EXT, crf_site  # noqa: E402

CAP = Path.home() / "TRDP/external/n_side/official_f_rows"
BR = Path.home() / "TRDP/external/n_side/brace_repro"
METHODS = ("ovam_w", "ovam_tf", "daam", "seediff", "sclip")
SOFT = {"ovam_w": "ovam_w", "ovam_t": "ovam_t", "ovam_tf": "ovam_tf", "daam": "daam", "seediff": "seediff", "sclip": "sclip_prob"}
OK = {"daam": "daam_ok", "seediff": "seediff_ok"}


def own_mask(m, z):
    if m in ("ovam_w", "ovam_t", "ovam_tf"):
        return z[m].astype(np.float32) > 0.40          # OVAM: fused > 0.40 (ovam_full_prediction), then DenseCRF
    if m == "daam":
        return z["daam"].astype(np.float32) > 0.4       # DAAM expand_as (min-max) > tau 0.4
    if m == "seediff":
        return z["seediff_mask"].astype(bool)           # SeeDiff: PAMR(map64 > 0.3) > 0.8
    return z["sclip_bin"].astype(bool)                  # SCLIP: argmax == target class


def one(job):
    mode, m, ds, iid, taus, site = job
    o = CAP / "scores" / m / ds / f"{iid}.npz"
    import torch
    torch.set_num_threads(1)
    import sg_steps_test as st
    import sg_class_comp as cc
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(site)
    zb = np.load(BR / ds / f"{iid}.npz")
    if m == "sclip":                                    # sclip_f.py ran on the BRACE reproduction's images
        z = dict(np.load(CAP / "sclip" / ds / f"{iid}.npz")); z["image"] = zb["image"]; z["same_as_brace"] = True
    else:
        z = np.load(CAP / ds / f"{iid}.npz")
    if m in OK and not bool(z[OK[m]]):
        return
    truth = zb["truth"].astype(bool); img = z["image"]
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    x = st.mm(z[SOFT[m]].astype(np.float32))
    res = dict(np.load(o)) if o.exists() else {}
    if mode == "raw":
        if "soft|raw" in res:
            return
        res["soft|raw"] = cc.counts(x[None] >= THR_EXT[:, None, None], truth, valid, other)
        om = own_mask(m, z)
        res["own|raw"] = cc.counts(om[None], truth, valid, other)
        res["own|crf"] = cc.counts(apply_ovam_densecrf(img, om, site)[None].astype(bool), truth, valid, other)
        res["same_as_brace"] = bool(z["same_as_brace"]); res["cls"] = str(z["cls"])
    else:
        todo = [t for t in taus if f"crf|{t}" not in res]
        if not todo:
            return
        for t in todo:
            res[f"crf|{t}"] = cc.counts(apply_ovam_densecrf(img, x >= THR_EXT[t], site)[None].astype(bool),
                                        truth, valid, other)
    o.parent.mkdir(parents=True, exist_ok=True)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def main(a):
    site = str(crf_site()); jobs = []
    taus = json.loads(a.taus.read_text()) if a.taus else {}
    for ds in a.sets:
        for m in a.methods:
            src = CAP / "sclip" / ds if m == "sclip" else CAP / ds
            ids = sorted(p.stem for p in src.glob("*.npz") if not p.stem.endswith(".tmp"))
            if a.mode == "crf" and m not in taus.get(ds, {}):
                continue
            for i in ids:
                jobs.append((a.mode, m, ds, i, taus.get(ds, {}).get(m, []), site))
    jobs = jobs[a.shard::a.num_shards]
    print(len(jobs), "jobs", flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(one, jobs, chunksize=1), 1):
            if n % 200 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("raw", "crf"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--sets", nargs="+", default=["voc", "coco"])
    ap.add_argument("--taus", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    main(ap.parse_args())
