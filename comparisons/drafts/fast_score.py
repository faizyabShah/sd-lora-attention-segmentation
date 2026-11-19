#!/usr/bin/env python3
"""CPU-saving scoring for runs/25 (side N) and the Table 1 ladder, giving the same reported numbers.

The DenseCRF masks cost about 4.3 s per call on these nodes. Both reports read COCO-Cap only at fixed thresholds
(0.45 and/or 0.55) and at the threshold that is best on all of VOC-Sim, so COCO-Cap DenseCRF counts are computed at
exactly those thresholds; DenseCRF counts at every other COCO-Cap threshold are stored as -1 (never read by either
report; `crf_thresholds` lists the scored ones). VOC-Sim is scored on the full grid with the committed scorers
(needed for out-of-fold selection), and raw counts are always on the full grid.

  python fast_score.py voc   <ladder|t1> --shard i --num-shards n --workers w   # committed scorer, VOC only
  python fast_score.py coco  <ladder|t1> --shard i --num-shards n --workers w   # needs all VOC scores first
"""
import argparse, os, runpy, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


_CACHE = {}


def load(which):
    if which not in _CACHE:
        _CACHE[which] = _load(which)
    return _CACHE[which]


def _load(which):
    if which == "ladder":
        g = runpy.run_path(str(REPO / "comparisons/token_ladder_nr/run_nr.py"), run_name="n_side_import")
        L = g["L"]
        crf_keys = [f"{s}|{m}+R2|raw" for s in L.SEEDS for m in L.MAPS]
        fixed = [0.55]
        return L, L.WORK, crf_keys, fixed
    sys.path.insert(0, str(REPO / "comparisons/drafts/table1_ladder"))
    import t1_ladder as T
    T._official_patch_rows()
    return T, T.WORK, [f"{m}|crf" for m in T.CRF_MAPS], [0.45, 0.55]


def needed(M, work, crf_keys, fixed):
    import sg_class_comp as cc
    S = cc.load_gen(work / "voc_scores"); one = np.ones(len(S["grp"]), bool)
    idx = {k: int(np.argmax(cc.miou(S[k], S["cls"], one))) for k in crf_keys}
    fx = [int(np.flatnonzero(np.isclose(M.THR, t))[0]) for t in fixed]
    return {k: sorted(set([idx[k]] + fx)) for k in crf_keys}


def coco_one(job):
    which, f, o, need = job
    f, o = Path(f), Path(o)
    if o.exists():
        return
    import torch
    torch.set_num_threads(1)
    M, work, crf_keys, fixed = load(which)
    import sg_class_comp as cc
    import sg_steps_test as st
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    site = Path.home() / "TRDP/external/n_side/deps/simplecrf_py311"
    _import_densecrf(site)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth); THR = M.THR
    blank = lambda: -np.ones((len(THR), 5), np.int64)
    if which == "ladder":
        res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"]),
               "img_vs_run20": float(z["img_vs_run20"]), "v3_vs_run20": float(z["v3_vs_run20"])}
        items = [((si, mi), f"{s}|{m}|raw", f"{s}|{m}+R2|raw") for si, s in enumerate(M.SEEDS) for mi, m in enumerate(M.MAPS)]
        get = lambda ij: z["maps"][ij[0], ij[1]]
    else:
        res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"])}
        items = [(k, f"{m}|raw", f"{m}|crf" if m in M.CRF_MAPS else None) for k, m in enumerate(M.MAPS)]
        get = lambda k: z["maps"][k]
    for ij, kraw, kcrf in items:
        x = st.up(st.mm(get(ij).astype(np.float32)), truth.shape)
        res[kraw] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if kcrf:
            c = blank(); ii = need[kcrf]
            ms = np.stack([apply_ovam_densecrf(img, x >= THR[i], site) for i in ii]).astype(bool)
            c[ii] = cc.counts(ms, truth, valid, other); res[kcrf] = c
            res["crfthr__" + kcrf.replace("|", "~")] = np.asarray(ii)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=("voc", "coco")); ap.add_argument("which", choices=("ladder", "t1"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    M, work, crf_keys, fixed = load(a.which)
    if a.ds == "voc":
        M.score(argparse.Namespace(sets=["voc"], tag="", workers=a.workers, shard=a.shard, num_shards=a.num_shards))
        return
    need = needed(M, work, crf_keys, fixed)
    print("COCO DenseCRF thresholds:", {k: [float(M.THR[i]) for i in v] for k, v in need.items()}, flush=True)
    sc = work / "coco_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(a.which, str(f), str(sc / f.name), need) for f in sorted((work / "coco").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    print("coco", len(jobs), flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(coco_one, jobs, chunksize=1), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"coco {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
