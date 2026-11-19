#!/usr/bin/env python3
"""BRACE per-image reproduction of runs/20_new_reruns (CAg seed x V3 hierarchy) for paired comparisons.

Read-only use of the committed code: `scripts/jc_newrerun.py` (corrected prompts, row handling, threshold grid),
`scripts/sg27_runtime.py` (frozen live generation), `src/segmentation/sg_combined_v2.py` (V0 / V3),
`scripts/sg_class_comp.py` (counts) and `src/segmentation/ovam_full.py` (DenseCRF R2). Nothing in those files
is changed. `jc_newrerun.py capture` cannot be restricted to the BRACE cell (it always also needs the NERVE
seed files, the template bank, the step-1 images and the stored paper capture, which live in side F's ignored
cache), so this script runs only the steps of `capture` that produce the CAg seed and its V0 / V3 maps, with
the same runtime settings (`recorder_class.enabled = ("A0", "A3")`, safety checker off, float16 map storage).

capture (GPU)  -> <OUT>/<ds>/<image_id>.npz   maps [3, 64, 64] = (CAg seed, V0 fine-only, V3 hierarchy)
score   (CPU)  -> <OUT>/<ds>_scores/<image_id>.npz   counts on the extended grid, raw and + DenseCRF R2

The only redirection: side F's `cc.gen_rows("coco")` reads `ovam_coco_cap.csv`, which is not in the released
archive. It is rebuilt with the committed `scripts/build_coco_cap_manifest.py` (output outside the repo) and
read from there (`--coco-csv`).
"""
from __future__ import annotations

import argparse, csv, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import jc_newrerun as jn  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import THR_EXT, crf_site  # noqa: E402

CELLS = ("seed", "V0", "V3")


def patch_rows(coco_csv: Path):
    orig = cc.gen_rows

    def gen_rows(dataset):
        if dataset != "coco":
            return orig(dataset)
        with coco_csv.open(newline="", encoding="utf-8") as h:
            rows = [r for r in csv.DictReader(h) if r["annotated"].lower() == "true"]
        for r in rows:
            r.setdefault("attribution_prompt", r["prompt"])
        return rows
    cc.gen_rows = gen_rows


@torch.no_grad()
def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    runtime["recorder_class"].enabled = ("A0", "A3"); pipe.safety_checker = None   # as jc_newrerun.capture
    out = a.out / a.dataset; out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    for r in jn.rows(a.dataset, a.shard, a.num_shards, a.max_images, a.ids):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(r), a.dataset) as (sample, sa, seed, _b, _p):
            w = runtime["recorders"][-1].seeds()
            if float((w["selected/A3"].float() - seed.float()).abs().max()) > 0:
                raise ValueError("A3 seed differs from the frozen runtime's seed")
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            ca_g = seed.float().clone()
            v, _ = sg_combined_v2.aggregate(sa, ca_g)
            maps = np.stack([ca_g.reshape(64, 64).cpu().numpy(), v["V0"].float().reshape(64, 64).cpu().numpy(),
                             v["V3"].float().reshape(64, 64).cpu().numpy()])
            del v
        torch.cuda.synchronize(); sec = time.perf_counter() - t0; peak = torch.cuda.max_memory_allocated() / 2**20
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        off = jn.official(a.dataset, r).astype(int); d = np.abs(img.astype(int) - off)
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=r["label"], changed=r["changed"], off_diff=float(d.mean()),
                 off_maxdiff=int(d.max()), truth=truth, image=img, seconds=sec, peak_mib=peak, gpu=gpu)
        os.replace(tmp, o)
        print(f"[brace:{a.dataset}] {iid} {r['label']} changed={r['changed']} vs official mean {d.mean():.2f} "
              f"max {d.max()} {sec:.1f}s peak {peak / 1024:.1f} GiB", flush=True)
        torch.cuda.empty_cache()


def score_one(job):
    f, o, site = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(site)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"]),
           "off_diff": float(z["off_diff"]), "off_maxdiff": int(z["off_maxdiff"]),
           "seconds": float(z["seconds"]), "peak_mib": float(z["peak_mib"])}
    for ci, c in enumerate(CELLS):
        x = st.up(st.mm(z["maps"][ci].astype(np.float32)), truth.shape)     # as jc_newrerun.score_one
        res[f"{c}|raw"] = cc.counts(x[None] >= THR_EXT[:, None, None], truth, valid, other)
        if c != "V3":
            continue                                                       # DenseCRF only on the BRACE map
        ms = np.stack([apply_ovam_densecrf(img, x >= t, site) for t in THR_EXT]).astype(bool)
        res[f"{c}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    site = str(crf_site())
    for ds in a.sets:
        sc = a.out / f"{ds}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name), site) for f in sorted((a.out / ds).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 50 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--ids", default="")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p = sub.add_parser("score"); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    for p in sub.choices.values():
        p.add_argument("--out", type=Path, default=Path.home() / "TRDP/external/n_side/brace_repro")
        p.add_argument("--coco-csv", type=Path, default=Path.home() / "TRDP/external/n_side/brace_repro/ovam_coco_cap.csv")
    a = ap.parse_args()
    patch_rows(a.coco_csv)
    {"capture": capture, "score": score}[a.cmd](a)
