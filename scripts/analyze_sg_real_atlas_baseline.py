#!/usr/bin/env python3
"""SG-real-atlas analysis 1 (Section 137): baseline table over conditions and label-free fusions.

For every captured pair, scores the stored 64x64 maps (A3, C16, V0, V1, V3) of each condition, and a set
of label-free fusions of conditions, against the real mask at 17 thresholds (raw) and, for V0/V3 of a
subset, after DenseCRF (R2) at 9 thresholds. No ground truth is used to build any map.
Output: one npz per pair with counts [variant, tau, 4] (tp fp fn tn over valid pixels) plus GT area.
Usage: analyze_sg_real_atlas_baseline.py CAPTURE_DIR OUT_DIR [--sets D] [--workers 16]
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
COCO = ROOT / "datasets/coco2017"
CRF_SITE = ROOT / "outputs/dependencies/simplecrf_py310"
THR = np.round(0.30 + 0.025 * np.arange(17), 3)
THR_CRF = THR[::2]                                     # 0.30, 0.35, ..., 0.70
MAPS = ("A3", "C16", "V0", "V1", "V3")
SINGLE = ("renoise/n0", "ddim", "renoise/n1", "renoise/n2", "prompt/the", "prompt/background", "prompt/multi")
FUSED = ("ens_noise", "ens_all", "ens_prompt", "quad", "quad_gm", "center_gm", "peak_gm")
CANDS = tuple(c.replace("renoise/", "").replace("prompt/", "p_") for c in SINGLE) + FUSED
CRF_CANDS = ("n0", "ddim", "ens_noise", "ens_all", "ens_prompt", "quad_gm", "peak_gm")
CRF_MAPS = ("V0", "V3")
VARIANTS = [f"{c}/{m}" for c in CANDS for m in MAPS] + [f"{c}/{m}+R2" for c in CRF_CANDS for m in CRF_MAPS]


def mm(x):
    x = np.asarray(x, np.float32); lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)


def up(x, hw):
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None]
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def paste(view64, box, hw):
    """View map (64x64 of a crop resized to 512^2) -> full-image canvas and coverage mask."""
    x0, y0, x1, y1 = box
    canvas, cov = np.zeros(hw, np.float32), np.zeros(hw, np.float32)
    canvas[y0:y1, x0:x1] = up(mm(view64), (y1 - y0, x1 - x0)); cov[y0:y1, x0:x1] = 1
    return canvas, cov


def load(entry_set, iid):
    if entry_set in ("D", "S"):
        return (np.asarray(Image.open(VOC / f"JPEGImages/{iid}.jpg").convert("RGB")),
                np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png")))
    if entry_set == "C":
        return (np.asarray(Image.open(COCO / f"val2017/{iid}.jpg").convert("RGB")),
                np.asarray(Image.open(COCO / f"sg_real_data_labels/{iid}.png")))
    raise ValueError(entry_set)


def candidate_maps(maps, meta, hw):
    """-> {cand: {mapname: full-res map in [0,1]}} built only from captured attention (label-free)."""
    out = {}
    for c in SINGLE:
        src = c if f"{c}/V3" in maps.files else "renoise/n0"     # single-class images have no multi prompt
        out[c.replace("renoise/", "").replace("prompt/", "p_")] = {m: up(mm(maps[f"{src}/{m}"]), hw) for m in MAPS}
    for m in MAPS:
        n0 = out["n0"][m]
        out.setdefault("ens_noise", {})[m] = mm((n0 + out["n1"][m] + out["n2"][m]) / 3)
        out.setdefault("ens_all", {})[m] = mm((n0 + out["n1"][m] + out["n2"][m] + out["ddim"][m]) / 4)
        out.setdefault("ens_prompt", {})[m] = mm((n0 + out["p_the"][m] + out["p_background"][m]) / 3)
        acc, cov = np.zeros(hw, np.float32), np.zeros(hw, np.float32)
        for q in ("q0", "q1", "q2", "q3"):
            a, c = paste(maps[f"view/{q}/{m}"], meta["boxes"][q], hw); acc += a; cov += c
        quad = mm(acc / np.maximum(cov, 1))
        out.setdefault("quad", {})[m] = quad
        out.setdefault("quad_gm", {})[m] = mm(np.sqrt(n0 * quad))
        for v in ("center", "peak"):
            a, c = paste(maps[f"view/{v}/{m}"], meta["boxes"][v], hw)
            out.setdefault(f"{v}_gm", {})[m] = mm(np.where(c > 0, np.sqrt(n0 * a), n0))
    return out


def counts(stack, truth, valid):
    p = stack & valid; t = truth & valid
    tp = (p & t).sum((1, 2)); fp = (p & ~t).sum((1, 2)); fn = (~p & t[None]).sum((1, 2))
    return np.stack([tp, fp, fn, valid.sum() - tp - fp - fn], 1).astype(np.int64)


def work(job):
    pdir, out = Path(job[0]), Path(job[1])
    if out.exists():
        return str(out)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(CRF_SITE)
    st, iid, k = pdir.parts[-3], pdir.parts[-2], int(pdir.parts[-1])
    img, lab = load(st, iid)
    truth, valid = lab == k, lab != 255
    meta = json.loads((pdir / "meta.json").read_text()); maps = np.load(pdir / "maps.npz")
    cm = candidate_maps(maps, meta, truth.shape)
    C = np.zeros((len(VARIANTS), len(THR), 4), np.int64)
    i = 0
    for c in CANDS:
        for m in MAPS:
            C[i] = counts(cm[c][m][None] >= THR[:, None, None], truth, valid); i += 1
    for c in CRF_CANDS:
        for m in CRF_MAPS:
            masks = np.stack([apply_ovam_densecrf(img, cm[c][m] >= t, CRF_SITE) for t in THR_CRF])
            C[i, ::2] = counts(masks.astype(bool), truth, valid); C[i, 1::2] = -1; i += 1
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, counts=C, area=float((truth & valid).sum() / valid.sum()), k=k)
    os.replace(tmp, out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--sets", nargs="+", default=["D"]); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = []
    for s in a.sets:
        for d in sorted((a.capture / s).glob("*/*/done.json")):
            p = d.parent; o = a.out / s / f"{p.parts[-2]}__{p.parts[-1]}.npz"; o.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((str(p), str(o)))
    (a.out / "variants.json").write_text(json.dumps({"variants": VARIANTS, "thr": THR.tolist(),
                                                     "crf_thr_every_other": True}))
    torch.set_num_threads(1)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, jobs), 1):
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
