#!/usr/bin/env python3
"""Run 44 scoring (env trdp). Ground truth is used here only.

score   (CPU) per (photo, class) primary item of a column: tp / fp / fn / tn counts of every method's binary mask
        -> <WORK>/scores_<set>/<item id>.npz
report  (CPU) class-macro mIoU, photo bootstrap (2,000, paired), primary contrasts -> runs/44_real_benchmarks/result.md

Ground truth: VOC = SegmentationClass == class (255 = void, excluded); Context = converted 59-class label == class
(0 = unlabelled, counted as not-the-class); COCO-Object = run 42 instance masks (crowd regions excluded).
Read-outs (PLAN.md section 8):
  ours        u-, uM, uL maps (V0): bilinear to photo size, min-max, threshold grid (0.55 primary); DenseCRF R2 at 0.55
  CLIP x4, NERVE, iSeg, OVDiff   their label map == the class's label (index found by name, jc_names44)
  DiffSegmenter 64 x 64 map -> bilinear, min-max, > 0.5 (fixed; owner decision)
  DAAM        word map -> bilinear, min-max (DAAM's expand_as), > 0.4
  OVAM        word map -> bilinear, / max, x self-attention rescaled to [0.85, 1], > 0.4 (OVAM's VOC-Sim settings); + DenseCRF
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_ov42 as O  # noqa: E402
from jc_real44 import WORK, VOC20, CTX59  # noqa: E402
from jc_names44 import to_canon, canon_sets  # noqa: E402

st, cc, THR, T55 = O.st, O.cc, O.THR, O.T55
VOC_SEG = ROOT / "datasets/voc2012/VOCdevkit/VOC2012/SegmentationClass"
CTX_SEG = ROOT / "datasets/pascal_context/SegmentationClass"
CLIPS = ("sclip", "clearclip", "naclip", "proxyclip")
OVK = {"voc": "voc44", "context": "context44", "coco": "coco44"}


def truth_of(s, x):
    from PIL import Image
    if s == "coco":
        z = np.load(O.WORK / "masks" / f"{x['id'][len('coco_'):]}.npz")
        return z["truth"], ~z["ignore"], z["other"]
    g = np.asarray(Image.open((VOC_SEG if s == "voc" else CTX_SEG) / f"{x['img']}.png")).astype(np.int64)
    t = g == x["ci"]; v = g != 255
    return t, v, v & (g > 0) & ~t


def label_of(lines, s, cls):
    """1-based label of canonical class `cls` in a method's foreground name list (0 = background)."""
    canon = canon_sets()[s]; hits = [i for i, l in enumerate(lines) if to_canon(l, s, canon) == cls]
    assert len(hits) == 1, (s, cls, hits); return hits[0] + 1


def names_tables(s):
    T = {}
    for r in CLIPS:
        f = WORK / f"clip_{r}" / f"names_{s}.json"
        if f.exists():
            T[r] = json.loads(f.read_text())["names"][1:]
    f = WORK / "nerve" / f"meta_{s}.json"
    if f.exists():
        T["nerve"] = json.loads(f.read_text())["categories"]
    f = WORK / "iseg" / f"names_{s}.json"
    if f.exists():
        T["iseg"] = json.loads(f.read_text())["names"]
    f = WORK / "ovdiff" / f"classes_{OVK[s]}.json"
    if f.exists():
        T["ovdiff"] = json.loads(f.read_text())[1:]
    return T


def score_one(job):
    s, x, o, tabs, ovd = job
    o = Path(o)
    if o.exists():
        return
    import torch
    torch.set_num_threads(1)
    from PIL import Image
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    truth, valid, other = truth_of(s, x); hw = truth.shape; res = {"cls": x["cls"], "img": str(x["img"])}
    cnt = lambda m: cc.counts(np.asarray(m, bool)[None], truth, valid, other)[0]
    rgb = None

    def crf(m):
        nonlocal rgb
        _import_densecrf(st.CRF_SITE)
        if rgb is None:
            p = ROOT / "datasets/coco2017" / x["path"] if s == "coco" else ROOT / x["path"]
            rgb = np.asarray(Image.open(p).convert("RGB"), np.uint8)
        return apply_ovam_densecrf(rgb, m, st.CRF_SITE).astype(bool)

    f = WORK / "ours" / f"cap_{s}" / f"{x['id']}.npz"
    if f.exists():
        z = np.load(f)
        for i, nm in enumerate(str(n) for n in z["names"]):
            if nm.endswith("|V0") or nm.endswith("|seed"):
                v = st.up(st.mm(z["maps"][i].astype(np.float32)), hw)
                res[f"ours:{nm}|raw"] = cc.counts(v[None] >= THR[:, None, None], truth, valid, other)
                if nm in ("u-|V0", "uL|V0"):
                    res[f"ours:{nm}|crf"] = cnt(crf(v >= THR[T55]))
    key = str(x["img"]); key12 = f"{int(x['img']):012d}" if s == "coco" else key
    for r in CLIPS:
        f = WORK / f"clip_{r}" / s / f"{key}.npz"
        if f.exists() and r in tabs:
            res[f"{r}|own"] = cnt(np.load(f)["pred"] == label_of(tabs[r], s, x["cls"]))
    f = WORK / "nerve" / s / f"{key}.npz"
    if f.exists() and "nerve" in tabs:
        res["nerve|own"] = cnt(np.load(f)["label"] == label_of(tabs["nerve"], s, x["cls"]))
    for sub, nm in ((s, "iseg|own"), (f"{s}_given", "iseg|given")):
        f = WORK / "iseg" / sub / f"{key12}.npz"
        if f.exists() and "iseg" in tabs:
            res[nm] = cnt(np.load(f)["label"] == label_of(tabs["iseg"], s, x["cls"]))
    if ovd is not None and "ovdiff" in tabs:
        f = WORK / "ovdiff" / "pred" / OVK[s] / f"{ovd:0>5d}.pt"
        if f.exists():
            p = torch.load(f, map_location="cpu")["preds"].numpy()
            res["ovdiff|own"] = cnt(p == label_of(tabs["ovdiff"], s, x["cls"]))
    f = WORK / "diffseg" / s / f"{x['id']}.npz"
    if f.exists():
        res["diffseg|0.5"] = cnt(st.up(st.mm(np.load(f)["m64"].astype(np.float32)), hw) > 0.5)
    f = WORK / "daam" / s / f"{x['id']}.npz"
    if f.exists():
        res["daam|0.4"] = cnt(st.up(st.mm(np.load(f)["daam"].astype(np.float32)), hw) > 0.4)
    f = WORK / "ovam" / s / f"{x['id']}.npz"
    if f.exists():
        z = np.load(f); h = st.up(z["ovam"].astype(np.float32), hw); sa = st.up(z["sa"].astype(np.float32), hw)
        sa = (sa - sa.min()) / max(float(sa.max() - sa.min()), 1e-12) * (1 - 0.85) + 0.85
        m = (h / max(float(h.max()), 1e-12)) * sa > 0.4
        res["ovam|0.4"] = cnt(m); res["ovam|0.4|crf"] = cnt(crf(m))
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    from multiprocessing import get_context
    X = [x for x in json.loads((WORK / "items.json").read_text())[a.set] if x["primary"]]
    tabs = names_tables(a.set); ovi = {}
    f = WORK / "ovdiff" / f"ids_{OVK[a.set]}.json"
    if f.exists():
        ovi = {str(i): k for k, i in enumerate(json.loads(f.read_text()))}
    out = WORK / f"scores_{a.set}"; out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.set, x, str(out / f"{x['id']}.npz"), tabs, ovi.get(str(x["img"]))) for x in X][a.shard::a.num_shards]
    if a.limit: jobs = jobs[:a.limit]
    with get_context("fork").Pool(a.workers) as p:
        for _ in p.imap_unordered(score_one, jobs, chunksize=8):
            pass
    print(f"scored {len(jobs)} items of {a.set}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score"); p.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); {"score": score}[a.cmd](a)
