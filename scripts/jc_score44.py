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
            if p.shape != tuple(hw):   # OVDiff predicts at its resized input (short side 448): nearest-neighbour back to the photo
                p = p[(np.arange(hw[0]) * p.shape[0]) // hw[0]][:, (np.arange(hw[1]) * p.shape[1]) // hw[1]]
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


ROWS = [("DAAM", "daam|0.4"), ("OVAM (no token opt.)", "ovam|0.4"), ("DiffSegmenter", "diffseg|0.5"), ("iSeg", "iseg|own"),
        ("SCLIP", "sclip|own"), ("ClearCLIP", "clearclip|own"), ("NACLIP", "naclip|own"), ("ProxyCLIP", "proxyclip|own"),
        ("NERVE", "nerve|own"), ("OVDiff", "ovdiff|own"), ("No gate (ours, ablation)", "ours:u-|V0|raw"),
        ("MK4 (ours, ablation)", "ours:uM|V0|raw"), ("**Ours (MK4-LR), raw**", "ours:uL|V0|raw"),
        ("**Ours + DenseCRF**", "ours:uL|V0|crf"), ("OVAM + DenseCRF", "ovam|0.4|crf"), ("No gate + DenseCRF", "ours:u-|V0|crf"),
        ("iSeg, class given", "iseg|given")]
COLS = [("voc", "PASCAL VOC (20)"), ("context", "PASCAL Context (59)"), ("coco", "COCO-Object (80)")]


def load(s):
    fs = sorted(f for f in (WORK / f"scores_{s}").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([str(z["img"]) for z in Z])}
    keys = sorted({k for z in Z for k in z.files if "|" in k})
    for k in keys:
        if all(k in z.files for z in Z):
            S[k] = np.stack([z[k] for z in Z])
    S["missing"] = {k: sum(k not in z.files for z in Z) for k in keys}
    return S


def report(a):
    L = []; w = L.append
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    f3 = lambda d: "—" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    rd = lambda d: "—" if d is None else "**helps**" if d[1] > 0 else "**hurts**" if d[2] < 0 else "not resolved"
    E = {}
    for s, _ in COLS:
        S = load(s); X = [x for x in json.loads((WORK / "items.json").read_text())[s] if x["primary"]]
        if not a.tag and len(S["id"]) != len(X):
            raise SystemExit(f"{s}: {len(S['id'])} scored of {len(X)}")
        E[s] = (S, O.Boot(S, a.boot))
    P = lambda S, k, t=T55: S[k][:, t] if S[k].ndim == 3 else S[k]
    stat = lambda s, k: E[s][1].stat(P(E[s][0], k)) if k in E[s][0] else None
    def dif(s, x, y):
        A, B = stat(s, x), stat(s, y)
        if A is None or B is None:
            return None
        return A[0] - B[0], float(np.percentile(A[1] - B[1], 2.5)), float(np.percentile(A[1] - B[1], 97.5))
    w("# Run 44: real-photo comparison on PASCAL VOC, PASCAL Context and COCO-Object (`PREREG.md`)\n")
    w("Class word given; one binary mask per (photo, class) primary item (class area >= 1%). Every method as released on the "
      "same photos (PREREG section 4); ours: MK4-LR, gamma 1, fine-only, fixed 0.55. Class-macro mIoU; 95% paired photo bootstrap "
      f"({a.boot} samples). Items: " + ", ".join(f"{n} {len(E[s][0]['id'])} ({len(set(E[s][0]['grp']))} photos)" for s, n in COLS) + ".\n")
    w("\n## 1. Primary contrasts (ours = MK4-LR, V0, raw 0.55)\n")
    w("| # | Contrast | Column | Delta mIoU [95% CI] | Reading |\n|---|---|---|---|---|")
    for i, (s, n) in enumerate(COLS):
        d = dif(s, "ours:uL|V0|raw", "ours:u-|V0|raw"); w(f"| P{i + 1} | ours - no gate | {n} | {f3(d)} | {rd(d)} |")
    for i, (s, n) in enumerate(COLS):
        d = dif(s, "ours:uL|V0|raw", "nerve|own"); w(f"| P{i + 4} | ours - NERVE | {n} | {f3(d)} | {rd(d)} |")
    w("\n## 2. Main table (mIoU)\n")
    w("| Method | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---:|" * len(COLS))
    for nm, k in ROWS:
        w(f"| {nm} | " + " | ".join((f"{stat(s, k)[0]:.3f}" if k in E[s][0] else "—") for s, _ in COLS) + " |")
    w("\n## 3. Ours - each baseline (raw 0.55)\n")
    w("| Baseline | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---|" * len(COLS))
    for nm, k in ROWS:
        if not k.startswith("ours:uL"):
            w(f"| {nm} | " + " | ".join((f3(dif(s, 'ours:uL|V0|raw', k)) if k in E[s][0] else "—") for s, _ in COLS) + " |")
    w("\n## 4. Subsets (raw 0.55)\n")
    for s, nm, sel_fn in (("context", "Context, 20 VOC classes", lambda c: c in VOC20), ("context", "Context, other 39", lambda c: c not in VOC20),
                          ("coco", "COCO-Object, 20 VOC classes", lambda c: c in VOC20), ("coco", "COCO-Object, other 60", lambda c: c not in VOC20)):
        S = E[s][0]; sel = np.array([sel_fn(c) for c in S["cls"]])
        if not sel.any() or "ours:uL|V0|raw" not in S or "ours:u-|V0|raw" not in S:
            continue
        S2 = sub(S, sel); B2 = O.Boot(S2, a.boot)
        st2 = lambda k: B2.stat(P(S2, k))
        cells = [f"{lab} {st2(k)[0]:.3f}" for lab, k in (("ours", "ours:uL|V0|raw"), ("no gate", "ours:u-|V0|raw"), ("NERVE", "nerve|own"),
                                                          ("ProxyCLIP", "proxyclip|own"), ("iSeg", "iseg|own")) if k in S2]
        d = (lambda A, B: (A[0] - B[0], float(np.percentile(A[1] - B[1], 2.5)), float(np.percentile(A[1] - B[1], 97.5))))(st2("ours:uL|V0|raw"), st2("ours:u-|V0|raw"))
        w(f"- **{nm}** ({int(sel.sum())} items): " + "; ".join(cells) + f"; ours - no gate {f3(d)}")
    w("\n## 5. Every threshold (ours, raw; descriptive)\n")
    w("| Column | Map | " + " | ".join(f"{t:.3f}" for t in THR) + " |\n|---|---|" + "---:|" * len(THR))
    for s, n in COLS:
        for k in [k for k in ("ours:uL|V0|raw", "ours:u-|V0|raw") if k in E[s][0]]:
            w(f"| {n} | {k[5:]} | " + " | ".join(f"{E[s][1].stat(P(E[s][0], k, t))[0]:.3f}" for t in range(len(THR))) + " |")
    w("\n## 6. Per class (raw 0.55)\n")
    for s, n in COLS:
        S = E[s][0]; ks = [(nm, k) for nm, k in ROWS[:13] if k in S]
        w(f"\n{n}:\n\n| Class | items | " + " | ".join(nm.replace("*", "") for nm, _ in ks) + " |\n|---|---:|" + "---:|" * len(ks))
        for c in sorted(set(S["cls"]), key=lambda c: (c not in VOC20, c)):
            sel = S["cls"] == c
            iou = lambda k: (lambda x: x[0] / max(x[:3].sum(), 1))(P(S, k)[sel].sum(0))
            w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{iou(k):.3f}" for _, k in ks) + " |")
    w("\n## 7. Coverage\n")
    for s, n in COLS:
        miss = {k: v for k, v in E[s][0]["missing"].items() if v}
        w(f"- {n}: methods with missing items: {miss if miss else 'none'}")
    w("\nBRACE stays frozen; nothing here is a promotion.")
    out = ROOT / "runs/44_real_benchmarks" / ("result.md" if not a.tag else f"result{a.tag}.md")
    out.write_text("\n".join(L) + "\n"); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score"); p.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report"); p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"score": score, "report": report}[a.cmd](a)
