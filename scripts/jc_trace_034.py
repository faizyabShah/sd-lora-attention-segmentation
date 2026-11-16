#!/usr/bin/env python3
"""Trace of the draft's COCO-Cap hierarchy gain (+0.034): recomputation from the runs/15 capture (diagnostic).

Uses the class-word maps of <BRACE_CACHE_DIR>/jc_ladder (fine-only V0 and capped hierarchy V3, same generation).
DenseCRF at the original 17-threshold grid (0.30..0.70 step 0.025), which the runs/15 scoring (step 0.05) lacks.

score   (CPU)  counts for V0, V3 raw and +R2 at the 17 thresholds
report  (CPU)  runs/16_trace_034/result.md
"""
from __future__ import annotations

import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import jc_ladder as jl  # noqa: E402

THR = np.round(0.30 + 0.025 * np.arange(17), 3)
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
OUT = ROOT / "runs/16_trace_034"
CELLS = ("V0", "V3", "V0+R2", "V3+R2")


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"])}
    w = z["maps"][jl.SEEDS.index("wA3")]
    for m in ("V0", "V3"):
        x = st.up(st.mm(w[jl.MAPS.index(m)]), truth.shape)
        res[f"{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
        res[f"{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in ("voc", "coco"):
        sc = jl.WORK / f"trace034_{ds}"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((jl.WORK / ds).glob("*.npz")) if not f.stem.endswith(".tmp")]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    S = {ds: cc.load_gen(jl.WORK / f"trace034_{ds}") for ds in ("voc", "coco")}
    one = {ds: np.ones(len(S[ds]["grp"]), bool) for ds in S}
    ti = lambda t: int(np.flatnonzero(np.isclose(THR, t))[0])

    def curve(ds, c):
        return cc.miou(S[ds][f"{c}|raw"], S[ds]["cls"], one[ds])

    def at(ds, c, t):
        return S[ds][f"{c}|raw"][:, ti(t)]

    def mi(ds, P, s=None):
        return cc.miou(P[:, None], S[ds]["cls"], one[ds] if s is None else s)[0]

    def d(ds, Pa, Pb):
        lo, hi = cc.boot(S[ds], Pa, Pb, n=2000); return f"{mi(ds, Pa) - mi(ds, Pb):+.4f} [{lo:+.4f}, {hi:+.4f}]"

    def prf(P):
        p, r, _ = cc.prf(P); return f"{p:.3f} / {r:.3f}"

    best = {c: float(THR[int(np.argmax(curve("voc", c)))]) for c in CELLS}          # full-VOC optimum (frozen for COCO)
    cbest = {c: float(THR[int(np.argmax(curve("coco", c)))]) for c in CELLS}        # COCO optimum (diagnostic only)
    L = ["# Where the draft's COCO-Cap hierarchy gain of +0.034 comes from (diagnostic)\n",
         "`scripts/jc_trace_034.py`. Recomputed from the class-word maps of the `runs/15_token_ladder` capture "
         "(490 VOC-Sim, 707 COCO-Cap images, one generation per image). Fine-only = two-pass 64-px propagation; "
         "capped hierarchy = the frozen method's map. DenseCRF with the frozen settings at the original 17-threshold "
         "grid (0.30 to 0.70, step 0.025). Class-macro mIoU; Δ with image-bootstrap 95% interval (2000). "
         "Post hoc analysis of existing results; nothing is selected here.\n",
         "## Thresholds\n",
         "| Map | Full-VOC-Sim best threshold (this capture) | Original run | COCO-Cap best threshold (diagnostic) |",
         "|---|---:|---:|---:|",
         f"| fine-only, raw | {best['V0']:.3f} | 0.450 | {cbest['V0']:.3f} |",
         f"| fine-only + DenseCRF | {best['V0+R2']:.3f} | 0.425 | {cbest['V0+R2']:.3f} |",
         f"| capped hierarchy, raw | {best['V3']:.3f} | 0.550 | {cbest['V3']:.3f} |",
         f"| capped hierarchy + DenseCRF | {best['V3+R2']:.3f} | 0.550 | {cbest['V3+R2']:.3f} |", ""]
    for ds in ("voc", "coco"):
        L.append(f"## {NAME[ds]}: mIoU against the threshold\n")
        L.append("| threshold | fine-only raw | hierarchy raw | fine-only + DenseCRF | hierarchy + DenseCRF | Δ (+DenseCRF, same threshold) |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        C = {c: curve(ds, c) for c in CELLS}
        for i, t in enumerate(THR):
            L.append(f"| {t:.3f} | {C['V0'][i]:.4f} | {C['V3'][i]:.4f} | {C['V0+R2'][i]:.4f} | {C['V3+R2'][i]:.4f} | {C['V3+R2'][i] - C['V0+R2'][i]:+.4f} |")
        L.append("")
    L.append("## COCO-Cap: capped hierarchy minus fine-only under each convention\n")
    L.append("| Convention | hierarchy (thr) | fine-only (thr) | Δ [95% interval] | precision / recall: hierarchy; fine-only |")
    L.append("|---|---:|---:|---|---|")
    conv = [("**Draft's convention**: each map at the original run's full-VOC-Sim threshold, + DenseCRF", "V3+R2", 0.55, "V0+R2", 0.425),
            ("Each map at this capture's full-VOC-Sim threshold, + DenseCRF", "V3+R2", best["V3+R2"], "V0+R2", best["V0+R2"]),
            ("Same threshold 0.55, + DenseCRF", "V3+R2", 0.55, "V0+R2", 0.55),
            ("Same threshold 0.425, + DenseCRF", "V3+R2", 0.425, "V0+R2", 0.425),
            ("Each map at its COCO-Cap best threshold, + DenseCRF (diagnostic, uses COCO-Cap labels)", "V3+R2", cbest["V3+R2"], "V0+R2", cbest["V0+R2"]),
            ("Raw, original full-VOC-Sim thresholds (0.55 / 0.45)", "V3", 0.55, "V0", 0.45),
            ("Raw, same threshold 0.55", "V3", 0.55, "V0", 0.55),
            ("Raw, each at its COCO-Cap best threshold (diagnostic)", "V3", cbest["V3"], "V0", cbest["V0"])]
    for lab, ca, ta, cb, tb in conv:
        Pa, Pb = at("coco", ca, ta), at("coco", cb, tb)
        L.append(f"| {lab} | {mi('coco', Pa):.4f} ({ta:.3f}) | {mi('coco', Pb):.4f} ({tb:.3f}) | {d('coco', Pa, Pb)} | {prf(Pa)}; {prf(Pb)} |")
    L.append("\n## VOC-Sim: the same conventions\n")
    L.append("| Convention | hierarchy (thr) | fine-only (thr) | Δ [95% interval] |"); L.append("|---|---:|---:|---|")
    for lab, ca, ta, cb, tb in conv[:4]:
        Pa, Pb = at("voc", ca, ta), at("voc", cb, tb)
        L.append(f"| {lab} | {mi('voc', Pa):.4f} ({ta:.3f}) | {mi('voc', Pb):.4f} ({tb:.3f}) | {d('voc', Pa, Pb)} |")
    oa = cc.oof(S["voc"], S["voc"]["V3+R2|raw"]); ob = cc.oof(S["voc"], S["voc"]["V0+R2|raw"])
    lo, hi = cc.boot(S["voc"], oa[1], ob[1], n=2000)
    L.append(f"| 5-fold out-of-fold threshold per map, + DenseCRF | {oa[0]:.4f} | {ob[0]:.4f} | {oa[0] - ob[0]:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    L.append("\n## COCO-Cap per class, + DenseCRF: fine-only at 0.425, fine-only at 0.55, hierarchy at 0.55\n")
    L.append("| class | fine-only @0.425 | fine-only @0.55 | hierarchy @0.55 | Δ draft's convention | Δ same threshold |"); L.append("|---|---:|---:|---:|---:|---:|")
    for c in np.unique(S["coco"]["cls"]):
        s = S["coco"]["cls"] == c
        x, y, z = mi("coco", at("coco", "V0+R2", 0.425), s), mi("coco", at("coco", "V0+R2", 0.55), s), mi("coco", at("coco", "V3+R2", 0.55), s)
        L.append(f"| {c} | {x:.3f} | {y:.3f} | {z:.3f} | {z - x:+.3f} | {z - y:+.3f} |")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); {"score": score, "report": report}[a.cmd](a)
