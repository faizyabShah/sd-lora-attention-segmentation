"""Shared scoring and reporting for the operator experiments (scripts/ana_ops_a.py, scripts/ana_ops_bc.py).

Per image, every result is a confusion-count array (cc.counts layout: tp, fp, fn, tn, fp_other):
  '<name>|raw'      [25, 5]  raw map on the tau grid 0.30..0.90 step 0.025
  '<name>+R2|raw'   [13, 5]  DenseCRF R2 on the tau grid step 0.05
  '<name>|mask'     [1, 5]   one label-free mask (no tau)
Exploratory diagnostics: ground truth is used only to score.
"""
from __future__ import annotations

import sys, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402

TAU = 0.55
J55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def crf_init():
    from src.segmentation.ovam_full import _import_densecrf
    _import_densecrf(st.CRF_SITE)


def crf(img, mask):
    from src.segmentation.ovam_full import apply_ovam_densecrf
    return apply_ovam_densecrf(img, mask, st.CRF_SITE).astype(bool)


def crf_grid(args):
    img, x = args
    return np.stack([crf(img, x >= t) for t in cc.THR_CRF])


def grid(x, truth):
    v = np.ones_like(truth); o = np.zeros_like(truth)
    return cc.counts(x[None] >= cc.THR[:, None, None], truth, v, o)


def cnt(stack, truth):
    stack = stack if stack.ndim == 3 else stack[None]
    return cc.counts(stack, truth, np.ones_like(truth), np.zeros_like(truth))


def load(d, ids=None):
    fs = sorted(f for f in Path(d).glob("*.npz") if not f.stem.endswith(".tmp") and (ids is None or f.stem in ids))
    if not fs:
        return None
    Z = [np.load(f, allow_pickle=True) for f in fs]; grp = np.array([f.stem for f in fs])
    keys = set.intersection(*[{k for k in z.files if "|" in k} for z in Z])
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": grp, "fold": np.array([zlib.crc32(g.encode()) % 5 for g in grp]),
         "_Z": Z}
    S.update({k: np.stack([z[k] for z in Z]) for k in keys})
    return S


def evaluate(S, name, ref="brace"):
    """(score, per-image counts used, label). Grid maps: OOF + R2; single masks: as is."""
    if f"{name}+R2|raw" in S:
        m, P, _ = cc.oof(S, S[f"{name}+R2|raw"]); return m, P, "OOF + R2"
    C = S[f"{name}|mask"][:, 0]
    return cc.miou(C[:, None], S["cls"], np.ones(len(C), bool))[0], C, "mask"


def table(S, names, ref="brace", groups=None):
    one = np.ones(len(S["grp"]), bool)
    rm, rP, _ = cc.oof(S, S[f"{ref}+R2|raw"]); r55 = S[f"{ref}+R2|raw"][:, J55]
    L = ["| group | map | score | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |", "|---|---|---:|---|---|---:|---:|"]
    L.append(f"| — | BRACE (V3 + R2, OOF) | {rm:.3f} | — | {cc.prf(rP)[0]:.3f} / {cc.prf(rP)[1]:.3f} | "
             f"{cc.oof(S, S[f'{ref}|raw'])[0]:.3f} | {cc.miou(r55[:, None], S['cls'], one)[0]:.3f} |")
    for g, ns in (groups or {"": names}).items():
        for n in ns:
            if f"{n}+R2|raw" not in S and f"{n}|mask" not in S:
                continue
            m, P, how = evaluate(S, n)
            base = rP if how == "OOF + R2" else r55; bm = rm if how == "OOF + R2" else cc.miou(r55[:, None], S["cls"], one)[0]
            lo, hi = cc.boot(S, P, base)
            raw = f"{cc.oof(S, S[f'{n}|raw'])[0]:.3f}" if f"{n}|raw" in S else "—"
            f55 = f"{cc.miou(S[f'{n}+R2|raw'][:, J55][:, None], S['cls'], one)[0]:.3f}" if f"{n}+R2|raw" in S else "—"
            pr = cc.prf(P)
            L.append(f"| {g} | {n} ({how}) | {m:.3f} | {m - bm:+.3f} ({lo:+.3f}…{hi:+.3f}) | {pr[0]:.3f} / {pr[1]:.3f} | {raw} | {f55} |")
    L.append("\nΔ for OOF maps is against BRACE OOF + R2; Δ for single label-free masks is against BRACE @ 0.55 + R2.")
    return L
