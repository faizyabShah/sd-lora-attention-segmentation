#!/usr/bin/env python3
"""Post hoc descriptive readout of runs 37 and 38 from their saved scores (no new capture, no new fit):
person-excluded mIoU, per-class sign counts, and gate differences across the whole threshold grid.
-> runs/38_gen_gates_factorial/readout_person_thr.md"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_realvoc37 as R37  # noqa: E402
import jc_gen38 as G38  # noqa: E402

R, cc = R37.R, R37.cc
THR = R37.THR
OUT = ROOT / "runs/38_gen_gates_factorial/readout_person_thr.md"
NB = 2000
L = []; w = L.append
f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def sub(S, sel):
    return {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}


def stat_diff(BS, Pa, Pb):
    a, b = BS.stat(Pa), BS.stat(Pb); d = a[1] - b[1]
    return float(a[0] - b[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def block(title, S, P, names, thr_grid=True):
    """S: scores; P(nm, rule): predictions; names: {label: map name}. rule 'crf' = primary, else threshold index."""
    noP = S["cls"] != "person"; Sx = sub(S, noP); BSa, BSx = R.Boot(S, NB), R.Boot(Sx, NB)
    w(f"\n### {title}\n")
    w("| Contrast | all classes | person excluded | classes LR/RW better / worse (of those present) |"); w("|---|---|---|---|")
    for lab, (x, y) in names.items():
        da = stat_diff(BSa, P(S, x, "crf"), P(S, y, "crf")); dx = stat_diff(BSx, P(Sx, x, "crf"), P(Sx, y, "crf"))
        per = [cc.miou(P(S, x, "crf")[:, None], S["cls"], S["cls"] == c)[0] - cc.miou(P(S, y, "crf")[:, None], S["cls"], S["cls"] == c)[0]
               for c in np.unique(S["cls"])]
        w(f"| {lab} | {f3(da)} | {f3(dx)} | {sum(p > 0 for p in per)} / {sum(p < 0 for p in per)} (median Δ {np.median(per):+.3f}) |")
    if thr_grid:
        w("\nRaw maps (no DenseCRF), difference at every threshold, all classes (point estimates; † = 95% CI excludes 0):\n")
        w("| Contrast | " + " | ".join(f"{t:.3f}" for t in THR) + " |"); w("|---|" + "---:|" * len(THR))
        for lab, (x, y) in names.items():
            row = []
            for i in range(len(THR)):
                d = stat_diff(BSa, P(S, x, i), P(S, y, i)); row.append(f"{d[0]:+.3f}{'†' if d[1] > 0 or d[2] < 0 else ''}")
            w(f"| {lab} | " + " | ".join(row) + " |")


# ---------------- run 37 (real VOC val, source C, SA weights on)
Tr = json.loads((R37.WORK / "thr.json").read_text()); gam = Tr["gamma"]; S37 = R37.load_scores("val", "")


def n37(g, p):
    f = R37.fam(True, "C", g)
    return f"{f}|{p}" if g == "-" else f"{f}{gam[f'{f}|{p}']}|{p}"


def P37(S, nm, rule):
    return S[f"{nm}|crf"][:, 0] if rule == "crf" else S[f"{nm}|raw"][:, rule]


w("# Runs 37 and 38: person-excluded and threshold-range readout (post hoc, descriptive)\n")
w("Computed from the saved scores of runs 37 and 38; nothing was recaptured or refitted. γ and the primary threshold are the "
  "runs' own train / dev selections. **Descriptive only**: this readout was not pre-registered. Bootstrap: 2000 resamples "
  "(photos for real VOC, images for the generated sets).\n")
w("\n## Run 37: real VOC val (source C, SA weights on)\n")
for p in ("V3", "V0"):
    block(f"{p} ({'capped hierarchy' if p == 'V3' else 'fine-only'}), + DenseCRF at the train threshold", S37, P37,
          {"MK4-LR − MK4": (n37("L", p), n37("M", p)), "MK4-RW − MK4": (n37("R", p), n37("M", p)),
           "MK4 − none": (n37("M", p), n37("-", p)), "MK4-LR − none": (n37("L", p), n37("-", p))})

# ---------------- run 38 (generated, test halves, SA weights on)
T8 = json.loads((G38.WORK / "thr.json").read_text()); g8 = T8["gamma"]
S8 = {}
for ds in G38.SETS:
    raw = G38.load_scores(ds, "scores", ""); tst = raw["part"] == "test"; S8[ds] = sub(raw, tst)
    cr = G38.load_scores(ds, "crf", ""); S8[ds].update({k: v for k, v in cr.items() if k.endswith("|crf")})

for r in G38.REGIMES:
    def n38(g, p, r=r):
        f = G38.fam(r, True, g)
        return f"{f}|{p}" if g == "-" else f"{f}{g8[r][f'{f}|{p}']}|{p}"

    def P38(S, nm, rule, r=r):
        return S[f"{r}|{nm}|crf"][:, 0] if rule == "crf" else S[f"{nm}|raw"][:, rule]
    w(f"\n## Run 38: regime {r} ({'VOC-Sim dev profiles; COCO-Cap is transfer' if r == 'V' else 'VOC-Sim + COCO-Cap dev'})\n")
    for ds in G38.SETS:
        for p in ("V3", "V0"):
            block(f"{G38.SETN[ds]} test, {p}, + DenseCRF at the dev threshold", S8[ds], P38,
                  {"MK4-LR − MK4": (n38("L", p), n38("M", p)), "MK4-RW − MK4": (n38("R", p), n38("M", p)),
                   "MK4 − none": (n38("M", p), n38("-", p)), "MK4-LR − none": (n38("L", p), n38("-", p))})

txt = "\n".join(L) + "\n"; OUT.write_text(txt); print(txt)
