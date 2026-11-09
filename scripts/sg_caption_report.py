#!/usr/bin/env python3
"""Report for sg_caption.py (D, exploratory): OOF class-macro mIoU per prompt condition and map, paired photo-bootstrap
Δ vs the base condition, split by whether the class word was already in the Florence-2 caption."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import sg_class_comp as cc  # noqa: E402

SC = Path("/home/shafa/sg_comp/cap_D_scores"); MP = Path("/home/shafa/sg_comp/cap_D")
CONDS = ("base", "inv", "cap", "both")
KEYS = [("base", "seed", "A3 seed (BRACE)"), ("base", "V3+R2", "BRACE V3+R2"), ("base", "C16", "A3 → C16"),
        ("base", "P1+R2", "A3 → P1+R2"), ("clip", "C16", "CLIP seed → C16"), ("clip", "V3+R2", "CLIP seed → V3+R2")]
S = {c: cc.load(SC / c, "D") for c in CONDS}
names = [f.name for f in sorted((SC / "base").glob("*.npz"))]
for c in CONDS:
    assert [f.name for f in sorted((SC / c).glob("*.npz"))] == names, c
inc = np.array([bool(np.load(MP / "base" / n)["class_in_caption"]) for n in names])
caps = json.loads((ROOT / "outputs/segmentation/sg_real_atlas/analysis/florence_captions_D.json").read_text())
L = ["# Florence-2 captions as SD prompts on real VOC D (exploratory, 2026-11-09)\n",
     "`scripts/sg_caption.py`. Captions: Florence-2-large `<CAPTION>` (label-free). Conditions per pair:",
     "- **base**: inversion with \"\", capture prompt \"a photograph of a {cls}\" (the stored D pipeline)",
     "- **inv**: inversion with the caption, capture prompt as base",
     "- **cap**: inversion with \"\", capture prompt = the caption (\"{caption}, a {cls}\" if the class word is missing)",
     "- **both**: caption for inversion and capture\n",
     f"Pairs: {len(names)}; class word already in the caption for {inc.sum()} ({inc.mean() * 100:.0f}%). "
     f"Replay of base vs stored D maps: max {np.nanmax(S['base']['check']):.1e}.\n",
     "OOF class-macro mIoU (τ chosen on 4 folds, applied to the 5th), Δ vs base with photo-bootstrap 95% CI:\n",
     "| map | base | inv | cap | both | Δ inv | Δ cap | Δ both |", "|---|---:|---:|---:|---:|---|---|---|"]
for v, m, lbl in KEYS:
    r = {c: cc.oof(S[c], S[c][f"{v}|{m}|raw"]) for c in CONDS}
    cells = []
    for c in CONDS[1:]:
        lo, hi = cc.boot(S["base"], r[c][1], r["base"][1]); cells.append(f"{r[c][0] - r['base'][0]:+.3f} ({lo:+.3f}…{hi:+.3f})")
    L.append(f"| {lbl} | " + " | ".join(f"{r[c][0]:.3f}" for c in CONDS) + " | " + " | ".join(cells) + " |")
L.append("\nSame, split by whether the class word was in the caption (OOF predictions from the full run):\n")
L.append("| map | subset | n | base | inv | cap | both |\n|---|---|---:|---:|---:|---:|---:|")
for v, m, lbl in KEYS:
    r = {c: cc.oof(S[c], S[c][f"{v}|{m}|raw"])[1] for c in CONDS}
    for nm, sel in (("in caption", inc), ("appended", ~inc)):
        L.append(f"| {lbl} | {nm} | {sel.sum()} | " + " | ".join(f"{cc.miou(r[c][:, None], S[c]['cls'], sel)[0]:.3f}" for c in CONDS) + " |")
L.append("\nSeed separation (A3 seed, mean over pairs): AUROC target vs other objects / vs background\n")
for c in CONDS:
    sp = S[c]["base|seed|sep"]; L.append(f"- {c}: {np.nanmean(sp[:, 0]):.3f} / {np.nanmean(sp[:, 1]):.3f}")
L.append("\nExample captions: " + "; ".join(f"`{k}`: {caps[k]['<CAPTION>']}" for k in list(caps)[:5]))
txt = "\n".join(L) + "\n"; print(txt)
(ROOT / "outputs/segmentation/sg_real_atlas/analysis/florence_captions_D.md").write_text(txt)
