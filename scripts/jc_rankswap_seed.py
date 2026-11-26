#!/usr/bin/env python3
"""Seed-alone rank-distribution swap from the runs/17 caches (runs/19 amendment 1, CPU part).

Seeds per image: CA (BRACE live seed, runs/17 capture), CLIP (NERVE's seed resized to 64 as its refinement does),
hA = CA ranking with CLIP values, hB = CLIP ranking with CA values (scripts/jc_srcrule.rank_swap).
Each seed is min-max scaled, upsampled and thresholded (0.30-0.70); no propagation, no DenseCRF.
"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st, sg_class_comp as cc  # noqa
from jc_srcrule import rank_swap, THR  # noqa
from brace_paths import CACHE  # noqa

SW = CACHE / "jc_swap"; SEEDS = ("ca", "clip", "hA", "hB"); NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
L = ["# Seed alone: rank–distribution swap (runs/19 amendment 1, CPU part)\n",
     "Seeds from the runs/17 caches (CLIP seed on the stored image; CA seed from the runs/17 live generation). "
     "No propagation, no DenseCRF. Class-macro mIoU at the VOC-selected threshold (5-fold out-of-fold on VOC-Sim; "
     "full-VOC-Sim-best threshold on COCO-Cap). Δ with image-bootstrap 95% interval (2000).\n"]
S = {}
for ds in ("voc", "coco"):
    rows = []
    for f in sorted((SW / "cap" / ds).glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        z = np.load(f); zs = np.load(SW / "nerve_seed" / ds / f.name)
        ca = torch.from_numpy(z["seed_ca"]).float(); clip = torch.from_numpy(zs["seed64"][0]).float()
        sd = {"ca": ca, "clip": clip, "hA": rank_swap(ca, clip), "hB": rank_swap(clip, ca)}
        truth = z["truth"].astype(bool); valid = np.ones_like(truth); other = np.zeros_like(truth)
        r = {"grp": f.stem, "cls": str(z["cls"]), "area": float(truth.mean())}
        for s in SEEDS:
            x = st.up(st.mm(sd[s].numpy()), truth.shape)
            r[s] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        rows.append(r)
    import zlib
    S[ds] = {"cls": np.array([r["cls"] for r in rows]), "grp": np.array([r["grp"] for r in rows]),
             "fold": np.array([zlib.crc32(r["grp"].encode()) % 5 for r in rows]), "area": np.array([r["area"] for r in rows])}
    for s in SEEDS:
        S[ds][s] = np.stack([r[s] for r in rows])
one = {ds: np.ones(len(S[ds]["grp"]), bool) for ds in S}
mi = lambda ds, P: cc.miou(P[:, None], S[ds]["cls"], one[ds])[0]
def sel(ds, s):
    if ds == "voc":
        return cc.oof(S["voc"], S["voc"][s])[1]
    return S["coco"][s][:, int(np.argmax(cc.miou(S["voc"][s], S["voc"]["cls"], one["voc"])))]
def dci(ds, a, b):
    lo, hi = cc.boot(S[ds], sel(ds, a), sel(ds, b), n=2000); return f"{mi(ds, sel(ds, a)) - mi(ds, sel(ds, b)):+.3f} [{lo:+.3f}, {hi:+.3f}]"
L.append("| Seed | VOC-Sim | COCO-Cap | VOC-best threshold |"); L.append("|---|---:|---:|---:|")
lab = {"ca": "CA", "clip": "CLIP", "hA": "CA ranking + CLIP values (hA)", "hB": "CLIP ranking + CA values (hB)"}
for s in SEEDS:
    t = THR[int(np.argmax(cc.miou(S["voc"][s], S["voc"]["cls"], one["voc"])))]
    L.append(f"| {lab[s]} | {mi('voc', sel('voc', s)):.3f} | {mi('coco', sel('coco', s)):.3f} | {t:.3f} |")
L.append("\n| Contrast | VOC-Sim | COCO-Cap |"); L.append("|---|---|---|")
for nm, a, b in (("values: hA − CA", "hA", "ca"), ("ranking: CLIP − hA", "clip", "hA"),
                 ("values: hB − CLIP", "hB", "clip"), ("ranking: hB − CA", "hB", "ca")):
    L.append(f"| {nm} | {dci('voc', a, b)} | {dci('coco', a, b)} |")
area = S["coco"]["area"]; tb = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
L.append("\nCOCO-Cap per-image IoU by object-size third (small / middle / large) and share below 0.3 IoU:\n")
for s in SEEDS:
    P = sel("coco", s); x = P[:, 0] / np.maximum(P[:, :3].sum(1), 1)
    L.append(f"- {lab[s]}: " + " / ".join(f"{x[tb == k].mean():.3f}" for k in range(3)) + f"; below 0.3: {np.mean(x < 0.3):.3f}")
txt = "\n".join(L) + "\n"; print(txt); (ROOT / "runs/19_source_rule/seed_alone_rankswap.md").write_text(txt)
