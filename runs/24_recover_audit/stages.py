#!/usr/bin/env python3
"""Stage audit (diagnostic only). Where in BRACE's frozen hierarchy (V3) does seed completeness survive or vanish?

Images: crc32 positions 10-39 per set (as runs/22), one live capture each (frozen runtime, runs/20 prompts).
Seeds: BR (BRACE seed) and three GT-built PROBES that are never inference methods:
  OF  oracle fill:      max(S, 0.9 * object interior)          (complete, even object)
  OB  oracle clean:     S * (1 - far background)                (no far background)
  OFB both.
The V3 path of sg_combined_v2.aggregate is replayed stage by stage (fidelity check vs aggregate for BR):
  s64 seed -> s16 (resize) -> r16 (16 SA) -> r32 (32 SA) -> z32 (cap) -> V3 (64 SA).
At every stage, at its resolution: AUC vs GT, mass share on GT, interior and far-background mean (min-max).
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_relgate as rg  # noqa: E402
import jc_layergate as lg  # noqa: E402

STAGES = ("s64", "s16", "r16", "r32", "z32", "V3")
RESO = {"s64": 64, "s16": 16, "r16": 16, "r32": 32, "z32": 32, "V3": 64}
THR = np.round(np.arange(0.05, 0.951, 0.05), 2)


def mmn(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def auc(s, g):
    pos, neg = s[g], s[~g]
    if pos.size < 2 or neg.size < 2:
        return np.nan
    rk = rankdata(np.concatenate([pos, neg]))
    return (rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def blk(x, k):
    s = x.shape[0] // k
    return x.reshape(k, s, k, s).mean((1, 3))


@torch.no_grad()
def stages(sa, seed):
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
    c64 = seed.float().reshape(-1); c16 = _resize(c64, 64, 16).reshape(-1)
    acc = {k: 0 for k in ("r16", "r32", "z32", "V3")}
    for call in range(11):
        h16 = _attention(sa[16][call], 16); h32 = _attention(sa[32][call], 32); h64 = _attention(sa[64][call], 64)
        r16 = _apply(h16, c16).mean(0); r32 = _apply(h32, _resize(r16, 16, 32).reshape(-1)).mean(0)
        z32, _ = cap(r16, r32, 16); v3 = _apply(h64, _resize(z32, 32, 64).reshape(-1)).mean(0)
        for k, v in (("r16", r16), ("r32", r32), ("z32", z32), ("V3", v3)):
            acc[k] = acc[k] + v / 11
        del h16, h32, h64
    out = {"s64": c64, "s16": c16, **acc}
    return {k: v.reshape(RESO[k], RESO[k]).cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def main(n, start, tag):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    runtime = sg27_runtime.make_runtime("stable-diffusion-v1-5/stable-diffusion-v1-5"); pipe = runtime["pipe"]
    pipe.safety_checker = None; dev = pipe.device
    res = {}
    for ds in ("voc", "coco"):
        res[ds] = []
        for r in rg.pick(ds, start, n):
            with sg27_runtime.live_maps(runtime, dict(r), ds) as (sample, sa, seed, _b, _p):
                truth = np.asarray(sample.target_mask, bool); H = truth.shape[0] // 64
                gf = truth.reshape(64, H, 64, H).mean((1, 3)); g = gf >= 0.5
                if g.sum() < 10 or (~g).sum() < 10:
                    continue
                inner = ndi.binary_erosion(g, iterations=2); inner = inner if inner.any() else g
                far = ~ndi.binary_dilation(g, iterations=4)
                S = mmn(seed.float().cpu().numpy())
                probes = {"BR": S, "OF": np.maximum(S, 0.9 * inner), "OB": S * (~far), "OFB": np.maximum(S, 0.9 * inner) * (~far)}
                ref = sg_combined_v2.aggregate(sa, seed.float())[0]["V3"].cpu().numpy()
                o = {"id": r["image_id"], "cls": r["label"]}
                for pn, x in probes.items():
                    st = stages(sa, torch.from_numpy(x.astype(np.float32)).to(dev))
                    if pn == "BR":
                        o["fidelity"] = float(np.abs(mmn(st["V3"]) - mmn(ref)).max())
                    for k, m in st.items():
                        k_ = RESO[k]; gk = blk(gf, k_) >= 0.5; ik = blk(inner.astype(float), k_) >= 0.5; fk = blk(far.astype(float), k_) >= 0.5
                        y = mmn(m)
                        o[f"{pn}|{k}|auc"] = float(auc(y.ravel(), gk.ravel())); o[f"{pn}|{k}|mass"] = float((y * gk).sum() / max(y.sum(), 1e-12))
                        o[f"{pn}|{k}|interior"] = float(y[ik].mean()) if ik.any() else np.nan
                        o[f"{pn}|{k}|far"] = float(y[fk].mean()) if fk.any() else np.nan
                    y = mmn(st["V3"])
                    o[f"{pn}|V3|I"] = [int(((y >= t) & g).sum()) for t in THR]; o[f"{pn}|V3|U"] = [int(((y >= t) | g).sum()) for t in THR]
            res[ds].append(o)
            print(ds, r["image_id"], f"fid {o['fidelity']:.1e}", " ".join(f"{p}:{o[f'{p}|V3|auc']:.3f}" for p in probes), flush=True)
            torch.cuda.empty_cache()
    summ = {}
    for ds, rows in res.items():
        s = {"n": len(rows), "max_fidelity": max(x["fidelity"] for x in rows)}
        cls = np.array([x["cls"] for x in rows])
        for pn in ("BR", "OF", "OB", "OFB"):
            for k in STAGES:
                for q in ("auc", "mass", "interior", "far"):
                    s[f"{pn}|{k}|{q}"] = float(np.nanmean([x[f"{pn}|{k}|{q}"] for x in rows]))
            I = np.array([x[f"{pn}|V3|I"] for x in rows]); U = np.array([x[f"{pn}|V3|U"] for x in rows])
            mi = np.array([np.mean([I[cls == c, j].sum() / U[cls == c, j].sum() for c in np.unique(cls)]) for j in range(len(THR))])
            s[f"{pn}|V3|miou_best"] = float(mi.max()); s[f"{pn}|V3|miou_055"] = float(mi[int(np.flatnonzero(np.isclose(THR, 0.55))[0])])
        summ[ds] = s
    Path(__file__).with_name(f"stages{tag}.json").write_text(json.dumps({"summary": summ, "per_image": res}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n, 10, sys.argv[2] if len(sys.argv) > 2 else "")
