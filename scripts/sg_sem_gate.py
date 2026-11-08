#!/usr/bin/env python3
"""Semantic gating inside the capped hierarchy, real VOC D only (exploratory, CPU, offline).

Question: does a label-free semantic gate change PROPAGATION (applied between the 16 px step and the 32/64 px
steps, so later SA operations and the V3 cap see the gated map), or is it equivalent to post-processing the
final map?

Inputs (all stored, nothing is re-captured):
  * per-word raw A3 seeds from the class-competition capture (/home/shafa/sg_comp/D_maps/<iid>__<k>.npz);
  * the target run's DDIM SA queries/keys at 16/32/64 (../sg_real_atlas/D/<iid>/<k>/qk_ddim.npz).

Pre-registered definitions (fixed before any result):
  c   = min-max A3 seed of the target word (64x64)
  b   = max over the 8 generic words (wall floor ground sky grass tree building road) of each word's
        min-max A3 seed
  g   = (c + eps) / (c + b + 2 eps), eps = 1e-6, lambda = 1 (no free parameter)
  g16 = the same formula on c, b downsampled to 16x16 with the pipeline's own _resize; g64 = at 64x64
  comp seed = marg.5/mm/gen = minmax(clamp(c - 0.5 b, 0)) (the real-photo winner, unchanged)
  Intermediate placement (one position, registered): per call, raw16 = A16 c16 -> g16 * raw16, which feeds
  A32 AND the V3 cap (V3), and the 16->64 C16 path (C16 = A64 up(g16 * A16 c16)).
  Output placement: the final 64x64 maps multiplied by g64.
  Shift control: g16 rolled by (8, 8) (same values, broken spatial alignment), intermediate placement.

Variants (stack = [seed, P1, V0, V3, C16], same layout as sg_class_comp so its scorer is reused):
  base        A3 seed, no gate                    (replay check vs stored map/base)
  comp        comp seed, no gate                  (replay check vs stored map/marg.5/mm/gen)
  out         A3 seed, output gate g64 on every map
  mid         A3 seed, intermediate gate g16 (V3, C16; seed/P1/V0 have no 16 px stage and stay ungated)
  comp+out    comp seed, output gate
  comp+mid    comp seed, intermediate gate (gate values from c, b as above, i.e. identical to mid)
  shift       A3 seed, rolled intermediate gate

Registered comparisons (photo-bootstrap 95% CI, OOF tau over the 5 grouped folds), judged on V3+R2 and C16:
  M (mechanism)  mid - out and comp+mid - comp+out: passes if the CI excludes 0 (> 0)
  X (spatial)    mid - shift: passes if CI > 0
  P (practical)  best gated variant (max OOF over {out, mid, comp+out, comp+mid} x {V3+R2, C16}) -
                 comp/C16 (registered baseline, also shown vs comp/V3+R2): passes if CI > 0
  The line continues only if M and P both pass; D is exploration only, any survivor is frozen and confirmed
  on unused photos (not S). If gains appear they are background/calibration gains, not object separation,
  since the generic words carry no other-object evidence.
"""
from __future__ import annotations

import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

GENERIC = ["wall", "floor", "ground", "sky", "grass", "tree", "building", "road"]
EPS_G = 1e-6
STORED = ROOT.parent / "sg_real_atlas" / "D"
VARIANTS = ("base", "comp", "out", "mid", "comp+out", "comp+mid", "shift")


def mm(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def gate(c, b):
    return (c + EPS_G) / (c + b + 2 * EPS_G)


@torch.inference_mode()
def paths(sa, seeds, gates16):
    """seeds: name -> [64,64]; gates16: name -> [256] or None. One SA softmax per call, shared by all variants.
    Mirrors sg_combined_v2.aggregate (V0, V3) and sg27_fusion.base_paths (C16) and the P1 of sg_class_comp."""
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
    acc = {n: torch.zeros((5, 4096)) for n in seeds}
    c64 = {n: s.float().reshape(-1) for n, s in seeds.items()}
    c16 = {n: _resize(c64[n], 64, 16).reshape(-1) for n in seeds}
    for call in range(11):
        h16 = _attention(sa[16][call], 16); h32 = _attention(sa[32][call], 32); h64 = _attention(sa[64][call], 64)
        for n in seeds:
            u = _apply(h64, c64[n])
            acc[n][1] += u.mean(0) / 11                                    # P1
            acc[n][2] += _apply(h64, u).mean(0) / 11                       # V0
            raw16 = _apply(h16, c16[n]).mean(0)
            if gates16[n] is not None:
                raw16 = gates16[n] * raw16
            raw32 = _apply(h32, _resize(raw16, 16, 32).reshape(-1)).mean(0)
            z32, _ = cap(raw16, raw32, 16)
            acc[n][3] += _apply(h64, _resize(z32, 32, 64).reshape(-1)).mean(0) / 11      # V3
            acc[n][4] += _apply(h64, _resize(raw16, 16, 64).reshape(-1)).sum(0) / 88     # C16
        del h16, h32, h64
    out = {}
    for n in seeds:
        a = acc[n]; a[0] = c64[n]
        out[n] = a.reshape(5, 64, 64)
    return out


def build_one(job):
    f, out = Path(job[0]), Path(job[1])
    if out.exists():
        return None
    torch.set_num_threads(int(os.environ.get("SG_THREADS", "2")))
    from src.segmentation.sg25_sa import _resize
    z = np.load(f); cls = str(z["cls"]); iid, k = f.stem.split("__")
    q = np.load(STORED / iid / k / "qk_ddim.npz")
    sa = {s: [(torch.from_numpy(q[f"s{s}"][c, 0]), torch.from_numpy(q[f"s{s}"][c, 1])) for c in range(11)]
          for s in (16, 32, 64)}
    c = mm(torch.from_numpy(z[f"word/{cls}"]).float())
    b = torch.stack([mm(torch.from_numpy(z[f"word/{w}"]).float()) for w in GENERIC]).max(0).values
    comp = mm((c - 0.5 * b).clamp_min(0))
    g64 = gate(c, b)
    g16 = gate(_resize(c.reshape(-1), 64, 16).reshape(-1), _resize(b.reshape(-1), 64, 16).reshape(-1))
    g16s = torch.roll(g16.reshape(16, 16), shifts=(8, 8), dims=(0, 1)).reshape(-1)
    seeds = {"base": c, "comp": comp, "mid": c, "comp+mid": comp, "shift": c}
    gates = {"base": None, "comp": None, "mid": g16, "comp+mid": g16, "shift": g16s}
    P = paths(sa, seeds, gates)
    P["out"] = P["base"] * g64[None]
    P["comp+out"] = P["comp"] * g64[None]
    chk = {}
    for n, ref in (("base", "map/base"), ("comp", "map/marg.5/mm/gen")):
        chk[n] = max(float((mm(P[n][i]) - mm(torch.from_numpy(z[ref][i]))).abs().max()) for i in range(5))
    check = max(chk.values())
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **{f"map/{n}": P[n].numpy().astype(np.float32) for n in VARIANTS},
             cls=cls, k=int(k), check=check, set="D",
             gate_mean=float(g64.mean()), gate_min=float(g64.min()))
    os.replace(tmp, out)
    return f.stem, chk


def build(a):
    fs = sorted(f for f in a.src.glob("*.npz") if not f.stem.endswith(".tmp"))
    if a.max_pairs:
        fs = fs[: a.max_pairs]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(a.out / f.name)) for f in fs]
    print(len(jobs), "pairs", flush=True)
    worst = 0.0
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(build_one, jobs), 1):
            if r:
                worst = max(worst, *r[1].values())
                if n <= 5 or n % 25 == 0:
                    print(f"{n}/{len(jobs)} {r[0]} replay base {r[1]['base']:.1e} comp {r[1]['comp']:.1e}", flush=True)
    print(f"worst replay |diff| {worst:.2e}", flush=True)
    # Reference maps were propagated with fresh GPU SA; this replay uses the stored qk_ddim from the original
    # capture. Their known cross-run gap is 1.27e-3 (class-competition capture check), so the tolerance is 2e-3.
    if worst > 2e-3:
        sys.exit(f"REPLAY CHECK FAILED ({worst:.2e} > 2e-3)")


# ------------------------------------------------------------------ report
def report(a):
    import sg_class_comp as cc
    S = cc.load(a.scores, "D")
    L = []; w = L.append
    w("# Semantic gating inside the capped hierarchy (real VOC D, exploratory)\n")
    w(f"{len(S['cls'])} pairs, {len(np.unique(S['grp']))} photos. Definitions and pass rules were fixed before "
      "the run (see `scripts/sg_sem_gate.py` docstring): gate g = (c+ε)/(c+b+2ε), c = target A3 seed, "
      "b = max of the 8 generic-word seeds (each min-max), λ = 1. `mid` gates the 16 px map that feeds A32, the "
      "V3 cap and the 16→64 C16 path; `out` multiplies the final maps by g; `shift` is `mid` with the gate "
      "rolled by (8, 8); `comp` = marg.5/mm/gen seed. OOF τ over 5 photo-grouped folds; Δ CIs photo-bootstrap "
      f"95%. Max replay |Δ| vs the stored maps: {S['check'].max():.1e}.\n")
    maps = ("V3+R2", "C16", "V3", "P1+R2")
    P = {}
    w("| variant | " + " | ".join(f"{m} OOF" for m in maps) + " | V3+R2 prec / rec / FP-on-other | "
      "V3 AUROC tgt-other / tgt-bg | C16 AUROC tgt-other / tgt-bg |")
    w("|---|" + "---:|" * (len(maps) + 3))
    for v in VARIANTS:
        cells = []
        for m in maps:
            key = f"{v}|{m}|raw"
            mi, Pm, taus = cc.oof(S, S[key]); P[(v, m)] = Pm
            tau = (cc.THR_CRF if "+R2" in m else cc.THR)[int(np.median(taus))]
            cells.append(f"{mi:.3f} (τ≈{tau:.2f})")
        pr, rc, fo = cc.prf(P[(v, "V3+R2")])
        sep = lambda m: (np.nanmean(S[f"{v}|{m}|sep"][S["has_other"], 0]), np.nanmean(S[f"{v}|{m}|sep"][:, 1]))
        s3, s16 = sep("V3"), sep("C16")
        w(f"| {v} | " + " | ".join(cells) + f" | {pr:.3f} / {rc:.3f} / {fo:.0%} | {s3[0]:.3f} / {s3[1]:.3f} | "
          f"{s16[0]:.3f} / {s16[1]:.3f} |")
    mi = lambda v, m: cc.miou(P[(v, m)][:, None], S["cls"], np.ones(len(S["cls"]), bool))[0]

    def cmp(va, ma, vb, mb):
        d = mi(va, ma) - mi(vb, mb); lo, hi = cc.boot(S, P[(va, ma)], P[(vb, mb)])
        return d, lo, hi, lo > 0

    w("\n## Registered comparisons\n")
    w("| test | comparison | map | Δ | 95% CI | pass |")
    w("|---|---|---|---:|---|---|")
    res = {}
    for tag, va, vb in (("M", "mid", "out"), ("M", "comp+mid", "comp+out"), ("X", "mid", "shift")):
        for m in ("V3+R2", "C16"):
            d, lo, hi, ok = cmp(va, m, vb, m); res[(tag, va, m)] = ok
            w(f"| {tag} | {va} − {vb} | {m} | {d:+.3f} | {lo:+.3f}…{hi:+.3f} | {'yes' if ok else 'no'} |")
    gated = [(v, m) for v in ("out", "mid", "comp+out", "comp+mid") for m in ("V3+R2", "C16")]
    best = max(gated, key=lambda x: mi(*x))
    for vb, mb in (("comp", "C16"), ("comp", "V3+R2")):
        d, lo, hi, ok = cmp(best[0], best[1], vb, mb); res[("P", vb, mb)] = ok
        w(f"| P | best gated ({best[0]} {best[1]}) − {vb} {mb}{' (registered)' if mb == 'C16' else ''} | — | "
          f"{d:+.3f} | {lo:+.3f}…{hi:+.3f} | {'yes' if ok else 'no'} |")
    m_pass = any(ok for (t, *_), ok in res.items() if t == "M")
    p_pass = res[("P", "comp", "C16")]
    w(f"\n**Mechanism (M):** {'PASS' if m_pass else 'FAIL'}. **Spatial control (X):** "
      f"{'PASS' if any(ok for (t, *_), ok in res.items() if t == 'X') else 'FAIL'}. "
      f"**Practical (P, vs comp/C16):** {'PASS' if p_pass else 'FAIL'}. "
      f"**Line continues:** {'yes — freeze and confirm on unused photos' if m_pass and p_pass else 'no — close'}.\n")

    w("## Per-class V3+R2 OOF IoU (base → comp → mid → comp+mid)\n")
    rows = []
    for c in np.unique(S["cls"]):
        sel = S["cls"] == c
        v = [cc.miou(P[(x, "V3+R2")][:, None], S["cls"], sel)[0] for x in ("base", "comp", "mid", "comp+mid")]
        rows.append(f"{c} {v[0]:.2f}→{v[1]:.2f}→{v[2]:.2f}→{v[3]:.2f}")
    w(", ".join(rows) + "\n")

    w("## By object size (V3+R2 OOF predictions, class-macro within bin)\n")
    w("| bin | n | base | comp | out | mid | comp+out | comp+mid | shift |")
    w("|---|---:|" + "---:|" * 7)
    for lo, hi, name in ((0, .05, "<5%"), (.05, .2, "5–20%"), (.2, 1.01, ">20%")):
        sel = (S["area"] >= lo) & (S["area"] < hi)
        if sel.sum() == 0:
            continue
        v = [cc.miou(P[(x, "V3+R2")][:, None], S["cls"], sel)[0] for x in VARIANTS]
        w(f"| {name} | {sel.sum()} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    a.report.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--src", type=Path, default=Path("/home/shafa/sg_comp/D_maps"))
    p.add_argument("--out", type=Path, default=Path("/home/shafa/sg_comp/gate_D_maps"))
    p.add_argument("--workers", type=int, default=8); p.add_argument("--max-pairs", type=int)
    p = sub.add_parser("report"); p.add_argument("--scores", type=Path, default=Path("/home/shafa/sg_comp/gate_D_scores"))
    p.add_argument("--report", type=Path,
                   default=ROOT / "outputs/segmentation/sg_real_atlas/analysis/semantic_gating.md")
    a = ap.parse_args()
    {"build": build, "report": report}[a.cmd](a)
