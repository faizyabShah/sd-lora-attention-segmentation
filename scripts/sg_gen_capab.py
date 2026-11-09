#!/usr/bin/env python3
"""Cap ablation of the 16→32→64 hierarchy on the generated sets, two seeds (exploratory, 2026-11-09).

Question: is BRACE's one-sided cap at 16→32 (V3) better than the alternatives also when the seed is not BRACE's own
cross-attention seed (A3) but SCLIP's target probability? Variants (frozen code, src/segmentation/sg_combined_v2.py):
  V1      16→32→64 hierarchy, no cap
  V2core  two-sided (exact generalized-KL) projection at both transitions, no completion
  V2      the same plus unconditional completion (the earlier SG-combined-v1 C6 cell)
  V3      one-sided cap at 16→32 only (BRACE)
  V4      one-sided cap at 16→32 and 32→64
  V0      fine-only two-step 64 graph (no hierarchy), reference
Seeds: base = A3 (BRACE), clip = SCLIP ViT-B/16 VOC21 target probability (mm), as in sg_gen_clip.py.
Every image is regenerated with the frozen live pipeline (replay-checked against the stored image and stored V3).

capture (GPU)  /home/shafa/sg_comp/gen_capab/<ds>/<id>.npz: maps/<seed> [len(VARS),64,64]
score   (CPU)  counts per variant raw (τ 0.30..0.90) and +R2 (τ step 0.05)
report  (CPU)  outputs/segmentation/sg_real_atlas/analysis/gen_cap_ablation.md
"""
from __future__ import annotations

import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402

WORK = Path("/home/shafa/sg_comp/gen_capab")
CLIPW = Path("/home/shafa/sg_comp/gen_clip")
STORED = {"voc": Path("/home/shafa/sg_comp/vocsim_maps"), "coco": Path("/home/shafa/sg_comp/cococap_maps")}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
VARS = ("V0", "V1", "V2core", "V2", "V3", "V4")
LABEL = {"V0": "fine-only 64 graph (no hierarchy)", "V1": "hierarchy, no cap",
         "V2core": "two-sided KL projection, both levels", "V2": "two-sided KL projection, both levels + completion",
         "V3": "one-sided cap at 16→32 (BRACE)", "V4": "one-sided cap at 16→32 and 32→64"}
SEEDS = ("base", "clip")
TAU = 0.55


def v2core(sa, seed):
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v1 import distribution, project
    c16 = _resize(seed.float().reshape(-1), 64, 16).reshape(-1); out = torch.zeros(4096, device=seed.device)
    for call in range(11):
        h16 = _attention(sa[16][call], 16); a32 = _attention(sa[32][call], 32).mean(0); a64 = _attention(sa[64][call], 64).mean(0)
        x16 = distribution(_apply(h16, c16).mean(0))
        q32 = distribution(a32 @ _resize(x16, 16, 32).reshape(-1)); p32, _ = project(x16, q32, 16, rho=1.)
        q64 = distribution(a64 @ _resize(p32, 32, 64).reshape(-1)); c4, _ = project(p32, q64, 32, rho=1.)
        out += c4 / 11
        del h16, a32, a64
    return out.reshape(64, 64)


def all_maps(sa, seed):
    from src.segmentation import sg_combined_v2
    v, _ = sg_combined_v2.aggregate(sa, seed.float())
    v["V2core"] = v2core(sa, seed)
    return np.stack([v[k].float().cpu().numpy() for k in VARS]).astype(np.float32)


def capture(a):
    from scripts import sg27_runtime
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    runtime = sg27_runtime.make_runtime(a.model)
    out = WORK / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        P = torch.from_numpy(np.load(CLIPW / f"sclip_{a.dataset}" / f"{iid}.npz")["prob21_64"]).float()
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed = seed.float().clone()
            maps = {"base": all_maps(sa, seed), "clip": all_maps(sa, cc.norm(P.to(seed.device), "mm"))}
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        img_mean = float(np.abs(img.astype(int) - stored["image"].astype(int)).mean())
        v3chk = float(np.abs(st.mm(maps["base"][VARS.index("V3")]) - st.mm(stored["map/base"][3])).max())
        v0chk = float(np.abs(st.mm(maps["base"][VARS.index("V0")]) - st.mm(stored["map/base"][2])).max())
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, **{f"maps/{s}": m for s, m in maps.items()}, cls=cls, check=max(v3chk, v0chk), img_maxdiff=img_diff, img_meandiff=img_mean,
                 truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[capab:{a.dataset}] {iid} {cls} img {img_diff} V3/V0 check {v3chk:.1e}/{v0chk:.1e} "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "check": float(z["check"]), "img_maxdiff": int(z["img_maxdiff"]),
           "img_meandiff": float(z["img_meandiff"]) if "img_meandiff" in z.files else np.nan}
    for s in SEEDS:
        for i, v in enumerate(VARS):
            x = st.up(st.mm(z[f"maps/{s}"][i]), truth.shape)
            res[f"{s}|{v}|raw"] = cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in cc.THR_CRF]).astype(bool)
            res[f"{s}|{v}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / ds).glob("*.npz")) if not f.stem.endswith(".tmp")]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    i55 = int(np.flatnonzero(np.isclose(cc.THR, TAU))[0]); j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    L = ["# Cap ablation of the 16→32→64 hierarchy, A3 seed vs SCLIP seed, generated sets (exploratory, 2026-11-09)\n",
         "`scripts/sg_gen_capab.py`. Each image is regenerated with the frozen live pipeline; both seeds are propagated "
         "through the image's own self-attention with the frozen variant code (`src/segmentation/sg_combined_v2.py`). "
         "Seeds: **A3** = BRACE's cross-attention seed; **SCLIP** = SCLIP ViT-B/16 VOC21 target-class probability "
         "(min-max). Class-macro mIoU; *OOF* = τ chosen on 4 folds (crc32(image_id) % 5) and applied to the 5th; "
         "*@0.55* = BRACE's fixed τ. R2 = BRACE's DenseCRF. Δ = row − V3 (same seed, same post-processing), "
         "image-bootstrap 95% CI. Exploratory: V3 was selected earlier on the A3 seed, so the A3 rows are not a fresh "
         "test of the cap; the SCLIP rows are.\n",
         "Variants: " + "; ".join(f"**{v}** {LABEL[v]}" for v in VARS) + ".\n"]
    for ds in a.sets:
        S = cc.load_gen(WORK / f"{ds}_scores"); n = len(S["grp"])
        Z = [np.load(f) for f in sorted((WORK / f"{ds}_scores").glob("*.npz"))]
        ex = np.array([int(z["img_maxdiff"]) == 0 for z in Z]); md = np.array([float(z["img_meandiff"]) for z in Z])
        L.append(f"## {NAME[ds]} ({n} images)\n")
        L.append(f"Regeneration: {ex.sum()}/{n} images bit-identical to the stored image; mean |Δpixel| over the rest "
                 f"{np.nanmean(md[~ex]) if (~ex).any() else 0:.2f}/255 (GPU kernel nondeterminism under shared-GPU "
                 "load; all variants and both seeds of an image come from the same capture). BRACE V3+R2 @0.55 from "
                 f"this capture: {cc.miou(S['base|V3+R2|raw'][:, j55][:, None], S['cls'], np.ones(n, bool))[0]:.3f}.\n")
        for post in ("+R2", ""):
            L.append(f"**{'with R2 (DenseCRF)' if post else 'raw maps, no CRF'}**\n")
            L.append("| variant | A3 OOF | A3 @0.55 | Δ vs V3 (A3, OOF) | SCLIP OOF | SCLIP @0.55 | Δ vs V3 (SCLIP, OOF) |")
            L.append("|---|---:|---:|---|---:|---:|---|")
            ref = {s: cc.oof(S, S[f"{s}|V3{post}|raw"]) for s in SEEDS}
            for v in VARS:
                cells = []
                for s in SEEDS:
                    X = S[f"{s}|{v}{post}|raw"]; m, P, _ = cc.oof(S, X)
                    fixed = cc.miou(X[:, (j55 if post else i55)][:, None], S["cls"], np.ones(n, bool))[0]
                    if v == "V3":
                        d = "—"
                    else:
                        lo, hi = cc.boot(S, P, ref[s][1]); d = f"{m - ref[s][0]:+.3f} ({lo:+.3f}…{hi:+.3f})"
                    cells += [f"{m:.3f}", f"{fixed:.3f}", d]
                lab = f"**{v}**" if v == "V3" else v
                L.append(f"| {lab} {LABEL[v]} | " + " | ".join(cells) + " |")
            L.append("")
        # per class, R2 OOF, SCLIP seed
        L.append("Per class, SCLIP seed, +R2 OOF (V1 / V2core / V3 / V4):\n")
        P = {v: cc.oof(S, S[f"clip|{v}+R2|raw"])[1] for v in ("V1", "V2core", "V3", "V4")}
        parts = []
        for c in np.unique(S["cls"]):
            sel = S["cls"] == c
            parts.append(f"{c} " + " / ".join(f"{cc.miou(P[v][:, None], S['cls'], sel)[0]:.2f}" for v in P))
        L.append(", ".join(parts) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    (ROOT / "outputs/segmentation/sg_real_atlas/analysis/gen_cap_ablation.md").write_text(txt)


# ------------------------------------------------------------------ late fusion (gen_fusion_prereg.md)
FUSE = ("geo", "mean", "min", "a", "c")


def fused(z):
    a = st.mm(z["maps/base"][VARS.index("V3")]); c = st.mm(z["maps/clip"][VARS.index("V3")])
    return {"geo": st.mm(np.sqrt(a * c)), "mean": st.mm((a + c) / 2), "min": st.mm(np.minimum(a, c)), "a": a, "c": c}


def fuse_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth); res = {"area": float(truth.mean()), "cls": str(z["cls"])}
    for k, m in fused(z).items():
        x = st.up(m, truth.shape)
        res[f"{k}|raw"] = cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other)
        res[f"{k}+R2|raw"] = cc.counts(np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in cc.THR_CRF]).astype(bool),
                                       truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def fuse_score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}_fusion_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / ds).glob("*.npz")) if not f.stem.endswith(".tmp")]
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(fuse_one, jobs, chunksize=2), 1):
                if n % 200 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def fuse_report(a):
    j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    lab = {"geo": "geo: sqrt(a·c)", "mean": "mean: (a+c)/2", "min": "min(a, c)", "a": "a = BRACE (cross-attention seed)",
           "c": "c = CLIP-seeded BRACE"}
    S = {ds: cc.load_gen(WORK / f"{ds}_fusion_scores") for ds in ("voc", "coco")}
    one = lambda ds: np.ones(len(S[ds]["grp"]), bool)
    fx = lambda ds, k: S[ds][f"{k}+R2|raw"][:, j55]
    m55 = lambda ds, k: cc.miou(fx(ds, k)[:, None], S[ds]["cls"], one(ds))[0]
    L = ["# Late fusion of cross-attention-seeded and CLIP-seeded BRACE (generated sets)\n",
         "Pre-registered in `gen_fusion_prereg.md`, committed before scoring. a, c = the capped-hierarchy maps of the two "
         "seeds from one capture per image; fused map min-max normalised, τ 0.55, then the frozen DenseCRF R2. Δ with "
         "image-bootstrap 95% CI.\n"]
    for ds in ("voc", "coco"):
        L += [f"## {NAME[ds]} ({len(S[ds]['grp'])} images)\n",
              "| map | mIoU @0.55 + R2 | prec / rec | Δ vs a (BRACE) | Δ vs c (CLIP-seeded) | OOF + R2 | raw @0.55 |",
              "|---|---:|---|---|---|---:|---:|"]
        i55 = int(np.flatnonzero(np.isclose(cc.THR, TAU))[0])
        for k in FUSE:
            cells = []
            for r in ("a", "c"):
                if k == r:
                    cells.append("—")
                else:
                    lo, hi = cc.boot(S[ds], fx(ds, k), fx(ds, r)); cells.append(f"{m55(ds, k) - m55(ds, r):+.3f} ({lo:+.3f}…{hi:+.3f})")
            pr = cc.prf(fx(ds, k)); oo = cc.oof(S[ds], S[ds][f"{k}+R2|raw"])[0]
            raw = cc.miou(S[ds][f"{k}|raw"][:, i55][:, None], S[ds]["cls"], one(ds))[0]
            L.append(f"| {lab[k]} | {m55(ds, k):.3f} | {pr[0]:.3f} / {pr[1]:.3f} | {cells[0]} | {cells[1]} | {oo:.3f} | {raw:.3f} |")
        L.append("")
        L.append("Per class @0.55 + R2 (a / c / geo / mean / min): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(fx(ds, k)[:, None], S[ds]['cls'], S[ds]['cls'] == c)[0]:.2f}" for k in ("a", "c", "geo", "mean", "min"))
            for c in np.unique(S[ds]["cls"])) + "\n")
    ch = max(("geo", "mean", "min"), key=lambda k: m55("voc", k))
    lo, hi = cc.boot(S["coco"], fx("coco", ch), fx("coco", "c")); d = m55("coco", ch) - m55("coco", "c")
    L.append(f"## Registered test\n\nChosen on VOC-Sim: **{ch}** ({m55('voc', ch):.3f} vs c {m55('voc', 'c'):.3f}). COCO-Cap: "
             f"{ch} {m55('coco', ch):.3f} vs CLIP-seeded BRACE {m55('coco', 'c'):.3f}, Δ {d:+.3f} (CI {lo:+.3f}…{hi:+.3f}) → "
             f"**{'fusion helps' if lo > 0 else 'no gain over CLIP-seeded BRACE'}**.")
    txt = "\n".join(L) + "\n"; print(txt)
    (ROOT / "outputs/segmentation/sg_real_atlas/analysis/gen_fusion.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report", "fuse-score", "fuse-report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report, "fuse-score": fuse_score, "fuse-report": fuse_report}[a.cmd](a)
