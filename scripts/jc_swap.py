#!/usr/bin/env python3
"""BRACE half, scoring and report of the seed x propagation swap (runs/17_seed_walk_swap/PREREG.md).

capture (GPU)  one live generation per image with the frozen runtime; the CA seed (frozen A3) and NERVE's CLIP seed
               (from scripts/jc_swap_nerve.py seed, min-max) are propagated with the frozen fine-only (V0) and capped
               hierarchy (V3) code:  <BRACE_CACHE_DIR>/jc_swap/cap/<ds>/<id>.npz
score   (CPU)  8 cells = {ca, clip} x {seed, V0, V3, walk}; raw and + DenseCRF at 17 thresholds
report  (CPU)  runs/17_seed_walk_swap/result.md
"""
from __future__ import annotations

import argparse, csv, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

SWAP = CACHE / "jc_swap"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
SEEDS = ("ca", "clip")
MAPS = ("seed", "V0", "V3")
PROPS = ("seed", "V0", "V3", "walk")
PLAB = {"seed": "none (seed alone)", "V0": "BRACE fine-only", "V3": "BRACE capped hierarchy", "walk": "NERVE walk"}
SLAB = {"ca": "cross-attention seed (BRACE)", "clip": "CLIP seed (NERVE)"}
THR = np.round(0.30 + 0.025 * np.arange(17), 3)
OUT = ROOT / "runs/17_seed_walk_swap"
NSIDE = ROOT / "outputs/segmentation/comparisons/n_side/nerve_sd15_A/per_item_counts.csv"


def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    sdir = SWAP / ("nerve_seed" + a.tag) / a.dataset
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:                       # smoke runs: only the images the seed smoke produced
        rows = [r for r in rows if (sdir / f"{r['image_id']}.npz").exists()][: a.max_images]
    runtime = sg27_runtime.make_runtime(a.model)
    out = SWAP / ("cap" + a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        zs = np.load(sdir / f"{iid}.npz")
        if str(zs["cls"]).replace("-", " ") != cls.replace("-", " "):
            raise ValueError(f"class mismatch for {iid}")
        clip_seed = torch.from_numpy(zs["seed64"][0]).float()
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            sd = {"ca": seed.float().clone(), "clip": cc.norm(clip_seed.to(seed.device), "mm")}
            maps = []
            for s in SEEDS:
                v, _ = sg_combined_v2.aggregate(sa, sd[s])
                maps.append(np.stack([sd[s].cpu().numpy()] + [v[k].float().cpu().numpy() for k in MAPS[1:]])); del v
            maps = np.stack(maps).astype(np.float32)
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        img_mean = float(np.abs(img.astype(int) - stored["image"].astype(int)).mean())
        c = maps[SEEDS.index("ca")]
        chk = max(float(np.abs(st.mm(c[MAPS.index("V3")]) - st.mm(stored["map/base"][3])).max()),
                  float(np.abs(st.mm(c[MAPS.index("V0")]) - st.mm(stored["map/base"][2])).max()))
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps, seed_ca=maps[0, 0], cls=cls, check=chk, img_maxdiff=img_diff, img_meandiff=img_mean,
                 truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[swap:{a.dataset}] {iid} {cls} img {img_diff} check {chk:.1e} {time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def score_one(job):
    f, w, o = Path(job[0]), Path(job[1]), Path(job[2])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); zw = np.load(w); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "check": float(z["check"]),
           "img_maxdiff": int(z["img_maxdiff"]), "img_meandiff": float(z["img_meandiff"])}
    for si, s in enumerate(SEEDS):
        for p in PROPS:
            m64 = zw[f"walk_{s}"] if p == "walk" else z["maps"][si, MAPS.index(p)]
            x = st.up(st.mm(m64), truth.shape)
            res[f"{s}|{p}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{s}|{p}+R2|raw"] = cc.counts(ms, truth, valid, other)
    # side N's sanity convention for the NERVE cell: bilinear upsample, then min-max, > 0.5
    u = F.interpolate(torch.from_numpy(zw["walk_clip"].astype(np.float32))[None, None], size=truth.shape, mode="bilinear")[0, 0].numpy()
    m = st.mm(u) > 0.5
    res["nside_tp_fp_fn"] = np.array([(m & truth).sum(), (m & ~truth).sum(), (~m & truth).sum()], np.int64)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = SWAP / f"scores{a.tag}" / ds; sc.mkdir(parents=True, exist_ok=True)
        jobs = []
        for f in sorted((SWAP / ("cap" + a.tag) / ds).glob("*.npz")):
            w = SWAP / ("walk" + a.tag) / ds / f.name
            if f.stem.endswith(".tmp"):
                continue
            if not w.exists():
                raise FileNotFoundError(f"walk output missing for {f.stem}")
            jobs.append((str(f), str(w), str(sc / f.name)))
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    S = {ds: cc.load_gen(SWAP / f"scores{a.tag}" / ds) for ds in ("voc", "coco")}
    Z = {ds: [np.load(f) for f in sorted((SWAP / f"scores{a.tag}" / ds).glob("*.npz"))] for ds in S}
    n = {ds: len(S[ds]["grp"]) for ds in S}
    one = {ds: np.ones(n[ds], bool) for ds in S}
    exact = {ds: np.array([int(z["img_maxdiff"]) == 0 for z in Z[ds]]) for ds in S}
    ti = lambda t: int(np.flatnonzero(np.isclose(THR, t))[0])

    def mi(ds, P, s=None):
        s = one[ds] if s is None else s
        return cc.miou(P[:, None], S[ds]["cls"], s)[0] if s.any() else float("nan")

    def sel(ds, key):            # primary: VOC-selected threshold (OOF on VOC-Sim, full-VOC-best transferred to COCO-Cap)
        X = S["voc"][f"{key}|raw"]
        if ds == "voc":
            return cc.oof(S["voc"], X)[1]
        return S["coco"][f"{key}|raw"][:, int(np.argmax(cc.miou(X, S["voc"]["cls"], one["voc"])))]

    def best_t(key):
        return float(THR[int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))])

    def boot(ds, fn, nb=2000, sub=None):
        """fn(index array) -> statistic; resamples images."""
        rng = np.random.default_rng(0); idx = np.arange(n[ds]) if sub is None else np.flatnonzero(sub); d = []
        for _ in range(nb):
            d.append(fn(rng.choice(idx, len(idx))))
        return np.percentile(d, [2.5, 97.5])

    def stat(ds, P):             # class-macro mIoU on an index sample
        return lambda s: cc.miou(P[s][:, None], S[ds]["cls"][s], np.ones(len(s), bool))[0]

    def contrast(ds, terms, sub=None):
        """terms: list of (sign, counts array). Returns text and (point, lo, hi)."""
        fs = [(sg, stat(ds, P)) for sg, P in terms]
        idx = np.arange(n[ds]) if sub is None else np.flatnonzero(sub)
        pt = sum(sg * f(idx) for sg, f in fs)
        lo, hi = boot(ds, lambda s: sum(sg * f(s) for sg, f in fs), sub=sub)
        return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", (pt, lo, hi)

    L = ["# Seed × propagation swap between BRACE and NERVE (`PREREG.md`)\n",
         "`scripts/jc_swap.py`, `scripts/jc_swap_nerve.py`. Seeds: **CA** = BRACE's cross-attention seed (live "
         "generation); **CLIP** = NERVE's CLIP ViT-L/14 seed for {class word} + NERVE's background words (stored image). "
         "Propagations: BRACE fine-only and capped hierarchy (generation self-attention, frozen code); **NERVE walk** = "
         "NERVE's released refinement, unchanged (stored image, one noised pass, closed-form walk). CA × hierarchy is "
         "BRACE; CLIP × walk is NERVE (side N's variant A). Class-macro mIoU. *VOC-sel.* = threshold selected on "
         "VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold applied unchanged on COCO-Cap). "
         "Primary number: VOC-sel. with DenseCRF. Δ with image-bootstrap 95% interval (2000). CLIP-seed cells use a "
         "second model (CLIP) and are not SD-only.\n"]
    for ds in ("voc", "coco"):
        chk = np.array([float(z["check"]) for z in Z[ds]])
        L.append(f"## {NAME[ds]} ({n[ds]} images)\n")
        L.append(f"Regeneration: {exact[ds].sum()}/{n[ds]} images bit-identical to the stored image; BRACE maps vs the "
                 f"stored capture: median max |Δ| {np.median(chk):.1e}.\n")
        for post, plab in (("+R2", "VOC-sel. threshold, with DenseCRF (primary)"), ("", "VOC-sel. threshold, raw")):
            L.append(f"**{plab}**\n"); L.append("| Propagation | " + " | ".join(SLAB[s] for s in SEEDS) + " | CLIP − CA |")
            L.append("|---|---:|---:|---|")
            for p in PROPS:
                P = {s: sel(ds, f"{s}|{p}{post}") for s in SEEDS}
                L.append(f"| {PLAB[p]} | " + " | ".join(f"{mi(ds, P[s]):.3f}" for s in SEEDS) +
                         f" | {contrast(ds, [(1, P['clip']), (-1, P['ca'])])[0]} |")
            L.append("")
        L.append("**Fixed threshold 0.5, raw (side N's sanity convention)**\n")
        L.append("| Propagation | " + " | ".join(SLAB[s] for s in SEEDS) + " |"); L.append("|---|---:|---:|")
        for p in PROPS:
            L.append(f"| {PLAB[p]} | " + " | ".join(f"{mi(ds, S[ds][f'{s}|{p}|raw'][:, ti(0.5)]):.3f}" for s in SEEDS) + " |")
        L.append("")
    L.append("## Thresholds selected on full VOC-Sim\n")
    L.append("| Cell | raw | with DenseCRF |"); L.append("|---|---:|---:|")
    for s in SEEDS:
        for p in PROPS:
            L.append(f"| {SLAB[s]} × {PLAB[p]} | {best_t(f'{s}|{p}'):.3f} | {best_t(f'{s}|{p}+R2'):.3f} |")
    L.append("\n## Registered contrasts (VOC-sel. threshold, with DenseCRF)\n")
    L.append("| ID | Contrast | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images only | COCO-Cap, raw |")
    L.append("|---|---|---|---|---|---|")
    C = {ds: {f"{s}|{p}{post}": sel(ds, f"{s}|{p}{post}") for s in SEEDS for p in ("V3", "walk") for post in ("+R2", "")}
         for ds in S}
    defs = [("P-CLIP", "walk − hierarchy, CLIP seed", lambda c, q: [(1, c[f"clip|walk{q}"]), (-1, c[f"clip|V3{q}"])]),
            ("P-CA", "walk − hierarchy, CA seed", lambda c, q: [(1, c[f"ca|walk{q}"]), (-1, c[f"ca|V3{q}"])]),
            ("S-H", "CLIP − CA seed, hierarchy", lambda c, q: [(1, c[f"clip|V3{q}"]), (-1, c[f"ca|V3{q}"])]),
            ("S-W", "CLIP − CA seed, walk", lambda c, q: [(1, c[f"clip|walk{q}"]), (-1, c[f"ca|walk{q}"])]),
            ("I", "interaction (P-CLIP − P-CA)", lambda c, q: [(1, c[f"clip|walk{q}"]), (-1, c[f"clip|V3{q}"]),
                                                               (-1, c[f"ca|walk{q}"]), (1, c[f"ca|V3{q}"])])]
    R = {}
    for cid, lab, fn in defs:
        v = contrast("voc", fn(C["voc"], "+R2")); c = contrast("coco", fn(C["coco"], "+R2")); R[cid] = c[1]
        ce = contrast("coco", fn(C["coco"], "+R2"), sub=exact["coco"]); cr = contrast("coco", fn(C["coco"], ""))
        L.append(f"| {cid} | {lab} | {v[0]} | {c[0]} | {ce[0]} | {cr[0]} |")
    seed_ok = R["S-H"][1] > 0 and R["S-W"][1] > 0; walk_ok = R["P-CLIP"][1] > 0 and R["P-CA"][1] > 0
    L.append(f"\n**Reading rule (COCO-Cap).** Seed explains NERVE's lead (S-H and S-W both above zero): "
             f"**{'yes' if seed_ok else 'no'}**. Walk explains NERVE's lead (P-CLIP and P-CA both above zero): "
             f"**{'yes' if walk_ok else 'no'}**." + ("" if seed_ok or walk_ok else " Result: mixed.") + "\n")
    L.append("## Fidelity checks\n")
    i55 = ti(0.55)
    L.append(f"- BRACE cell (CA × hierarchy, 0.55, DenseCRF): {mi('voc', S['voc']['ca|V3+R2|raw'][:, i55]):.3f} VOC-Sim, "
             f"{mi('coco', S['coco']['ca|V3+R2|raw'][:, i55]):.3f} COCO-Cap (paper: 0.830 / 0.696).")
    ns = {r["key"]: (int(r["tp_t0.5"]), int(r["fp_t0.5"]), int(r["fn_t0.5"])) for r in csv.DictReader(open(NSIDE))}
    for ds in ("voc", "coco"):
        mine = np.stack([z["nside_tp_fp_fn"] for z in Z[ds]]); theirs = np.array([ns[g] for g in S[ds]["grp"]])
        def macro(T):
            return np.mean([T[S[ds]["cls"] == c, 0].sum() / max(T[S[ds]["cls"] == c].sum(), 1) for c in np.unique(S[ds]["cls"])])
        iou = lambda T: T[:, 0] / np.maximum(T.sum(1), 1)
        L.append(f"- NERVE cell (CLIP × walk, 0.5 raw, side N's convention), {NAME[ds]}: {macro(mine):.4f} here vs "
                 f"{macro(theirs):.4f} from side N's committed counts; per-image IoU: identical counts on "
                 f"{int((mine == theirs).all(1).sum())}/{n[ds]} images, max |ΔIoU| {np.abs(iou(mine) - iou(theirs)).max():.4f}, "
                 f"mean |ΔIoU| {np.abs(iou(mine) - iou(theirs)).mean():.5f}.")
    L.append("")
    for ds in ("voc", "coco"):
        L.append(f"Per class, {NAME[ds]}, VOC-sel. + DenseCRF (CA×hierarchy / CA×walk / CLIP×hierarchy / CLIP×walk): " + ", ".join(
            f"{c} " + " / ".join(f"{mi(ds, C[ds][k], S[ds]['cls'] == c):.2f}" for k in ("ca|V3+R2", "ca|walk+R2", "clip|V3+R2", "clip|walk+R2"))
            for c in np.unique(S[ds]["cls"])) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=16); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"capture": capture, "score": score, "report": report}[a.cmd](a)
