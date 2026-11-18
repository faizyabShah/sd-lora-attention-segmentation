#!/usr/bin/env python3
"""runs/25_token_ladder_nr/PREREG.md: stage ladder, class word (W) vs OVAM optimized token (T), corrected data.

Capture reuses runs/15's LadderRecorder unchanged; rows come from runs/20's corrected prompts; safety checker off.
capture (GPU)  <BRACE_CACHE_DIR>/jc_ladder_nr/<ds>/<id>.npz: maps [4 seeds (wA0,wA3,tA0,tA3), 3 maps (seed,V0,V3), 64, 64]
score   (CPU)  counts raw and + DenseCRF R2 at runs/20 thresholds (0.30-0.70, step 0.025)
report  (CPU)  runs/25_token_ladder_nr/result{tag}.md
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
import jc_ladder as jl  # noqa: E402
import jc_newrerun as nr  # noqa: E402
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_ladder_nr"
NR = CACHE / "jc_newrerun"
OUT = ROOT / "runs/25_token_ladder_nr"
SEEDS = jl.SEEDS                          # wA0, wA3, tA0, tA3
MAPS = ("seed", "V0", "V3")
THR = nr.THR
T55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
STAGES = (("seed", "CA"), ("V0", "CA + fine SA"), ("V3", "CA + hierarchy (BRACE)"))


@torch.no_grad()
def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    R = nr.rows(a.dataset, a.shard, a.num_shards, a.max_images, a.ids)
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=jl.LadderRecorder)
    jl.LadderRecorder.enabled = ("A0", "A3"); runtime["pipe"].safety_checker = None
    released = np.load(jl.TOKENS, allow_pickle=True).item()
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for row in R:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        jl.LadderRecorder.token_states = jl.token_states(runtime["pipe"], released, cls)
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            word = rec.seeds(); tok = rec.token_seeds()
            if float((word["selected/A3"].float() - seed.float()).abs().max()) > 0:
                raise ValueError("ladder: A3 seed differs from the frozen runtime's seed")
            sd = {"wA0": word["selected/A0"], "wA3": word["selected/A3"], "tA0": tok["A0"], "tA3": tok["A3"]}
            maps = []
            for s in SEEDS:
                x = sd[s].float().clone(); v, _ = sg_combined_v2.aggregate(sa, x)
                maps.append(np.stack([x.cpu().numpy(), v["V0"].float().cpu().numpy(), v["V3"].float().cpu().numpy()]))
                del v
            maps = np.stack(maps).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        prev = NR / a.dataset / f"{iid}.npz"; idiff, v3d = float("nan"), float("nan")
        if prev.exists():
            z = np.load(prev); idiff = int(np.abs(img.astype(int) - z["image"].astype(int)).max())
            v3d = float(np.abs(st.mm(maps[1, 2]) - st.mm(z["maps"][nr.SEEDS.index("CAg"), nr.CELLS.index("V3")].astype(np.float32))).max())
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=row["label"], token_class=cls, changed=row["changed"],
                 img_vs_run20=idiff, v3_vs_run20=v3d, truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[lnr:{a.dataset}] {iid} {row['label']} changed={row['changed']} img-vs-r20 {idiff} V3-vs-r20 {v3d:.1e} "
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
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"]),
           "img_vs_run20": float(z["img_vs_run20"]), "v3_vs_run20": float(z["v3_vs_run20"])}
    for si, s in enumerate(SEEDS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][si, mi].astype(np.float32)), truth.shape)
            res[f"{s}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{s}|{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 25 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    mi = lambda ds, P: cc.miou(P[:, None], S[ds]["cls"], one[ds])[0]
    tb = lambda key: int(np.argmax(cc.miou(S["voc"][key], S["voc"]["cls"], one["voc"])))

    def at(ds, key, mode):
        X = S[ds][key]
        if mode == "0.55":
            return X[:, T55]
        return cc.oof(S["voc"], X)[1] if ds == "voc" else X[:, tb(key)]

    def dci(ds, Pa, Pb):
        lo, hi = cc.boot(S[ds], Pa, Pb, n=a.boot); d = mi(ds, Pa) - mi(ds, Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo

    key = lambda q, w, m, crf: f"{q}{w}|{m}{'+R2' if crf else ''}|raw"
    L = ["# Stage ladder: class word vs OVAM's optimized token, corrected data (`PREREG.md`)\n",
         "`scripts/jc_ladder_nr.py`. One live generation per image (frozen runtime, runs/20 corrected prompts, safety "
         "checker off); both query columns and all stages from that generation. **Class word** = frozen attribution "
         "(BRACE). **Optimized token** = OVAM's released VOC-Sim token (few-shot: needs annotated images; the same "
         "VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold "
         f"applied unchanged on COCO-Cap. Δ with paired image-bootstrap 95% interval ({a.boot}). Class-word results are "
         "in-sample for the method design. Supersedes runs/15 (old prompts).\n", "## Data checks\n"]
    for ds in S:
        im = S[ds].get("img_vs_run20"); Z = [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")]
        im = np.array([float(z["img_vs_run20"]) for z in Z]); v3 = np.array([float(z["v3_vs_run20"]) for z in Z])
        L.append(f"- {NAME[ds]}: {n[ds]} images; image identical to the runs/20 capture on {(im == 0).sum()}/{n[ds]}; "
                 f"class-word BRACE map vs runs/20: median max |Δ| {np.nanmedian(v3):.1e}.")
    for crf in (True, False):
        L.append(f"\n## Stage table, {'with DenseCRF (primary)' if crf else 'raw (secondary)'}, SA-guided layer weights\n")
        L.append("| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | "
                 "COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |")
        L.append("|---|" + "---:|" * 8)
        for m, lab in STAGES:
            L.append(f"| {lab} | " + " | ".join(f"{mi(ds, at(ds, key(q, 'A3', m, crf), mode)):.3f}"
                                               for ds in S for mode in ("sel", "0.55") for q in "wt") + " |")
    L.append("\n## Uniform layer weights, with DenseCRF (secondary)\n")
    L.append("| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | "
             "COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |")
    L.append("|---|" + "---:|" * 8)
    for m, lab in STAGES:
        L.append(f"| {lab} | " + " | ".join(f"{mi(ds, at(ds, key(q, 'A0', m, True), mode)):.3f}"
                                           for ds in S for mode in ("sel", "0.55") for q in "wt") + " |")
    L.append("\n## Registered comparisons (with DenseCRF)\n")
    L.append("| ID | Comparison | Query | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 | Verdict (0.55) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tid, lab, ma, mb in (("F", "CA + fine − CA", "V0", "seed"), ("H", "CA + hierarchy − CA + fine", "V3", "V0"),
                             ("—", "CA + hierarchy − CA", "V3", "seed")):
        for q, qn in (("w", "class word"), ("t", "optimized token")):
            c = {(ds, mode): dci(ds, at(ds, key(q, "A3", ma, True), mode), at(ds, key(q, "A3", mb, True), mode))
                 for mode in ("sel", "0.55") for ds in S}
            v = "—"
            if tid != "—":
                v = "holds" if (c[("coco", "0.55")][2] > 0 and c[("voc", "0.55")][1] >= -0.005) else "does not hold"
            L.append(f"| {tid} | {lab} | {qn} | {c[('voc', 'sel')][0]} | {c[('coco', 'sel')][0]} | {c[('voc', '0.55')][0]} | "
                     f"{c[('coco', '0.55')][0]} | {v} |")
    for m, lab in STAGES:
        c = {(ds, mode): dci(ds, at(ds, key("t", "A3", m, True), mode), at(ds, key("w", "A3", m, True), mode))
             for mode in ("sel", "0.55") for ds in S}
        L.append(f"| Q | token − word: {lab} | — | {c[('voc', 'sel')][0]} | {c[('coco', 'sel')][0]} | {c[('voc', '0.55')][0]} | "
                 f"{c[('coco', '0.55')][0]} | descriptive |")
    L.append("\nRule (F, H, per query): holds if the COCO-Cap Δ at 0.55 is positive with interval above zero and the VOC-Sim "
             "Δ at 0.55 is not below −0.005.\n")
    for ds in S:
        k_ = key("w", "A3", "V3", True); kt = key("t", "A3", "V3", True); cl = S[ds]["cls"]
        L.append(f"Per class, {NAME[ds]}, CA + hierarchy + DenseCRF @0.55 (word / token): " + ", ".join(
            f"{c} {cc.miou(at(ds, k_, '0.55')[:, None], cl, cl == c)[0]:.2f} / {cc.miou(at(ds, kt, '0.55')[:, None], cl, cl == c)[0]:.2f}"
            for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--ids", default=""); p.add_argument("--tag", default="")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"]); p.add_argument("--tag", default="")
        p.add_argument("--workers", type=int, default=16); p.add_argument("--shard", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
