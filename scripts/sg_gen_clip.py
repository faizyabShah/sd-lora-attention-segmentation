#!/usr/bin/env python3
"""SCLIP baselines and mask-reliability study on the generated sets (VOC-Sim 490, COCO-Cap 707), 2026-11-09.
BRACE (V3+R2 @ τ 0.55) stays frozen; nothing here changes it.

sclip   (GPU)  SCLIP ViT-B/16 (scripts/sg_clip_dense.py port) on every stored generated image, two closed
               vocabularies: VOC21 (logit 65, prob_thd .1, area_thd .1) and COCO-Object (repo cls_coco_object.txt,
               logit 50, prob_thd .1). Stores per image: the repo prediction == target (binary), the target-class
               probability (full size, fp16) and its 64x64 pool. Label-free.
capture (GPU)  Regenerates every image with the frozen live pipeline (deterministic; checked against the stored
               image and V3), keeps the SA graphs and stores in sg_class_comp format: map/base (A3 seed through
               the frozen paths, replay of the stored maps), map/clip (SCLIP VOC21 target probability, mm, through
               the same paths), plus per-call fine (P1) and C16 maps of the A3 seed [11,64,64] for call stability.
score   (CPU)  Class-given binary protocol as BRACE: SCLIP binary and binary+R2 (no τ); SCLIP probability with
               τ (OOF like BRACE) and +R2.
report  (CPU)  BRACE vs SCLIP table (+ SCLIP seed → SD C16 / V3+R2 once captured), per class, FP/FN, runtime.
reliab  (CPU)  Image- and pixel-level reliability of the frozen BRACE mask (see voc_gen_reliab_prereg.md).
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402

AN = ROOT / "outputs/segmentation/sg_real_atlas/analysis"
STORED = {"voc": Path("/home/shafa/sg_comp/vocsim_maps"), "coco": Path("/home/shafa/sg_comp/cococap_maps")}
BASESC = {"voc": Path("/home/shafa/sg_comp/vocsim_scores"), "coco": Path("/home/shafa/sg_comp/cococap_scores")}
WORK = Path("/home/shafa/sg_comp/gen_clip")
COCO_LINES = (ROOT / "scripts/sclip_cls_coco_object.txt").read_text().strip().split("\n")
VOC_ORDER = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining-table", "dog",
             "horse", "motorbike", "person", "potted-plant", "sheep", "sofa", "train", "tv"]
TO_COCO = {"aeroplane": "airplane", "motorbike": "motorcycle", "sofa": "couch", "dining-table": "dining table",
           "potted-plant": "potted plant"}
TAU = 0.55
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def canon(c):
    return c.replace(" ", "-")


def voc_k(c):
    return VOC_ORDER.index(canon(c)) + 1


def coco_k(c):
    c = canon(c); n = TO_COCO.get(c, c)
    return [l.split(", ")[0] for l in COCO_LINES].index(n)


def files(ds):
    return sorted(f for f in STORED[ds].glob("*.npz") if not f.stem.endswith(".tmp"))


# ------------------------------------------------------------------ SCLIP (GPU)
def sclip(a):
    import sg_clip_dense as cd
    dev = torch.device("cuda"); M = cd.DenseCLIP("B16", dev)
    n21, i21 = cd.split([cd.VOC21_BG] + cd.VOC20_NAMES); nC, iC = cd.split(COCO_LINES)
    T21, TC = M.text(n21), M.text(nC)
    for ds in a.sets:
        out = WORK / f"sclip_{ds}"; out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter(); n = 0
        for f in files(ds):
            o = out / f.name
            if o.exists():
                continue
            z = np.load(f); img = z["image"]; cls = str(z["cls"]); k21, kC = voc_k(cls), coco_k(cls)
            t1 = time.perf_counter()
            p21, P21 = cd.postprocess(M.logits(img, T21), i21, 65, 0.1, 0.1)
            pC, PC = cd.postprocess(M.logits(img, TC), iC, 50, 0.1, None)
            torch.cuda.synchronize(); dt = time.perf_counter() - t1
            to64 = lambda x: F.adaptive_avg_pool2d(x[None, None].float(), (64, 64))[0, 0].cpu().numpy()
            np.savez(o, bin21=(p21 == k21).cpu().numpy(), binC=(pC == kC).cpu().numpy(),
                     prob21=P21[k21].cpu().numpy().astype(np.float16), probC=PC[kC].cpu().numpy().astype(np.float16),
                     prob21_64=to64(P21[k21]), probC_64=to64(PC[kC]), cls=cls, k21=k21, kC=kC, sec=dt)
            n += 1
        print(f"[sclip:{ds}] {n} images, {(time.perf_counter() - t0) / max(n, 1):.2f}s/img incl. io", flush=True)


# ------------------------------------------------------------------ SD capture (GPU)
def capture(a):
    from scripts import sg27_runtime
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    runtime = sg27_runtime.make_runtime(a.model)
    out = WORK / f"sd_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        P = torch.from_numpy(np.load(WORK / f"sclip_{a.dataset}" / f"{iid}.npz")["prob21_64"]).float()
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, base, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed = seed.float()
            maps = {"base": cc.propagate(sa, seed), "clip": cc.propagate(sa, cc.norm(P.to(seed.device), "mm"))}
            c64 = seed.reshape(-1); c16 = _resize(c64, 64, 16).reshape(-1); pc_p1, pc_c16 = [], []
            for call in range(11):
                a64 = _attention(sa[64][call], 64); a16 = _attention(sa[16][call], 16)
                pc_p1.append(_apply(a64, c64).mean(0).reshape(64, 64))
                pc_c16.append(_apply(a64, _resize(_apply(a16, c16).mean(0), 16, 64).reshape(-1)).sum(0).reshape(64, 64) / 8)
                del a64, a16
            pc_p1 = torch.stack(pc_p1).cpu().numpy(); pc_c16 = torch.stack(pc_c16).cpu().numpy()
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        v3chk = float(np.abs(st.mm(maps["base"][3]) - st.mm(stored["map/base"][3])).max())
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, **{f"map/{n}": v for n, v in maps.items()}, percall_P1=pc_p1.astype(np.float16),
                 percall_C16=pc_c16.astype(np.float16), cls=cls, check=v3chk, img_maxdiff=img_diff, truth=truth,
                 image=img, set=a.dataset)
        os.replace(tmp, o)
        print(f"[gen-sd:{a.dataset}] {iid} {cls} img maxdiff {img_diff} V3 check {v3chk:.1e} "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ SCLIP scoring (CPU)
def score_one(job):
    f, sc = Path(job[0]), Path(job[1]); o = sc / f.name
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(STORED[job[2]] / f.name); c = np.load(f)
    img, truth = z["image"], z["truth"].astype(bool); valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"])}
    for v in ("21", "C"):
        b = c[f"bin{v}"].astype(bool); p = st.mm(c[f"prob{v}"].astype(np.float32))
        res[f"sclip{v}|bin|raw"] = cc.counts(b[None], truth, valid, other)
        res[f"sclip{v}|bin+R2|raw"] = cc.counts(apply_ovam_densecrf(img, b, st.CRF_SITE)[None], truth, valid, other)
        res[f"sclip{v}|prob|raw"] = cc.counts(p[None] >= cc.THR[:, None, None], truth, valid, other)
        res[f"sclip{v}|prob+R2|raw"] = cc.counts(np.stack([apply_ovam_densecrf(img, p >= t, st.CRF_SITE) for t in cc.THR_CRF]),
                                                 truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"sclip_{ds}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc), ds) for f in sorted((WORK / f"sclip_{ds}").glob("*.npz"))]
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(score_one, jobs, chunksize=4))
        print(f"[score:{ds}] {len(jobs)}", flush=True)


# ------------------------------------------------------------------ report (CPU)
def prf(P):
    tp, fp, fn = P[:, 0].sum(), P[:, 1].sum(), P[:, 2].sum()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def report(a):
    i55 = int(np.flatnonzero(np.isclose(cc.THR, TAU))[0]); j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    L = ["# SCLIP vs BRACE on generated images (class-given binary protocol, as in the paper)\n",
         "BRACE = frozen V3+R2 at τ 0.55 (paper protocol). SCLIP ViT-B/16 (our port, reproduces the repo on VOC val). "
         "*bin* = SCLIP's own closed-vocabulary prediction == target class (no τ); *prob* = SCLIP's target-class "
         "probability, min-max, thresholded like BRACE (OOF = τ chosen on 4 folds, crc32(image_id) % 5). R2 = the "
         "same DenseCRF as BRACE. SCLIP seed → SD = the SCLIP VOC21 target probability used as the seed of the frozen "
         "SD paths of the image's own generation run. Δ = row − BRACE, image-bootstrap 95% CI.\n"]
    for ds in a.sets:
        B = cc.load_gen(BASESC[ds]); S = cc.load_gen(WORK / f"sclip_{ds}_scores")
        assert list(B["grp"]) == list(S["grp"]), "image order mismatch"
        for k in S:
            if "|" in k:
                B[k] = S[k]
        sd = WORK / f"sd_{ds}_scores"
        if sd.exists() and len(list(sd.glob("*.npz"))) == len(B["grp"]):
            D = cc.load_gen(sd); assert list(D["grp"]) == list(B["grp"])
            for k in D:
                if k.startswith("clip|"):
                    B[k] = D[k]
                if k.startswith("base|"):
                    B["replay_" + k] = D[k]
        allp = np.ones(len(B["cls"]), bool); brace = B["base|V3+R2|raw"][:, j55]
        rows = [("BRACE V3+R2 @0.55 (paper)", brace, None), ("BRACE V3+R2 OOF", None, "base|V3+R2|raw"),
                ("SD fine-only P1+R2 OOF", None, "base|P1+R2|raw")]
        for v, vl in (("21", "VOC21 vocab"), ("C", "COCO-Object vocab")):
            rows += [(f"SCLIP {vl}, bin", B[f"sclip{v}|bin|raw"][:, 0], None),
                     (f"SCLIP {vl}, bin + R2", B[f"sclip{v}|bin+R2|raw"][:, 0], None),
                     (f"SCLIP {vl}, prob OOF", None, f"sclip{v}|prob|raw"),
                     (f"SCLIP {vl}, prob + R2 OOF", None, f"sclip{v}|prob+R2|raw")]
        if "clip|C16|raw" in B:
            rows += [("SCLIP seed → SD C16 OOF", None, "clip|C16|raw"), ("SCLIP seed → SD V3+R2 OOF", None, "clip|V3+R2|raw"),
                     ("SCLIP seed → SD P1+R2 OOF", None, "clip|P1+R2|raw")]
        L.append(f"## {NAME[ds]} ({len(B['cls'])} images)\n")
        L.append("| pipeline | mIoU | prec / rec | Δ vs BRACE @0.55 | CI |\n|---|---:|---|---:|---|")
        preds = {}
        for lbl, P, key in rows:
            if P is None:
                mi, P, _ = cc.oof(B, B[key])
            mi = cc.miou(P[:, None], B["cls"], allp)[0]; preds[lbl] = P; pr, rc = prf(P)
            if lbl.startswith("BRACE V3+R2 @"):
                L.append(f"| **{lbl}** | **{mi:.3f}** | {pr:.3f} / {rc:.3f} | — | — |"); m0 = mi
            else:
                lo, hi = cc.boot(B, P, brace)
                L.append(f"| {lbl} | {mi:.3f} | {pr:.3f} / {rc:.3f} | {mi - m0:+.3f} | {lo:+.3f}…{hi:+.3f} |")
        best = max((l for l in preds if l.startswith("SCLIP") and "→" not in l), key=lambda l: cc.miou(preds[l][:, None], B["cls"], allp)[0])
        L.append(f"\nPer class (BRACE @0.55 / best SCLIP-only = {best}" + (" / SCLIP seed → SD C16" if "SCLIP seed → SD C16 OOF" in preds else "") + "):\n")
        cells = []
        for c in np.unique(B["cls"]):
            s = B["cls"] == c; x = [cc.miou(preds[l][:, None], B["cls"], s)[0] for l in
                                    ["BRACE V3+R2 @0.55 (paper)", best] + (["SCLIP seed → SD C16 OOF"] if "SCLIP seed → SD C16 OOF" in preds else [])]
            cells.append(f"{c} " + " / ".join(f"{v:.2f}" for v in x))
        L.append(", ".join(cells))
        if "replay_base|V3+R2|raw" in B:
            r = cc.miou(B["replay_base|V3+R2|raw"][:, j55:j55 + 1], B["cls"], allp)[0]
            L.append(f"\nRegeneration replay: BRACE V3+R2 @0.55 from the new capture = {r:.3f} (stored {m0:.3f}).")
        secs = [float(np.load(f)["sec"]) for f in sorted((WORK / f"sclip_{ds}").glob("*.npz"))]
        L.append(f"\nSCLIP runtime: {np.mean(secs):.2f} s/image for both vocabularies (GPU, excl. CRF).\n")
    txt = "\n".join(L) + "\n"; (AN / "sclip_generated.md").write_text(txt); print(txt)


# ------------------------------------------------------------------ reliability (CPU)
def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg]); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def rel_one(job):
    ds, name = job; torch.set_num_threads(1)
    from scipy.ndimage import distance_transform_edt
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(STORED[ds] / name); c = np.load(WORK / f"sclip_{ds}" / name)
    sdf = WORK / f"sd_{ds}" / name; sd = np.load(sdf) if sdf.exists() else None
    img, truth = z["image"], z["truth"].astype(bool); hw = truth.shape; M = z["map/base"]
    v3 = st.up(st.mm(M[3]), hw); p1 = st.up(st.mm(M[1]), hw); c16 = st.up(st.mm(M[4]), hw); seed = st.up(st.mm(M[0]), hw)
    raw = v3 >= TAU; mask = apply_ovam_densecrf(img, raw, st.CRF_SITE).astype(bool)
    pk = st.mm(c["prob21"].astype(np.float32)); sb = c["bin21"].astype(bool)
    r = {"name": name, "cls": str(z["cls"]), "iou": iou(mask, truth), "tp": int((mask & truth).sum()),
         "fp": int((mask & ~truth).sum()), "fn": int((~mask & truth).sum())}
    # image-level signals (label-free); higher = predicted more reliable
    r["s_area"] = float(mask.mean())
    r["s_seedpeak"] = float(np.quantile(st.mm(M[0]), 0.99) - np.median(st.mm(M[0])))
    r["s_clip_iou"] = iou(mask, sb)
    r["s_clip_soft"] = float(pk[mask].mean() - pk[~mask].mean()) if mask.any() and (~mask).any() else 0.0
    r["s_conf"] = -float(((v3 > TAU - 0.1) & (v3 < TAU + 0.1)).mean() / max(mask.mean(), 1e-3))
    r["s_cf_iou"] = iou(p1 >= TAU, raw)
    r["s_c16_iou"] = iou(c16 >= TAU, raw)
    r["s_crf_iou"] = iou(raw, mask)
    if sd is not None:
        pc = np.stack([st.up(st.mm(x.astype(np.float32)), hw) >= TAU for x in sd["percall_P1"]])
        r["s_call_iou"] = float(np.mean([iou(pc[i], pc[j]) for i in range(11) for j in range(i + 1, 11)]))
    # pixel-level: which pixels of the BRACE mask are wrong?
    err = mask != truth
    bd = np.where(mask, distance_transform_edt(mask), distance_transform_edt(~mask))  # distance to the mask boundary
    U = {"u_boundary": -bd, "u_conf": -np.abs(v3 - TAU), "u_cf": np.abs(p1 - v3), "u_c16": np.abs(c16 - v3),
         "u_clip": np.abs(mask.astype(np.float32) - pk), "u_seed": np.abs(mask.astype(np.float32) - seed)}
    if sd is not None:
        U["u_call"] = np.stack([st.up(st.mm(x.astype(np.float32)), hw) for x in sd["percall_P1"]]).std(0)
    rng = np.random.default_rng(zlib.crc32(name.encode())); idx = rng.choice(err.size, min(err.size, 40000), replace=False)
    e = err.ravel()[idx]; b = bd.ravel()[idx]; bins = np.digitize(b, [2, 5, 10, 20])
    for k, u in U.items():
        x = u.ravel()[idx]; r[k] = auroc(x[e], x[~e])
        ws = [(auroc(x[e & (bins == q)], x[~e & (bins == q)]), (e & (bins == q)).sum()) for q in range(5)]
        ws = [(v, w) for v, w in ws if not np.isnan(v) and w > 0]
        r[k + "|strat"] = float(np.average([v for v, _ in ws], weights=[w for _, w in ws])) if ws else np.nan
    r["err_frac"] = float(err.mean())
    return r


def reliab(a):
    L = ["# Reliability of the frozen BRACE mask on generated images (pre-registered in `gen_reliab_prereg.md`)\n"]
    R = {}
    for ds in ("voc", "coco"):
        names = [f.name for f in files(ds)]
        with Pool(a.workers) as pool:
            R[ds] = pool.map(rel_one, [(ds, n) for n in names], chunksize=4)
    sig = [k for k in R["voc"][0] if k.startswith("s_")]
    cand = [s for s in sig if s not in ("s_area", "s_seedpeak", "s_clip_iou", "s_clip_soft")]
    ok = lambda s: all(s in r for r in R["coco"])
    cand = [s for s in cand if ok(s)]
    stats = {}
    for ds in ("voc", "coco"):
        rs = R[ds]; cls = np.array([r["cls"] for r in rs]); y = np.array([r["iou"] for r in rs])
        area = np.log(np.array([r["s_area"] for r in rs]) + 1e-4)
        miou_all = np.mean([y[cls == c].mean() for c in np.unique(cls)])
        tp = np.array([r["tp"] for r in rs]); fp = np.array([r["fp"] for r in rs]); fn = np.array([r["fn"] for r in rs])
        macro = lambda sel: np.mean([tp[sel & (cls == c)].sum() / max((tp + fp + fn)[sel & (cls == c)].sum(), 1) for c in np.unique(cls)])
        L.append(f"## {NAME[ds]} ({len(rs)} images; frozen BRACE mask class-macro IoU {macro(np.ones(len(rs), bool)):.3f}; "
                 f"{(y < 0.5).sum()} masks with IoU < 0.5)\n")
        L.append("Image level. ρ = Spearman with the mask's true IoU; ρ_wc = mean within-class ρ; ρ_wc|size = within-class "
                 "partial ρ controlling log predicted area; AUROC_wc = detecting IoU < 0.5 with scores ranked within class; "
                 "retained class-macro mIoU when keeping the top q of each class (random keep = all-data mIoU in expectation).\n")
        L.append("| signal | ρ | ρ_wc | ρ_wc\\|size | AUROC_wc | keep 90% | keep 70% | keep 50% |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        for s in sig + ["oracle"]:
            if s != "oracle" and not all(s in r for r in rs):
                continue
            x = y.copy() if s == "oracle" else np.array([r[s] for r in rs])
            if s == "random":
                continue
            rwc, pwc, rk = [], [], np.zeros(len(rs))
            from scipy.stats import rankdata
            for c in np.unique(cls):
                m = cls == c; rwc.append(spearman(x[m], y[m]))
                rx, ry, ra = rankdata(x[m]), rankdata(y[m]), rankdata(area[m])
                res = lambda v: v - np.polyval(np.polyfit(ra, v, 1), ra)
                pwc.append(float(np.corrcoef(res(rx), res(ry))[0, 1])); rk[m] = rankdata(x[m]) / m.sum()
            low = y < 0.5; au = auroc(-rk[low], -rk[~low])
            keep = lambda q: macro(rk > 1 - q + 1e-9)
            stats[(ds, s)] = (np.nanmean(rwc), np.nanmean(pwc))
            L.append(f"| {s} | {spearman(x, y):+.3f} | {np.nanmean(rwc):+.3f} | {np.nanmean(pwc):+.3f} | {au:.3f} | "
                     f"{keep(.9):.3f} | {keep(.7):.3f} | {keep(.5):.3f} |")
        L.append(f"\nAll images: {macro(np.ones(len(rs), bool)):.3f}.\n")
        L.append("Pixel level: which pixels of the BRACE mask are wrong? AUROC of each uncertainty map for error pixels, "
                 "mean over images (40k pixels each); *strat* = the same within boundary-distance bins (<2, 2–5, 5–10, "
                 "10–20, >20 px), i.e. what a map adds beyond 'errors sit at the boundary'.\n")
        L.append("| uncertainty | AUROC | AUROC within distance bins |\n|---|---:|---:|")
        for k in [k for k in rs[0] if k.startswith("u_") and "|" not in k]:
            if not all(k in r for r in rs):
                continue
            L.append(f"| {k} | {np.nanmean([r[k] for r in rs]):.3f} | {np.nanmean([r[k + '|strat'] for r in rs]):.3f} |")
        L.append("")
    if cand:
        ch = max(cand, key=lambda s: stats[("voc", s)][1])
        rv, rc = stats[("coco", ch)][1], stats[("coco", "s_clip_soft")][1]
        L.append(f"## Registered transfer test\n\nBRACE-internal signal chosen on VOC-Sim by within-class size-controlled ρ: "
                 f"**{ch}** (VOC-Sim {stats[('voc', ch)][1]:+.3f}). COCO-Cap: {ch} {rv:+.3f} vs CLIP-agreement "
                 f"(s_clip_soft, prior-art style) {rc:+.3f}.")
        # bootstrap over images for COCO-Cap difference
        rs = R["coco"]; cls = np.array([r["cls"] for r in rs]); y = np.array([r["iou"] for r in rs])
        area = np.log(np.array([r["s_area"] for r in rs]) + 1e-4); X1 = np.array([r[ch] for r in rs]); X2 = np.array([r["s_clip_soft"] for r in rs])
        from scipy.stats import rankdata
        def pwc(idx, X):
            out = []
            for c in np.unique(cls[idx]):
                m = idx[cls[idx] == c]
                if len(m) < 5:
                    continue
                rx, ry, ra = rankdata(X[m]), rankdata(y[m]), rankdata(area[m])
                res = lambda v: v - np.polyval(np.polyfit(ra, v, 1), ra)
                v = np.corrcoef(res(rx), res(ry))[0, 1]
                if not np.isnan(v):
                    out.append(v)
            return np.mean(out)
        rng = np.random.default_rng(0); D = []
        for _ in range(1000):
            idx = rng.integers(0, len(rs), len(rs)); D.append(pwc(idx, X1) - pwc(idx, X2))
        lo, hi = np.percentile(D, [2.5, 97.5])
        L.append(f"Δ (BRACE-internal − CLIP agreement) = {rv - rc:+.3f}, image-bootstrap 95% CI {lo:+.3f}…{hi:+.3f} → "
                 f"**{'BRACE-internal signal better' if lo > 0 else ('CLIP agreement better' if hi < 0 else 'no difference')}**.\n")
    txt = "\n".join(L) + "\n"; (AN / "gen_reliability.md").write_text(txt); print(txt)
    json.dump({ds: R[ds] for ds in R}, open(WORK / "reliab_rows.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("sclip", "score", "report"):
        p = sub.add_parser(n); p.add_argument("--sets", nargs="+", default=["voc", "coco"]); p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("capture"); p.add_argument("--dataset", required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p = sub.add_parser("reliab"); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"sclip": sclip, "capture": capture, "score": score, "report": report, "reliab": reliab}[a.cmd](a)
