#!/usr/bin/env python3
"""New reruns (runs/20_new_reruns/PREREG.md): every component on the prompt-corrected generated sets + seed calibration.

gen      (GPU, trdp)  corrected-prompt generation (the live runtime's exact call, no recording), safety checker off
                      -> <CACHE>/jc_newrerun/img/<ds>/<id>.png   (input of NERVE's CLIP seed)
  then   jc_swap_nerve.py seed --image-dir <CACHE>/jc_newrerun/img --tag _nr   (NERVE env)
bank     (CPU)        sorted min-max NERVE CLIP values of every VOC-Sim image + fold + class -> jc_newrerun/bank.npz
capture  (GPU, trdp)  one live generation per image (frozen runtime, corrected prompt), all seeds x {seed, V0, V3, walk}
score    (CPU)        counts raw and + DenseCRF at 17 thresholds for every cell
report   (CPU)        runs/20_new_reruns/result.md
"""
from __future__ import annotations

import argparse, csv, json, os, random, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import nerve_walk_port as port  # noqa: E402
import jc_srcrule as sr  # noqa: E402
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_newrerun"
NSEED = CACHE / "jc_swap" / "nerve_seed_nr"            # + --ntag suffix for smoke
SWAP = CACHE / "jc_swap"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
EV = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam"
OFFDIR = {"voc": EV / "voc_sim", "coco": EV / "coco_captions"}
OUT = ROOT / "runs/20_new_reruns"
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
PROMPTS = json.loads((ROOT / "configs/new_reruns_prompts.json").read_text())
BASE = ("CAu", "CAg", "NV", "SC")
PW = (0.5, 1.5, 2.0, 3.0); FL = (50, 70, 80, 90); TK = (5, 10, 20, 40)
CAL = ("hA", "TG", "TC") + tuple(f"PW{g:g}" for g in PW) + tuple(f"FL{p}" for p in FL) + tuple(f"TK{k}" for k in TK) + ("FO",)
SEEDS = BASE + CAL
CELLS = ("seed", "V0", "V3", "W")
LAB = {"seed": "seed alone", "V0": "fine-only", "V3": "BRACE hierarchy", "W": "NERVE walk"}
DESC = {"CAu": "cross-attention, uniform layer weights", "CAg": "cross-attention, GOCA-style weights (BRACE seed)",
        "NV": "NERVE CLIP seed (ViT-L/14)", "SC": "SCLIP seed (ViT-B/16)", "hA": "CAg ranking + per-image NERVE CLIP values",
        "TG": "global VOC template (CLIP offline)", "TC": "per-class VOC template (CLIP offline)", "FO": "Otsu floor (SD only)"}
DESC.update({f"PW{g:g}": f"power s^{g:g} (SD only)" for g in PW}); DESC.update({f"FL{p}": f"percentile floor p{p} (SD only)" for p in FL})
DESC.update({f"TK{k}": f"top {k}% (SD only)" for k in TK})
FAMILIES = {"PW": [f"PW{g:g}" for g in PW], "FL": [f"FL{p}" for p in FL], "TK": [f"TK{k}" for k in TK]}
THR = np.round(0.30 + 0.025 * np.arange(17), 3)


def rows(ds, shard=0, num_shards=1, max_images=0):
    R = cc.gen_rows(ds)
    for r in R:
        r["label"] = r["classname"]; fix = PROMPTS[ds].get(r["image_id"])
        r["changed"] = fix is not None
        if fix:
            r["prompt"] = fix["prompt"]
            if ds == "voc":
                r["attribution_prompt"] = fix["prompt"]; r["classname"] = fix["classname"]
    R = R[shard::num_shards]
    if max_images:
        R = R[:: max(1, len(R) // max_images)][:max_images]
    return R


def official(ds, r):
    from PIL import Image
    return np.asarray(Image.open(OFFDIR[ds] / r["image"]).convert("RGB"))


def canon(c):
    return c.replace(" ", "-")


# ------------------------------------------------------------------ gen (GPU)
def gen(a):
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float32, local_files_only=True,
                                                   safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    out = WORK / f"img{a.tag}" / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for r in rows(a.dataset, a.shard, a.num_shards, a.max_images):
        o = out / f"{r['image_id']}.png"
        if o.exists():
            continue
        s = int(r["seed"]); t0 = time.perf_counter()
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)   # live._seed_everything
        with torch.inference_mode():
            pe, ne = pipe.encode_prompt(prompt=r["prompt"], device=pipe.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True, negative_prompt=None)
            noise = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=pipe.device, dtype=pe.dtype)
            img = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=noise, height=512, width=512,
                       num_inference_steps=int(r["num_inferece_steps"]), guidance_scale=7.5, output_type="pil").images[0]
        img.save(o.with_suffix(".tmp.png")); os.replace(o.with_suffix(".tmp.png"), o)
        d = float(np.abs(np.asarray(img).astype(int) - official(a.dataset, r).astype(int)).mean())
        print(f"[gen:{a.dataset}] {r['image_id']} changed={r['changed']} vs official {d:.2f} {time.perf_counter() - t0:.1f}s", flush=True)


# ------------------------------------------------------------------ bank (CPU)
def bank(a):
    R = rows("voc"); V, fold, cls, ids = [], [], [], []
    for r in R:
        f = Path(str(NSEED) + a.ntag) / "voc" / f"{r['image_id']}.npz"
        if not f.exists():
            if a.tag:
                continue                                        # smoke bank: only the smoke images exist
            raise FileNotFoundError(f)
        z = np.load(f)
        v = st.mm(z["seed64"][0].astype(np.float64)).ravel(); V.append(np.sort(v).astype(np.float32))
        fold.append(zlib.crc32(r["image_id"].encode()) % 5); cls.append(canon(r["label"])); ids.append(r["image_id"])
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / f"bank{a.tag}.npz", vals=np.stack(V), fold=np.array(fold), cls=np.array(cls), ids=np.array(ids))
    print("bank", len(V), "VOC-Sim images; per fold", np.bincount(fold))


def otsu(x: torch.Tensor) -> float:
    h = torch.histc(x.float(), bins=256, min=0.0, max=1.0); p = h / h.sum(); c = torch.arange(256, device=x.device) + 0.5
    w0 = torch.cumsum(p, 0); m = torch.cumsum(p * c, 0); mt = m[-1]
    var = (mt * w0 - m) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    return float((int(torch.argmax(var)) + 1) / 256.0)


def calib_seeds(s: torch.Tensor, nv64: torch.Tensor, tg: torch.Tensor, tc: torch.Tensor) -> dict:
    """s: min-max CAg seed [64, 64]. Monotone remaps (ranking of s kept); every output min-max scaled."""
    mm = lambda x: cc.norm(x, "mm")
    q = lambda p: torch.quantile(s.flatten().double(), p).to(s.dtype)
    out = {"hA": sr.rank_swap(s, mm(nv64)), "TG": sr.rank_swap(s, tg.reshape(64, 64)), "TC": sr.rank_swap(s, tc.reshape(64, 64))}
    for g in PW:
        out[f"PW{g:g}"] = s.clamp_min(0) ** g
    for p in FL:
        out[f"FL{p}"] = (s - q(p / 100)).clamp_min(0)
    for k in TK:
        out[f"TK{k}"] = s * (s >= q(1 - k / 100)).to(s.dtype)
    out["FO"] = (s - otsu(s)).clamp_min(0)
    return {k: mm(v.float()) for k, v in out.items()}


# ------------------------------------------------------------------ capture (GPU)
@torch.no_grad()
def capture(a):
    from PIL import Image
    import torch.nn.functional as F
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    import sg_clip_dense as cd
    import sg_gen_clip as gc
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    runtime["recorder_class"].enabled = ("A0", "A3"); pipe.safety_checker = None
    dev = pipe.device
    M = cd.DenseCLIP("B16", dev); n21, i21 = cd.split([cd.VOC21_BG] + cd.VOC20_NAMES); T21 = M.text(n21)
    B = np.load(WORK / f"bank{a.btag}.npz"); bvals = torch.from_numpy(B["vals"]).to(dev)
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for r in rows(a.dataset, a.shard, a.num_shards, a.max_images):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        zs = np.load(Path(str(NSEED) + a.ntag) / a.dataset / f"{iid}.npz")
        if canon(str(zs["cls"])) != canon(r["label"]):
            raise ValueError(f"class mismatch for {iid}")
        lab = canon(r["label"])
        allowed = np.ones(len(B["fold"]), bool) if a.dataset == "coco" else B["fold"] != zlib.crc32(iid.encode()) % 5
        tg = bvals[torch.from_numpy(allowed).to(dev)].mean(0)
        tcm = allowed & (B["cls"] == lab)
        if not tcm.any():
            if not a.max_images:
                raise ValueError(f"no VOC template images for class {lab}")
            tcm = allowed                                       # smoke only: too few bank images per class
        tc = bvals[torch.from_numpy(tcm).to(dev)].mean(0)
        with sg27_runtime.live_maps(runtime, dict(r), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]; w = rec.seeds()
            if float((w["selected/A3"].float() - seed.float()).abs().max()) > 0:
                raise ValueError("A3 seed differs from the frozen runtime's seed")
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            ca_g = seed.float().clone(); ca_u = w["selected/A0"].float().clone().reshape(64, 64)
            nv64 = torch.from_numpy(zs["seed64"][0]).float().to(dev); nvseg = torch.from_numpy(zs["seg"]).float().to(dev)
            _, P21 = cd.postprocess(M.logits(img, T21), i21, 65, 0.1, 0.1)
            sc64 = F.adaptive_avg_pool2d(P21[gc.voc_k(r["label"])][None, None].float(), (64, 64))[0, 0].to(dev)
            sd = {"CAu": ca_u, "CAg": ca_g, "NV": cc.norm(nv64, "mm"), "SC": cc.norm(sc64, "mm")}
            sd.update(calib_seeds(cc.norm(ca_g, "mm"), nv64, tg, tc))
            res = {s: {"seed": sd[s]} for s in SEEDS}
            for s in SEEDS:
                v, _ = sg_combined_v2.aggregate(sa, sd[s]); res[s]["V0"], res[s]["V3"] = v["V0"], v["V3"]; del v
        nstore = sr.noised_pass(pipe, Image.fromarray(img), sr.NERVE_LAYERS)
        d = int(nstore[sr.NERVE_LAYERS[0]][0]["query"].shape[-1])
        op = port.walk_operator(port.chained_attention_product([nstore[n][0] for n in sr.NERVE_LAYERS],
                                                                torch.ones(4096, 1, device=dev), d=d))
        for s in SEEDS:
            res[s]["W"] = port.apply_walk(op, nvseg if s == "NV" else sd[s].reshape(1, 64, 64))[0]
        del op, nstore
        maps = np.stack([np.stack([res[s][c].float().reshape(64, 64).cpu().numpy() for c in CELLS]) for s in SEEDS])
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        g1 = np.asarray(Image.open(WORK / f"img{a.itag}" / a.dataset / f"{iid}.png").convert("RGB"))
        stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        offd = float(np.abs(img.astype(int) - official(a.dataset, r).astype(int)).mean())
        g1d = int(np.abs(img.astype(int) - g1.astype(int)).max())
        chk = float(np.abs(st.mm(maps[SEEDS.index("CAg"), CELLS.index("V3")]) - st.mm(stored["map/base"][3])).max())
        prev = SWAP / "walk" / a.dataset / f"{iid}.npz"
        wchk = float(np.abs(st.mm(maps[SEEDS.index("NV"), CELLS.index("W")]) - st.mm(np.load(prev)["walk_clip"].squeeze())).max()) if prev.exists() else float("nan")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=r["label"], changed=r["changed"], off_diff=offd, gen1_maxdiff=g1d,
                 check=chk, walk_check=wchk, truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[nr:{a.dataset}] {iid} {r['label']} changed={r['changed']} vs official {offd:.2f} vs gen {g1d} "
              f"V3chk {chk:.1e} walkchk {wchk:.1e} {time.perf_counter() - t0:.1f}s peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ score (CPU)
def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"]), "off_diff": float(z["off_diff"]),
           "check": float(z["check"]), "walk_check": float(z["walk_check"]), "gen1_maxdiff": int(z["gen1_maxdiff"])}
    for si, s in enumerate(SEEDS):
        for ci, c in enumerate(CELLS):
            x = st.up(st.mm(z["maps"][si, ci].astype(np.float32)), truth.shape)
            res[f"{s}|{c}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{s}|{c}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 50 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------ report (CPU)
def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    Z = {ds: [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")] for ds in S}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    offd = {ds: np.array([float(z["off_diff"]) for z in Z[ds]]) for ds in S}
    resolved = {ds: offd[ds] < 2 for ds in S}
    t55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
    k = lambda s, c, crf: f"{s}|{c}{'+R2' if crf else ''}"

    def mi(ds, P, sub=None):
        sub = one[ds] if sub is None else sub
        return cc.miou(P[:, None], S[ds]["cls"], sub)[0]

    def tbest(key):
        return int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))

    def sel(ds, key):
        return cc.oof(S["voc"], S["voc"][f"{key}|raw"])[1] if ds == "voc" else S["coco"][f"{key}|raw"][:, tbest(key)]

    def fx(ds, key):
        return S[ds][f"{key}|raw"][:, t55]

    def best(ds, key, sub=None):
        sub = one[ds] if sub is None else sub
        return float(cc.miou(S[ds][f"{key}|raw"], S[ds]["cls"], sub).max())

    def dci(ds, Pa, Pb, sub=None):
        sub = one[ds] if sub is None else sub
        pt = mi(ds, Pa, sub) - mi(ds, Pb, sub); idx = np.flatnonzero(sub); rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            s = rng.choice(idx, len(idx)); o1 = np.ones(len(s), bool)
            d.append(cc.miou(Pa[s][:, None], S[ds]["cls"][s], o1)[0] - cc.miou(Pb[s][:, None], S[ds]["cls"][s], o1)[0])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", (pt, lo, hi)

    L = ["# New reruns: components on the prompt-corrected generated sets (`PREREG.md`)\n",
         "`scripts/jc_newrerun.py`. One live generation per image with the frozen runtime, corrected prompts "
         "(`configs/new_reruns_prompts.json`), safety checker off. Class-macro mIoU. *VOC-sel.* = threshold selected on "
         "VOC-Sim (5-fold OOF on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap). Δ with image-bootstrap 95% "
         "interval (2000). Seeds: " + "; ".join(f"**{s}** = {DESC[s]}" for s in BASE) + ".\n"]
    L.append("## Data checks\n")
    for ds in S:
        ch = np.array([bool(z["changed"]) for z in Z[ds]]); chk = np.array([float(z["check"]) for z in Z[ds]])
        g1 = np.array([int(z["gen1_maxdiff"]) for z in Z[ds]]); wc = np.array([float(z["walk_check"]) for z in Z[ds]])
        L.append(f"- {NAME[ds]}: {n[ds]} images, {ch.sum()} with corrected prompt. Generated vs official PNG: median "
                 f"{np.median(offd[ds]):.2f} grey levels; < 1 on {(offd[ds] < 1).sum()}, unresolved (≥ 2) {(~resolved[ds]).sum()}"
                 f" ({', '.join(str(g) for g in S[ds]['grp'][~resolved[ds]])}). Capture vs gen step image: max |Δ| 0 on "
                 f"{(g1 == 0).sum()}/{n[ds]}. Unchanged-prompt images: BRACE map vs stored paper capture median max |Δ| "
                 f"{np.median(chk[~ch]):.1e}. NERVE walk vs runs/17 (min–max), images < 1 grey level from official: median max |Δ| "
                 f"{np.nanmedian(wc[offd[ds] < 1]):.1e}.")
    L.append("")
    convs = (("VOC-sel. threshold, with DenseCRF (primary)", sel, True), ("fixed 0.55, with DenseCRF", fx, True),
             ("VOC-sel. threshold, raw", sel, False), ("fixed 0.55, raw", fx, False))
    for ds in S:
        L.append(f"## Components, {NAME[ds]} ({n[ds]} images)\n")
        for title, f, crf in convs:
            L.append(f"**{title}**\n"); L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |")
            L.append("|---|" + "---:|" * len(CELLS))
            for s in BASE:
                L.append(f"| {DESC[s]} | " + " | ".join(f"{mi(ds, f(ds, k(s, c, crf))):.3f}" for c in CELLS) + " |")
            L.append("")
        L.append("**Best threshold on this set, with DenseCRF (audit only)**\n")
        L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---:|" * len(CELLS))
        for s in BASE:
            L.append(f"| {DESC[s]} | " + " | ".join(f"{best(ds, k(s, c, True)):.3f}" for c in CELLS) + " |")
        L.append("")
    L.append("**Headline:** BRACE (CAg × hierarchy + DenseCRF, 0.55): " + ", ".join(
        f"{NAME[ds]} {mi(ds, fx(ds, k('CAg', 'V3', True))):.3f}" for ds in S) + "; NERVE (NV × walk, VOC-sel. + DenseCRF): " +
        ", ".join(f"{NAME[ds]} {mi(ds, sel(ds, k('NV', 'W', True))):.3f}" for ds in S) + ".\n")
    L.append("## Thresholds selected on full VOC-Sim (with DenseCRF)\n")
    L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---:|" * len(CELLS))
    for s in SEEDS:
        L.append(f"| {s} | " + " | ".join(f"{THR[tbest(k(s, c, True))]:.3f}" for c in CELLS) + " |")
    # registered contrasts
    L.append("\n## Registered contrasts\n")
    L.append("| ID | Contrast | VOC-Sim, VOC-sel. + CRF | COCO-Cap, VOC-sel. + CRF | VOC-Sim, 0.55 + CRF | COCO-Cap, 0.55 + CRF |")
    L.append("|---|---|---|---|---|---|")
    con = []
    for s in BASE:
        con.append(("F", f"fine-only − seed, {s}", (s, "V0", True), (s, "seed", True)))
    for s in BASE:
        con.append(("H", f"hierarchy − fine-only, {s}", (s, "V3", True), (s, "V0", True)))
    for c in ("seed", "V0", "V3"):
        con.append(("G", f"CAg − CAu (GOCA weighting), {LAB[c]}", ("CAg", c, True), ("CAu", c, True)))
    for s in ("NV", "SC"):
        for c in ("seed", "V0", "V3", "W"):
            con.append(("C", f"{s} − CAg, {LAB[c]}", (s, c, True), ("CAg", c, True)))
    for s in BASE:
        con.append(("R", f"DenseCRF − raw, hierarchy, {s}", (s, "V3", True), (s, "V3", False)))
    con.append(("N", "NERVE (NV × walk) − BRACE (CAg × hierarchy)", ("NV", "W", True), ("CAg", "V3", True)))
    con.append(("N", "BRACE hierarchy on NERVE's seed (NV × hierarchy) − NERVE", ("NV", "V3", True), ("NV", "W", True)))
    for cid, lab, A, Bq in con:
        cells = []
        for f in (sel, fx):
            for ds in S:
                cells.append(dci(ds, f(ds, k(*A)), f(ds, k(*Bq)))[0])
        L.append(f"| {cid} | {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    L.append("\nSensitivity, resolved images only (generated within 2 grey levels of the official PNG), VOC-sel. + CRF:\n")
    L.append("| Contrast | VOC-Sim | COCO-Cap |"); L.append("|---|---|---|")
    for cid, lab, A, Bq in con:
        if cid in ("H", "N", "G"):
            L.append(f"| {cid}: {lab} | " + " | ".join(dci(ds, sel(ds, k(*A)), sel(ds, k(*Bq)), resolved[ds])[0] for ds in S) + " |")
    L.append("\nComponent table on resolved images only (VOC-sel. + CRF):\n")
    L.append("| Seed | set | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|---|" + "---:|" * len(CELLS))
    for ds in S:
        for s in BASE:
            L.append(f"| {s} | {NAME[ds]} | " + " | ".join(f"{mi(ds, sel(ds, k(s, c, True)), resolved[ds]):.3f}" for c in CELLS) + " |")
    # calibration arm
    L.append("\n## Calibration arm (exploratory)\n")
    chosen = {}
    for fam, mem in FAMILIES.items():
        v = {m: mi("voc", sel("voc", k(m, "V3", True))) for m in mem}
        chosen[fam] = max(v, key=v.get)
        L.append(f"- {fam}: VOC-Sim OOF (hierarchy + CRF) " + ", ".join(f"{m} {x:.3f}" for m, x in v.items()) + f" → **{chosen[fam]}**")
    cand = ["hA", "TG", "TC", chosen["PW"], chosen["FL"], chosen["TK"], "FO"]
    L.append("")
    for c in ("V0", "V3", "W"):
        L.append(f"**{LAB[c]}, with DenseCRF**\n")
        L.append("| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |")
        L.append("|---|---:|---:|---:|---:|---|---|---|")
        br = k("CAg", "V3", True)
        for s in ("CAg",) + tuple(cand):
            kk = k(s, c, True)
            row = [f"{mi('voc', sel('voc', kk)):.3f}", f"{mi('coco', sel('coco', kk)):.3f}", f"{mi('coco', fx('coco', kk)):.3f}", f"{best('coco', kk):.3f}"]
            if s == "CAg" and c == "V3":
                row += ["—", "—", "—"]
            else:
                row += [dci("voc", sel("voc", kk), sel("voc", br))[0], dci("coco", sel("coco", kk), sel("coco", br))[0],
                        f"{best('coco', kk) - best('coco', br):+.3f}"]
            L.append(f"| {s}: {DESC[s]} | " + " | ".join(row) + " |")
        L.append("")
    L.append("**Reading (hierarchy + CRF vs BRACE; helps = COCO Δ CI > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0):**\n")
    br = k("CAg", "V3", True)
    for s in cand:
        kk = k(s, "V3", True); _, (pc, lo, _) = dci("coco", sel("coco", kk), sel("coco", br)); pv = mi("voc", sel("voc", kk)) - mi("voc", sel("voc", br))
        bd = best("coco", kk) - best("coco", br)
        verdict = "helps" if (lo > 0 and pv >= 0 and bd > 0) else ("calibration only" if (lo > 0 and pv >= 0) else "does not help")
        L.append(f"- {s} ({DESC[s]}): COCO {pc:+.3f} (lower {lo:+.3f}), VOC {pv:+.3f}, COCO best-thr. {bd:+.3f} → **{verdict}**")
    L.append("\n**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**\n")
    L.append("| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |"); L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    area = S["coco"]["area"]; q1, q2 = np.quantile(area, [1 / 3, 2 / 3])
    for s in ("CAg", "NV", "SC") + tuple(cand):
        for c in ("V3", "W"):
            P = sel("coco", k(s, c, True)); iou = P[:, 0] / np.maximum(P[:, :3].sum(1), 1)
            cls_ = S["coco"]["cls"]; U = np.unique(cls_)   # counts columns: tp, fp, fn
            prec = np.mean([P[cls_ == cl, 0].sum() / max(P[cls_ == cl][:, [0, 1]].sum(), 1) for cl in U])
            rec = np.mean([P[cls_ == cl, 0].sum() / max(P[cls_ == cl][:, [0, 2]].sum(), 1) for cl in U])
            L.append(f"| {s} | {LAB[c]} | {mi('coco', P):.3f} | {iou[area <= q1].mean():.3f} | {iou[(area > q1) & (area <= q2)].mean():.3f} | "
                     f"{iou[area > q2].mean():.3f} | {(iou < 0.3).mean():.3f} | {P[:, 1].mean() / (512 * 512):.4f} | {prec:.3f} | {rec:.3f} |")
    L.append("\n## Per class, VOC-sel. + CRF (CAg × hierarchy / NV × walk / SC × hierarchy)\n")
    for ds in S:
        cl = S[ds]["cls"]
        L.append(f"{NAME[ds]}: " + ", ".join(f"{c} " + " / ".join(f"{mi(ds, sel(ds, k(s, p, True)), cl == c):.2f}" for s, p in (("CAg", "V3"), ("NV", "W"), ("SC", "V3"))) for c in np.unique(cl)) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"result{a.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("gen", "capture"):
        p = sub.add_parser(nm); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
        p.add_argument("--itag", default="", help="capture: tag of the gen images"); p.add_argument("--btag", default="")
        p.add_argument("--ntag", default="", help="capture: suffix of the NERVE seed folder (smoke)")
    p = sub.add_parser("bank"); p.add_argument("--tag", default=""); p.add_argument("--ntag", default="")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=16); p.add_argument("--tag", default="")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    {"gen": gen, "bank": bank, "capture": capture, "score": score, "report": report}[a.cmd](a)
