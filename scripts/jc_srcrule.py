#!/usr/bin/env python3
"""Seed x attention source x propagation rule (runs/19_source_rule/PREREG.md).

One live generation per image with the frozen runtime. Two self-attention sources from that image:
  G  generation: the frozen runtime's self-attention, denoising calls 10-20, conditional branch
  N  NERVE's source: one pass of the generated image, noised to t=101, empty prompt (NERVE's one_step)
Two propagation rules, each with its own layers:
  H  BRACE: fine-only (V0) and capped 16->32->64 hierarchy (V3), frozen code, BRACE's SA modules
  W  NERVE: global+local fusion and closed-form walk (scripts/nerve_walk_port.py), NERVE's layers -4 and -6;
     on G the fused matrix is averaged over the 11 calls before the walk
Two seeds: CA (BRACE's live seed) and CLIP (NERVE's seed from runs/17, computed on the stored image).

capture (GPU)  <BRACE_CACHE_DIR>/jc_srcrule/<ds>/<id>.npz  maps [2 seeds, 7 cells, 64, 64]
score   (CPU)  counts per cell, raw and + DenseCRF, 17 thresholds
report  (CPU)  runs/19_source_rule/result.md
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
import nerve_walk_port as port  # noqa: E402
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_srcrule"
SWAP = CACHE / "jc_swap"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
OUT = ROOT / "runs/19_source_rule"
SEEDS = ("ca", "clip")
CELLS = ("seed", "G|V0", "G|V3", "G|W", "N|V0", "N|V3", "N|W")
NERVE_LAYERS = ("up_blocks.3.attentions.1.transformer_blocks.0.attn1",    # NERVE index -4
                "up_blocks.3.attentions.0.transformer_blocks.0.attn1")    # NERVE index -6
T_NOISE, NOISE_SEED = 101, 0
THR = np.round(0.30 + 0.025 * np.arange(17), 3)


class QKHook:
    """Forward hooks on to_q / to_k of chosen self-attention modules; keeps the last batch element per call."""

    def __init__(self, unet, names, keep_calls=None):
        self.names, self.keep, self.store, self.count, self.handles = names, keep_calls, {}, {}, []
        for n in names:
            m = unet.get_submodule(n); self.store[n] = []; self.count[n] = 0; self._pending = {}
            self.handles.append(m.to_q.register_forward_hook(self._make(n, "q", m.heads)))
            self.handles.append(m.to_k.register_forward_hook(self._make(n, "k", m.heads)))

    def _make(self, name, kind, heads):
        def hook(_mod, _inp, out):
            b, n, c = out.shape
            x = out[-1].reshape(n, heads, c // heads).permute(1, 0, 2).detach().half().clone()   # [heads, N, d]
            if kind == "q":
                self._pending[name] = x
            else:
                call = self.count[name]; self.count[name] += 1
                if self.keep is None or call in self.keep:
                    self.store[name].append({"query": self._pending.pop(name), "key": x})
                else:
                    self._pending.pop(name, None)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def nerve_fused(stores, d):
    """stores: per call, [layer -4 entry, layer -6 entry]. Mean over calls of NERVE's fused matrix."""
    acc = None
    dummy = torch.ones(4096, 1, device=stores[0][0]["query"].device)   # K=1: head weights uniform, seed-independent
    for entry in stores:
        f = port.chained_attention_product(entry, dummy, d=d)
        acc = f if acc is None else acc + f
        del f
    return acc / len(stores)


@torch.no_grad()
def noised_pass(pipe, image, names):
    """NERVE's diffusion.forward/one_step on the generated image: VAE mean latent, noise to t=101, empty prompt."""
    from diffusers.models.attention_processor import AttnProcessor2_0
    saved = pipe.unet.attn_processors
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    hook = QKHook(pipe.unet, names)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            x = pipe.image_processor.preprocess(image, height=512, width=512).to(pipe.device)
            lat = pipe.vae.encode(x)[0].mean * pipe.vae.config.scaling_factor
            torch.manual_seed(NOISE_SEED)
            noise = torch.randn(lat.shape, device=lat.device, dtype=lat.dtype)
            ac = pipe.scheduler.alphas_cumprod.to(lat.device)[T_NOISE].to(lat.dtype)
            noisy = ac.sqrt() * lat + (1 - ac).sqrt() * noise
            emb = pipe.encode_prompt("", pipe.device, 1, do_classifier_free_guidance=False)[0]
            pipe.unet(noisy, torch.tensor(T_NOISE, device=pipe.device), encoder_hidden_states=emb, return_dict=False)
    finally:
        hook.remove(); pipe.unet.set_attn_processor(saved)
    return hook.store


def capture(a):
    from PIL import Image
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    import json
    sa_mods = {int(k): v for k, v in json.loads((ROOT / "configs/sg25.json").read_text())["sa_modules"].items()}
    rows = cc.gen_rows(a.dataset)
    if a.max_images:
        rows = rows[:: max(1, len(rows) // a.max_images)][: a.max_images]
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    sdir = SWAP / "nerve_seed" / a.dataset
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz"); zs = np.load(sdir / f"{iid}.npz")
        if str(zs["cls"]).replace("-", " ") != cls.replace("-", " "):
            raise ValueError(f"class mismatch for {iid}")
        hook = QKHook(pipe.unet, NERVE_LAYERS + (sa_mods[64],))    # every forward of to_q/to_k; aligned to steps below
        try:
            with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
                hook.remove()
                rec = runtime["recorders"][-1]
                # The recording processor computes Q/K itself and then calls the original processor, which computes
                # them again, so a module's to_q/to_k can fire more than once per denoising step. Align by the
                # recorder's own per-module call count; the first firing of each step is kept.
                for name in hook.store:
                    steps = rec.call_counts[name]; f = hook.count[name] // steps
                    if steps < 21 or f < 1 or hook.count[name] != f * steps:
                        raise ValueError(f"hook/call mismatch for {name}: {hook.count[name]} firings, {steps} steps")
                    hook.store[name] = [hook.store[name][i * f] for i in range(10, 21)]
                img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
                # hook check: the hooked BRACE 64-px Q/K must equal the runtime's own record, call by call
                hk = hook.store[sa_mods[64]]
                if len(hk) != 11 or any(not (torch.equal(h["query"], s[0]) and torch.equal(h["key"], s[1]))
                                        for h, s in zip(hk, sa[64])):
                    raise ValueError(f"hooked Q/K differ from the runtime's SA record for {iid}")
                d = int(hk[0]["query"].shape[-1])
                sd = {"ca": seed.float().clone(),
                      "clip": cc.norm(torch.from_numpy(zs["seed64"][0]).float().to(seed.device), "mm")}
                walk_in = {"ca": seed.float()[None], "clip": torch.from_numpy(zs["seg"]).float().to(seed.device)}
                res = {s: {"seed": sd[s]} for s in SEEDS}
                for s in SEEDS:                                         # G, BRACE rule
                    v, _ = sg_combined_v2.aggregate(sa, sd[s]); res[s]["G|V0"], res[s]["G|V3"] = v["V0"], v["V3"]; del v
                gstores = [[hook.store[NERVE_LAYERS[0]][c], hook.store[NERVE_LAYERS[1]][c]] for c in range(11)]
                op = port.walk_operator(nerve_fused(gstores, d)); del gstores; hook.store.clear()
                for s in SEEDS:                                         # G, NERVE rule
                    res[s]["G|W"] = port.apply_walk(op, walk_in[s])[0]
                del op
            nstore = noised_pass(pipe, Image.fromarray(img), NERVE_LAYERS + tuple(sa_mods[r] for r in (16, 32, 64)))
            nsa = {r: [(nstore[sa_mods[r]][0]["query"], nstore[sa_mods[r]][0]["key"])] * 11 for r in (16, 32, 64)}
            for s in SEEDS:                                             # N, BRACE rule (one call, repeated)
                v, _ = sg_combined_v2.aggregate(nsa, res[s]["seed"]); res[s]["N|V0"], res[s]["N|V3"] = v["V0"], v["V3"]; del v
            op = port.walk_operator(port.chained_attention_product([nstore[n][0] for n in NERVE_LAYERS],
                                                                    torch.ones(4096, 1, device=pipe.device), d=d))
            for s in SEEDS:                                             # N, NERVE rule
                res[s]["N|W"] = port.apply_walk(op, walk_in[s])[0]
            del op, nstore, nsa
            maps = np.stack([np.stack([res[s][c].float().cpu().numpy() for c in CELLS]) for s in SEEDS]).astype(np.float32)
        finally:
            hook.remove()
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        chk = float(np.abs(st.mm(maps[0, CELLS.index("G|V3")]) - st.mm(stored["map/base"][3])).max())
        prev = SWAP / "walk" / a.dataset / f"{iid}.npz"
        wchk = float(np.abs(st.mm(maps[1, CELLS.index("N|W")]) - st.mm(np.load(prev)["walk_clip"])).max()) if prev.exists() else float("nan")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps, cls=cls, check=chk, walk_check=wchk, img_maxdiff=img_diff, truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[srcrule:{a.dataset}] {iid} {cls} img {img_diff} V3 check {chk:.1e} N|W vs runs/17 {wchk:.1e} "
              f"{time.perf_counter() - t0:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
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
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "check": float(z["check"]),
           "walk_check": float(z["walk_check"]), "img_maxdiff": int(z["img_maxdiff"])}
    for si, s in enumerate(SEEDS):
        for ci, c in enumerate(CELLS):
            x = st.up(st.mm(z["maps"][si, ci]), truth.shape)
            res[f"{s}|{c}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            if c != "seed":
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
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


LAB = {"seed": "seed alone", "G|V0": "generation SA, BRACE fine-only", "G|V3": "generation SA, BRACE hierarchy",
       "G|W": "generation SA, NERVE walk", "N|V0": "NERVE pass SA, BRACE fine-only",
       "N|V3": "NERVE pass SA, BRACE hierarchy", "N|W": "NERVE pass SA, NERVE walk"}
CONTRASTS = (  # (id, label, seed, a, b): a − b
    ("R-G", "rule: NERVE walk − BRACE hierarchy, generation SA", "W", "G|W", "G|V3"),
    ("R-N", "rule: NERVE walk − BRACE hierarchy, NERVE pass SA", "W", "N|W", "N|V3"),
    ("S-H", "source: NERVE pass − generation, BRACE hierarchy", "S", "N|V3", "G|V3"),
    ("S-W", "source: NERVE pass − generation, NERVE walk", "S", "N|W", "G|W"),
)


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    t55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0]); t50 = int(np.flatnonzero(np.isclose(THR, 0.5))[0])
    Z = {ds: [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz"))] for ds in S}
    exact = {ds: np.array([int(z["img_maxdiff"]) == 0 for z in Z[ds]]) for ds in S}

    def mi(ds, P, s=None):
        s = one[ds] if s is None else s
        return cc.miou(P[:, None], S[ds]["cls"], s)[0] if s.any() else float("nan")

    def tbest(key):
        return int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))

    def sel(ds, key):
        return cc.oof(S["voc"], S["voc"][f"{key}|raw"])[1] if ds == "voc" else S["coco"][f"{key}|raw"][:, tbest(key)]

    def dci(ds, Pa, Pb, sub=None):
        sub = one[ds] if sub is None else sub
        pt = mi(ds, Pa, sub) - mi(ds, Pb, sub)
        if sub.sum() < 50:
            return f"{pt:+.3f}", (pt, float("nan"), float("nan"))
        idx = np.flatnonzero(sub); rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            s = rng.choice(idx, len(idx)); m = np.zeros(n[ds], bool)
            d.append(cc.miou(Pa[s][:, None], S[ds]["cls"][s], np.ones(len(s), bool))[0]
                     - cc.miou(Pb[s][:, None], S[ds]["cls"][s], np.ones(len(s), bool))[0])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", (pt, lo, hi)

    key = lambda s, c: f"{s}|{c}" if c == "seed" else f"{s}|{c}+R2"
    L = ["# Seed × attention source × propagation rule (`PREREG.md`)\n",
         "`scripts/jc_srcrule.py`. One live generation per image. Sources: **G** = the frozen runtime's generation "
         "self-attention (calls 10–20); **N** = NERVE's source, one pass of the generated image noised to t = 101 with "
         "an empty prompt. Rules: BRACE fine-only / capped hierarchy (frozen code, BRACE's modules) and NERVE's walk "
         "(released code, ported; NERVE's layers; on G the fused matrix is averaged over the 11 calls). Seeds: **CA** "
         "(live) and **CLIP** (NERVE's, from runs/17, stored image). Class-macro mIoU. Primary: threshold selected on "
         "VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap), with DenseCRF. "
         "Δ with image-bootstrap 95% interval (2000).\n"]
    for ds in ("voc", "coco"):
        chk = np.array([float(z["check"]) for z in Z[ds]]); wc = np.array([float(z["walk_check"]) for z in Z[ds]])
        L.append(f"## {NAME[ds]} ({n[ds]} images)\n")
        L.append(f"Regeneration bit-identical to the stored image: {exact[ds].sum()}/{n[ds]}. CA × generation × hierarchy "
                 f"map vs stored capture: median max |Δ| {np.median(chk):.1e}. CLIP × NERVE pass × walk vs runs/17's "
                 f"NERVE walk (min–max maps): median max |Δ| {np.nanmedian(wc):.1e} on all images, "
                 f"{np.nanmedian(wc[exact[ds]]) if exact[ds].any() else float('nan'):.1e} on bit-identical images.\n")
        for conv, f in (("VOC-sel. threshold, with DenseCRF (primary)", lambda k: sel(ds, k)),
                        ("fixed threshold 0.55, with DenseCRF", lambda k: S[ds][f"{k}|raw"][:, t55])):
            L.append(f"**{conv}**\n"); L.append("| Cell | CA seed | CLIP seed | CLIP − CA |"); L.append("|---|---:|---:|---|")
            for c in CELLS:
                pa, pb = f(key("clip", c)), f(key("ca", c))
                L.append(f"| {LAB[c]} | {mi(ds, pb):.3f} | {mi(ds, pa):.3f} | {dci(ds, pa, pb)[0]} |")
            L.append("")
        L.append("**Raw (no DenseCRF), VOC-sel. threshold**\n"); L.append("| Cell | CA seed | CLIP seed |"); L.append("|---|---:|---:|")
        for c in CELLS:
            L.append(f"| {LAB[c]} | {mi(ds, sel(ds, f'ca|{c}')):.3f} | {mi(ds, sel(ds, f'clip|{c}')):.3f} |")
        L.append("")
    L.append("## Thresholds selected on full VOC-Sim (with DenseCRF)\n")
    L.append("| Cell | CA | CLIP |"); L.append("|---|---:|---:|")
    for c in CELLS:
        L.append(f"| {LAB[c]} | {THR[tbest(key('ca', c))]:.3f} | {THR[tbest(key('clip', c))]:.3f} |")
    L.append("\n## Registered contrasts (VOC-sel. threshold, with DenseCRF)\n")
    L.append("| ID | Contrast | Seed | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images |"); L.append("|---|---|---|---|---|---|")
    res = {}
    for cid, lab, _, ca_, cb_ in CONTRASTS:
        for s in SEEDS:
            v = dci("voc", sel("voc", key(s, ca_)), sel("voc", key(s, cb_)))[0]
            c, pt = dci("coco", sel("coco", key(s, ca_)), sel("coco", key(s, cb_)))
            e = dci("coco", sel("coco", key(s, ca_)), sel("coco", key(s, cb_)), exact["coco"])[0]
            res[(cid, s)] = pt
            L.append(f"| {cid} | {lab} | {s.upper()} | {v} | {c} | {e} |")
    for src, a_, b_ in (("G", "G|W", "G|V3"), ("N", "N|W", "N|V3")):
        terms = [sel("coco", key("clip", a_)), sel("coco", key("clip", b_)), sel("coco", key("ca", a_)), sel("coco", key("ca", b_))]
        pt = (mi("coco", terms[0]) - mi("coco", terms[1])) - (mi("coco", terms[2]) - mi("coco", terms[3]))
        rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            s = rng.choice(n["coco"], n["coco"]); o1 = np.ones(len(s), bool); cl = S["coco"]["cls"][s]
            m = [cc.miou(T[s][:, None], cl, o1)[0] for T in terms]; d.append((m[0] - m[1]) - (m[2] - m[3]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        L.append(f"| I-{src} | interaction: (walk − hierarchy) on CLIP minus on CA, source {src} | — | — | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] | — |")
    pos = lambda k: res[k][1] > 0
    rule = pos(("R-G", "clip")) and pos(("R-N", "clip")); source = pos(("S-H", "clip")) and pos(("S-W", "clip"))
    verdict = ("rule and source" if rule and source else "the rule" if rule else "the source" if source else "neither alone")
    L.append(f"\n**Reading rule (COCO-Cap, CLIP seed).** The walk's advantage is carried by the rule if R-G and R-N are both "
             f"above zero: **{'yes' if rule else 'no'}**. By the attention source if S-H and S-W are both above zero: "
             f"**{'yes' if source else 'no'}**. Result: **{verdict}**.\n")
    L.append(f"Reference (runs/17, CLIP seed): NERVE walk on NERVE pass − BRACE hierarchy on generation = "
             f"{dci('coco', sel('coco', key('clip', 'N|W')), sel('coco', key('clip', 'G|V3')))[0]} here (runs/17: +0.033).\n")
    L.append("## N's sanity convention for the NERVE cell (CLIP × NERVE pass × walk, raw, threshold 0.5)\n")
    for ds in ("voc", "coco"):
        L.append(f"- {NAME[ds]}: {mi(ds, S[ds]['clip|N|W|raw'][:, t50]):.4f} (runs/17 used bilinear-then-min–max; "
                 f"here min–max-then-bilinear, the project's scoring)")
    L.append("")
    for ds in ("voc", "coco"):
        L.append(f"Per class, {NAME[ds]}, VOC-sel. + DenseCRF (" + " / ".join(f"{s.upper()} {c}" for s in SEEDS for c in ("G|V3", "G|W", "N|V3", "N|W")) + "): "
                 + ", ".join(f"{cl} " + " / ".join(f"{mi(ds, sel(ds, key(s, c)), S[ds]['cls'] == cl):.2f}" for s in SEEDS for c in ("G|V3", "G|W", "N|V3", "N|W"))
                             for cl in np.unique(S[ds]["cls"])) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=16); p.add_argument("--tag", default="")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
