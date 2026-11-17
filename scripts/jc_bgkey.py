#!/usr/bin/env python3
"""Background-aware cross-attention seed (runs/18_bg_key_seed/PREREG.md).

One live generation per image with the frozen runtime. Seeds differ only in the attribution keys:
  W   frozen class-word attribution (BRACE)
  S1  [<SoT>, class]            B8 / BN  [<SoT>, class, background words]      R8 / RN  [class, background words]
Class map = sum of the softmax columns of the class word's pieces; conditional query, head mean, frozen layers,
calls and SA-guided layer weights. Each seed goes through the frozen propagation (sg_combined_v2.aggregate).

capture (GPU)  <BRACE_CACHE_DIR>/jc_bgkey/<ds>/<id>.npz: maps [12 seeds, 3 maps, 64, 64]
score   (CPU)  counts per cell at 17 thresholds, raw and + DenseCRF
report  (CPU)  runs/18_bg_key_seed/result.md
"""
from __future__ import annotations

import argparse, math, os, sys, time
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
from src.segmentation import sg25_sa  # noqa: E402

WORK = CACHE / "jc_bgkey"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
OUT = ROOT / "runs/18_bg_key_seed"
GEN8 = ["wall", "floor", "ground", "sky", "grass", "tree", "building", "road"]
NERVE25 = ("ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, railroad, keyboard, "
           "helmet, cloud, house, mountain, ocean, road, rock, street, valley, bridge, sign").split(", ")
CAND = ("S1", "B8", "R8", "BN", "RN")
UNI = tuple(c + "u" for c in CAND)          # the same key sets with uniform layer weights
SEEDS = ("W",) + CAND + ("W0",) + UNI
FORMS = {"S1": (True, ()), "B8": (True, GEN8), "R8": (False, GEN8), "BN": (True, NERVE25), "RN": (False, NERVE25)}
MAPS = ("seed", "V0", "V3")
THR = np.round(0.30 + 0.025 * np.arange(17), 3)
TAU = 0.55
DESC = {"W": "frozen class-word attribution (BRACE)", "S1": "⟨SoT⟩ + class word", "B8": "⟨SoT⟩ + class + 8 generic words",
        "R8": "class + 8 generic words", "BN": "⟨SoT⟩ + class + NERVE's 25 words", "RN": "class + NERVE's 25 words",
        "W0": "frozen class-word attribution, uniform layer weights"}
DESC.update({c + "u": DESC[c] + ", uniform layer weights" for c in CAND})


class BgKeyRecorder(sg25_sa.SG25Recorder):
    """Frozen seeds plus class-share seeds for several key sets, in one generation."""

    key_sets = None   # {name: (states [1, n, 768], number of class columns, class columns start)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_bg = {}
        self.bg_sums = {k: torch.zeros((64, 64), device=self.pipe.device) for k in CAND + UNI}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)
        if kind == "self" or module_name not in self.ca_modules:
            return
        with torch.no_grad():
            module = self.pipe.unet.get_submodule(module_name); q = query[1].float(); res = {}
            for name, (states, lo, hi) in self.key_sets.items():
                s = states.to(module.to_k.weight.dtype)
                if module.norm_cross:
                    s = module.norm_encoder_hidden_states(s)
                k = module.to_k(s).reshape(1, s.shape[1], module.heads, -1).permute(0, 2, 1, 3)[0].float()  # [heads, n, d]
                prob = (q @ k.transpose(-1, -2) * scale).softmax(-1)[..., lo:hi].sum(-1).mean(0)            # [P]
                side = math.isqrt(prob.shape[0]); res[name] = prob.reshape(1, 1, side, side)
            self.pending_bg[module_name] = res

    def flush_step(self, step, timestep, latent):
        pend = dict(self.pending_bg); self.pending_bg.clear()
        super().flush_step(step, timestep, latent)
        _, label, w = self.weights[-1]
        if label != "selected" or set(pend) != set(self.selected_modules):
            raise ValueError("bgkey: layer weights or maps missing")
        w = torch.as_tensor(w, device=self.pipe.device).float()
        for i, m in enumerate(self.selected_modules):
            for name in CAND:
                up = F.interpolate(pend[m][name], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
                self.bg_sums[name] += up * w[i] / 31
                self.bg_sums[name + "u"] += up / len(self.selected_modules) / 31

    def bg_seeds(self):
        out = {}
        for k, x in self.bg_sums.items():
            lo, hi = x.min(), x.max(); out[k] = (x - lo) / (hi - lo).clamp_min(1e-20)
        return out


def word_states(pipe, word):
    """Text-encoder outputs of a word encoded on its own: (<SoT> state, [pieces, 768])."""
    with torch.inference_mode():
        ids = pipe.tokenizer(word, return_tensors="pt").input_ids.to(pipe.device)     # [<SoT>, pieces..., <EoT>]
        h = pipe.text_encoder(ids)[0][0].float()
    return h[0], h[1:-1]


def key_sets(pipe, classname, cache):
    if "sot" not in cache:
        cache["sot"] = word_states(pipe, "")[0]
        for w in dict.fromkeys(GEN8 + NERVE25):
            cache[w] = word_states(pipe, w)[1]
    cw = word_states(pipe, classname.replace("-", " "))[1]; n = cw.shape[0]; out = {}
    for name, (sot, words) in FORMS.items():
        parts = ([cache["sot"][None]] if sot else []) + [cw] + [cache[w] for w in words]
        lo = 1 if sot else 0
        out[name] = (torch.cat(parts)[None], lo, lo + n)
    return out, n


def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:
        rows = rows[:: max(1, len(rows) // a.max_images)][: a.max_images]
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=BgKeyRecorder)
    BgKeyRecorder.enabled = ("A0", "A3")
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True); cache = {}
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        BgKeyRecorder.key_sets, npiece = key_sets(runtime["pipe"], cls, cache)
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            word = rec.seeds(); sd = dict(rec.bg_seeds()); sd["W"] = word["selected/A3"]; sd["W0"] = word["selected/A0"]
            if float((sd["W"].float() - seed.float()).abs().max()) > 0:
                raise ValueError("bgkey: A3 seed differs from the frozen runtime's seed")
            maps = []
            for s in SEEDS:
                x = sd[s].float().clone()
                if not torch.isfinite(x).all():
                    raise ValueError(f"bgkey: non-finite seed {s} for {iid}")
                v, _ = sg_combined_v2.aggregate(sa, x)
                maps.append(np.stack([x.cpu().numpy()] + [v[k].float().cpu().numpy() for k in MAPS[1:]])); del v
            maps = np.stack(maps).astype(np.float32)
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        w = maps[0]
        chk = max(float(np.abs(st.mm(w[2]) - st.mm(stored["map/base"][3])).max()),
                  float(np.abs(st.mm(w[1]) - st.mm(stored["map/base"][2])).max()))
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps, cls=cls, check=chk, img_maxdiff=img_diff, pieces=npiece, truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[bgkey:{a.dataset}] {iid} {cls} pieces {npiece} img {img_diff} check {chk:.1e} "
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
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "check": float(z["check"]),
           "img_maxdiff": int(z["img_maxdiff"]), "pieces": int(z["pieces"])}
    for si, s in enumerate(SEEDS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][si, mi]), truth.shape)
            res[f"{s}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            if m != "seed":
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
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


CELLS = (("seed alone", "seed"), ("fine-only", "V0"), ("capped hierarchy", "V3"), ("fine-only + DenseCRF", "V0+R2"),
         ("capped hierarchy + DenseCRF", "V3+R2"))


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    t55 = int(np.flatnonzero(np.isclose(THR, TAU))[0])

    def mi(ds, P, s=None):
        s = one[ds] if s is None else s
        return cc.miou(P[:, None], S[ds]["cls"], s)[0] if s.any() else float("nan")

    def tbest(key):
        return int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))

    def sel(ds, key):
        if ds == "voc":
            return cc.oof(S["voc"], S["voc"][f"{key}|raw"])[1]
        return S["coco"][f"{key}|raw"][:, tbest(key)]

    def fx(ds, key):
        return S[ds][f"{key}|raw"][:, t55]

    def dci(ds, Pa, Pb):
        if n[ds] < 50:
            return f"{mi(ds, Pa) - mi(ds, Pb):+.3f}", float("nan")
        lo, hi = cc.boot(S[ds], Pa, Pb, n=2000); return f"{mi(ds, Pa) - mi(ds, Pb):+.3f} [{lo:+.3f}, {hi:+.3f}]", lo

    iou = lambda P: P[:, 0] / np.maximum(P[:, :3].sum(1), 1)
    L = ["# Background-aware cross-attention seed (`PREREG.md`)\n",
         "`scripts/jc_bgkey.py`. One live generation per image with the frozen runtime; seeds differ only in the "
         "attribution keys. Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; "
         "full-VOC-Sim-best threshold applied unchanged on COCO-Cap). Δ vs W with image-bootstrap 95% interval (2000). "
         "Stable Diffusion only.\n", "| Seed | Keys |", "|---|---|"] + [f"| {s} | {DESC[s]} |" for s in SEEDS] + [""]
    for ds in ("voc", "coco"):
        Z = [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz"))]
        ex = sum(int(z["img_maxdiff"]) == 0 for z in Z); chk = np.median([float(z["check"]) for z in Z])
        L.append(f"## {NAME[ds]} ({n[ds]} images)\n")
        L.append(f"Regeneration: {ex}/{n[ds]} images bit-identical to the stored image; W maps vs the stored capture: "
                 f"median max |Δ| {chk:.1e}. W full pipeline at 0.55: {mi(ds, fx(ds, 'W|V3+R2')):.3f}.\n")
        for conv, f in (("VOC-sel. threshold", sel), ("fixed threshold 0.55", fx)):
            L.append(f"**{conv}**\n"); L.append("| Cell | " + " | ".join(SEEDS) + " |"); L.append("|---|" + "---:|" * len(SEEDS))
            for lab, m in CELLS:
                L.append(f"| {lab} | " + " | ".join(f"{mi(ds, f(ds, f'{s}|{m}')):.3f}" for s in SEEDS) + " |")
            L.append("")
        L.append("**Δ vs W, full pipeline (capped hierarchy + DenseCRF)**\n")
        L.append("| Seed | VOC-sel. | fixed 0.55 | hierarchy − fine-only on this seed (both + DenseCRF, VOC-sel.) | hierarchy − seed alone (raw, VOC-sel.) | SA-guided − uniform layer weights (full pipeline, VOC-sel.) |"); L.append("|---|---|---|---|---|---|")
        for s in SEEDS:
            h = dci(ds, sel(ds, f"{s}|V3+R2"), sel(ds, f"{s}|V0+R2"))[0]
            g = dci(ds, sel(ds, f"{s}|V3"), sel(ds, f"{s}|seed"))[0]
            u = "W0" if s == "W" else s + "u"
            lw = dci(ds, sel(ds, f"{s}|V3+R2"), sel(ds, f"{u}|V3+R2"))[0] if u in SEEDS else "—"
            if s == "W":
                L.append(f"| W | — | — | {h} | {g} | {lw} |"); continue
            L.append(f"| {s} | {dci(ds, sel(ds, f'{s}|V3+R2'), sel(ds, 'W|V3+R2'))[0]} | {dci(ds, fx(ds, f'{s}|V3+R2'), fx(ds, 'W|V3+R2'))[0]} | {h} | {g} | {lw} |")
        L.append("")
    oofm = {s: mi("voc", sel("voc", f"{s}|V3+R2")) for s in SEEDS}
    win = max(CAND, key=lambda s: oofm[s])
    dv = oofm[win] - oofm["W"]; txt_c, lo_c = dci("coco", sel("coco", f"{win}|V3+R2"), sel("coco", "W|V3+R2"))
    ok = (lo_c > 0) and dv >= -0.005
    L.append("## Registered selection and test\n")
    L.append("VOC-Sim out-of-fold mIoU, full pipeline: " + ", ".join(f"{s} {oofm[s]:.4f}" for s in SEEDS) + ".\n")
    L.append(f"- **Selected on VOC-Sim:** {win} ({DESC[win]}); VOC-Sim Δ vs W {dci('voc', sel('voc', f'{win}|V3+R2'), sel('voc', 'W|V3+R2'))[0]}.")
    L.append(f"- **COCO-Cap test:** {win} − W = {txt_c} (thresholds {THR[tbest(f'{win}|V3+R2')]:.3f} / {THR[tbest('W|V3+R2')]:.3f}).")
    L.append(f"- **Verdict: {'PASS' if ok else 'FAIL'}** (rule: COCO-Cap interval above zero and VOC-Sim Δ ≥ −0.005).\n")
    L.append("## Registered secondary readouts, COCO-Cap, full pipeline at the VOC-sel. threshold (descriptive)\n")
    area = S["coco"]["area"]; b = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3])); cls = S["coco"]["cls"]
    L.append("| Seed | thr | mIoU | per-image IoU: small third | middle | large | images below 0.3 IoU | macro precision | macro recall |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        P = sel("coco", f"{s}|V3+R2"); x = iou(P); pr, rc = [], []
        for c in np.unique(cls):
            q = P[cls == c].sum(0); pr.append(q[0] / max(q[0] + q[1], 1)); rc.append(q[0] / max(q[0] + q[2], 1))
        L.append(f"| {s} | {THR[tbest(f'{s}|V3+R2')]:.3f} | {mi('coco', P):.3f} | " + " | ".join(f"{x[b == k].mean():.3f}" for k in range(3))
                 + f" | {np.mean(x < 0.3):.3f} | {np.mean(pr):.3f} | {np.mean(rc):.3f} |")
    L.append("")
    for ds in ("voc", "coco"):
        L.append(f"Per class, {NAME[ds]}, full pipeline, VOC-sel. ({' / '.join(SEEDS)}): " + ", ".join(
            f"{c} " + " / ".join(f"{mi(ds, sel(ds, f'{s}|V3+R2'), S[ds]['cls'] == c):.2f}" for s in SEEDS)
            for c in np.unique(S[ds]["cls"])) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--workers", type=int, default=16); p.add_argument("--tag", default="")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
