#!/usr/bin/env python3
"""Component ladder with the class word and with OVAM's optimized token (runs/15_token_ladder/PREREG.md).

One live generation per image with the frozen runtime. From that generation, two query columns:
  W  class word, frozen attribution (unchanged)
  T  OVAM's released VOC-Sim token: keys [<SoT>, token], softmax over the two, conditional query, head mean
Seeds per column: A0 (uniform layer weights) and A3 (SA-guided layer weights; the weights are text-independent,
so both columns share them). Each seed is propagated with the frozen code (sg_combined_v2.aggregate).

capture (GPU)  <BRACE_CACHE_DIR>/jc_ladder/<ds>/<id>.npz: maps [4 seeds, 4 maps, 64, 64]
score   (CPU)  counts per cell, raw (tau 0.30..0.90) and +R2 (tau step 0.05)
report  (CPU)  runs/15_token_ladder/result.md
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

WORK = CACHE / "jc_ladder"
STORED = {"voc": CACHE / "vocsim_maps", "coco": CACHE / "cococap_maps"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
TOKENS = ROOT / "configs/ovam/tokens_VOC.npy"
SEEDS = ("wA0", "wA3", "tA0", "tA3")          # query (w/t) x layer weights (uniform / SA-guided)
MAPS = ("seed", "V0", "V1", "V3")
TAU = 0.55
OUT = ROOT / "runs/15_token_ladder"


class LadderRecorder(sg25_sa.SG25Recorder):
    """Frozen A0/A3 class-word seeds plus the same two seeds for a token attribution, in one generation."""

    token_states = None  # [1, 2, 768]: (<SoT>, token)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_tok = {}
        self.tok_sums = {k: torch.zeros((64, 64), device=self.pipe.device) for k in ("A0", "A3")}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)
        if kind == "self" or module_name not in self.ca_modules:
            return
        with torch.no_grad():
            module = self.pipe.unet.get_submodule(module_name)
            states = self.token_states.to(module.to_k.weight.dtype)
            if module.norm_cross:
                states = module.norm_encoder_hidden_states(states)
            k = module.to_k(states).reshape(1, 2, module.heads, -1).permute(0, 2, 1, 3)[0].float()  # [heads, 2, d]
            prob = (query[1].float() @ k.transpose(-1, -2) * scale).softmax(-1)[..., 1].mean(0)      # [P]
            side = math.isqrt(prob.shape[0])
            self.pending_tok[module_name] = prob.reshape(1, 1, side, side)

    def flush_step(self, step, timestep, latent):
        pend = dict(self.pending_tok); self.pending_tok.clear()
        super().flush_step(step, timestep, latent)
        _, label, w = self.weights[-1]
        if label != "selected" or set(pend) != set(self.selected_modules):
            raise ValueError("ladder: layer weights or token maps missing")
        w = torch.as_tensor(w, device=self.pipe.device).float()
        for i, m in enumerate(self.selected_modules):
            up = F.interpolate(pend[m], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
            self.tok_sums["A3"] += up * w[i] / 31
            self.tok_sums["A0"] += up / len(self.selected_modules) / 31

    def token_seeds(self):
        out = {}
        for k, x in self.tok_sums.items():
            lo, hi = x.min(), x.max(); out[k] = (x - lo) / (hi - lo).clamp_min(1e-20)
        return out


def token_states(pipe, released, classname):
    tok = torch.from_numpy(np.asarray(released[classname.replace(" ", "-")], np.float32)).to(pipe.device)
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(pipe.device)
        sot = pipe.text_encoder(empty)[0][0, 0].float()  # CLIP is causal: <SoT> output is prompt-independent
    return torch.stack((sot, tok))[None]


def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:
        rows = rows[: a.max_images]
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=LadderRecorder)
    LadderRecorder.enabled = ("A0", "A3")
    released = np.load(TOKENS, allow_pickle=True).item()
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); stored = np.load(STORED[a.dataset] / f"{iid}.npz")
        LadderRecorder.token_states = token_states(runtime["pipe"], released, cls)
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
                maps.append(np.stack([x.cpu().numpy()] + [v[k].float().cpu().numpy() for k in MAPS[1:]]))
                del v
            maps = np.stack(maps).astype(np.float32)
        img_diff = int(np.abs(img.astype(int) - stored["image"].astype(int)).max())
        img_mean = float(np.abs(img.astype(int) - stored["image"].astype(int)).mean())
        w3 = maps[SEEDS.index("wA3")]
        v3chk = float(np.abs(st.mm(w3[MAPS.index("V3")]) - st.mm(stored["map/base"][3])).max())
        v0chk = float(np.abs(st.mm(w3[MAPS.index("V0")]) - st.mm(stored["map/base"][2])).max())
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps, cls=cls, check=max(v3chk, v0chk), img_maxdiff=img_diff, img_meandiff=img_mean,
                 truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[ladder:{a.dataset}] {iid} {cls} img {img_diff} V3/V0 check {v3chk:.1e}/{v0chk:.1e} "
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
           "img_maxdiff": int(z["img_maxdiff"]), "img_meandiff": float(z["img_meandiff"])}
    for si, s in enumerate(SEEDS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][si, mi]), truth.shape)
            res[f"{s}|{m}|raw"] = cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other)
            if m in ("V0", "V3"):
                ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in cc.THR_CRF]).astype(bool)
                res[f"{s}|{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=2), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


# rows of the ladder: (label, cell suffix, reference row index or None)
ROWS = (("1", "CA seed, uniform layer weights", "A0|seed", None),
        ("2", "CA seed, SA-guided layer weights", "A3|seed", 0),
        ("3", "+ fine-only SA propagation (two passes)", "A3|V0", 1),
        ("4", "+ 16→32→64 hierarchy, no cap", "A3|V1", 2),
        ("5", "+ one-sided cap at 16→32", "A3|V3", 3),
        ("6", "+ DenseCRF (full pipeline)", "A3|V3+R2", 4),
        ("3c", "fine-only + DenseCRF", "A3|V0+R2", None),
        ("6u", "full pipeline, uniform layer weights", "A0|V3+R2", None))
TESTS = (("H", "capped hierarchy vs fine-only (both + DenseCRF)", "A3|V3+R2", "A3|V0+R2"),
         ("L", "SA-guided vs uniform layer weights (full pipeline)", "A3|V3+R2", "A0|V3+R2"),
         ("C", "cap vs no cap (raw)", "A3|V3", "A3|V1"))


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in a.sets}
    n = {ds: len(S[ds]["grp"]) for ds in a.sets}
    one = {ds: np.ones(n[ds], bool) for ds in a.sets}

    def grid(key):
        return cc.THR_CRF if "+R2" in key else cc.THR

    def fx(ds, key):                       # counts at fixed 0.55
        return S[ds][f"{key}|raw"][:, int(np.flatnonzero(np.isclose(grid(key), TAU))[0])]

    def sel(ds, key):                      # VOC-selected threshold: OOF on VOC-Sim, transferred to COCO-Cap
        if ds == "voc":
            m, P, _ = cc.oof(S["voc"], S["voc"][f"{key}|raw"]); return P, None
        ti = int(np.argmax(cc.miou(S["voc"][f"{key}|raw"], S["voc"]["cls"], one["voc"])))
        return S[ds][f"{key}|raw"][:, ti], float(grid(key)[ti])

    def mi(ds, P, s=None):
        s = one[ds] if s is None else s
        return cc.miou(P[:, None], S[ds]["cls"], s)[0] if s.any() else float("nan")  # empty fold (smoke runs only)

    def dci(ds, Pa, Pb):
        lo, hi = cc.boot(S[ds], Pa, Pb, n=2000); return f"{mi(ds, Pa) - mi(ds, Pb):+.3f} [{lo:+.3f}, {hi:+.3f}]", lo

    def folds(Pa, Pb):
        return sum(mi("voc", Pa, S["voc"]["fold"] == f) > mi("voc", Pb, S["voc"]["fold"] == f) for f in range(5))

    L = ["# Component ladder: class word vs OVAM's optimized token (`PREREG.md`)\n",
         "`scripts/jc_ladder.py`. One live generation per image with the frozen runtime; both query columns, all rows, "
         "from that generation. **W** = class word (frozen attribution). **T** = OVAM's released VOC-Sim token "
         "(few-shot: needs annotated images; the same VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *@0.55* = fixed "
         "threshold (primary). *VOC-sel.* = threshold selected on VOC-Sim: 5-fold out-of-fold on VOC-Sim, the "
         "VOC-Sim-best threshold applied unchanged on COCO-Cap. Δ with image-bootstrap 95% interval (2000). Class-word "
         "results are in-sample for the method design (these sets were used to develop it).\n"]
    for ds in a.sets:
        Z = [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz"))]
        ex = np.array([int(z["img_maxdiff"]) == 0 for z in Z]); chk = np.array([float(z["check"]) for z in Z])
        L.append(f"## {NAME[ds]} ({n[ds]} images)\n")
        L.append(f"Regeneration: {ex.sum()}/{n[ds]} images bit-identical to the stored image; class-word maps vs the "
                 f"stored capture: median max |Δ| {np.median(chk):.1e}, 95th percentile {np.percentile(chk, 95):.1e}.\n")
        L.append("| Row | Step | W @0.55 | Δ W vs ref | T @0.55 | Δ T vs ref | T − W @0.55 | W VOC-sel. | T VOC-sel. |"
                 + (" folds W | folds T |" if ds == "voc" else ""))
        L.append("|---|---|---:|---|---:|---|---|---:|---:|" + ("---:|---:|" if ds == "voc" else ""))
        for i, (rid, lab, key, ref) in enumerate(ROWS):
            cells = []
            for q in "wt":
                P = fx(ds, q + key); cells.append(f"{mi(ds, P):.3f}")
                cells.append("—" if ref is None else dci(ds, P, fx(ds, q + ROWS[ref][2]))[0])
            tw = dci(ds, fx(ds, "t" + key), fx(ds, "w" + key))[0]
            vs = [f"{mi(ds, sel(ds, q + key)[0]):.3f}" for q in "wt"]
            fo = ""
            if ds == "voc":
                fo = " " + " | ".join("—" if ref is None else f"{folds(sel('voc', q + key)[0], sel('voc', q + ROWS[ref][2])[0])}/5"
                                      for q in "wt") + " |"
            L.append(f"| {rid} | {lab} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {tw} | {vs[0]} | {vs[1]} |{fo}")
        L.append("\nReference of each Δ: the row above (rows 2–6). Row 6 − row 5 is the DenseCRF effect (raw to DenseCRF); "
                 "all other Δ compare like with like.\n")
    L.append("## Registered comparisons (@0.55)\n")
    L.append("| ID | Comparison | Query | VOC-Sim Δ | COCO-Cap Δ | Verdict |"); L.append("|---|---|---|---|---|---|")
    for tid, lab, ka, kb in TESTS:
        for q, qn in (("t", "optimized token (registered)"), ("w", "class word (rerun, reference)")):
            d = {ds: dci(ds, fx(ds, q + ka), fx(ds, q + kb)) for ds in ("voc", "coco") if ds in a.sets}
            verdict = "—"
            if len(d) == 2:
                dv = mi("voc", fx("voc", q + ka)) - mi("voc", fx("voc", q + kb))
                ok = d["coco"][1] > 0 and dv >= -0.005
                verdict = ("**holds**" if ok else "**does not hold**") if q == "t" else ("holds" if ok else "does not hold")
            L.append(f"| {tid} | {lab} | {qn} | {d.get('voc', ('—',))[0]} | {d.get('coco', ('—',))[0]} | {verdict} |")
    q6 = {ds: dci(ds, fx(ds, "tA3|V3+R2"), fx(ds, "wA3|V3+R2"))[0] for ds in a.sets}
    L.append(f"| Q | optimized token vs class word, full pipeline | descriptive | {q6.get('voc', '—')} | {q6.get('coco', '—')} | — |")
    L.append("\nRule (H, L, C separately): holds if the COCO-Cap Δ is positive with its interval above zero and the "
             "VOC-Sim Δ is not below −0.005.\n")
    for ds in a.sets:
        L.append(f"Per class, {NAME[ds]}, full pipeline @0.55 (W / T): " + ", ".join(
            f"{c} {mi(ds, fx(ds, 'wA3|V3+R2'), S[ds]['cls'] == c):.2f} / {mi(ds, fx(ds, 'tA3|V3+R2'), S[ds]['cls'] == c):.2f}"
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
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
