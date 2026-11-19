#!/usr/bin/env python3
"""Table 1 ladder on VOC-Sim and COCO-Cap from one live generation per image (PREREG.md in this folder).

Side N. Uses side F's committed frozen runtime and modules unchanged (sg27_runtime, sg25_sa, sg_combined_v2,
sg_class_comp, ovam_full, jc_newrerun); the only new code is a recorder subclass that additionally keeps
(a) the A0 class-word map of the five 32x32 CA modules the frozen seed drops, and (b) conditional SA Q/K of all
16 SA modules for all 31 calls.

capture (GPU)  ~/TRDP/external/n_side/table1_ladder/<ds>/<id>.npz   maps [9, 64, 64] in MAPS order
score   (CPU)  ~/TRDP/external/n_side/table1_ladder/<ds>_scores/<id>.npz   counts raw (+ DenseCRF R2 for CRF maps)
report  (CPU)  comparisons/drafts/table1_ladder/result.md, result.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
EXT = Path.home() / "TRDP/external/n_side"
os.environ.setdefault("BRACE_CACHE_DIR", str(EXT / "table1_ladder/cache"))
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import jc_newrerun as jn  # noqa: E402
from src.segmentation import sg25_sa  # noqa: E402
from src.segmentation.sg25_sa import _apply, _attention, _resize  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = EXT / "table1_ladder"
CRF_SITE = EXT / "deps/simplecrf_py311"
THR = jn.THR
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
MAPS = ("seed16", "seed11", "p1_16sa", "p1_m15", "p2_m15", "v0_a0", "v0_a3", "v3_a3", "seed_a3")
CRF_MAPS = ("p2_m15", "v0_a0", "v0_a3", "v3_a3")
ROWS = (  # (label, key, previous key or None)
    ("1 CA, all 16 modules (reference)", "seed16|raw", None),
    ("2 + drop the five 32-px CA modules", "seed11|raw", "seed16|raw"),
    ("3a one SA pass, all 16 SA modules", "p1_16sa|raw", "seed11|raw"),
    ("3 + one SA pass, 64-px module only", "p1_m15|raw", "seed11|raw"),
    ("4 + second SA pass", "p2_m15|raw", "p1_m15|raw"),
    ("5 + DenseCRF", "p2_m15|crf", "p2_m15|raw"),
    ("6 + SA calls 10-20", "v0_a0|crf", "p2_m15|crf"),
    ("7 + SA-guided CA weights (fine-only reference)", "v0_a3|crf", "v0_a0|crf"),
    ("8 + hierarchy with one-sided cap (BRACE)", "v3_a3|crf", "v0_a3|crf"),
)


def _official_patch_rows():
    import importlib.util
    spec = importlib.util.spec_from_file_location("n_official_brace_repro", ROOT / "comparisons/official/brace_repro.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.patch_rows(EXT / "brace_repro/ovam_coco_cap.csv")


class T1Recorder(sg25_sa.SG25Recorder):
    all_ca: tuple[str, ...] = ()
    all_sa: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_ca = tuple(m for m in self.all_ca if m not in self.ca_modules)
        self.pending_extra = {}
        self.sum16 = torch.zeros((64, 64), device=self.pipe.device)
        self.sa_all = {m: [] for m in self.all_sa}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind == "self":
            if module_name in self.sa_all:
                self.sa_all[module_name].append((query[1].detach().half().clone(), key[1].detach().half().clone()))
        elif module_name in self.extra_ca:      # A0 exactly as SG24Recorder.record builds it
            k = key
            mask = attention_mask
            if attribution_key is not None:
                k = torch.stack((key[0], attribution_key[0])); mask = None
            scores = query[1].float() @ k[1].float().transpose(-1, -2) * scale
            if mask is not None:
                scores += mask[1].float()
            prob = scores.softmax(-1)
            a0 = prob[..., list(self.target)].mean(-1).half().float().mean(0)
            side = math.isqrt(a0.shape[0])
            self.pending_extra[module_name] = a0.reshape(1, 1, side, side)
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)

    def flush_step(self, step, timestep, latent):
        if set(self.pending_extra) != set(self.extra_ca) or not set(self.ca_modules) <= set(self.pending_ca):
            raise ValueError("T1: missing CA maps")
        for m in self.all_ca:
            src = self.pending_extra[m] if m in self.pending_extra else self.pending_ca[m]["A0"]
            up = F.interpolate(src, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
            self.sum16 += up / len(self.all_ca) / 31
        self.pending_extra.clear()
        super().flush_step(step, timestep, latent)

    def clear(self):
        super().clear(); self.pending_extra.clear()
        for v in self.sa_all.values():
            v.clear()


def mm_t(x):
    lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo).clamp_min(1e-20)


def passes(pairs, side, c, n):
    s = torch.zeros(side * side, device=c.device)
    for pair in pairs:
        h = _attention(pair, side); u = _apply(h, c)
        if n == 2:
            u = _apply(h, u)
        s += u.mean(0) / len(pairs); del h, u
    return s


def p1_all_modules(sa_all, c64):
    s = torch.zeros(4096, device=c64.device); n = 0
    for m, pairs in sa_all.items():
        side = math.isqrt(pairs[0][0].shape[1])
        c = c64.reshape(-1) if side == 64 else _resize(c64.reshape(-1), 64, side).reshape(-1)
        for pair in pairs:
            h = _attention(pair, side); y = _apply(h, c).mean(0)
            s += (y if side == 64 else _resize(y, side, 64).reshape(-1)); n += 1; del h, y
    return s / n


def hierarchy(sa, seed):
    from src.segmentation.sg_combined_v2 import cap
    c64 = seed.float().reshape(-1); c16 = _resize(c64, 64, 16).reshape(-1); s = torch.zeros(4096, device=seed.device)
    for call in range(11):
        h16 = _attention(sa[16][call], 16); h32 = _attention(sa[32][call], 32); h64 = _attention(sa[64][call], 64)
        r16 = _apply(h16, c16).mean(0); r32 = _apply(h32, _resize(r16, 16, 32).reshape(-1)).mean(0)
        z32, _ = cap(r16, r32, 16)
        s += _apply(h64, _resize(z32, 32, 64).reshape(-1)).mean(0) / 11
        del h16, h32, h64
    return s


@torch.no_grad()
def capture(a):
    from scripts import sg27_runtime
    _official_patch_rows()
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=T1Recorder); pipe = runtime["pipe"]
    names = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    T1Recorder.all_ca = tuple(sorted(n for n in names if n.endswith("attn2")))
    T1Recorder.all_sa = tuple(sorted(n for n in names if n.endswith("attn1")))
    m15 = T1Recorder.all_sa[15]
    if len(T1Recorder.all_ca) != 16 or len(T1Recorder.all_sa) != 16 or m15 != runtime["modules"][64]:
        raise ValueError("T1: module layout differs from the protocol")
    T1Recorder.enabled = ("A0", "A3"); pipe.safety_checker = None
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for r in jn.rows(a.dataset, a.shard, a.num_shards, a.max_images, a.ids):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(r), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]; sd = rec.seeds()
            if float((sd["selected/A3"].float() - seed.float()).abs().max()) > 0:
                raise ValueError("T1: A3 seed differs from the frozen runtime's seed")
            if any(len(v) != 31 for v in rec.sa_all.values()):
                raise ValueError("T1: expected 31 SA calls per module")
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            s11 = sd["selected/A0"].float().reshape(-1); s16 = mm_t(rec.sum16).reshape(-1); sa3 = seed.float().reshape(-1)
            mp = {"seed16": s16, "seed11": s11, "seed_a3": sa3,
                  "p1_16sa": p1_all_modules(rec.sa_all, s11),
                  "p1_m15": passes(rec.sa_all[m15], 64, s11, 1), "p2_m15": passes(rec.sa_all[m15], 64, s11, 2),
                  "v0_a0": passes(sa[64], 64, s11, 2), "v0_a3": passes(sa[64], 64, sa3, 2),
                  "v3_a3": hierarchy(sa, seed)}
            maps = np.stack([mp[k].reshape(64, 64).float().cpu().numpy() for k in MAPS]).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=r["label"], changed=r["changed"], truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[t1:{a.dataset}] {iid} {r['label']} {time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "changed": bool(z["changed"])}
    for k, m in enumerate(MAPS):
        x = st.up(st.mm(z["maps"][k].astype(np.float32)), truth.shape)
        res[f"{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if m in CRF_MAPS:
            ms = np.stack([apply_ovam_densecrf(img, x >= t, CRF_SITE) for t in THR]).astype(bool)
            res[f"{m}|crf"] = cc.counts(ms, truth, valid, other)
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


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    one = {ds: np.ones(len(S[ds]["grp"]), bool) for ds in S}
    mi = lambda ds, P: float(cc.miou(P[:, None], S[ds]["cls"], one[ds])[0])
    i45 = int(np.flatnonzero(np.isclose(THR, 0.45))[0]); i55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
    tb = lambda key: int(np.argmax(cc.miou(S["voc"][key], S["voc"]["cls"], one["voc"])))

    def at(ds, key, mode):
        X = S[ds][key]
        if mode == "0.45":
            return X[:, i45]
        if mode == "0.55":
            return X[:, i55]
        return cc.oof(S["voc"], X)[1] if ds == "voc" else X[:, tb(key)]

    def fold_wins(ka, kb):
        Pa, Pb = cc.oof(S["voc"], S["voc"][ka])[1], cc.oof(S["voc"], S["voc"][kb])[1]; w = 0
        for fo in range(5):
            sel = S["voc"]["fold"] == fo
            w += cc.miou(Pa[:, None], S["voc"]["cls"], sel)[0] > cc.miou(Pb[:, None], S["voc"]["cls"], sel)[0]
        return int(w)

    def dci(ds, ka, kb, mode):
        Pa, Pb = at(ds, ka, mode), at(ds, kb, mode); lo, hi = cc.boot(S[ds], Pa, Pb, n=a.boot)
        return {"d": mi(ds, Pa) - mi(ds, Pb), "lo": float(lo), "hi": float(hi)}

    J = {"n": {ds: int(len(S[ds]["grp"])) for ds in S}, "tau_voc_best": {}, "rows": []}
    for lab, key, prev in ROWS:
        row = {"label": lab, "key": key, "prev": prev}
        for ds in S:
            for mode in ("sel", "0.45", "0.55"):
                row[f"{ds}_{mode}"] = mi(ds, at(ds, key, mode))
        J["tau_voc_best"][key] = float(THR[tb(key)])
        if prev:
            row["folds_won"] = fold_wins(key, prev)
            for ds in S:
                for mode in ("sel", "0.55"):
                    row[f"d_{ds}_{mode}"] = dci(ds, key, prev, mode)
        J["rows"].append(row); print(lab, flush=True)
    f = lambda x: f"{x:.3f}"
    g = lambda c: f"{c['d']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
    L = ["# Table 1 ladder on both benchmarks, one capture (`PREREG.md`)\n",
         "Side N, `t1_ladder.py`. One live generation per image (frozen runtime, runs/20 corrected prompts, safety checker "
         f"off), {J['n']['voc']} VOC-Sim and {J['n']['coco']} COCO-Cap images. Class-macro mIoU. *VOC-sel.* = 5-fold OOF on "
         "VOC-Sim, full-VOC-Sim best threshold unchanged on COCO-Cap. Δ = row minus the row it builds on (column *vs*), "
         f"paired image bootstrap 95% interval ({a.boot}). Folds won: VOC-Sim held-out folds (of 5) where the row beats "
         "the row it builds on, thresholds chosen on the other four. Class-word results are in-sample for the method design.\n",
         "## Main table (VOC-sel.)\n",
         "| Row | vs | VOC-Sim | Δ VOC-Sim | folds won | COCO-Cap | Δ COCO-Cap | τ (VOC-best) |", "|---|---|---:|---|---:|---:|---|---:|"]
    num = {key: lab.split()[0] for lab, key, _ in ROWS}
    for r in J["rows"]:
        L.append(f"| {r['label']} | {num.get(r['prev'], '—') if r['prev'] else '—'} | {f(r['voc_sel'])} | "
                 f"{g(r['d_voc_sel']) if r['prev'] else '—'} | {r.get('folds_won', '—')} | {f(r['coco_sel'])} | "
                 f"{g(r['d_coco_sel']) if r['prev'] else '—'} | {J['tau_voc_best'][r['key']]:.3f} |")
    L += ["\n## Fixed thresholds (0.45 = early screens; 0.55 = BRACE's frozen threshold)\n",
          "| Row | VOC-Sim 0.45 | COCO-Cap 0.45 | VOC-Sim 0.55 | Δ VOC-Sim 0.55 | COCO-Cap 0.55 | Δ COCO-Cap 0.55 |",
          "|---|---:|---:|---:|---|---:|---|"]
    for r in J["rows"]:
        L.append(f"| {r['label']} | {f(r['voc_0.45'])} | {f(r['coco_0.45'])} | {f(r['voc_0.55'])} | "
                 f"{g(r['d_voc_0.55']) if r['prev'] else '—'} | {f(r['coco_0.55'])} | {g(r['d_coco_0.55']) if r['prev'] else '—'} |")
    raw = [("5 raw (no DenseCRF)", "p2_m15|raw"), ("6 raw", "v0_a0|raw"), ("7 raw", "v0_a3|raw"), ("8 raw", "v3_a3|raw")]
    L += ["\n## Raw versions of rows 5-8 (secondary)\n", "| Row | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | VOC-Sim 0.55 | COCO-Cap 0.55 |",
          "|---|---:|---:|---:|---:|"]
    for lab, key in raw:
        L.append(f"| {lab} | {f(mi('voc', at('voc', key, 'sel')))} | {f(mi('coco', at('coco', key, 'sel')))} | "
                 f"{f(mi('voc', at('voc', key, '0.55')))} | {f(mi('coco', at('coco', key, '0.55')))} |")
    txt = "\n".join(L) + "\n"; print(txt)
    (HERE / f"result{a.tag}.md").write_text(txt); (HERE / f"result{a.tag}.json").write_text(json.dumps(J, indent=1))


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
    if a.cmd != "capture":
        _official_patch_rows()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
