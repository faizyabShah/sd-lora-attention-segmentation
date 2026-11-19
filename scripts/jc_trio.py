#!/usr/bin/env python3
"""runs/28_trio_screen/PREREG.md: three training-free ideas, COCO-Cap screen (exploratory).

  A   BRACE (runtime A3 seed, V3)            X1  SDXL cross-attention seed -> V3
  X2  sqrt(BRACE seed * SDXL seed) -> V3     N1  absent-class-calibrated seed -> V3
  K1  V3 with class competition at 16/32/64  K0  competition on the final maps only

stats    (GPU, sharded) per-image absent/object query statistics  -> <CACHE>/jc_trio/stats{tag}/
fit      (CPU) aggregate statistics                                 -> <CACHE>/jc_trio/fit{tag}.pt
capture  (GPU, sharded) maps [6 arms, 2 maps (seed, V3), 64, 64]     -> <CACHE>/jc_trio/coco{tag}/
score    (CPU) raw and + DenseCRF R2 counts at 0.30-0.70
report   (CPU) runs/28_trio_screen/result{tag}.md
"""
from __future__ import annotations

import argparse, json, math, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_shadow as J  # noqa: E402

cc, st, nr = J.cc, J.st, J.nr
from src.segmentation.sg25_sa import _attention, _resize  # noqa: E402
from src.segmentation.sg_combined_v2 import cap  # noqa: E402

WORK = J.CACHE / "jc_trio"
OUT = ROOT / "runs/28_trio_screen"
ARMS = ("A", "X1", "X2", "N1", "K1", "K0")
MAPS = ("seed", "V3")
THR, T55 = J.THR, J.T55
CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog",
           "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv")
SYN = {"aeroplane": "aeroplane airplane plane jet aircraft airliner", "bicycle": "bicycle bike cyclist",
       "bird": "bird duck gull seagull pigeon parrot eagle owl goose geese swan", "boat": "boat ship sailboat kayak canoe yacht ferry",
       "bottle": "bottle", "bus": "bus", "car": "car taxi suv sedan", "cat": "cat kitten kitty",
       "chair": "chair", "cow": "cow cattle bull calf calves ox oxen", "dining table": "table",
       "dog": "dog puppy pup", "horse": "horse pony foal", "motorbike": "motorbike motorcycle scooter moped",
       "person": "person people man men woman women boy girl child children kid player guy lady rider skier surfer skateboarder baby",
       "potted plant": "plant houseplant", "sheep": "sheep lamb", "sofa": "sofa couch", "train": "train locomotive",
       "tv": "tv television monitor screen"}
XL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
XL_T = (100, 300, 500)
TOP_SPUR, TOP_OBJ, GAIN = 0.01, 0.05, 4.0


def mentions(caption, c):
    words = SYN[c].split()
    return re.search(r"\b(" + "|".join(words) + r")(s|es)?\b", caption.lower()) is not None


def _k(name):
    return name.replace(".", "_")


# ---------------------------------------------------------------- SD1.5 class keys and seeds
@torch.no_grad()
def class_keys(rt, classes):
    """{c: ({module: k [heads, 77, d] f32}, target positions)} from the class-word prompt (COCO-Cap attribution)."""
    from src.segmentation.attention_aggregation import find_token_subsequence
    pipe = rt["pipe"]; tok = pipe.tokenizer; K = {}
    for c in classes:
        ids = tok(c, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(pipe.device)
        emb = pipe.text_encoder(ids)[0]
        tgt = list(find_token_subsequence(ids[0].tolist(), tok(c, add_special_tokens=False)["input_ids"]))
        kd = {}
        for m in rt["ca_modules"]:
            mod = pipe.unet.get_submodule(m); k = mod.to_k(emb); d = k.shape[-1] // mod.heads
            kd[m] = k.view(1, -1, mod.heads, d).transpose(1, 2)[0].float()
        K[c] = (kd, tgt)
    return K


def qbar(rec):
    """Call-averaged head-merged conditional CA query per module, rows L2-normalised."""
    s, n = {}, {}
    for step, name, q, k, mask, scale in rec.ca_store:
        m = q.float().permute(1, 0, 2).reshape(q.shape[1], -1)
        s[name] = s.get(name, 0) + m; n[name] = n.get(name, 0) + 1
    return {m: F.normalize(s[m] / n[m], dim=-1) for m in s}


def ca_seeds(rec, K, gate=None, tbar=False):
    """A3 seeds for each entry of K ({label: (kdict or None for the recorded target key, target)}); min-max.
    gate: {label: {module: [N]}} multiplies the head-mean attention. tbar: also call-averaged maps per module."""
    wmap = {s: w for s, lab, w in rec.weights if lab == "selected"}
    if len(wmap) != 31:
        raise ValueError("expected 31 selected weight vectors")
    mods = list(rec.selected_modules); dev = rec.pipe.device
    acc = {c: torch.zeros((64, 64), device=dev) for c in K}; ts, tn = {}, {}
    for step, name, q, kst, mask, scale in rec.ca_store:
        q = q.float(); w = float(wmap[step][mods.index(name)])
        for c, (kd, tgt) in K.items():
            k = kst.float() if kd is None else kd[name]
            s = q @ k.transpose(-1, -2) * scale
            if kd is None and mask is not None:
                s = s + mask.float()
            t = s.softmax(-1)[..., tgt].mean(-1).mean(0)
            if gate is not None and c in gate:
                t = t * gate[c][name]
            if tbar:
                ts[(c, name)] = ts.get((c, name), 0) + t; tn[(c, name)] = tn.get((c, name), 0) + 1
            side = math.isqrt(t.shape[0])
            acc[c] = acc[c] + F.interpolate(t.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * w / 31
    out = {c: J.mm_t(v) for c, v in acc.items()}
    return (out, {k: v / tn[k] for k, v in ts.items()}) if tbar else out


# ---------------------------------------------------------------- SDXL reader
class XL:
    def __init__(self, dev):
        from diffusers import StableDiffusionXLPipeline
        p = StableDiffusionXLPipeline.from_pretrained(XL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        p.to(dev); p.vae.to(torch.float32); p.set_progress_bar_config(disable=True); self.p, self.dev = p, dev
        self.kattr, self.tgt, self.acc, self.n = None, None, None, 0
        self.mods = [m for n, m in p.unet.named_modules() if n.endswith("attn2")]
        for m in self.mods:
            m.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs):
        if self.kattr is None:
            return
        h = args[0] if args else kwargs["hidden_states"]
        q, k = mod.to_q(h), mod.to_k(self.kattr); d = q.shape[-1] // mod.heads
        q = q.view(1, -1, mod.heads, d).transpose(1, 2).float(); k = k.view(1, -1, mod.heads, d).transpose(1, 2).float()
        t = (q @ k.transpose(-1, -2) * mod.scale).softmax(-1)[..., self.tgt].mean(-1).mean(1)[0]
        side = math.isqrt(t.shape[0])
        self.acc = self.acc + F.interpolate(t.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        self.n += 1

    @torch.no_grad()
    def seed(self, image, caption, attr, classname):
        from PIL import Image
        from src.segmentation.attention_aggregation import find_token_subsequence
        p, dev = self.p, self.dev; tok = p.tokenizer
        emb, _, pooled, _ = p.encode_prompt(prompt=caption, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        attr_emb = p.encode_prompt(prompt=attr, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
        ids = tok(attr, padding="max_length", max_length=77, truncation=True).input_ids
        self.tgt = list(find_token_subsequence(ids, tok(classname, add_special_tokens=False).input_ids))
        tid = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=dev, dtype=emb.dtype)
        x = p.image_processor.preprocess(Image.fromarray(image), height=1024, width=1024).to(dev, torch.float32)
        lat = p.vae.encode(x).latent_dist.mean * p.vae.config.scaling_factor
        self.acc, self.n, self.kattr = torch.zeros((64, 64), device=dev), 0, attr_emb
        try:
            for t in XL_T:
                g = torch.Generator(device=dev).manual_seed(0)
                noise = torch.randn(lat.shape, generator=g, device=dev, dtype=lat.dtype)
                noisy = p.scheduler.add_noise(lat, noise, torch.tensor([t], device=dev)).half()
                p.unet(noisy, torch.tensor(t, device=dev), encoder_hidden_states=emb,
                       added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}, return_dict=False)
        finally:
            self.kattr = None
        if self.n == 0 or not torch.isfinite(self.acc).all():
            raise ValueError("SDXL seed failed")
        return J.mm_t(self.acc / self.n)


# ---------------------------------------------------------------- competition inside the hierarchy
def _share(R):
    m = torch.stack([J.mm_t(r) for r in R]); return R * (m / (m.sum(0, keepdim=True) + 1e-8))


def comp_v3(rec, rt, seeds, inside):
    """seeds [C, 64, 64] (target first). inside=True: shares after 16-SA, after the cap, after 64-SA (K1).
    inside=False: plain V3 per class, share on the final maps (K0). Returns the target map."""
    md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
    C = seeds.shape[0]; x16 = _resize(seeds.reshape(C, -1), 64, 16); acc = 0
    sh = _share if inside else (lambda R: R)
    for i in range(11):
        a16, a32, a64 = _attention(sa[16][i], 16), _attention(sa[32][i], 32), _attention(sa[64][i], 64)
        r16 = sh(torch.einsum("hij,cj->ci", a16, x16) / 8)
        r32 = torch.einsum("hij,cj->ci", a32, _resize(r16, 16, 32)) / 8
        z32 = sh(torch.stack([cap(r16[c], r32[c], 16)[0].reshape(-1) for c in range(C)]))
        v = sh(torch.einsum("hij,cj->ci", a64, _resize(z32, 32, 64)) / 8)
        acc = acc + v / 11; del a16, a32, a64
    if not inside:
        acc = _share(acc)
    return acc[0].reshape(64, 64)


# ---------------------------------------------------------------- passes
def rows(a):
    R = nr.rows("coco")[a.shard::a.num_shards]
    return R[:: max(1, len(R) // a.max_images)][: a.max_images] if a.max_images else R      # smoke: spread over classes


@torch.no_grad()
def stats(a):
    rt = J.make_runtime(a.model); K = class_keys(rt, CLASSES)
    out = WORK / f"stats{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows(a):
        o = out / f"{row['image_id']}.pt"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, "coco")
        Q = qbar(rec); _, T = ca_seeds(rec, K, tbar=True); lab = row["label"]; cap_ = row["prompt"]
        res = {"label": lab, "obj": {}, "spur": {}}
        for m in Q:
            t = T[(lab, m)]; idx = t.topk(max(1, int(round(TOP_OBJ * t.numel())))).indices
            res["obj"][m] = (Q[m][idx].sum(0).cpu(), len(idx))
            for c in CLASSES:
                if c == lab or mentions(cap_, c):
                    continue
                t = T[(c, m)]; idx = t.topk(max(1, int(round(TOP_SPUR * t.numel())))).indices
                res["spur"][(c, m)] = (Q[m][idx].sum(0).cpu(), len(idx))
        torch.save(res, o)
        print(f"[trio:stats] {row['image_id']} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


def fit(a):
    fs = sorted((WORK / f"stats{a.tag}").glob("*.pt")); spur, obj, img = {}, {}, {}
    for f in fs:
        z = torch.load(f); lab = z["label"]; img[f.stem] = z["obj"]
        for m, (v, n) in z["obj"].items():
            s = obj.setdefault((lab, m), [0, 0]); s[0] = s[0] + v; s[1] += n
        for key, (v, n) in z["spur"].items():
            s = spur.setdefault(key, [0, 0]); s[0] = s[0] + v; s[1] += n
    torch.save({"spur": spur, "obj": obj, "img": img}, WORK / f"fit{a.tag}.pt")
    print(f"fit: {len(fs)} images, {len(obj)} object and {len(spur)} spurious (class, module) cells", flush=True)


def gates(Fd, iid, lab, Q):
    """Absent-class gate per module for the target class; None where a profile is missing."""
    g, miss = {}, 0
    for m, q in Q.items():
        so, sp = Fd["obj"].get((lab, m)), Fd["spur"].get((lab, m))
        vo, no = (so[0].clone(), so[1]) if so else (None, 0)
        if vo is not None and iid in Fd["img"] and m in Fd["img"][iid]:
            vi, ni = Fd["img"][iid][m]; vo = vo - vi; no -= ni
        if vo is None or no <= 0 or sp is None or sp[1] <= 0:
            g[m] = torch.ones(q.shape[0], device=q.device); miss += 1; continue
        mo, ms = (vo / no).to(q.device), (sp[0] / sp[1]).to(q.device)
        w = F.normalize(mo - ms, dim=0); po, ps = float(w @ mo), float(w @ ms)
        g[m] = torch.sigmoid(GAIN * (q @ w - (po + ps) / 2) / max(po - ps, 1e-6))
    return g, miss


@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(a.model); dev = rt["pipe"].device; K = class_keys(rt, CLASSES)
    Fd = torch.load(WORK / f"fit{a.tag}.pt"); xl = XL(dev)
    out = WORK / f"coco{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    agg = lambda sa, s: sg_combined_v2.aggregate(sa, s)[0]["V3"].float()
    for row in rows(a):
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, "coco"); lab = row["label"]
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); v3 = agg(sa, seed)
        comp = [c for c in CLASSES if c != lab and mentions(row["prompt"], c)]
        Q = qbar(rec); g, miss = gates(Fd, row["image_id"], lab, Q)
        S = ca_seeds(rec, {"T": (None, list(rec.target)), "N": (None, list(rec.target)), **{c: K[c] for c in comp}},
                     gate={"N": g})
        fid = float((S["T"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"recomputed seed differs from runtime seed by {fid:.2e} for {row['image_id']}")
        sx = xl.seed(np.asarray(sample.generated_image, np.uint8), row["prompt"], row["attribution_prompt"], row["classname"])
        s2 = J.mm_t(torch.sqrt(J.mm_t(seed) * sx))
        if comp:
            allS = torch.stack([seed] + [S[c] for c in comp])
            k1, k0 = comp_v3(rec, rt, allS, True), comp_v3(rec, rt, allS, False)
        else:
            k1 = k0 = v3
        maps = [[seed, v3], [sx, agg(sa, sx)], [s2, agg(sa, s2)], [S["N"], agg(sa, S["N"])], [seed, k1], [seed, k0]]
        maps = np.stack([np.stack([x.float().cpu().numpy() for x in p]) for p in maps]).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {row['image_id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid, ncomp=len(comp),
                 comp=",".join(comp), gate_miss=miss, gate_mean=float(np.mean([float(x.mean()) for x in g.values()])))
        os.replace(tmp, o)
        print(f"[trio:coco] {row['image_id']} fid {fid:.1e} comp {len(comp)} gate {float(np.mean([float(x.mean()) for x in g.values()])):.2f}"
              f" miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


# ---------------------------------------------------------------- score / report
def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "fid_seed": float(z["fid_seed"]), "ncomp": int(z["ncomp"]),
           "gate_mean": float(z["gate_mean"]), "gate_miss": int(z["gate_miss"])}
    for ai, arm in enumerate(ARMS):
        for mi, m in enumerate(MAPS):
            if arm in ("K1", "K0") and m == "seed":
                continue
            x = st.up(st.mm(z["maps"][ai, mi].astype(np.float32)), truth.shape)
            res[f"{arm}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{arm}|{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    sc = WORK / f"coco{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / f"coco{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    print(len(jobs), flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def _sub(S, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in S.items()}


def report(a):
    d = WORK / f"coco{a.tag}_scores"; S0 = cc.load_gen(d)
    Z = [np.load(f) for f in sorted(d.glob("*.npz")) if not f.stem.endswith(".tmp")]
    nc = np.array([int(z["ncomp"]) for z in Z]); S = {"all": S0, "comp": _sub(S0, nc >= 1) if (nc >= 1).sum() >= 2 else S0}
    one = {s: np.ones(len(S[s]["grp"]), bool) for s in S}
    mi = lambda s, P: cc.miou(P[:, None], S[s]["cls"], one[s])[0]
    key = lambda arm, m, crf: f"{('A' if arm in ('K1', 'K0') else arm) if m == 'seed' else arm}|{m}{'+R2' if crf else ''}|raw"
    best = lambda s, k: int(np.argmax(cc.miou(S[s][k], S[s]["cls"], one[s])))

    def dci(s, x, y, crf=True, m="V3"):
        Pa, Pb = S[s][key(x, m, crf)][:, T55], S[s][key(y, m, crf)][:, T55]
        lo, hi = cc.boot(S[s], Pa, Pb, n=a.boot); dd = mi(s, Pa) - mi(s, Pb)
        return f"{dd:+.3f} [{lo:+.3f}, {hi:+.3f}]", dd, lo

    lab = {"A": "A BRACE", "X1": "X1 SDXL seed", "X2": "X2 √(BRACE·SDXL) seed", "N1": "N1 absent-class-calibrated seed",
           "K1": "K1 competition inside hierarchy", "K0": "K0 competition on final maps"}
    L = ["# Three training-free ideas: COCO-Cap screen (`PREREG.md`)\n",
         "`scripts/jc_trio.py`. **Exploratory screen on COCO-Cap** (owner's choice; COCO-Cap is not untuned transfer for "
         "these ideas). One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; Δ at 0.55 with "
         f"paired image-bootstrap 95% interval ({a.boot}). *own best* = best threshold on COCO-Cap itself (in-sample).\n",
         "## Data and checks\n",
         f"- {len(Z)} images, {len(np.unique(S0['cls']))} classes. Recomputed vs runtime BRACE seed: max |Δ| "
         f"{max(float(z['fid_seed']) for z in Z):.1e}.",
         f"- Images with ≥ 1 caption competitor (K1/K0 differ from A): {int((nc >= 1).sum())}; mean competitors "
         f"{nc.mean():.2f}.",
         f"- Absent-class gate: mean {np.mean([float(z['gate_mean']) for z in Z]):.3f}; modules without a profile "
         f"{sum(int(z['gate_miss']) for z in Z)} of {11 * len(Z)}."]
    for crf in (True, False):
        L.append(f"\n## {'With DenseCRF (primary)' if crf else 'Raw (secondary)'}\n")
        L.append("| Arm | Seed @0.55 | Seed own best | Hierarchy @0.55 | Hierarchy own best | Hierarchy @0.55, competitor images |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            ks, kv = key(arm, "seed", crf), key(arm, "V3", crf)
            bs, bv = best("all", ks), best("all", kv)
            L.append(f"| {lab[arm]} | {mi('all', S['all'][ks][:, T55]):.3f} | {mi('all', S['all'][ks][:, bs]):.3f} ({THR[bs]:.3f}) | "
                     f"{mi('all', S['all'][kv][:, T55]):.3f} | {mi('all', S['all'][kv][:, bv]):.3f} ({THR[bv]:.3f}) | "
                     f"{mi('comp', S['comp'][kv][:, T55]):.3f} |")
    L.append("\n## Registered comparisons (hierarchy map + DenseCRF, 0.55)\n")
    L.append("| ID | Comparison | Images | Δ [95% CI] | Promising? |")
    L.append("|---|---|---|---|---|")
    for tid, lb, x, y, s in (("T1", "X2 − A", "X2", "A", "all"), ("T2", "N1 − A", "N1", "A", "all"),
                             ("T3", "K1 − A", "K1", "A", "all"), ("T3b", "K1 − A", "K1", "A", "comp"),
                             ("T4", "K1 − K0", "K1", "K0", "comp"), ("—", "X1 − A (descriptive)", "X1", "A", "all"),
                             ("—", "K0 − A (descriptive)", "K0", "A", "comp")):
        c = dci(s, x, y)
        L.append(f"| {tid} | {lb} | {'all' if s == 'all' else 'competitor images'} ({len(S[s]['grp'])}) | {c[0]} | "
                 f"{'yes' if round(c[2], 3) > 0 else 'no'} |")
    L.append("\nPromising = interval above zero; licenses a VOC-Sim development check only.\n")
    L.append("Seed + DenseCRF @0.55 vs A: " + "; ".join(f"{x} {dci('all', x, 'A', m='seed')[0]}" for x in ("X1", "X2", "N1")) + "\n")
    cl = S0["cls"]
    L.append("Per class, hierarchy + DenseCRF @0.55 (" + " / ".join(ARMS) + "): " + ", ".join(
        f"{c} " + " / ".join(f"{cc.miou(S0[key(arm, 'V3', True)][:, T55][:, None], cl, cl == c)[0]:.2f}" for arm in ARMS)
        for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    M = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    for nm in ("stats", "capture"):
        p = sub.add_parser(nm); p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p = sub.add_parser("fit"); p.add_argument("--tag", default="")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=16)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"stats": stats, "fit": fit, "capture": capture, "score": score, "report": report}[a.cmd](a)
