#!/usr/bin/env python3
"""runs/32_realvoc_factorial/PREREG.md: component factorial on real PASCAL VOC photos, one noising pass (exploratory).

items    (CPU) item lists, one per (photo, class present), for VOC train / val      -> <WORK>/items_{split}.json
dump     (GPU) profile-fit statistics, one pass per item: VOC train (source A), VOC-Sim (source B)
fit      (CPU) object profiles and absent-class centroids (K = 1, 4) for A_all, A_f0, A_f1, B_all -> <WORK>/fit{tag}.pt
capture  (GPU) every map for the train and val items                                 -> <WORK>/cap_{split}{tag}/
score    (CPU) raw counts; val only: DenseCRF at the train-selected threshold and at 0.55
select   (CPU) train-selected threshold per map                                     -> <WORK>/thr{tag}.json
report   (CPU)                                                                       -> runs/32_realvoc_factorial/result{tag}.md

Components (all applied after the UNet call to recorded attention, so every cell comes from one capture):
  W  self-attention-guided layer weights (A3) vs uniform over the same 11 CA modules
  H  raw 16->32->64 hierarchy without cap (sg_combined_v2 V1) vs fine-only (V0)
  G  run 31 MK4 gate (4 absent-class centroids, hard nearest) vs no gate
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_mneg as M  # noqa: E402  (run 31: cluster, g_single, g_multi)

T, cc, st, nr, AB = M.T, M.cc, M.st, M.nr, M.AB
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_realvoc"
OUT = ROOT / "runs/32_realvoc_factorial"
VOC = st.VOC
VOCSIM = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim"
CLASSES = M.CLASSES                      # VOC order: GT index = position + 1
THR, T55 = M.THR, M.T55
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
T_NOISE, NOISE_SEED, PX = 100, 0, 512
CA_IDX = json.loads((ROOT / "configs/controlled_sa_seed_frozen.json").read_text())["modules"]
SA = {int(k): v for k, v in json.loads((ROOT / "configs/sg25.json").read_text())["sa_modules"].items()}
REF = "up_blocks.2.attentions.1.transformer_blocks.0.attn2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seeds: (name, layer weights, gate (source, K) or None)
SEEDS = (("u-", "u", None), ("w-", "w", None), ("uA4", "u", ("A", 4)), ("wA4", "w", ("A", 4)),
         ("uB4", "u", ("B", 4)), ("wB4", "w", ("B", 4)), ("wA1", "w", ("A", 1)), ("wB1", "w", ("B", 1)))
MAPS = tuple(f"{s}|{p}" for s, _, _ in SEEDS for p in ("seed", "V0", "V1")) + ("w-|V3",)
CRF_MAPS = tuple(f"{s}|{p}" for s in ("u-", "w-", "uA4", "wA4", "uB4", "wB4") for p in ("V0", "V1"))
COMP = ("W", "H", "G")
CNAME = {"W": "SA-guided layer weights", "H": "hierarchy (no cap)", "G": "4 absent-class profiles"}
SRC_NAME = {"A": "A: VOC train labels", "B": "B: VOC-Sim captions (label-free)"}


def cell(src, W, H, G):
    return f"{'w' if W else 'u'}{src + '4' if G else '-'}|{'V1' if H else 'V0'}"


def h(s):
    return zlib.crc32(s.encode())


def word_map():
    return {AB.label(r): r["classname"] for r in nr.rows("voc")}


# ---------------------------------------------------------------- items
def items(a):
    from PIL import Image
    WORK.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        out = []
        for i in (VOC / f"ImageSets/Segmentation/{split}.txt").read_text().split():
            gt = np.asarray(Image.open(VOC / f"SegmentationClass/{i}.png"))
            present = sorted(set(np.unique(gt).tolist()) - {0, 255})
            for ci in present:
                out.append({"id": f"{i}_{ci:02d}", "img": i, "cls": CLASSES[ci - 1], "ci": ci, "fold": h(i) % 2,
                            "present": [CLASSES[p - 1] for p in present]})
        (WORK / f"items_{split}.json").write_text(json.dumps(out))
        print(split, "photos", len({x["img"] for x in out}), "items", len(out), flush=True)


def load_items(split, a):
    R = json.loads((WORK / f"items_{split}.json").read_text())[a.shard::a.num_shards]
    if a.max_items:
        R = R[:: max(1, len(R) // a.max_items)][: a.max_items]
    return R


# ---------------------------------------------------------------- attention recording (as comparisons/backbones/single_pass.py, SD1.5)
class Rec:
    def __init__(self):
        self.ca_order, self.ca_names, self.pair, self.grid = [], set(), set(), set()
        self.target = ()
        self.clear()

    def clear(self):
        self.ca, self.q, self.scale, self.pair_qk, self.grid_qk, self.ref_q = {}, {}, {}, {}, {}, None


class Proc:
    """Plain scaled-dot-product attention that records what the components read."""

    def __init__(self, name, rec):
        self.name, self.rec = name, rec

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        rec, name = self.rec, self.name
        if attn.spatial_norm is not None or attn.group_norm is not None or hidden_states.ndim != 3 or attention_mask is not None:
            raise ValueError(f"unexpected attention configuration at {name}")
        b, n, _ = hidden_states.shape
        cross = encoder_hidden_states is not None
        ctx = encoder_hidden_states if cross else hidden_states
        if cross and attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        hd = attn.heads; d = q.shape[-1] // hd
        sp = lambda x: x.view(b, -1, hd, d).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        if not cross and name in rec.pair:
            rec.pair_qk[name] = (q[0].detach().half().clone(), k[0].detach().half().clone())
        if not cross and name in rec.grid:
            rec.grid_qk[math.isqrt(n)] = (q[0].detach().half().clone(), k[0].detach().half().clone())
        if cross and name == REF:
            rec.ref_q = q[0].detach().permute(1, 0, 2).reshape(n, -1).float()
        if cross and name in rec.ca_names:
            prob = (q[0].float() @ k[0].float().transpose(-1, -2) * attn.scale).softmax(-1)
            rec.ca[name] = prob[..., list(rec.target)].mean(-1).mean(0)
            rec.q[name] = q[0].detach().float(); rec.scale[name] = attn.scale
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, hd * d).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            raise ValueError("unexpected residual attention")
        return out / attn.rescale_output_factor


def load():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, safety_checker=None,
                                                   requires_safety_checker=False).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    rec = Rec()
    attn2 = sorted(n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith("attn2.processor"))
    rec.ca_order = [attn2[i] for i in CA_IDX]; rec.ca_names = set(rec.ca_order)
    rec.pair = {m[:-5] + "attn1" for m in rec.ca_order}; rec.grid = set(SA.values())
    names = {n.removesuffix(".processor") for n in pipe.unet.attn_processors}
    missing = (rec.ca_names | rec.pair | rec.grid | {REF}) - names
    if missing:
        raise ValueError(f"modules missing {sorted(missing)}")
    pipe.unet.set_attn_processor({n: Proc(n.removesuffix(".processor"), rec) for n in pipe.unet.attn_processors})
    return pipe, rec


@torch.no_grad()
def one_pass(pipe, rec, img_u8, prompt, word):
    from src.segmentation.attention_aggregation import find_token_subsequence
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
    x = F.interpolate(x, size=(PX, PX), mode="bicubic", align_corners=False).clamp(-1, 1)
    z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(NOISE_SEED)).to(DEV, z0.dtype)
    ac = pipe.scheduler.alphas_cumprod.to(DEV)
    zt = ac[T_NOISE].sqrt() * z0 + (1 - ac[T_NOISE]).sqrt() * eps
    pe, _ = pipe.encode_prompt(prompt=prompt, device=DEV, num_images_per_prompt=1, do_classifier_free_guidance=False)
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    rec.target = find_token_subsequence(ids, pipe.tokenizer(word, add_special_tokens=False)["input_ids"])
    rec.clear()
    pipe.unet(zt, torch.tensor(T_NOISE, device=DEV), encoder_hidden_states=pe.float(), return_dict=False)
    if set(rec.ca) != rec.ca_names or rec.ref_q is None or set(rec.grid_qk) != set(SA) or set(rec.pair_qk) != rec.pair:
        raise ValueError("missing recorded attention")


def layer_weights(rec):
    """A3 weights for one call (= sg24_ca selected weights, historical orientation; as single_pass.seed_map)."""
    from src.segmentation.sg24_ca import affinity_similarity
    ref = rec.ref_q
    pseudo = torch.softmax(ref @ ref.T / math.sqrt(ref.shape[-1]), dim=-1)
    sims = torch.stack([affinity_similarity(*rec.pair_qk[m[:-5] + "attn1"], pseudo) for m in rec.ca_order]).clamp_min(0)
    if float(sims.sum()) <= 1e-20:
        return torch.full((len(rec.ca_order),), 1 / len(rec.ca_order), device=DEV)
    return sims / sims.sum()


def make_seed(rec, w, gate=None):
    from src.segmentation import controlled_cv as cv
    s = torch.zeros((64, 64), device=DEV)
    for i, m in enumerate(rec.ca_order):
        t = rec.ca[m] if gate is None else rec.ca[m] * gate[m]
        side = math.isqrt(t.numel())
        s += F.interpolate(t.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * w[i]
    return cv.normalize_batch(s[None], "minmax")[0]


def qbar(rec):
    return {m: F.normalize(q.permute(1, 0, 2).reshape(q.shape[1], -1), dim=-1) for m, q in rec.q.items()}


def tbar(rec, K):
    """{module: [20, N]} head-mean attention to each class word (keys from the prompt "c" alone; jc_trio.class_keys)."""
    out = {}
    for m in rec.ca_order:
        q = rec.q[m]
        out[m] = torch.stack([(q @ K[c][0][m].transpose(-1, -2) * rec.scale[m]).softmax(-1)[..., K[c][1]].mean(-1).mean(0)
                              for c in CLASSES])
    return out


def propagate(H, seed, v3=False):
    """One call of sg_combined_v2.aggregate: V0 fine-only, V1 raw hierarchy (no cap), V3 capped at 16->32."""
    from src.segmentation.sg25_sa import _apply, _resize
    from src.segmentation.sg_combined_v2 import cap
    c64 = seed.float().reshape(-1); c16 = _resize(c64, 64, 16).reshape(-1)
    u = _apply(H[64], c64); out = {"V0": _apply(H[64], u).mean(0)}
    raw16 = _apply(H[16], c16).mean(0)
    raw32 = _apply(H[32], _resize(raw16, 16, 32).reshape(-1)).mean(0)
    out["V1"] = _apply(H[64], _resize(raw32, 32, 64).reshape(-1)).mean(0)
    if v3:
        z32, _ = cap(raw16, raw32, 16)
        out["V3"] = _apply(H[64], _resize(z32, 32, 64).reshape(-1)).mean(0)
    return {k: v.reshape(64, 64) for k, v in out.items()}


# ---------------------------------------------------------------- dump (profile-fit statistics)
def dump_rows(a):
    W = word_map()
    if a.src == "A":
        for it in load_items("train", a):
            yield it["id"], VOC / f"JPEGImages/{it['img']}.jpg", f"A photograph of a {W[it['cls']]}", W[it["cls"]], it["cls"], \
                [c in it["present"] for c in CLASSES], it["fold"]
    else:
        R = nr.rows("voc")[a.shard::a.num_shards]
        if a.max_items:
            R = R[:: max(1, len(R) // a.max_items)][: a.max_items]
        for r in R:
            yield r["image_id"], VOCSIM / r["image"], r["prompt"], r["classname"], AB.label(r), \
                [T.mentions(r["prompt"], c) for c in CLASSES], -1


@torch.no_grad()
def dump(a):
    from PIL import Image
    pipe, rec = load(); K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, CLASSES)
    out = WORK / f"dump_{a.src}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for iid, path, prompt, word, lab, mention, fold in dump_rows(a):
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        one_pass(pipe, rec, np.asarray(Image.open(path).convert("RGB")), prompt, word)
        Q = qbar(rec); Tm = tbar(rec, K); ci = CLASSES.index(lab)
        d = {"label": lab, "src": a.src, "fold": fold, "mention": np.array(mention), "prompt": prompt,
             "modules": np.array(rec.ca_order)}
        for mi, m in enumerate(rec.ca_order):
            q, t = Q[m], Tm[m]; N = q.shape[0]
            ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            d[f"o_{mi:02d}"] = q[t[ci].topk(ko).indices].sum(0).cpu().numpy().astype(np.float32); d[f"ko_{mi:02d}"] = ko
            for cj, c in enumerate(CLASSES):
                if c != lab and not mention[cj]:
                    d[f"n_{mi:02d}_{cj:02d}"] = q[t[cj].topk(kn).indices].cpu().numpy().astype(np.float16)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[realvoc:dump:{a.src}] {iid} {lab} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear()


# ---------------------------------------------------------------- fit
def fit(a):
    torch.set_num_threads(a.workers)
    fA = sorted(f for f in (WORK / f"dump_A{a.tag}").glob("*.npz") if not f.stem.endswith(".tmp"))
    fB = sorted(f for f in (WORK / f"dump_B{a.tag}").glob("*.npz") if not f.stem.endswith(".tmp"))
    meta = {f: (str(np.load(f)["label"]), int(np.load(f)["fold"])) for f in fA + fB}
    SETS = {"A_all": fA, "A_f0": [f for f in fA if meta[f][1] == 0], "A_f1": [f for f in fA if meta[f][1] == 1], "B_all": fB}
    print("fit sets", {k: len(v) for k, v in SETS.items()}, flush=True)
    mods = [str(x) for x in np.load((fA or fB)[0])["modules"]]
    FIT = {s: {"mo": {}, "neg": {}} for s in SETS}; STAT = {s: {} for s in SETS}
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); obj, ko, neg = {}, {}, {}
        for f in fA + fB:
            z = np.load(f); obj[f] = torch.from_numpy(z[f"o_{mi:02d}"]); ko[f] = int(z[f"ko_{mi:02d}"])
            neg[f] = {cj: torch.from_numpy(z[f"n_{mi:02d}_{cj:02d}"].astype(np.float32)) for cj in range(len(CLASSES))
                      if f"n_{mi:02d}_{cj:02d}" in z.files}
        for s, fs in SETS.items():
            for cj, c in enumerate(CLASSES):
                oi = [f for f in fs if meta[f][0] == c]
                if oi:
                    FIT[s]["mo"][(c, m)] = sum(obj[f] for f in oi) / sum(ko[f] for f in oi)
                parts = [neg[f][cj] for f in fs if cj in neg[f]]
                if not parts:
                    continue
                X = F.normalize(torch.cat(parts), dim=1)
                C1, _, _ = M.cluster(X, 1); C4, lab4, _ = M.cluster(X, 4)
                FIT[s]["neg"][(c, m)] = {1: C1, 4: C4}
                STAT[s][(c, m)] = {"n": int(X.shape[0]), "k4": int(C4.shape[0]), "sizes4": torch.bincount(lab4).tolist(),
                                   "n_obj": len(oi)}
        print(f"fit module {mi} ({m}): {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"fit": FIT, "stat": STAT, "modules": mods}, WORK / f"fit{a.tag}.pt")


# ---------------------------------------------------------------- capture
def gate_for(P, c, m, q, K):
    mo, neg = P["mo"].get((c, m)), P["neg"].get((c, m))
    if mo is None or neg is None:
        return None
    return M.g_multi(q, mo, neg[4]) if K == 4 else M.g_single(q, mo, neg[1][0])


@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = load(); W = word_map()
    B = torch.load(WORK / f"fit{a.tag}.pt"); dev_p = {}
    prof = lambda s: dev_p.setdefault(s, M.to_dev(B["fit"][s]))
    out = WORK / f"cap_{a.split}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    w_u = torch.full((len(CA_IDX),), 1 / len(CA_IDX), device=DEV)
    for it in load_items(a.split, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        one_pass(pipe, rec, np.asarray(Image.open(VOC / f"JPEGImages/{it['img']}.jpg").convert("RGB")),
                 f"A photograph of a {W[c]}", W[c])
        w_a3 = layer_weights(rec); Q = qbar(rec); G = {}; miss = {}
        sets = {"A": "A_all" if a.split == "val" else ("A_f1" if it["fold"] == 0 else "A_f0"), "B": "B_all"}
        for src, sname in sets.items():
            P = prof(sname)
            for K in (1, 4):
                gd, n = {}, 0
                for m, q in Q.items():
                    g = gate_for(P, c, m, q, K)
                    if g is None:
                        g = torch.ones(q.shape[0], device=DEV); n += 1
                    gd[m] = g
                    if a.tag and K == 1 and (c, m) in P["neg"] and (c, m) in P["mo"]:   # smoke: K = 1 hard-nearest == single
                        d = float((M.g_multi(q, P["mo"][(c, m)], P["neg"][(c, m)][1]) - g).abs().max())
                        assert d < 1e-5, f"K=1 gate mismatch {d:.2e}"
                G[(src, K)] = gd; miss[f"{src}{K}"] = n
        H = {g: _attention(rec.grid_qk[g], g) for g in SA}
        maps = {}
        for name, ca, gk in SEEDS:
            s = make_seed(rec, w_a3 if ca == "w" else w_u, G[gk] if gk else None)
            pr = propagate(H, s, v3=(name == "w-"))
            maps[f"{name}|seed"] = s
            for k, v in pr.items():
                maps[f"{name}|{k}"] = v
        if a.tag and not getattr(capture, "_checked", False):                         # smoke: identity with aggregate
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [rec.grid_qk[g]] * 11 for g in SA}, maps["w-|seed"])
            worst = max(float(np.abs(st.mm(maps[f"w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max())
                        for k in ("V0", "V1", "V3"))
            print(f"identity vs sg_combined_v2.aggregate (V0, V1, V3): max |d| {worst:.2e}", flush=True)
            assert worst < 1e-5, "single-call propagation differs from aggregate"
            capture._checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, ci=it["ci"], img=it["img"], fold=it["fold"],
                 split=a.split, w_a3=w_a3.cpu().numpy(), **{f"miss_{k}": v for k, v in miss.items()},
                 gate_mean=np.array([float(torch.stack([G[(s, 4)][m].mean() for m in Q]).mean()) for s in ("A", "B")]))
        os.replace(tmp, o)
        print(f"[realvoc:{a.split}] {it['id']} {c} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score
def score_one(job):
    f, o, thr = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); img = str(z["img"]); ci = int(z["ci"])
    gt = np.asarray(Image.open(VOC / f"SegmentationClass/{img}.png")).astype(np.int64)
    truth = gt == ci; valid = gt != 255; other = valid & (gt > 0) & ~truth
    res = {"cls": str(z["cls"]), "img": img, "fold": int(z["fold"]), "area": float(truth.sum() / max(valid.sum(), 1))}
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if thr is not None and nm in CRF_MAPS:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(VOC / f"JPEGImages/{img}.jpg").convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr = None
    if a.split == "val":
        thr = json.loads((WORK / f"thr{a.tag}.json").read_text())["thr"]
    sc = WORK / f"cap_{a.split}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name), thr) for f in sorted((WORK / f"cap_{a.split}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[1]).exists()]; print(f"attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 100 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    left = [j for j in jobs if not Path(j[1]).exists()]
    if left:
        raise SystemExit(f"{len(left)} items not scored")


def load_scores(split, tag):
    fs = sorted(f for f in (WORK / f"cap_{split}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp"))
    Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]),
         "grp": np.array([str(z["img"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k:
            S[k] = np.stack([z[k] for z in Z])
    return S


# ---------------------------------------------------------------- select
def select(a):
    S = load_scores("train", a.tag); one = np.ones(len(S["cls"]), bool); thr, tr = {}, {}
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv))
        thr[nm] = float(THR[i]); tr[nm] = float(cv[i])
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "train_miou": tr, "items": int(len(S["cls"]))}, indent=1))
    print(json.dumps(thr, indent=1))


# ---------------------------------------------------------------- report
class Boot:
    """Class-macro mIoU on photo-bootstrap resamples (all items of a photo together); same resamples for every map."""

    def __init__(self, S, n=2000):
        self.g, gi = np.unique(S["grp"], return_inverse=True); self.ci = np.array([CLASSES.index(c) for c in S["cls"]])
        self.gi = gi; G, C = len(self.g), len(CLASSES)
        self.present = np.zeros((G, C)); np.add.at(self.present, (gi, self.ci), 1)
        rng = np.random.default_rng(0)
        self.W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n)]).astype(np.float64)

    def per_group(self, P):
        X = np.zeros((len(self.g), len(CLASSES), 3)); np.add.at(X, (self.gi, self.ci), P[:, :3].astype(np.float64)); return X

    def miou(self, P, weights=None):
        X = self.per_group(P); pres = self.present
        if weights is None:
            weights = np.ones((1, len(self.g)))
        tot = np.einsum("bg,gcx->bcx", weights, X); n = weights @ pres
        iou = tot[..., 0] / np.maximum(tot.sum(-1), 1)
        return np.where(n > 0, iou, np.nan), n

    def stat(self, P):
        """(point, bootstrap draws) of class-macro mIoU."""
        iou0, _ = self.miou(P); ioub, _ = self.miou(P, self.W)
        return np.nanmean(iou0[0]), np.nanmean(ioub, 1)


def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr = Tr["thr"]
    S = load_scores("val", a.tag); BS = Boot(S, a.boot); L = []; w = L.append
    ti = {nm: int(np.flatnonzero(np.isclose(THR, thr[nm]))[0]) for nm in MAPS}
    RULES = {"sel_crf": "+ DenseCRF, train-selected threshold (primary)", "055_crf": "+ DenseCRF, fixed 0.55",
             "sel_raw": "raw, train-selected threshold", "055_raw": "raw, fixed 0.55"}

    def P(nm, rule):
        if rule == "sel_raw":
            return S[f"{nm}|raw"][:, ti[nm]]
        if rule == "055_raw":
            return S[f"{nm}|raw"][:, T55]
        return S[f"{nm}|crf"][:, 0 if rule == "sel_crf" else 1]
    cache = {}

    def m(nm, rule):
        if (nm, rule) not in cache:
            cache[(nm, rule)] = BS.stat(P(nm, rule))
        return cache[(nm, rule)]
    ci = lambda d: (float(np.mean(d[0])), float(np.percentile(d[1], 2.5)), float(np.percentile(d[1], 97.5)))
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"

    def effect(src, comp, rule, kind):
        idx = COMP.index(comp)
        if kind == "alone":
            on = [False] * 3; on[idx] = True; pairs = [(tuple(on), (False,) * 3)]
        elif kind == "stack":
            off = [True] * 3; off[idx] = False; pairs = [((True,) * 3, tuple(off))]
        else:
            pairs = []
            for o1 in (False, True):
                for o2 in (False, True):
                    rest = iter((o1, o2)); base = [None] * 3
                    for j in range(3):
                        base[j] = None if j == idx else next(rest)
                    on = list(base); on[idx] = True; off = list(base); off[idx] = False
                    pairs.append((tuple(on), tuple(off)))
        pt = np.mean([m(cell(src, *x), rule)[0] - m(cell(src, *y), rule)[0] for x, y in pairs])
        bs = np.mean([m(cell(src, *x), rule)[1] - m(cell(src, *y), rule)[1] for x, y in pairs], 0)
        return (pt, bs)

    nimg = len(np.unique(S["grp"]))
    w("# Run 32: component factorial on real PASCAL VOC photos, one noising pass (`PREREG.md`)\n")
    w(f"**Exploratory.** VOC 2012 val: {nimg} photos, {len(S['cls'])} (photo, class) items. Thresholds chosen on VOC train "
      f"({Tr['items']} items, raw maps). One SD1.5 pass per item at t = {T_NOISE}, prompt `A photograph of a {{class}}`. "
      "Class-macro mIoU over items (our benchmark metric, not the VOC21 dataset metric). 95% photo-bootstrap "
      f"intervals ({a.boot} samples). DenseCRF is applied at the threshold selected on raw train maps.\n")
    for src in ("A", "B"):
        w(f"\n## Profile source {SRC_NAME[src]}\n")
        w("| SA-guided weights | Hierarchy | 4 absent-class profiles | " + " | ".join(RULES.values()) + " | train threshold |")
        w("|:-:|:-:|:-:|" + "---:|" * len(RULES) + "---:|")
        for W_ in (False, True):
            for H_ in (False, True):
                for G_ in (False, True):
                    nm = cell(src, W_, H_, G_)
                    w(f"| {'✓' if W_ else ''} | {'✓' if H_ else ''} | {'✓' if G_ else ''} | "
                      + " | ".join(f"{m(nm, r)[0]:.3f}" for r in RULES) + f" | {thr[nm]:.3f} |")
        w("\nEffect of each component (Δ mIoU). *Alone*: only it on, minus all off. *In the stack*: all on, minus all on "
          "without it. *Average*: mean of the 4 paired differences that change only it.\n")
        w("| Component | Effect | " + " | ".join(RULES.values()) + " |")
        w("|---|---|" + "---|" * len(RULES))
        for comp in COMP:
            for kind, lab in (("alone", "alone"), ("stack", "in the stack"), ("avg", "average")):
                w(f"| {CNAME[comp]} | {lab} | " + " | ".join(f3(ci(effect(src, comp, r, kind))) for r in RULES) + " |")
        w("\nReading (primary rule, average effect; fixed in PREREG): " + "; ".join(
            f"{CNAME[c]} " + ("**contributes**" if ci(effect(src, c, 'sel_crf', 'avg'))[1] > 0 else
                             "**hurts**" if ci(effect(src, c, 'sel_crf', 'avg'))[2] < 0 else "not resolved") for c in COMP) + ".\n")
    w("\n## Descriptive\n")
    w("| Map | raw, train-selected | raw, 0.55 |"); w("|---|---:|---:|")
    for nm, lab in (("u-|seed", "seed only, uniform weights"), ("w-|seed", "seed only, SA-guided weights"),
                    ("w-|V3", "SA-guided weights + capped hierarchy (BRACE, one pass)"),
                    ("wA1|V1", "SA-guided + hierarchy + 1 absent-class profile, source A"),
                    ("wA4|V1", "SA-guided + hierarchy + 4 absent-class profiles, source A"),
                    ("wB1|V1", "SA-guided + hierarchy + 1 absent-class profile, source B"),
                    ("wB4|V1", "SA-guided + hierarchy + 4 absent-class profiles, source B")):
        w(f"| {lab} | {m(nm, 'sel_raw')[0]:.3f} | {m(nm, '055_raw')[0]:.3f} |")
    w("\nPer class IoU, primary rule (all off / all on, source A / all on, source B):\n")
    w("| Class | items | all off | all on (A) | all on (B) |"); w("|---|---:|---:|---:|---:|")
    one = np.ones(len(S["cls"]), bool)
    for c in CLASSES:
        sel = S["cls"] == c
        if not sel.any():
            continue
        v = [cc.miou(P(nm, "sel_crf")[:, None], S["cls"], sel)[0] for nm in (cell("A", 0, 0, 0), cell("A", 1, 1, 1), cell("B", 1, 1, 1))]
        w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    w("\nChecks: profiles with no absent-class pool fall back to g = 1; gate misses per item are in the capture files.\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("items", "dump", "fit", "capture", "score", "select", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--src", choices=("A", "B"), default="A"); p.add_argument("--split", choices=("train", "val"), default="val")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    {"items": items, "dump": dump, "fit": fit, "capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
