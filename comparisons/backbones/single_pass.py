#!/usr/bin/env python3
"""JC #5 (side N): fine-only and BRACE in one noising pass on SD 1.5, SD 2.1-base and SDXL. Rules: PREREG.md.

capture  (GPU)  -> OUT/<backbone>/<ds>/<key>.npz   maps [3, 64, 64] = (A3 seed, V0 fine-only, V3 BRACE), timings
check    (GPU)  SD 1.5 identity test of the single-call hierarchy against sg_combined_v2.aggregate (11 equal calls)
score    (CPU)  -> OUT/<backbone>/<ds>_scores/<key>.npz   counts on THR_EXT, raw and + DenseCRF R2, both maps
report   (CPU)  -> comparisons/backbones/{result.md, summary.json}

Side F's code is imported unchanged; nothing in scripts/ or src/ is edited.
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "comparisons/official"))
from common import EXT, OFFICIAL_OUT, THR_EXT, crf_site  # noqa: E402

OUT = EXT / "backbones"
HERE = Path(__file__).resolve().parent
MAN = ROOT / "comparisons/manifests"
DSN = {"voc": "voc_sim", "coco": "coco_cap"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
T_NOISE, NOISE_SEED = 100, 0
CRF_THR = (0.55,)   # PREREG Amendment A1: DenseCRF only at the frozen threshold
FROZEN_CA = [0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15]
BACKBONES = {
    "sd15": dict(repo="stable-diffusion-v1-5/stable-diffusion-v1-5", px=512, xl=False, variant=None,
                 ca="frozen", ref="up_blocks.2.attentions.1.transformer_blocks.0.attn2",
                 sa={16: "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
                     32: "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
                     64: "up_blocks.3.attentions.2.transformer_blocks.0.attn1"}),
    # the cache has no refs/main, so the snapshot (revision 4e63672c) is loaded by path
    "sd21": dict(repo=str(Path.home() / ".cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/"
                          "snapshots/4e63672c03103b6c636b8fb4119ba982469b2955"), px=512, xl=False, variant=None,
                 ca="frozen", ref="up_blocks.2.attentions.1.transformer_blocks.0.attn2",
                 sa={16: "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
                     32: "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
                     64: "up_blocks.3.attentions.2.transformer_blocks.0.attn1"}),
    "sdxl": dict(repo="stabilityai/stable-diffusion-xl-base-1.0", px=1024, xl=True, variant="fp16",
                 ca="all", ref="up_blocks.1.attentions.0.transformer_blocks.0.attn2",
                 sa={32: "up_blocks.0.attentions.2.transformer_blocks.0.attn1",
                     64: "up_blocks.1.attentions.2.transformer_blocks.0.attn1"}),
}
GOCA = {"sd15": "goca_sd15", "sd21": "goca_sd21", "sdxl": "goca_sdxl"}


def items(ds):
    return list(csv.DictReader(open(MAN / f"manifest_full_{DSN[ds]}.csv")))


def rows_by_key(ds, coco_csv):
    """runs/20 rows (corrected prompts) keyed by image id; same redirection as comparisons/official/brace_repro."""
    import jc_newrerun as jn
    sys.path.insert(0, str(ROOT / "comparisons/official"))
    import brace_repro
    brace_repro.patch_rows(Path(coco_csv))
    return {r["image_id"]: r for r in jn.rows(ds)}


# ---------------------------------------------------------------- attention recording
class Rec:
    def __init__(self):
        self.ca_names, self.sa_pair, self.sa_grid, self.ref = set(), set(), set(), ""
        self.target, self.attr_states = (), None
        self.clear()

    def clear(self):
        self.ca, self.pair_qk, self.grid_qk, self.ref_q = {}, {}, {}, None


class Proc:
    """Plain scaled-dot-product attention (as diffusers AttnProcessor2_0 for 3-D inputs) that records what BRACE reads."""

    def __init__(self, name, rec):
        self.name, self.rec = name, rec

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        import torch
        import torch.nn.functional as F
        rec, name = self.rec, self.name
        if attn.spatial_norm is not None or attn.group_norm is not None or hidden_states.ndim != 3:
            raise ValueError(f"unexpected attention configuration at {name}")
        b, n, _ = hidden_states.shape
        cross = encoder_hidden_states is not None
        ctx = encoder_hidden_states if cross else hidden_states
        if cross and attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        h = attn.heads; d = q.shape[-1] // h
        sp = lambda x: x.view(b, -1, h, d).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        side = math.isqrt(n)
        if not cross and (name in rec.sa_pair or name in rec.sa_grid):
            qq, kk = q[0].detach().half().clone(), k[0].detach().half().clone()
            if name in rec.sa_pair:
                rec.pair_qk[name] = (qq, kk)
            if name in rec.sa_grid:
                rec.grid_qk[side] = (qq, kk)
        if cross and name == rec.ref:
            rec.ref_q = q[0].detach().permute(1, 0, 2).reshape(n, -1).float()
        if cross and name in rec.ca_names:
            ka = k
            if rec.attr_states is not None:  # COCO: attribution-prompt keys, as SG24Recorder.record
                st = rec.attr_states
                if attn.norm_cross:
                    st = attn.norm_encoder_hidden_states(st)
                ka = attn.to_k(st).view(1, -1, h, d).transpose(1, 2)
            prob = (q[0].float() @ ka[0].float().transpose(-1, -2) * attn.scale).softmax(-1)
            rec.ca[name] = prob[..., list(rec.target)].mean(-1).mean(0).reshape(1, 1, side, side)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, h * d).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            raise ValueError("unexpected residual attention")
        return out / attn.rescale_output_factor


def load(bb):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    cfg = BACKBONES[bb]
    cls = StableDiffusionXLPipeline if cfg["xl"] else StableDiffusionPipeline
    kw = {"variant": cfg["variant"], "torch_dtype": torch.float16} if cfg["variant"] else {"torch_dtype": torch.float32}
    if not cfg["xl"]:
        kw.update(safety_checker=None, requires_safety_checker=False)
    pipe = cls.from_pretrained(cfg["repo"], **kw).to("cuda")
    pipe.unet.float(); pipe.vae.float(); pipe.text_encoder.float()
    if cfg["xl"]:
        pipe.text_encoder_2.float()
    rec = Rec()
    attn2 = sorted(n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith("attn2.processor"))
    rec.ca_names = set(attn2[i] for i in FROZEN_CA) if cfg["ca"] == "frozen" else set(attn2)
    rec.ca_order = [m for m in attn2 if m in rec.ca_names]
    rec.sa_pair = {m[:-5] + "attn1" for m in rec.ca_names}
    rec.sa_grid = set(cfg["sa"].values()); rec.ref = cfg["ref"]
    names = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    missing = (rec.ca_names | rec.sa_pair | rec.sa_grid | {rec.ref}) - set(names)
    if missing:
        raise ValueError(f"{bb}: modules missing {sorted(missing)}")
    pipe.unet.set_attn_processor({n: Proc(n.removesuffix(".processor"), rec) for n in pipe.unet.attn_processors})
    return pipe, rec, cfg


def encode(pipe, cfg, prompt):
    import torch
    if cfg["xl"]:
        pe, _, pooled, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1,
                                              do_classifier_free_guidance=False)
        return pe.float(), pooled.float()
    pe, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    return pe.float(), None


def target_positions(pipe, cfg, prompt, word):
    from src.segmentation.attention_aggregation import find_token_subsequence
    toks = [pipe.tokenizer] + ([pipe.tokenizer_2] if cfg["xl"] else [])
    pos = []
    for tk in toks:
        ids = tk(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
        pos.append(find_token_subsequence(ids, tk(word, add_special_tokens=False)["input_ids"]))
    if len(set(pos)) != 1:
        raise ValueError(f"tokenizers disagree on target positions: {pos}")
    return pos[0]


def att(pair):
    import torch
    q, k = pair[0].float(), pair[1].float()
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)


def hierarchy(grid_qk, seed):
    """Single-call V0 (FINE-P2) and V3 (capped coarse-to-fine, cap at the first transition) on the available grids."""
    import torch
    from src.segmentation.sg25_sa import _apply, _resize
    from src.segmentation.sg_combined_v2 import cap
    grids = sorted(grid_qk)
    c64 = seed.float().reshape(-1)
    fine = att(grid_qk[64]); torch.cuda.synchronize()
    t0 = time.perf_counter()
    u = _apply(fine, c64); v0 = _apply(fine, u).mean(0)
    torch.cuda.synchronize(); t_v0 = time.perf_counter() - t0
    t0 = time.perf_counter()
    g0 = grids[0]
    raw = _apply(att(grid_qk[g0]), _resize(c64, 64, g0).reshape(-1)).mean(0); prev = g0
    capped = False
    for g in grids[1:]:
        ops = fine if g == 64 else att(grid_qk[g])
        prop = _apply(ops, _resize(raw, prev, g).reshape(-1)).mean(0)
        if not capped:
            prop, _ = cap(raw, prop, prev); capped = True
        raw, prev = prop, g
    torch.cuda.synchronize(); t_v3 = time.perf_counter() - t0
    return v0.reshape(64, 64), raw.reshape(64, 64), t_v0, t_v3


def seed_map(rec):
    import torch
    import torch.nn.functional as F
    from src.segmentation.sg24_ca import affinity_similarity
    from src.segmentation import controlled_cv as cv
    ref = rec.ref_q
    pseudo = torch.softmax(ref @ ref.T / math.sqrt(ref.shape[-1]), dim=-1)
    sims = torch.stack([affinity_similarity(*rec.pair_qk[m[:-5] + "attn1"], pseudo) for m in rec.ca_order]).clamp_min(0)
    w = sims / sims.sum().clamp_min(1e-20)
    if float(sims.sum()) <= 1e-20:
        w.fill_(1 / len(rec.ca_order))
    s = torch.zeros((64, 64), device=ref.device)
    for i, m in enumerate(rec.ca_order):
        s += F.interpolate(rec.ca[m], size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * w[i]
    return cv.normalize_batch(s[None], "minmax")[0], w.cpu().numpy()


def one_pass(pipe, rec, cfg, img_u8, r):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    x = torch.from_numpy(img_u8).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
    if cfg["px"] != x.shape[-1]:
        x = F.interpolate(x, size=(cfg["px"], cfg["px"]), mode="bicubic", align_corners=False).clamp(-1, 1)
    z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(NOISE_SEED)).to(dev, z0.dtype)
    ac = pipe.scheduler.alphas_cumprod.to(dev)
    zt = ac[T_NOISE].sqrt() * z0 + (1 - ac[T_NOISE]).sqrt() * eps
    pe, pooled = encode(pipe, cfg, r["prompt"])
    attr = r["attribution_prompt"]
    rec.attr_states = None if attr == r["prompt"] else encode(pipe, cfg, attr)[0]
    rec.target = target_positions(pipe, cfg, attr, r["classname"])
    rec.clear()
    kw = {}
    if cfg["xl"]:
        tid = torch.tensor([[cfg["px"], cfg["px"], 0, 0, cfg["px"], cfg["px"]]], device=dev, dtype=pe.dtype)
        kw["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": tid}
    pipe.unet(zt, torch.tensor(T_NOISE, device=dev), encoder_hidden_states=pe, return_dict=False, **kw)
    if set(rec.ca) != rec.ca_names or rec.ref_q is None or set(rec.grid_qk) != set(cfg["sa"]):
        raise ValueError("missing recorded attention")
    return rec


def capture(a):
    import torch
    from PIL import Image
    pipe, rec, cfg = load(a.backbone)
    gpu = torch.cuda.get_device_name(0)
    for ds in a.sets:
        R = rows_by_key(ds, a.coco_csv)
        out = OUT / a.backbone / ds; out.mkdir(parents=True, exist_ok=True)
        its = items(ds)[a.shard::a.num_shards]
        if a.max_images:
            its = its[:: max(1, len(its) // a.max_images)][: a.max_images]
        for it in its:
            o = out / f"{it['key']}.npz"
            if o.exists():
                continue
            r = R[it["key"]]
            if r["label"] != it["classname"]:
                raise ValueError(f"class mismatch {it['key']}")
            img = np.asarray(Image.open(it["image"]).convert("RGB"))
            truth = np.asarray(Image.open(it["annotation"]).convert("RGB")).any(-1)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            with torch.no_grad():
                one_pass(pipe, rec, cfg, img, r)
                seed, w = seed_map(rec)
                torch.cuda.synchronize(); t_pass = time.perf_counter() - t0
                v0, v3, t_v0, t_v3 = hierarchy(rec.grid_qk, seed)
            peak = torch.cuda.max_memory_allocated() / 2**20
            maps = np.stack([seed.cpu().numpy(), v0.float().cpu().numpy(), v3.float().cpu().numpy()])
            rec.clear()
            if not np.isfinite(maps).all():
                raise ValueError(f"non-finite map for {it['key']}")
            tmp = o.with_name(o.stem + ".tmp.npz")
            np.savez(tmp, maps=maps.astype(np.float16), cls=it["classname"], truth=truth, image_path=it["image"],
                     layer_w=w.astype(np.float32), t_pass=t_pass, t_v0=t_v0, t_v3=t_v3, peak_mib=peak, gpu=gpu,
                     prompt=r["prompt"], attribution_prompt=r["attribution_prompt"], word=r["classname"])
            os.replace(tmp, o)
            print(f"[{a.backbone}:{ds}] {it['key']} {it['classname']} pass {t_pass:.2f}s v0 {t_v0:.3f}s v3 {t_v3:.3f}s "
                  f"peak {peak / 1024:.1f} GiB", flush=True)
            torch.cuda.empty_cache()


def check(a):
    """Identity test: the single-call hierarchy equals sg_combined_v2.aggregate fed the same call 11 times (SD 1.5)."""
    import torch
    from PIL import Image
    from src.segmentation import sg_combined_v2
    import sg_steps_test as st
    pipe, rec, cfg = load("sd15")
    R = rows_by_key("voc", a.coco_csv)
    worst = 0.
    for it in items("voc")[:: 97][:5]:
        img = np.asarray(Image.open(it["image"]).convert("RGB"))
        with torch.no_grad():
            one_pass(pipe, rec, cfg, img, R[it["key"]])
            seed, _ = seed_map(rec)
            v0, v3, _, _ = hierarchy(rec.grid_qk, seed)
            sa = {g: [rec.grid_qk[g]] * 11 for g in (16, 32, 64)}
            ref, _ = sg_combined_v2.aggregate(sa, seed.reshape(64, 64))
        for a_, b_ in ((v0, ref["V0"]), (v3, ref["V3"])):
            worst = max(worst, float(np.abs(st.mm(a_.cpu().numpy()) - st.mm(b_.float().cpu().numpy())).max()))
        rec.clear()
    print(json.dumps({"identity_max_abs_diff_after_minmax": worst, "pass": worst < 1e-5}))
    (OUT / "identity_check.json").write_text(json.dumps({"max_abs_diff": worst, "pass": worst < 1e-5, "images": 5}))
    if worst >= 1e-5:
        raise SystemExit("identity check failed")


def score_one(job):
    f, o, site = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    import torch
    torch.set_num_threads(1)
    from PIL import Image
    import sg_steps_test as st
    import sg_class_comp as cc
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(site)
    z = np.load(f)
    truth = z["truth"].astype(bool); img = np.asarray(Image.open(str(z["image_path"])).convert("RGB"))
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"cls": str(z["cls"]), "t_pass": float(z["t_pass"]), "t_v0": float(z["t_v0"]), "t_v3": float(z["t_v3"]),
           "peak_mib": float(z["peak_mib"]), "truth_crc": zlib.crc32(np.packbits(truth).tobytes())}
    for ci, c in enumerate(("seed", "V0", "V3")):
        x = st.up(st.mm(z["maps"][ci].astype(np.float32)), truth.shape)
        res[f"{c}|raw"] = cc.counts(x[None] >= THR_EXT[:, None, None], truth, valid, other)
        if c == "seed":
            continue
        ms = np.stack([apply_ovam_densecrf(img, x >= t, site) for t in CRF_THR]).astype(bool)   # Amendment A1
        res[f"{c}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    site = str(crf_site())
    for ds in a.sets:
        src = OUT / a.backbone / ds; sc = OUT / a.backbone / f"{ds}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name), site) for f in sorted(src.glob("*.npz")) if not f.stem.endswith(".tmp")]
        jobs = jobs[a.shard::a.num_shards]
        print(a.backbone, ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 100 == 0 or n == len(jobs):
                    print(f"{a.backbone} {ds} {n}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- report
def report(a):
    import sg_class_comp as cc
    NB = 2000
    i55 = int(np.flatnonzero(np.isclose(THR_EXT, 0.55))[0])
    L = []; w = L.append; J = {"label": "exploratory; single noising pass, t=100, noise seed 0; see PREREG.md", "rows": [], "paired": []}
    S, C, INFO = {}, {}, {}
    bbs = [b for b in BACKBONES if all((OUT / b / f"{ds}_scores").exists() for ds in ("voc", "coco"))]
    for ds in ("voc", "coco"):
        man = {r["key"]: r for r in items(ds)}
        ids = sorted(k for k in man if all((OUT / b / f"{ds}_scores" / f"{k}.npz").exists() for b in bbs))
        if len(ids) != len(man):
            print(f"WARNING {ds}: {len(ids)} of {len(man)} items scored for all backbones", flush=True)
        S[ds] = {"grp": np.array(ids), "cls": np.array([man[i]["classname"] for i in ids]),
                 "fold": np.array([zlib.crc32(i.encode()) % 5 for i in ids])}
        for b in bbs:
            z = {i: np.load(OUT / b / f"{ds}_scores" / f"{i}.npz") for i in ids}
            for c in ("seed", "V0", "V3"):
                C[(b, c, False, ds)] = np.stack([z[i][f"{c}|raw"] for i in ids])
                if c != "seed":
                    C[(b, c, True, ds)] = np.stack([z[i][f"{c}+R2|raw"] for i in ids])
            INFO[(b, ds)] = {k: np.array([float(z[i][k]) for i in ids]) for k in ("t_pass", "t_v0", "t_v3", "peak_mib")}
            INFO[(b, ds)]["crc"] = np.array([int(z[i]["truth_crc"]) for i in ids])
            g = GOCA[b]
            if all((OFFICIAL_OUT / g / ds / f"{i}.npz").exists() for i in ids):
                zg = {i: np.load(OFFICIAL_OUT / g / ds / f"{i}.npz") for i in ids}
                C[(g, "soft", False, ds)] = np.stack([zg[i]["soft|raw"] for i in ids])
                C[(g, "soft", True, ds)] = np.stack([zg[i]["soft+R2|raw"] for i in ids])
                INFO[(g, ds)] = {"crc": np.array([int(zg[i]["truth_crc"]) for i in ids])}
    miou = lambda ds, P: float(cc.miou(P[:, None], S[ds]["cls"], np.ones(len(P), bool))[0])

    def sel(m, c, crf):
        Xv, Xc = C[(m, c, crf, "voc")], C[(m, c, crf, "coco")]
        _, Pv, taus = cc.oof(S["voc"], Xv)
        tb = int(np.argmax(cc.miou(Xv, S["voc"]["cls"], np.ones(len(Xv), bool))))
        return {"voc": Pv, "coco": Xc[:, tb]}, float(THR_EXT[tb]), {"voc": Xv[:, i55], "coco": Xc[:, i55]}

    def boot(ds, Pa, Pb):
        lo, hi = cc.boot(S[ds], Pa, Pb, n=NB)
        return miou(ds, Pa) - miou(ds, Pb), float(lo), float(hi)
    f3 = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    LAB = {"sd15": "SD 1.5", "sd21": "SD 2.1-base (community copy)", "sdxl": "SDXL base 1.0"}

    w("# JC #5: fine-only and BRACE in one noising pass on three backbones (side N, exploratory)\n")
    w("Rules fixed in `PREREG.md` before capture. Stored OVAM benchmark images (490 VOC-Sim, 707 COCO-Cap), VAE encode, "
      f"noise at t = {T_NOISE} (seed {NOISE_SEED}), one conditional UNet call, the runs/20 prompts and class-word query. "
      "Class-macro mIoU. *VOC-sel.* = VOC-Sim 5-fold OOF threshold; COCO-Cap gets the full-VOC-Sim-best threshold "
      "unchanged. Paired Δ: 2000-sample image bootstrap, 95% interval. **This is a different setting from the "
      "generation-time BRACE numbers (runs/20, 0.832 / 0.709) and is never mixed with them.**\n")
    chk = OUT / "identity_check.json"
    if chk.exists():
        w(f"Identity check of the single-call re-implementation vs `sg_combined_v2.aggregate` (SD 1.5, 5 images): "
          f"`{chk.read_text().strip()}`.\n")
    gref = {}
    w("## Raw (no DenseCRF)\n")
    w("| Backbone | Map | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | τ (full VOC) | VOC-Sim 0.55 | COCO-Cap 0.55 |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for b in bbs:
        for c, lab in (("seed", "seed alone"), ("V0", "fine-only"), ("V3", "BRACE hierarchy")):
            P, tau, F55 = sel(b, c, False)
            row = {"backbone": b, "map": c, "densecrf": False, "voc_sel": miou("voc", P["voc"]), "coco_sel": miou("coco", P["coco"]),
                   "tau": tau, "voc_055": miou("voc", F55["voc"]), "coco_055": miou("coco", F55["coco"])}
            J["rows"].append(row)
            w(f"| {LAB[b]} | {lab} | {row['voc_sel']:.3f} | {row['coco_sel']:.3f} | {tau:.3f} | {row['voc_055']:.3f} | {row['coco_055']:.3f} |")
        g = GOCA[b]
        if all((g, "soft", False, ds_) in C for ds_ in ("voc", "coco")):
            P, tau, F55 = sel(g, "soft", False)
            J["rows"].append({"backbone": b, "map": "GoCA", "densecrf": False, "voc_sel": miou("voc", P["voc"]), "coco_sel": miou("coco", P["coco"]), "tau": tau})
            w(f"| {LAB[b]} | GoCA (reference, N run) | {miou('voc', P['voc']):.3f} | {miou('coco', P['coco']):.3f} | {tau:.3f} | {miou('voc', F55['voc']):.3f} | {miou('coco', F55['coco']):.3f} |")
    w("")
    w("## With DenseCRF R2 (fixed 0.55 only, Amendment A1)\n")
    w("| Backbone | Map | VOC-Sim 0.55 + R2 | COCO-Cap 0.55 + R2 |")
    w("|---|---|---:|---:|")
    for b in bbs:
        for c, lab in (("V0", "fine-only"), ("V3", "BRACE hierarchy")):
            v, cc_ = miou("voc", C[(b, c, True, "voc")][:, 0]), miou("coco", C[(b, c, True, "coco")][:, 0])
            J["rows"].append({"backbone": b, "map": c, "densecrf": True, "voc_055": v, "coco_055": cc_})
            w(f"| {LAB[b]} | {lab} | {v:.3f} | {cc_:.3f} |")
        g = GOCA[b]
        if all((g, "soft", True, ds_) in C for ds_ in ("voc", "coco")):
            P, tau, _ = sel(g, "soft", True)
            gref[b] = P
            w(f"| {LAB[b]} | GoCA + R2, VOC-sel. (τ {tau:.3f}; reference) | {miou('voc', P['voc']):.3f} | {miou('coco', P['coco']):.3f} |")
    w("")
    w("## Paired differences\n")
    w("| Backbone | Comparison | Rule | Δ VOC-Sim | Δ COCO-Cap |")
    w("|---|---|---|---:|---:|")
    for b in bbs:
        Pa, _, Fa = sel(b, "V3", False); Pb, _, Fb = sel(b, "V0", False)
        for rule, A, B in (("raw, VOC-sel.", Pa, Pb), ("raw, fixed 0.55", Fa, Fb)):
            d = {ds: boot(ds, A[ds], B[ds]) for ds in ("voc", "coco")}
            J["paired"].append({"backbone": b, "cmp": "V3-V0", "rule": rule, **d})
            w(f"| {LAB[b]} | BRACE − fine-only | {rule} | {f3(d['voc'])} | {f3(d['coco'])} |")
        A = {ds: C[(b, "V3", True, ds)][:, 0] for ds in ("voc", "coco")}; B = {ds: C[(b, "V0", True, ds)][:, 0] for ds in ("voc", "coco")}
        d = {ds: boot(ds, A[ds], B[ds]) for ds in ("voc", "coco")}
        J["paired"].append({"backbone": b, "cmp": "V3-V0", "rule": "0.55 + R2", **d})
        w(f"| {LAB[b]} | BRACE − fine-only | 0.55 + DenseCRF | {f3(d['voc'])} | {f3(d['coco'])} |")
        g = GOCA[b]
        if all((g, "soft", False, ds_) in C for ds_ in ("voc", "coco")):
            Pg, _, _ = sel(g, "soft", False)
            d = {ds: boot(ds, Pa[ds], Pg[ds]) for ds in ("voc", "coco")}
            J["paired"].append({"backbone": b, "cmp": "V3-GoCA", "rule": "raw, VOC-sel.", **d})
            w(f"| {LAB[b]} | BRACE − GoCA | raw, VOC-sel. (both) | {f3(d['voc'])} | {f3(d['coco'])} |")
        if b in gref:
            d = {ds: boot(ds, A[ds], gref[b][ds]) for ds in ("voc", "coco")}
            J["paired"].append({"backbone": b, "cmp": "V3-GoCA", "rule": "BRACE 0.55+R2 vs GoCA VOC-sel.+R2 (mixed)", **d})
            w(f"| {LAB[b]} | BRACE − GoCA | DenseCRF; BRACE 0.55, GoCA VOC-sel. (mixed rule) | {f3(d['voc'])} | {f3(d['coco'])} |")
    w("\nBRACE − GoCA pairs use the same stored images and masks but different captures (GoCA's own code and noise "
      "draw); they are same-image, not same-capture, pairs.\n")
    w("## Runtime and memory per image (capture GPU)\n")
    w("| Backbone | GPU | UNet pass + seed, median s | fine-only step, median s | hierarchy step, median s | peak GPU memory, GiB (max) |")
    w("|---|---|---:|---:|---:|---:|")
    for b in bbs:
        t = {k: np.concatenate([INFO[(b, ds)][k] for ds in ("voc", "coco")]) for k in ("t_pass", "t_v0", "t_v3", "peak_mib")}
        gpu = str(np.load(next((OUT / b / "voc").glob("*.npz")))["gpu"])
        J.setdefault("runtime", {})[b] = {k: float(np.median(v)) for k, v in t.items()} | {"peak_mib_max": float(t["peak_mib"].max()), "gpu": gpu}
        w(f"| {LAB[b]} | {gpu} | {np.median(t['t_pass']):.2f} | {np.median(t['t_v0']):.3f} | {np.median(t['t_v3']):.3f} | {t['peak_mib'].max() / 1024:.1f} |")
    w("\nThe UNet pass + seed is shared: fine-only and BRACE come from the same call. Timings exclude model loading and "
      "DenseCRF.\n")
    w("## Checks\n")
    for b in bbs:
        for ds in ("voc", "coco"):
            ok = (INFO[(b, ds)]["crc"] == INFO[(GOCA[b], ds)]["crc"]).all() if (GOCA[b], ds) in INFO else None
            w(f"- {LAB[b]}, {NAME[ds]}: {len(S[ds]['grp'])} items; ground-truth masks identical to the GoCA run's: {ok}")
    (HERE / "result.md").write_text("\n".join(L) + "\n"); (HERE / "summary.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("capture", "check", "score", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--backbone", choices=tuple(BACKBONES), default="sd15")
        p.add_argument("--sets", nargs="+", default=["voc", "coco"])
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--workers", type=int, default=16)
        p.add_argument("--coco-csv", type=Path, default=EXT / "brace_repro/ovam_coco_cap.csv")
        p.add_argument("--out", type=Path, default=EXT / "backbones")
    a = ap.parse_args()
    OUT = a.out
    {"capture": capture, "check": check, "score": score, "report": report}[a.cmd](a)
