#!/usr/bin/env python3
"""Map analysis, rich capture (GPU, exploratory diagnostics) on a fixed development subset of the generated sets.

Subset: the 150 VOC-Sim and 150 COCO-Cap annotated rows with the smallest crc32(image_id) (fixed, label-free).
Per image:
  live   frozen live capture -> A3 seed, V3 (BRACE map), image, truth; oracle seed (truth area-pooled to 64) -> V3
  base   own regeneration (manual PNDM loop == pipe(), checked against the live image) with recording processors:
         - self-attention at the three frozen SA modules (16/32/64), head-mean per step block (0-9, 10-19, 20-30) and
           branch (uncond / cond) -> top-64 eigenpairs of D^-1/2 W D^-1/2 (W = sym. attention); full matrices at 16/32;
           seed diffusion A^d s for d in {1,2,4,8,16} at 64; per-head object/background mass-retention statistics
         - cross-attention at every attn2 layer, conditional, per step block: SOT, all prompt tokens, EOT and the mean of
           the padding tokens; per head at 16/32, head-mean at 64
         - decoder features (up_blocks[1] 640x32x32, up_blocks[2] 320x64x64) at steps 5, 15, 25 (conditional)
         - x0 prediction at every step (4x64x64)
Ground truth is used only for the oracle seed and the per-head statistics (both diagnostics).

python scripts/ana_capture.py run --dataset voc|coco [--n 150] [--shard i --num-shards n]   -> CACHE/ana_dump/<ds>/<id>.npz
"""
from __future__ import annotations

import argparse, copy, os, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import idea1_knockout as k1  # noqa: E402
from idea2_dose import choose_tokens  # noqa: E402

MODEL = k1.MODEL
BLOCKS = ((0, 10), (10, 20), (20, 31))
FEAT_STEPS = (5, 15, 25)
DEPTHS = (1, 2, 4, 8, 16)
N_EIG = 64


def subset(ds, n):
    rows = cc.gen_rows(ds)
    return sorted(rows, key=lambda r: zlib.crc32(r["image_id"].encode()))[:n]


def block_of(i):
    return next(b for b, (lo, hi) in enumerate(BLOCKS) if lo <= i < hi)


class Rec:
    def __init__(self, sa_names, gt, keep_tokens, _unused, eot):
        self.step = 0; self.sa_names = sa_names; self.gt = gt            # gt: {N: bool tensor [N]}
        self.keep = keep_tokens; self.eot = eot
        self.sa = {}; self.sa_n = {}; self.head = {}; self.ca = {}; self.ca_n = {}


class AnaProc:
    """diffusers AttnProcessor arithmetic with explicit probabilities and recording."""

    def __init__(self, rec, name, kind):
        self.rec, self.name, self.kind = rec, name, kind

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query); key = attn.head_to_batch_dim(key); value = attn.head_to_batch_dim(value)
        probs = attn.get_attention_scores(query, key, attention_mask)
        H = attn.heads
        if probs.shape[0] == 2 * H:
            self.record(probs, H)
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states); hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

    @torch.no_grad()
    def record(self, probs, H):
        R = self.rec; blk = block_of(R.step); N = probs.shape[1]
        if self.kind == "sa" and self.name in R.sa_names:
            for br in (0, 1):
                P = probs[br * H:(br + 1) * H].float()                     # [H, N, N]
                k = (N, blk, br)
                R.sa[k] = R.sa.get(k, 0) + P.mean(0); R.sa_n[k] = R.sa_n.get(k, 0) + 1
                g = R.gt[N].float(); stay_obj = (P @ g)[:, R.gt[N]].mean(1); stay_bg = (P @ (1 - g))[:, ~R.gt[N]].mean(1)
                R.head[k] = R.head.get(k, 0) + torch.stack([stay_obj, stay_bg], 1)   # [H, 2]
        elif self.kind == "ca":
            P = probs[H:].float()                                            # conditional [H, N, 77]
            if N not in (256, 1024, 4096):
                return
            pad = P[:, :, R.eot + 1:].mean(2, keepdim=True) if R.eot + 1 < P.shape[2] else torch.zeros_like(P[:, :, :1])
            x = torch.cat([P[:, :, R.keep], pad], 2)                       # [H, N, len(keep) + 1]
            if N == 4096:
                x = x.mean(0, keepdim=True)                                  # head-mean only at 64
            k = (self.name, blk)
            R.ca[k] = R.ca.get(k, 0) + x; R.ca_n[k] = R.ca_n.get(k, 0) + 1


def install(pipe, rec):
    saved = dict(pipe.unet.attn_processors); new = {}
    for name, p in saved.items():
        mod = name.removesuffix(".processor")
        new[name] = AnaProc(rec, mod, "sa" if mod.endswith("attn1") else "ca")
    pipe.unet.set_attn_processor(new)
    return saved


@torch.no_grad()
def base_run(pipe, rec, row):
    steps = int(row["num_inferece_steps"]); pe, ne, noise = k1.embeds(pipe, row)
    sched = k1.new_sched(pipe, steps); ts = sched.timesteps; ac = sched.alphas_cumprod.to(pipe.device)
    feats = {}; cur = {}

    def hook(i):
        def f(_m, _a, out):
            if rec.step in FEAT_STEPS:
                feats[(i, rec.step)] = out[1].half().cpu().numpy()          # conditional half
        return f
    hs = [pipe.unet.up_blocks[i].register_forward_hook(hook(i)) for i in (1, 2)]
    z = noise * sched.init_noise_sigma; enc = torch.cat([ne, pe]); x0 = []
    try:
        for i in range(len(ts)):
            t = ts[i]; rec.step = i
            eps = pipe.unet(sched.scale_model_input(torch.cat([z] * 2), t), t, encoder_hidden_states=enc, return_dict=False)[0]
            eu, ec = eps.chunk(2); eg = eu + k1.CFG * (ec - eu); a = ac[int(t)]
            x0.append(((z - (1 - a).sqrt() * eg) / a.sqrt())[0].half().cpu().numpy())
            z = sched.step(eg, t, z, return_dict=False)[0]
    finally:
        for h in hs:
            h.remove()
    return z, np.stack(x0), feats


def pool_gt(truth, side):
    t = torch.from_numpy(truth.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, side)[0, 0]


@torch.no_grad()
def spectral(A, k=N_EIG):
    W = (A + A.T) / 2; d = W.sum(1).clamp_min(1e-12).rsqrt(); S = d[:, None] * W * d[None]
    try:
        if not S.is_cuda:
            raise RuntimeError("CPU: use SciPy")
        vals, vecs = torch.linalg.eigh(S.float())
        vals, vecs = vals[-k:].flip(0).cpu().numpy(), vecs[:, -k:].flip(1).T.cpu().numpy()
    except RuntimeError:                                  # CPU MKL eigh fails at 4096; top-k only with SciPy
        from scipy.linalg import eigh
        n = S.shape[0]; w, v = eigh(S.double().cpu().numpy(), subset_by_index=[n - k, n - 1])
        vals, vecs = w[::-1].copy(), v[:, ::-1].T.copy()
    return vals.astype(np.float32), vecs.astype(np.float16)


def run(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2, sg24_ca
    rows = subset(a.dataset, a.n)[a.shard::a.num_shards]
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]
    sa_names = tuple(runtime["modules"][s] for s in (16, 32, 64))
    out = CACHE / "ana_dump" / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); res = {"cls": cls, "prompt": row["prompt"]}
        # live capture -> BRACE, oracle seed
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed64 = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, seed64)[0]["V3"].float().cpu().numpy()
            gt64 = pool_gt(truth, 64).to(seed64.device)
            v3_gt = sg_combined_v2.aggregate(sa, gt64)[0]["V3"].float().cpu().numpy()
        res.update(image=img, truth=truth, seed=seed64.cpu().numpy(), v3=v3, v3_gt=v3_gt)
        # tokens
        ids = pipe.tokenizer(row["prompt"], padding="max_length", max_length=77, truncation=True)["input_ids"]
        eot = ids.index(pipe.tokenizer.eos_token_id)
        try:                                                          # class word in the prompt (always on VOC-Sim)
            tokens, how = choose_tokens(pipe, row, "voc", None, None)
        except ValueError:                                            # COCO-Cap: label-free fallback after the base run
            tokens, how = None, "fallback"
        try:
            content = list(sg24_ca.content_positions(pipe.tokenizer, row["prompt"], tuple(tokens or ())))
        except ValueError:
            content = list(range(1, eot))
        keep = list(range(0, eot + 1))                                 # SOT, prompt tokens, EOT
        gts = {N: (pool_gt(truth, int(N ** .5)).reshape(-1) >= 0.5).to(pipe.device) for N in (256, 1024, 4096)}
        rec = Rec(sa_names, gts, keep, (), eot)
        saved = install(pipe, rec)
        try:
            zb, x0, feats = base_run(pipe, rec, row)
            base_img = k1.to_uint8(k1.decode(pipe, zb))
        finally:
            pipe.unet.set_attn_processor(saved)
        if tokens is None:                                            # COCO fallback: token most similar to the A3 seed
            T16 = sum(v[:, :, :len(keep)].mean(0) / rec.ca_n[k] for k, v in rec.ca.items() if v.shape[1] == 256)
            s16 = F.adaptive_avg_pool2d(seed64[None, None], 16).reshape(-1); s16 = s16 / s16.norm()
            cand = [p for p in content if 0 < p < eot] or list(range(1, eot))
            T = T16[:, cand]; T = T / (T.norm(dim=0, keepdim=True) + 1e-12)
            tokens = (cand[int(torch.argmax(s16 @ T))],)
        res.update(tokens=np.array(tokens), token_rule=how, content=np.array(content), eot=eot, keep=np.array(keep),
                   img_meandiff=float(np.abs(base_img.astype(int) - img.astype(int)).mean()), x0=x0)
        # self-attention: eigenpairs, full 16/32, diffusion at 64, per-head stats
        s = torch.from_numpy(st.mm(seed64.cpu().numpy())).to(pipe.device).reshape(-1)
        for (N, blk, br), A in rec.sa.items():
            A = A / rec.sa_n[(N, blk, br)]; tag = f"{int(N ** .5)}_b{blk}_{'uc'[br]}"
            res[f"sa_eval/{tag}"], res[f"sa_evec/{tag}"] = spectral(A)
            res[f"sa_head/{tag}"] = (rec.head[(N, blk, br)] / rec.sa_n[(N, blk, br)]).cpu().numpy()
            if N in (256, 1024):
                res[f"sa_full/{tag}"] = A.half().cpu().numpy()
            else:
                x = s.clone(); D = []
                for d in range(1, max(DEPTHS) + 1):
                    x = A @ x
                    if d in DEPTHS:
                        D.append(x.reshape(64, 64).cpu().numpy())
                res[f"sa_diff/{tag}"] = np.stack(D)
        for (name, blk), X in rec.ca.items():
            res[f"ca/{name}/b{blk}"] = (X / rec.ca_n[(name, blk)]).half().cpu().numpy()
        for (i, stp), Fm in feats.items():
            res[f"feat/up{i}/s{stp}"] = Fm
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[ana:{a.dataset}] {iid} {cls} tokens {tokens} ({how}) |Δpix| {res['img_meandiff']:.2f} "
              f"{res['sec']:.1f}s {o.stat().st_size / 2**20:.0f} MB", flush=True)
        del rec, res, feats; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--n", type=int, default=150); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args(); run(a)
