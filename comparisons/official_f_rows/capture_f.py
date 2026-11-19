#!/usr/bin/env python3
"""JC #3 remaining rows: OVAM (class word, optimized token), DAAM and SCLIP from ONE generation per item.

See PREREG.md. The generation replicates `src/attention_capture/live.generate_attention_sample` call for call
(seeding, prompt encoding, initial noise, pipeline call), with the frozen runtime's pipeline load (SD 1.5 fp32,
safety checker off), so the image matches the BRACE reproduction (`~/TRDP/external/n_side/brace_repro/`).
A light recorder (RecordingAttnProcessor interface, `src/attention_capture/recorder.py`, unchanged) accumulates
in the same pass:
  OVAM class word   as `memory.CompactAttentionRecorder` + `ovam_baseline.ovam_cross_attention_map`
  OVAM token        same, keys from [SoT output, OVAM's released VOC token], target position 1
  OVAM SA           as `memory.CompactAttentionRecorder` (all-ones value) + `ovam_full.ovam_self_attention_map`
  DAAM              as castorini/daam v0.2.0 (trace.py UNetCrossAttentionHooker / compute_global_heat_map)
  SeeDiff           BAIKLAB/SeeDiff (commit 28d26f4) `src/seediff_app.py`: prompt-to-prompt AttentionStore filled in the
                    same pass (conditional softmax attention averaged over steps; all cross-attention layers, the first
                    self-attention layer per (block, resolution), which is the only one its code reads), then its
                    `aggregate_attention` (imported unchanged) and the body of its `save_cross_attention` (replicated
                    line for line; PAMR imported unchanged, run on the GPU)
SCLIP runs separately (`sclip_f.py`, CPU, transformers 4.x env) on the BRACE reproduction's stored images.

-> ~/TRDP/external/n_side/official_f_rows/<ds>/<image_id>.npz
"""
from __future__ import annotations

import argparse, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "comparisons/official"))
import brace_repro as br  # noqa: E402  (patch_rows: COCO-Cap manifest rebuilt outside the repo)
import jc_newrerun as jn  # noqa: E402
from src.attention_capture.live import _install  # noqa: E402
from src.segmentation.attention_aggregation import resize_map  # noqa: E402

OUT = Path.home() / "TRDP/external/n_side/official_f_rows"
BR = Path.home() / "TRDP/external/n_side/brace_repro"
TOKENS = np.load(ROOT / "configs/ovam/tokens_VOC.npy", allow_pickle=True).item()
# OVAM optimizes a 2-row embedding (rows (0, -2) of encode_text: <SoT> and the class token; experiments/voc_simple/
# 2_token_optimization.py) and evaluates the 2-row per-image embeddings (3_generate_voc_simple_dataset.py).
# The released tokens_VOC.npy is the class mean of row 1; TOKENS2 is the class mean of both rows (row 1 equals
# tokens_VOC exactly).
_TD = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim_train/tokens"
TOKENS2 = {}
for _f in sorted(_TD.glob("sd-15_*.npy")):
    if not _f.name.endswith("_losses.npy"):
        TOKENS2.setdefault(_f.name[6:].rsplit("_", 2)[0], []).append(np.load(_f))
TOKENS2 = {k: np.mean(v, 0) for k, v in TOKENS2.items()}
assert all(np.abs(TOKENS2[k][1] - TOKENS[k]).max() < 1e-5 for k in TOKENS)


def canon(c):
    return c.replace(" ", "-")


def daam_merge(tokenizer, prompt, word):
    """daam/utils.py compute_token_merge_indices (word_idx None, offset 0), +1 for SoT. [] if absent."""
    tokens = [x.replace("</w>", "") for x in tokenizer.tokenize(prompt.lower())]
    search = [x.replace("</w>", "") for x in tokenizer.tokenize(word.lower())]
    idx = []
    for s in [x for x in range(len(tokens)) if tokens[x:x + len(search)] == search]:
        idx += [i + s for i in range(len(search))]
    return [x + 1 for x in idx]


class FRecorder:
    """Per-image accumulator for the RecordingAttnProcessor interface (no tensors kept per step)."""

    def __init__(self, pipe, attribution_prompt, gen_prompt, word, token_vec, token2, voc):
        self.pipe = pipe; dev = pipe.device; tok = pipe.tokenizer
        self.module_metadata, self.call_counts = {}, {}
        self.attribution_hidden_states = None            # RecordingAttnProcessor: no attribution_key
        ids = tok(attribution_prompt, padding=False, return_tensors="pt").input_ids.to(dev)
        self.hs = pipe.text_encoder(ids)[0]              # [1, n, 768], unpadded (live.py short_tokens)
        # OVAM's released index: VOC-Sim heatmap[0, len(prompt.split())] (experiments/voc_simple/3_generate_voc_simple_
        # dataset.py:107), COCO-Cap class name heatmap[0, -2] (experiments/coco_captions/2_generate_examples_coco.py:110)
        self.n = int(self.hs.shape[1]); self.pos = len(attribution_prompt.split()) if voc else self.n - 2
        self.hs_tok = torch.cat([self.hs[:, :1], torch.as_tensor(token_vec, dtype=self.hs.dtype, device=dev)[None, None]], 1)
        self.hs_tok2 = torch.as_tensor(token2, dtype=self.hs.dtype, device=dev)[None]   # [1, 2, 768]
        self.T2 = torch.zeros(64, 64, device=dev)
        self.keys = {}
        self.W = torch.zeros(64, 64, device=dev); self.T = torch.zeros(64, 64, device=dev)
        self.sa = {}
        self.didx = daam_merge(tok, gen_prompt, word); self.daam = {}
        tt = tok.encode(word); pt = tok.encode(gen_prompt)          # SeeDiff target_location (save_cross_attention)
        hit = np.where(np.array(pt) == tt[1])[0]
        self.sd_loc = int(hit[0]) if len(hit) else None
        self.sd_store, self.sd_order, self.sd_self = {}, {}, set()

    def register_module(self, name, kind, heads, scale, processor_class):
        self.call_counts[name] = 0
        self.module_metadata[name] = {"kind": kind, "heads": heads}

    def _keys(self, name):
        if name not in self.keys:
            m = self.pipe.unet.get_submodule(name); out = []
            for h in (self.hs, self.hs_tok, self.hs_tok2):
                s = m.norm_encoder_hidden_states(h) if m.norm_cross else h
                k = m.to_k(s); k = k.view(1, -1, m.heads, k.shape[-1] // m.heads).transpose(1, 2)
                if m.norm_k is not None:
                    k = m.norm_k(k)
                out.append(k[0].float())
            self.keys[name] = out
        return self.keys[name]

    def _sd_add(self, name, kind, r, p):
        place = "down" if name.startswith("down_blocks") else "up" if name.startswith("up_blocks") else "mid"
        if name not in self.sd_store:
            if kind == "self":
                if (place, r) in self.sd_self:
                    return
                self.sd_self.add((place, r))
            self.sd_order.setdefault(f"{place}_{kind}", []).append(name)
            self.sd_store[name] = p.clone()
        else:
            self.sd_store[name] += p

    def record(self, name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        self.call_counts[name] += 1
        r = math.isqrt(query.shape[-2])
        if self.sd_loc is not None:                         # SeeDiff store: conditional half, P2P softmax
            place = "down" if name.startswith("down_blocks") else "up" if name.startswith("up_blocks") else "mid"
            if kind == "cross" or name in self.sd_store or (place, r) not in self.sd_self:
                sc = query[1].float() @ key[1].float().transpose(-1, -2) * scale
                if attention_mask is not None:
                    sc = sc + attention_mask[1].float()
                self._sd_add(name, kind, r, sc.softmax(-1)); del sc
        if kind == "cross":
            if name.startswith("mid_block"):
                return                                    # OVAM (StableDiffusionHooker) and DAAM skip the mid block
            kw, kt, kt2 = self._keys(name); q0 = query[0].float()  # OVAM: unconditional query (memory.py)
            pw = (q0 @ kw.transpose(-1, -2) * scale).softmax(-1)[..., self.pos]
            self.W += resize_map(pw.sum(0).reshape(r, r), 64)
            pt = (q0 @ kt.transpose(-1, -2) * scale).softmax(-1)[..., 1]
            self.T += resize_map(pt.sum(0).reshape(r, r), 64)
            pt2 = (q0 @ kt2.transpose(-1, -2) * scale).softmax(-1)[..., 1]
            self.T2 += resize_map(pt2.sum(0).reshape(r, r), 64)
            if self.didx and r != 8:                       # DAAM: conditional half, skip factor 8
                sc = query[1].float() @ key[1].float().transpose(-1, -2) * scale
                if attention_mask is not None:
                    sc = sc + attention_mask[1].float()
                p = sc.softmax(-1)[..., self.didx]         # [heads, hw, n_idx]
                x = p.permute(2, 0, 1).reshape(len(self.didx), -1, r, r)
                self.daam[name] = self.daam.get(name, 0) + x
        elif kind == "self" and r == 64:                   # OVAM: highest-resolution self-attention modules
            o = F.scaled_dot_product_attention(query, key, torch.ones_like(key), attn_mask=attention_mask,
                                               dropout_p=0.0, is_causal=False)
            v = o.sum(dim=0).sum(dim=-1).float()
            s, c = self.sa.get(name, (0, 0)); self.sa[name] = (s + v, c + 1)

    def results(self):
        sa = torch.stack([resize_map((s / c).sum(0).reshape(64, 64), 64) for _, (s, c) in sorted(self.sa.items())])
        sa = -sa.mean(0)
        daam = None
        if self.daam:
            merges = []
            for name in sorted(self.daam):
                x = self.daam[name]                        # [n_idx, heads, r, r], summed over steps
                for h in range(x.shape[1]):
                    merges.append(F.interpolate(x[:, h][:, None], size=(64, 64), mode="bicubic").clamp_(min=0)[:, 0])
            daam = torch.stack(merges).mean(0).mean(0)    # mean over (module, head), then over the word's tokens
        return self.W, self.T2, self.T, sa, daam

    def get_average_attention(self):                        # prompt-to-prompt AttentionStore interface
        steps = self.call_counts[next(iter(self.sd_store))]
        out = {k: [] for k in ("down_cross", "mid_cross", "up_cross", "down_self", "mid_self", "up_self")}
        for k, names in self.sd_order.items():
            out[k] = [self.sd_store[n] / steps for n in names]
        return out


def load_seediff():
    """aggregate_attention from SeeDiff's src/seediff_app.py, unchanged (extracted with ast; the module itself
    imports gradio/IPython at top level). PAMR from its src/pamr.py, unchanged, wrapped to run on the GPU."""
    import ast, cv2
    from typing import List
    from PIL import Image
    src = Path.home() / "TRDP/external/n_side/official_f_rows/src/seediff/repo/src"
    sys.path.insert(0, str(src)); from pamr import PAMR as _PAMR
    code = (src / "seediff_app.py").read_text(); tree = ast.parse(code)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "aggregate_attention"][0]
    ns = {"torch": torch, "np": np, "cv2": cv2, "List": List, "Image": Image, "AttentionStore": object}
    exec(compile(ast.Module([fn], []), str(src / "seediff_app.py"), "exec"), ns)

    class PAMR(torch.nn.Module):
        def __init__(self, num_iter=1, dilations=[1]):
            super().__init__(); self.m = _PAMR(num_iter, dilations).cuda()

        def forward(self, x, mask):
            return self.m(x.cuda(), mask.cuda()).cpu()
    return ns["aggregate_attention"], PAMR


def seediff_masks(agg, PAMR, store, original_image, prompts, target_location, class_one):
    """Body of SeeDiff's save_cross_attention (seediff_app.py lines 1308-1378), replicated; returns its soft
    64-level map (bicubic to 512, min-max) and its final mask (PAMR 5000 iterations, > 0.8)."""
    import cv2
    select, from_where = 0, ("up", "down")
    _a8, sa8 = agg(store, 8, ("up", "mid", "down"), True, select, prompts=prompts, mask=None, ca_init_map=None,
                   class_one=class_one, target_location=target_location)
    _a16, sa16 = agg(store, 16, from_where, True, select, prompts=prompts, mask=None, ca_init_map=None,
                     class_one=class_one, target_location=target_location)
    ca32 = cv2.resize(sa16.numpy(), (32, 32), interpolation=cv2.INTER_CUBIC)
    _a32, sa32 = agg(store, 32, from_where, True, select, prompts=prompts, mask=None, ca_init_map=ca32,
                     class_one=class_one, target_location=target_location)
    ca64 = cv2.resize(sa32.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)
    _a64, sa64 = agg(store, 64, ["up"], True, select, prompts=prompts, mask=None, ca_init_map=ca64,
                     class_one=class_one, target_location=target_location)
    m64 = cv2.resize(sa64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
    m64 = (m64 - m64.min()) / (m64.max() - m64.min())
    sa_agg_map_mask = m64 > 0.3
    x = torch.tensor(np.array(original_image)).permute(2, 0, 1).unsqueeze(0).type(torch.float32)
    mt = torch.tensor(sa_agg_map_mask).unsqueeze(0).unsqueeze(0)
    pamr_tensor = PAMR(num_iter=5000)(x, mt * 1)[0][0]
    mo = (pamr_tensor - pamr_tensor.min()) / (pamr_tensor.max() - pamr_tensor.min())
    return m64.astype(np.float32), (mo > 0.8).numpy()


def daam_expand(m):
    """daam WordHeatMap.expand_as(image) with absolute=False: bicubic to 512, min-max."""
    im = F.interpolate(m[None, None].float(), size=(512, 512), mode="bicubic")
    return ((im - im.min()) / (im.max() - im.min() + 1e-8))[0, 0]


@torch.inference_mode()
def generate(pipe, row, rec):
    s = int(row["seed"])
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)   # live._seed_everything
    _install(pipe, rec)
    pe, ne = pipe.encode_prompt(prompt=row["prompt"], device=pipe.device, num_images_per_prompt=1,
                                do_classifier_free_guidance=True, negative_prompt=None)
    lat = torch.randn((1, pipe.unet.config.in_channels, 512 // pipe.vae_scale_factor, 512 // pipe.vae_scale_factor),
                      device=pipe.device, dtype=pe.dtype)
    res = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=lat, height=512, width=512,
               num_inference_steps=int(row["num_inferece_steps"]), guidance_scale=7.5, output_type="pil",
               callback_on_step_end=lambda _p, i, t, kw: kw, callback_on_step_end_tensor_inputs=["latents"])
    return np.asarray(res.images[0].convert("RGB")).copy()


def main(a):
    from diffusers import StableDiffusionPipeline
    from src.segmentation.ovam_full import fuse_ovam_maps
    pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float32, local_files_only=True).to("cuda")
    pipe.safety_checker = None; pipe.set_progress_bar_config(disable=True)
    agg, PAMR = load_seediff()
    gpu = torch.cuda.get_device_name(0)
    out = a.out / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for r in jn.rows(a.dataset, a.shard, a.num_shards, a.max_images, a.ids):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        lab = canon(r["label"])
        word = (r["classname"] if a.dataset == "voc" else r["label"]).replace("-", " ")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        rec = FRecorder(pipe, r["attribution_prompt"], r["prompt"], word, TOKENS[lab], TOKENS2[lab], a.dataset == "voc")
        img = generate(pipe, r, rec)
        W, T, Tf, SA, D = rec.results()
        torch.cuda.synchronize(); t_gen = time.perf_counter() - t0
        fw = fuse_ovam_maps(W, SA, (512, 512)).cpu().numpy(); ft = fuse_ovam_maps(T, SA, (512, 512)).cpu().numpy()
        ftf = fuse_ovam_maps(Tf, SA, (512, 512)).cpu().numpy()
        daam = daam_expand(D).cpu().numpy() if D is not None else None
        t2 = time.perf_counter(); sd_soft = sd_mask = None
        if rec.sd_loc is not None:
            sd_soft, sd_mask = seediff_masks(agg, PAMR, rec, img, [r["prompt"]], rec.sd_loc, word)
        rec.sd_store.clear(); t_sd = time.perf_counter() - t2
        t_sclip = 0.0
        peak = torch.cuda.max_memory_allocated() / 2**20
        zb = np.load(BR / a.dataset / f"{iid}.npz")
        same = bool(np.array_equal(zb["image"], img)); dmax = int(np.abs(zb["image"].astype(int) - img.astype(int)).max())
        maps = [fw, ft, ftf] + ([daam] if daam is not None else []) + ([sd_soft] if sd_soft is not None else [])
        if not all(np.isfinite(x).all() for x in maps):
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, ovam_w=fw.astype(np.float16), ovam_t=ft.astype(np.float16), ovam_tf=ftf.astype(np.float16),
                 ca_w=W.cpu().numpy(), ca_t=T.cpu().numpy(), ca_tf=Tf.cpu().numpy(), sa=SA.cpu().numpy(),
                 daam=(daam.astype(np.float16) if daam is not None else np.zeros(0, np.float16)),
                 daam_ok=daam is not None, daam_idx=np.array(rec.didx), ovam_pos=rec.pos,
                 seediff=(sd_soft.astype(np.float16) if sd_soft is not None else np.zeros(0, np.float16)),
                 seediff_mask=(sd_mask if sd_mask is not None else np.zeros(0, bool)), seediff_ok=sd_soft is not None,
                 seediff_loc=(-1 if rec.sd_loc is None else rec.sd_loc), sec_seediff=t_sd,
                 image=img, cls=r["label"],
                 same_as_brace=same, maxdiff_brace=dmax, sec_gen=t_gen, sec_sclip=t_sclip, peak_mib=peak, gpu=gpu)
        os.replace(tmp, o)
        print(f"[f:{a.dataset}] {iid} {lab} daam={'y' if daam is not None else 'n'} seediff={'y' if sd_soft is not None else 'n'} ({t_sd:.1f}s) same_as_brace={same} (max {dmax}) "
              f"{t_gen:.1f}s peak {peak / 1024:.1f} GiB", flush=True)
        del rec; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("voc", "coco"), required=True)
    ap.add_argument("--max-images", type=int, default=0); ap.add_argument("--ids", default="")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--coco-csv", type=Path, default=BR / "ovam_coco_cap.csv")
    a = ap.parse_args()
    br.patch_rows(a.coco_csv)
    main(a)
