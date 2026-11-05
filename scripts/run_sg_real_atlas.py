#!/usr/bin/env python3
"""SG-real-atlas capture (protocol Section 137).

One replay-complete pass per image-class pair. Every condition runs the unchanged SG25/A3 recorder
through ``sg27_runtime.live_maps``, so the A3 seed, SA calls 10-20 and the V0/V1/V3 maps are the frozen
implementations. What a condition stores depends on its level:

  full  BRACE SA Q/K (calls 10-20; see --ddim-qk-sets), live GT R/L over every SA module x call x head,
        CA logits / log-sum-exp / |q| per head for every CA module x call, per-call head-mean token probs
  seed  second identical pass of renoise noise 0: label-free R/L with the pair's own A3 seed
  lite  live maps, A3 layer weights, call-window head-mean token probs per CA module

Conditions: renoise/n0 (full + seed), ddim (full), generate (VOC-Sim only; full), renoise/n1-n2, six views,
up to three prompt variants (lite). Inference inputs are the image and class names only; GT enters only
the diagnostic R/L statistics. Output per pair: <out>/<set>/<image>/<k>/ written to a temporary
directory and renamed when complete, so an interrupted run resumes without partial or duplicate records.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_coco_cap_transfer as transfer, sg27_runtime  # noqa: E402
from src.attention_capture import recorder as rec_module  # noqa: E402
from src.segmentation import sg_combined_v2  # noqa: E402
from src.segmentation.sg25_sa import SG25Recorder  # noqa: E402

VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
COCO = ROOT / "datasets/coco2017"
SETS = ROOT / "outputs/segmentation/sg_real_atlas"
VOC_CLASSES = [transfer.voc_rows(i)[0] for i in range(20)]
COCO_CLASSES = (ROOT / "outputs/segmentation/sg_real_data/coco_classes.txt").read_text().splitlines()
VIEWS = {"q0": (0, 0, .6, .6), "q1": (.4, 0, 1, .6), "q2": (0, .4, .6, 1), "q3": (.4, .4, 1, 1),
         "center": (.2, .2, .8, .8)}
NOISES = 3
WINDOW = (10, 20)

# ------------------------------------------------------------------ pause recording for raw UNet calls
PAUSED = {"on": False}
_orig_call = rec_module.RecordingAttnProcessor.__call__


def _call(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
    if PAUSED["on"]:
        return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *a, **k)
    return _orig_call(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, *a, **k)


rec_module.RecordingAttnProcessor.__call__ = _call


# ------------------------------------------------------------------ recorder
class AtlasRecorder(SG25Recorder):
    """SG25 recorder plus atlas statistics. Class attributes are set per condition."""

    level = "lite"
    rl_masks: dict[int, torch.Tensor] = {}      # side -> [side*side, M] soft masks for live R/L
    all_qk = False                              # store Q/K of every SA module in the call window
    n_tokens = 0                                # attribution prompt length incl. SOT/EOT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atlas = {"rl": {}, "ca": {}, "tok": {}, "qk": {}}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        step = self.call_counts[module_name]
        try:
            if kind == "self":
                self._self(module_name, step, query[1], key[1], scale)
            else:
                self._cross(module_name, step, query, key, value, attention_mask, scale, attribution_key)
        finally:
            super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)

    @torch.inference_mode()
    def _self(self, name, step, q, k, scale):
        side = math.isqrt(q.shape[-2])
        if self.level in ("full", "seed") and side in self.rl_masks:
            a = torch.softmax(torch.bmm(q.float(), k.float().transpose(1, 2)) * scale, -1)  # [8,N,N]
            m = self.rl_masks[side]                                                       # [N,M]
            y = torch.matmul(a, m)                                                        # [8,N,M]
            keep = (m[None] * y).sum(1) / m.sum(0).clamp_min(1e-8)
            leak = ((1 - m)[None] * y).sum(1) / (1 - m).sum(0).clamp_min(1e-8)
            self.atlas["rl"].setdefault(name, {})[step] = torch.stack((keep, leak), -1).half().cpu()
            del a, y
        if self.all_qk and self.level == "full" and WINDOW[0] <= step <= WINDOW[1]:
            self.atlas["qk"].setdefault(name, []).append(
                torch.stack((q.detach().half(), k.detach().half())).cpu())

    @torch.inference_mode()
    def _cross(self, name, step, query, key, value, attention_mask, scale, attribution_key):
        if attribution_key is not None:
            k = attribution_key[0]
            module = self.pipe.unet.get_submodule(name)
            states = self.attribution_hidden_states
            if module.norm_cross:
                states = module.norm_encoder_hidden_states(states)
            v = module.to_v(states).reshape(1, states.shape[1], module.heads, -1).permute(0, 2, 1, 3)[0]
            mask = None
        else:
            k, v = key[1], value[1]
            mask = None if attention_mask is None else attention_mask[1].float()
        q = query[1].float()
        scores = q @ k.float().transpose(-1, -2) * scale
        if mask is not None:
            scores = scores + mask
        n = min(self.n_tokens + 4, scores.shape[-1])                           # SOT..EOT + 4 pads
        prob = scores.softmax(-1)
        tokens = prob[..., :n].mean(0)                                          # [N, n] head mean
        target = list(self.target)
        if self.level == "full":
            module = self.pipe.unet.get_submodule(name)
            w = module.to_out[0].weight.float()
            d = v.shape[-1]
            vn = torch.stack([(v[h, target].float() @ w[:, h * d:(h + 1) * d].T).norm(dim=-1)
                              for h in range(v.shape[0])], 1)                    # [T, 8]
            self.atlas["ca"].setdefault(name, []).append({
                "logit": scores[..., target].permute(2, 0, 1).half().cpu(),     # [T, 8, N]
                "lse": torch.logsumexp(scores, -1).half().cpu(),                # [8, N]
                "qnorm": q.norm(dim=-1).half().cpu(),                           # [8, N]
                "knorm": k[:, target].float().norm(dim=-1).T.half().cpu(),      # [T, 8]
                "vnorm": vn.half().cpu()})
            self.atlas["tok"].setdefault(name, []).append(tokens.T.half().cpu())  # [n, N]
        else:
            acc = self.atlas["tok"].setdefault(name, torch.zeros((2,) + tokens.T.shape, device=tokens.device))
            acc[0] += tokens.T / 31
            if WINDOW[0] <= step <= WINDOW[1]:
                acc[1] += tokens.T / (WINDOW[1] - WINDOW[0] + 1)


# ------------------------------------------------------------------ trajectories
@contextlib.contextmanager
def trajectory(latent_at):
    """Replace every UNet input of the pipeline call with latent_at(pipe, t) (as Section 136)."""
    orig = StableDiffusionPipeline.__call__

    def patched(self, *args, latents=None, callback_on_step_end=None, num_inference_steps=50, **kw):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        ts = self.scheduler.timesteps.tolist()

        def cb(p, i, t, ckw):
            out = callback_on_step_end(p, i, t, ckw) if callback_on_step_end else ckw
            if i + 1 < len(ts):
                out["latents"] = latent_at(self, ts[i + 1])
            return out
        return orig(self, *args, latents=latent_at(self, ts[0]), callback_on_step_end=cb,
                    num_inference_steps=num_inference_steps, **kw)

    StableDiffusionPipeline.__call__ = patched
    try:
        yield
    finally:
        StableDiffusionPipeline.__call__ = orig


def renoise_fn(z0, noise):
    def at(pipe, t):
        a = pipe.scheduler.alphas_cumprod[int(t)].to(z0.device)
        return (a.sqrt() * z0.float() + (1 - a).sqrt() * noise).to(z0.dtype)
    return at


def table_fn(table):
    return lambda pipe, t: table[int(t)]


def noise_for(iid, k, z0):
    gen = torch.Generator(device=z0.device).manual_seed(zlib.crc32(iid.encode()) + 1000003 * k)
    return torch.randn(z0.shape, generator=gen, device=z0.device, dtype=torch.float32)


@torch.inference_mode()
def unet_eps(pipe, x, t, emb):
    PAUSED["on"] = True
    try:
        return pipe.unet(x, int(t), encoder_hidden_states=emb, return_dict=False)[0]
    finally:
        PAUSED["on"] = False


@torch.inference_mode()
def ddim_invert(pipe, z0, ts_unique, emb):
    """DDIM inversion at the PNDM timesteps (ascending); returns {t: x_t} and the DDIM reconstruction."""
    ac = pipe.scheduler.alphas_cumprod.to(z0.device).float()
    x, a_prev, table = z0.float(), torch.tensor(1.0, device=z0.device), {}
    for t in ts_unique:
        eps = unet_eps(pipe, x.to(z0.dtype), t, emb).float()
        x0 = (x - (1 - a_prev).sqrt() * eps) / a_prev.sqrt()
        x = ac[t].sqrt() * x0 + (1 - ac[t]).sqrt() * eps
        a_prev = ac[t]
        table[int(t)] = x.to(z0.dtype)
    y = x
    for i in range(len(ts_unique) - 1, -1, -1):
        t = ts_unique[i]
        a = ac[t]; a_next = ac[ts_unique[i - 1]] if i > 0 else torch.tensor(1.0, device=z0.device)
        eps = unet_eps(pipe, y.to(z0.dtype), t, emb).float()
        x0 = (y - (1 - a).sqrt() * eps) / a.sqrt()
        y = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return table, y.to(z0.dtype)


# ------------------------------------------------------------------ data
def load_pair_source(entry):
    """entry: dict(set, image, dataset, [row]) -> image uint8 [H,W,3], label map [H,W] (255 ignore)."""
    if entry["dataset"] == "voc":
        img = np.asarray(Image.open(VOC / f"JPEGImages/{entry['image']}.jpg").convert("RGB"))
        lab = np.asarray(Image.open(VOC / f"SegmentationClass/{entry['image']}.png"))
        return img, lab, VOC_CLASSES
    if entry["dataset"] == "coco":
        img = np.asarray(Image.open(COCO / f"val2017/{entry['image']}.jpg").convert("RGB"))
        lab = np.asarray(Image.open(COCO / f"sg_real_data_labels/{entry['image']}.png"))
        return img, lab, COCO_CLASSES
    raise ValueError(entry)


def crop(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(box[0] * w)), int(round(box[1] * h)), int(round(box[2] * w)), int(round(box[3] * h)))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def to512(img):
    return np.asarray(Image.fromarray(img).resize((512, 512), Image.Resampling.BICUBIC))


@torch.inference_mode()
def encode(pipe, img512):
    x = torch.from_numpy(img512).permute(2, 0, 1)[None].to(pipe.device, pipe.vae.dtype) / 127.5 - 1
    return pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor


@torch.inference_mode()
def decode(pipe, z):
    x = pipe.vae.decode(z / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()


def rl_masks(truth, valid, device, seed64=None):
    """Soft masks per SA side. GT mode: exact, eroded, dilated (fractional occupancy over valid pixels).
    Seed mode: the A3 seed (soft) and its >=0.55 mask, area-downsampled."""
    out = {}
    if seed64 is None:
        t = torch.as_tensor(truth & valid, dtype=torch.float32, device=device)[None, None]
        t64 = F.interpolate(t, size=(64, 64), mode="area")
        base = (t64 >= .5).float()
        er = 1 - F.max_pool2d(1 - base, 5, 1, 2)
        di = F.max_pool2d(base, 5, 1, 2)
        stack64 = torch.cat([t64, er, di], 1)                                        # [1,3,64,64]
    else:
        s = seed64.float()[None, None]
        stack64 = torch.cat([s, (s >= .55).float()], 1)
    for side in (8, 16, 32, 64):
        st = stack64 if side == 64 else F.interpolate(stack64, size=(side, side), mode="area")
        out[side] = st[0].reshape(st.shape[1], -1).T.contiguous()                    # [N, M]
    return out


def gt_mask_full(truth, valid):
    return truth & valid


# ------------------------------------------------------------------ one condition
def run_condition(runtime, pipe, row, latent_at, level, masks=None, all_qk=False, sets_qk=True):
    rc = runtime["recorder_class"]
    rc.level, rc.rl_masks, rc.all_qk = level, (masks or {}), all_qk
    rc.n_tokens = len(pipe.tokenizer(row["attribution_prompt"], padding=False)["input_ids"])
    t0 = time.perf_counter()
    with trajectory(latent_at), sg27_runtime.live_maps(runtime, row, "voc") as (sample, sa, seed, base, previous):
        torch.cuda.synchronize(); t1 = time.perf_counter()
        rec = runtime["recorders"][-1]
        v2, _ = sg_combined_v2.aggregate(sa, seed)
        maps = {"A3": seed.float().cpu().numpy(), "C16": base["C16"].reshape(64, 64).float().cpu().numpy(),
                "V0": v2["V0"].float().cpu().numpy(), "V1": v2["V1"].float().cpu().numpy(),
                "V3": v2["V3"].float().cpu().numpy()}
        weights = np.stack([w for (_, lab, w) in rec.weights if lab == "selected"]).astype(np.float32)
        out = {"maps": maps, "weights": weights, "atlas": rec.atlas}
        if level == "full" and sets_qk:
            out["qk"] = {s: torch.stack([torch.stack(p) for p in sa[s]]).cpu() for s in (16, 32, 64)}  # [11,2,8,N,d]
        timesteps = list(sample.timesteps)
        torch.cuda.synchronize(); t2 = time.perf_counter()
    out["timesteps"] = timesteps
    out["time"] = {"unet_and_record": t1 - t0, "extract": t2 - t1}
    return out


def pack_atlas(atlas, level):
    arrays = {}
    for name, per in atlas["rl"].items():
        arrays[f"rl/{name}"] = torch.stack([per[s] for s in sorted(per)]).numpy()        # [calls,8,M,2]
    if level == "full":
        for name, seq in atlas["ca"].items():
            for key in ("logit", "lse", "qnorm", "knorm", "vnorm"):
                arrays[f"ca/{key}/{name}"] = torch.stack([d[key] for d in seq]).numpy()
        for name, seq in atlas["tok"].items():
            arrays[f"tok/{name}"] = torch.stack(seq).numpy()                                # [31,n,N]
    elif level == "lite":
        for name, acc in atlas["tok"].items():
            arrays[f"tok/{name}"] = acc.half().cpu().numpy()                                # [2,n,N]
    for name, seq in atlas["qk"].items():
        arrays[f"qk/{name}"] = torch.stack(seq).numpy()                                     # [11,2,8,N,d]
    return arrays


def save_npz(path, arrays, compress=True):
    (np.savez_compressed if compress else np.savez)(path, **arrays)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=SETS / "manifest.json")
    ap.add_argument("--sets", nargs="+", default=["D", "G", "C", "S"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = [e for s in args.sets for e in manifest[s]][args.shard::args.num_shards]
    if args.max_images:
        entries = entries[: args.max_images]
    runtime = sg27_runtime.make_runtime(args.model, recorder_class=AtlasRecorder)
    pipe = runtime["pipe"]
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(pipe.device)
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    pipe.scheduler.set_timesteps(30, device=pipe.device)
    ts_unique = sorted(set(pipe.scheduler.timesteps.tolist()))
    empty = pipe.text_encoder(pipe.tokenizer([""], padding="max_length", max_length=77,
                                             return_tensors="pt").input_ids.to(pipe.device))[0].to(pipe.unet.dtype)

    def recon_metrics(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(pipe.device) / 255
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().to(pipe.device) / 255
        mse = float(((ta - tb) ** 2).mean())
        return {"psnr": 10 * math.log10(1 / max(mse, 1e-12)), "ssim": float(ssim_fn(ta, tb, data_range=1.0)),
                "lpips": float(lp(ta * 2 - 1, tb * 2 - 1).mean())}

    for entry in entries:
        img_dir = args.output_dir / entry["set"] / entry["image"]
        pairs = entry["classes"]
        if all((img_dir / str(k) / "done.json").exists() for k in pairs):
            continue
        timing = {"image_setup": time.perf_counter()}
        if entry["dataset"] == "vocsim":
            row0 = entry["row"]
            img = lab = None
            names = None
        else:
            img, lab, names = load_pair_source(entry)
        # ---------- image-level: generated image (G), latents, DDIM inversion, reconstruction metrics
        gen = {}
        if entry["dataset"] == "vocsim":
            rc = runtime["recorder_class"]
            rc.level, rc.rl_masks, rc.all_qk = "lite", {}, False
            rc.n_tokens = len(pipe.tokenizer(row0["attribution_prompt"], padding=False)["input_ids"])
            with sg27_runtime.live_maps(runtime, dict(row0), "voc") as (sample, *_):
                img = np.asarray(sample.generated_image, np.uint8)
                truth_g = np.asarray(sample.target_mask, bool)
            lab = np.where(truth_g, 1, 0).astype(np.uint8)
        img512 = to512(img)
        z0 = encode(pipe, img512)
        image_info = {"vae": recon_metrics(decode(pipe, z0), img512)}
        table, zrec = ddim_invert(pipe, z0, ts_unique, empty)
        image_info["ddim_recon"] = recon_metrics(decode(pipe, zrec), img512)
        image_info["shape"] = list(img.shape[:2])
        timing["image_setup"] = time.perf_counter() - timing["image_setup"]
        for k in pairs:
            pair_dir = img_dir / str(k)
            if (pair_dir / "done.json").exists():
                continue
            tmp = img_dir / f".{k}.tmp"
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            t_pair = time.perf_counter()
            truth = lab == k
            valid = lab != 255
            if entry["dataset"] == "vocsim":
                cls = row0["classname"]
                base_row = dict(row0)
                queried = [cls]
            else:
                cls = names[k - 1]
                queried = [names[j - 1] for j in pairs]
                base_row = None
            def row_for(prompt, key):
                if base_row is not None and prompt is None:
                    return dict(base_row, image_id=key)
                return {"image_id": key, "classname": cls, "prompt": prompt, "attribution_prompt": prompt,
                        "seed": "0", "num_inferece_steps": "30", "annotation": str(tmp / "gt.png")}
            Image.fromarray((truth & valid).astype(np.uint8) * 255).save(tmp / "gt.png")
            prompt0 = f"a photograph of a {cls}"
            gt_masks = rl_masks(truth, valid, pipe.device)
            maps, meta, times, sizes = {}, {"class": cls, "k": int(k), "queried": queried, "prompts": {},
                                           "boxes": {}, "noise_seeds": {}}, {}, {}
            iid = entry["image"]
            first50 = entry.get("all_qk", False)

            def store(cond, res, level):
                maps.update({f"{cond}/{n}": v for n, v in res["maps"].items()})
                maps[f"{cond}/weights"] = res["weights"]
                arr = pack_atlas(res["atlas"], level)
                if arr:
                    f = tmp / f"atlas_{cond.replace('/', '_')}.npz"
                    save_npz(f, arr, compress=False)
                    sizes[cond] = f.stat().st_size
                if "qk" in res:
                    f = tmp / f"qk_{cond.replace('/', '_')}.npz"
                    save_npz(f, {f"s{s}": v.numpy() for s, v in res["qk"].items()}, compress=False)
                    sizes[cond + "/qk"] = f.stat().st_size
                times[cond] = res["time"]
                meta["timesteps"] = res["timesteps"]

            # generate (VOC-Sim only): the original trajectory
            if entry["dataset"] == "vocsim":
                res = run_condition_generate(runtime, pipe, row_for(None, f"{iid}/{k}"), gt_masks)
                store("generate", res, "full")
            # renoise n0 (full) and its label-free second pass (seed)
            n0 = noise_for(iid, 0, z0)
            meta["noise_seeds"]["n0"] = zlib.crc32(iid.encode())
            ref_row = row_for(prompt0, f"{iid}/{k}")
            meta["prompts"]["ref"] = prompt0
            res = run_condition(runtime, pipe, ref_row, renoise_fn(z0, n0), "full", gt_masks, all_qk=first50)
            store("renoise/n0", res, "full")
            seed64 = torch.as_tensor(res["maps"]["A3"], device=pipe.device)
            res = run_condition(runtime, pipe, ref_row, renoise_fn(z0, n0), "seed",
                                rl_masks(None, None, pipe.device, seed64), sets_qk=False)
            store("renoise/n0_seed", res, "seed")
            # ddim (full); Q/K only for sets listed in the manifest
            res = run_condition(runtime, pipe, ref_row, table_fn(table), "full", gt_masks,
                                sets_qk=entry.get("ddim_qk", False))
            store("ddim", res, "full")
            # lite: other noises
            for j in range(1, NOISES):
                meta["noise_seeds"][f"n{j}"] = zlib.crc32(iid.encode()) + 1000003 * j
                res = run_condition(runtime, pipe, ref_row, renoise_fn(z0, noise_for(iid, j, z0)), "lite")
                store(f"renoise/n{j}", res, "lite")
            # lite: views (fixed boxes + attention-guided crop from the label-free reference seed)
            yy, xx = np.unravel_index(int(np.argmax(maps["renoise/n0/A3"])), (64, 64))
            cx, cy = (xx + .5) / 64, (yy + .5) / 64
            boxes = dict(VIEWS)
            boxes["peak"] = (min(max(cx - .25, 0), .5), min(max(cy - .25, 0), .5),
                             min(max(cx - .25, 0), .5) + .5, min(max(cy - .25, 0), .5) + .5)
            for v, box in boxes.items():
                part, px = crop(img, box)
                meta["boxes"][v] = px
                zv = encode(pipe, to512(part))
                res = run_condition(runtime, pipe, ref_row, renoise_fn(zv, noise_for(iid, 0, zv)), "lite")
                store(f"view/{v}", res, "lite")
            # lite: prompt variants
            variants = {"the": f"a photo of the {cls}", "background": f"a photograph of a {cls} and background"}
            if len(queried) > 1:
                others = [c for c in queried if c != cls]
                ordered = [cls] + others[:2]
                variants["multi"] = "a photograph of " + ", ".join(f"a {c}" for c in ordered[:-1]) + f" and a {ordered[-1]}"
            for name, prompt in variants.items():
                meta["prompts"][name] = prompt
                res = run_condition(runtime, pipe, row_for(prompt, f"{iid}/{k}"), renoise_fn(z0, n0), "lite")
                store(f"prompt/{name}", res, "lite")
            t_w = time.perf_counter()
            save_npz(tmp / "maps.npz", maps)
            meta["image"] = image_info
            meta["time"] = {"conditions": times, "image_setup_once": timing["image_setup"],
                            "pair_total": time.perf_counter() - t_pair}
            meta["bytes"] = sizes
            meta["write_maps_s"] = time.perf_counter() - t_w
            (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
            (tmp / "done.json").write_text(json.dumps({"ok": True}))
            os.replace(tmp, pair_dir)
            print(f"[atlas:{entry['set']}] {iid}/{k} {cls} {meta['time']['pair_total']:.1f}s "
                  f"{sum(sizes.values()) / 1e6:.0f} MB peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                  f"V3@ref max {maps['renoise/n0/V3'].max():.3g}", flush=True)


def run_condition_generate(runtime, pipe, row, masks):
    """Original SD1.5 generation (no trajectory patch), full level."""
    rc = runtime["recorder_class"]
    rc.level, rc.rl_masks, rc.all_qk = "full", masks, False
    rc.n_tokens = len(pipe.tokenizer(row["attribution_prompt"], padding=False)["input_ids"])
    t0 = time.perf_counter()
    with sg27_runtime.live_maps(runtime, row, "voc") as (sample, sa, seed, base, previous):
        torch.cuda.synchronize(); t1 = time.perf_counter()
        rec = runtime["recorders"][-1]
        v2, _ = sg_combined_v2.aggregate(sa, seed)
        maps = {"A3": seed.float().cpu().numpy(), "C16": base["C16"].reshape(64, 64).float().cpu().numpy(),
                "V0": v2["V0"].float().cpu().numpy(), "V1": v2["V1"].float().cpu().numpy(),
                "V3": v2["V3"].float().cpu().numpy()}
        weights = np.stack([w for (_, lab, w) in rec.weights if lab == "selected"]).astype(np.float32)
        out = {"maps": maps, "weights": weights, "atlas": rec.atlas,
               "qk": {s: torch.stack([torch.stack(p) for p in sa[s]]).cpu() for s in (16, 32, 64)},
               "timesteps": list(sample.timesteps)}
        torch.cuda.synchronize(); t2 = time.perf_counter()
    out["time"] = {"unet_and_record": t1 - t0, "extract": t2 - t1}
    return out


if __name__ == "__main__":
    main()
