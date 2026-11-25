#!/usr/bin/env python3
"""Run 44: DAAM (castorini/daam @ c30493e, env daam44) and OVAM without token optimisation (vpulab/ovam @ 12afd87,
env ovam44) on run 44's real photos, through ONE noised SD 1.5 pass per item (owner decision 2026-11-25).

The pass is the one our method uses (scripts/jc_realvoc.one_pass): photo bicubic to 512, VAE posterior mean x scaling,
noise from torch.Generator(cpu).manual_seed(0), z_t at t = 100 (alphas_cumprod), prompt "A photograph of a {word}", fp32.
Both methods hook the UNet the way they hook a generation with classifier-free guidance, so the UNet is called with the
batch [unconditional, conditional] (DAAM keeps the second half of every attention map, OVAM stores the first sample's
queries by default); the conditional half is identical to our single conditional call.
Saved per item (64 x 64, float32): DAAM word heat map (compute_global_heat_map -> compute_word_heat_map) | OVAM heat map
for the word (get_ovam_callable defaults; channel of the word's token, as save_heatmap) and OVAM's self-attention map
(StableDiffusionHookerSA.get_self_attention_map, cross-attentions not hooked, as extract_sa).
Read-out at scoring (OVAM's evaluation notebooks): bilinear to photo size; DAAM: map / max > 0.4; OVAM: map / max x SA
rescaled to [0.85, 1] > 0.4 (VOC-Sim settings for every column), DenseCRF as OVAM (secondary).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
T_NOISE, NOISE_SEED, PX = 100, 0, 512
COCO_ROOT = ROOT / "datasets/coco2017"


def zt_and_embeds(pipe, img_u8, prompt, dev):
    import numpy as np, torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
    x = F.interpolate(x, size=(PX, PX), mode="bicubic", align_corners=False).clamp(-1, 1)
    z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(NOISE_SEED)).to(dev, z0.dtype)
    ac = pipe.scheduler.alphas_cumprod.to(dev)
    zt = ac[T_NOISE].sqrt() * z0 + (1 - ac[T_NOISE]).sqrt() * eps
    tok = lambda s: pipe.tokenizer(s, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        cond = pipe.text_encoder(tok(prompt))[0]; unc = pipe.text_encoder(tok(""))[0]
    return torch.cat([zt, zt]), torch.cat([unc, cond])


def run(a):
    import numpy as np, torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    sys.path.insert(0, str(ROOT / "scripts")); from jc_real44 import query_word
    dev = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, safety_checker=None,
                                                   requires_safety_checker=False, local_files_only=True).to(dev)
    it = [x for x in json.loads((WORK / "items.json").read_text())[a.set] if x["primary"]][a.shard::a.num_shards]
    if a.limit: it = it[:a.limit]
    out = WORK / a.method / a.set; out.mkdir(parents=True, exist_ok=True)
    for x in it:
        o = out / f"{x['id']}.npz"
        if o.exists():
            continue
        word = query_word(a.set, x["cls"]); prompt = f"A photograph of a {word}"
        path = COCO_ROOT / x["path"] if a.set == "coco" else ROOT / x["path"]
        img = np.asarray(Image.open(path).convert("RGB")); t0 = time.time()
        lat, emb = zt_and_embeds(pipe, img, prompt, dev); t = torch.tensor(T_NOISE, device=dev)
        res = {}
        with torch.no_grad():
            if a.method == "daam":
                from daam import trace
                with trace(pipe) as tc:
                    tc.all_heat_maps.clear(); tc.last_prompt = prompt
                    pipe.unet(lat, t, encoder_hidden_states=emb, return_dict=False)
                    hm = tc.compute_global_heat_map(prompt).compute_word_heat_map(word)
                res["daam"] = hm.heatmap.float().cpu().numpy()
            else:
                from ovam.stable_diffusion import StableDiffusionHooker
                from ovam.stable_diffusion_sa import StableDiffusionHookerSA
                with StableDiffusionHooker(pipe) as hk:
                    pipe.unet(lat, t, encoder_hidden_states=emb, return_dict=False)
                    hk._store_hidden_states()
                ov = hk.get_ovam_callable()
                e = ov.encode_text(prompt, remove_special_tokens=False)
                ids = pipe.tokenizer(prompt)["input_ids"]; wid = pipe.tokenizer(word, add_special_tokens=False)["input_ids"]
                pos = [i for i in range(len(ids) - len(wid) + 1) if ids[i:i + len(wid)] == wid][-1]
                hmap = ov(e).float().cpu().numpy()[0]
                res["ovam"] = hmap[pos]; res["ovam_tok"] = np.array([pos, len(wid)])
                with StableDiffusionHookerSA(pipe, extract_self_attentions=True, locator_kwargs={"locate_attn2": False}) as hs:   # extract_sa used the older `ovdaam` API, where this hooked every attn1
                    pipe.unet(lat, t, encoder_hidden_states=emb, return_dict=False)
                res["sa"] = hs.get_self_attention_map().float().cpu().numpy()
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez_compressed(tmp, prompt=prompt, size=np.array(img.shape[:2]),
                                                                  sec=time.time() - t0, **res); os.replace(tmp, o)
        print(f"[{a.method}:{a.set}] {x['id']} '{prompt}' {[(k, v.shape) for k, v in res.items()]} {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["daam", "ovam"]); ap.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
