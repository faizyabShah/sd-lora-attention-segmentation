"""Which generation setting reproduces the official OVAM benchmark images? Diagnostic, generation only, no scoring.

Replicates src/attention_capture/live.py (seed everything, CUDA randn initial noise, CFG 7.5, manifest steps, default
scheduler) without attention recording, under one variant per job:
  fp32       our runtime (fp32 weights), safety checker off
  fp16       fp16 weights, noise drawn in fp16 (diffusers' default prepare_latents for an fp16 pipe)
  fp16cast   fp16 weights, noise drawn in fp32 then cast
  fp16attn   fp16 weights, fp16 noise, classic AttnProcessor (pre-SDPA attention)
  fp32safety our runtime exactly, safety checker ON (checks the black images)
Images: every COCO-Cap / VOC-Sim image whose stored capture differs from the official PNG (mean abs diff >= 5) plus
20 + 10 matched controls. Writes outputs/claude_scratch/regen/<variant>.npz with mean abs diff vs official and vs stored.
"""
import csv, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); C = Path("/home/shafa/sg_comp")
EV = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam"
MAN = {"coco": (EV / "coco_captions", "ovam_coco_cap.csv", "cococap_maps"), "voc": (EV / "voc_sim", None, "vocsim_maps")}
OUT = ROOT / "outputs/claude_scratch/regen"; OUT.mkdir(parents=True, exist_ok=True)


def rows():
    out = []
    for ds, (d, name, sub) in MAN.items():
        name = name or next(p.name for p in d.glob("*.csv"))
        dd = dict(np.load(ROOT / f"outputs/claude_scratch/nerve_diag/imgdiff_{ds}.npy", allow_pickle=True).tolist())
        R = {r["image_id"]: r for r in csv.DictReader(open(d / name))}
        diff = sorted(k for k, v in dd.items() if float(v) >= 5)
        ctrl = sorted(k for k, v in dd.items() if float(v) < 1)[:: 25][: (20 if ds == "coco" else 10)]
        for k in diff + ctrl:
            r = dict(R[k]); r["ds"], r["dir"], r["sub"], r["diff0"] = ds, d, sub, float(dd[k]); out.append(r)
    return out


def main(tag):
    variant = tag.split(":")[0]
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention_processor import AttnProcessor
    fp16 = variant.startswith("fp16")
    kw = {} if variant == "fp32safety" else {"safety_checker": None, "requires_safety_checker": False}
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                                   torch_dtype=torch.float16 if fp16 else torch.float32, **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if variant in ("fp16attn", "fp32attn"):
        pipe.unet.set_attn_processor(AttnProcessor())
    print(torch.cuda.get_device_name(), pipe.scheduler.__class__.__name__, flush=True)
    res = {}
    for r in rows():
        seed = int(r["seed"]); t0 = time.time()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            pe, ne = pipe.encode_prompt(prompt=r["prompt"], device=pipe.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True, negative_prompt=None)
            shape = (1, pipe.unet.config.in_channels, 64, 64)
            if variant == "fp16cast":
                noise = torch.randn(shape, device="cuda", dtype=torch.float32).half()
            else:
                noise = torch.randn(shape, device="cuda", dtype=pe.dtype)
            img = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=noise, height=512, width=512,
                       num_inference_steps=int(r["num_inferece_steps"]), guidance_scale=7.5, output_type="pil").images[0]
        a = np.asarray(img.convert("RGB")).astype(int)
        off = np.asarray(Image.open(r["dir"] / r["image"]).convert("RGB")).astype(int)
        sto = np.load(C / r["sub"] / f"{r['image_id']}.npz")["image"].astype(int)
        res[r["image_id"]] = np.array([np.abs(a - off).mean(), np.abs(a - sto).mean(), r["diff0"]])
        print(f"{tag} {r['image_id'][:44]:44s} vs official {res[r['image_id']][0]:6.2f}  vs stored "
              f"{res[r['image_id']][1]:6.2f}  (stored vs official {r['diff0']:6.2f})  {time.time() - t0:.1f}s", flush=True)
    np.savez(OUT / f"{tag.replace(':', '_')}.npz", **res)


if __name__ == "__main__":
    main(sys.argv[1])
