"""Regenerate with OVAM's own stack and call (env ovam_stack: torch 2.1.0 cu118, diffusers 0.20.0).
OVAM (experiments/*/generate*.py): StableDiffusionPipeline.from_pretrained(model_id).to("cuda"); set_seed(seed);
pipe(prompt, num_inference_steps=30). Same image list as regen_variants.py. Diagnostic only, no scoring.
Usage: python regen_ovam_stack.py <tag> [sdpa|classic]
"""
import csv, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); C = Path("/home/shafa/sg_comp")
EV = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam"
MAN = {"coco": (EV / "coco_captions", "ovam_coco_cap.csv", "cococap_maps"), "voc": (EV / "voc_sim", "ovam_voc_sim.csv", "vocsim_maps")}
OUT = ROOT / "outputs/claude_scratch/regen"


def rows():
    out = []
    for ds, (d, name, sub) in MAN.items():
        dd = dict(np.load(ROOT / f"outputs/claude_scratch/nerve_diag/imgdiff_{ds}.npy", allow_pickle=True).tolist())
        R = {r["image_id"]: r for r in csv.DictReader(open(d / name))}
        diff = sorted(k for k, v in dd.items() if float(v) >= 5)
        ctrl = sorted(k for k, v in dd.items() if float(v) < 1)[:: 25][: (20 if ds == "coco" else 10)]
        for k in diff + ctrl:
            r = dict(R[k]); r["ds"], r["dir"], r["sub"], r["diff0"] = ds, d, sub, float(dd[k]); out.append(r)
    return out


def main(tag, attn):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                                   safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if attn == "classic":
        from diffusers.models.attention_processor import AttnProcessor
        pipe.unet.set_attn_processor(AttnProcessor())
    import diffusers
    print(torch.__version__, diffusers.__version__, torch.cuda.get_device_name(), pipe.scheduler.__class__.__name__, attn, flush=True)
    res = {}
    for r in rows():
        seed = int(r["seed"]); t0 = time.time()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            img = pipe(r["prompt"], num_inference_steps=int(r["num_inferece_steps"])).images[0]
        a = np.asarray(img.convert("RGB")).astype(int)
        off = np.asarray(Image.open(r["dir"] / r["image"]).convert("RGB")).astype(int)
        sto = np.load(C / r["sub"] / f"{r['image_id']}.npz")["image"].astype(int)
        res[r["image_id"]] = np.array([np.abs(a - off).mean(), np.abs(a - sto).mean(), r["diff0"]])
        print(f"{tag} {r['image_id'][:44]:44s} vs official {res[r['image_id']][0]:6.2f}  vs stored {res[r['image_id']][1]:6.2f}  "
              f"(stored vs official {r['diff0']:6.2f})  {time.time() - t0:.1f}s", flush=True)
    np.savez(OUT / f"{tag}.npz", **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sdpa")
