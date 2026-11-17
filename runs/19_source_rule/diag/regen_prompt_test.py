"""Do hyphen-free VOC prompts reproduce the official dining-table / potted-plant images? (trdp, fp32, our live.py call)"""
import csv, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
EV = Path("/home/shafa/sd-lora-attention-segmentation/datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim")
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                               safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
R = [r for r in csv.DictReader(open(EV / "ovam_voc_sim.csv")) if r["classname"] in ("dining-table", "potted-plant")][::5]
for r in R:
    off = np.asarray(Image.open(EV / r["image"]).convert("RGB")).astype(int); s = int(r["seed"]); out = []
    for p in (r["prompt"], r["prompt"].replace("-", " "), r["prompt"].replace("-", "")):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            img = pipe(p, num_inference_steps=int(r["num_inferece_steps"])).images[0]
        out.append(f"'{p}' {np.abs(np.asarray(img).astype(int) - off).mean():.2f}")
    print(r["image_id"], " | ".join(out), flush=True)
