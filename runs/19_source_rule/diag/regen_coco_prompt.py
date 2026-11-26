"""Which caption text reproduces the official COCO-Cap images that our regeneration misses? Diagnostic only."""
import csv, random
from pathlib import Path
import numpy as np, torch
from PIL import Image
D = Path("/home/shafa/sd-lora-attention-segmentation/datasets/ovam_experiment_with_dataset/evaluation/ovam/coco_captions")
dd = dict(np.load("/home/shafa/sd-lora-attention-segmentation/outputs/claude_scratch/nerve_diag/imgdiff_coco.npy", allow_pickle=True).tolist())
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                               safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
R = [r for r in csv.DictReader(open(D / "ovam_coco_cap.csv")) if float(dd[r["image_id"]]) >= 5][::5][:20]
V = {"asis": lambda p, c: p, "trail_sp": lambda p, c: p + " ", "trail_dot": lambda p, c: p + ".", "trail_spdot": lambda p, c: p + " .",
     "Cap.": lambda p, c: p[0].upper() + p[1:] + ".", "Cap": lambda p, c: p[0].upper() + p[1:], "lead_sp": lambda p, c: " " + p,
     "+cls": lambda p, c: p + ", " + c, "cls+": lambda p, c: c + ", " + p}
win = {}
for r in R:
    off = np.asarray(Image.open(D / r["image"]).convert("RGB")).astype(int); s = int(r["seed"]); out = {}
    for k, f in V.items():
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            img = pipe(f(r["prompt"], r["classname"]), num_inference_steps=30).images[0]
        out[k] = np.abs(np.asarray(img).astype(int) - off).mean()
    b = min(out, key=out.get); win[b] = win.get(b, 0) + (out[b] < 2)
    print(r["image_id"][:40], " ".join(f"{k} {v:.1f}" for k, v in out.items()), "| best", b, flush=True)
print("variants reproducing (<2):", win)
