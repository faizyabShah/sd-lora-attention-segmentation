"""Leftover COCO images: does the pre-substitution caption (couch / motorcycle / airplane) reproduce the official image?"""
import csv, random
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); D = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/coco_captions"
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                               safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
left = [r for r in csv.DictReader(open(ROOT / "outputs/claude_scratch/regen/prompt_fix_coco.csv")) if float(r["mean_abs_diff"]) >= 2]
M = {r["image_id"]: r for r in csv.DictReader(open(D / "ovam_coco_cap.csv"))}
for x in left:
    r = M[x["image_id"]]; p = r["prompt"]; off = np.asarray(Image.open(D / r["image"]).convert("RGB")).astype(int); s = int(r["seed"])
    alts = {p.replace("sofa", "couch"), p.replace("motorbike", "motorcycle"), p.replace("aeroplane", "airplane")} - {p}
    cand = [q + e for q in ([p] + sorted(alts)) for e in ("", ".")]
    res = []
    for q in cand:
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            img = pipe(q, num_inference_steps=30).images[0]
        res.append((float(np.abs(np.asarray(img).astype(int) - off).mean()), q))
    b = min(res); print(x["image_id"][:40], f"best {b[0]:.2f}", repr(b[1]), flush=True)
