import os
import json
import argparse
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from cleanfid import fid
from torchmetrics.multimodal import CLIPScore
from transformers import CLIPTokenizer
from transformers import BlipProcessor, BlipForImageTextRetrieval
import lpips

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()

RANK = args.rank

# =========================
# PATHS
# =========================

REAL_DIR = "./real_cityscapes_512"  # resized dataset (ALL images)
GEN_DIR = f"./outputs/cityscapes-gen-lora-r{RANK}"
OUT_DIR = f"./outputs/cityscapes-metrics-lora-r{RANK}"
CAPTIONS_PATH = "./leftImg8bit/metadata.jsonl"

os.makedirs(OUT_DIR, exist_ok=True)

FID_STATS_PATH = "./outputs/cityscapes_fid_stats.npz"

NUM_SAMPLES = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD FILES
# =========================

def load_images(folder, limit=None):
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ])
    return files if limit is None else files[:limit]

real_images = load_images(REAL_DIR, limit=None)  # ALL
gen_images = load_images(GEN_DIR, limit=NUM_SAMPLES)

assert len(gen_images) >= NUM_SAMPLES, "Not enough generated images"

print(f"Using {NUM_SAMPLES} generated images")
print(f"Using {len(real_images)} real images (for FID)")

# =========================
# LOAD CAPTIONS (ORDERED)
# =========================

with open(CAPTIONS_PATH, "r") as f:
    entries = [json.loads(line) for line in f]

captions = [e["text"] for e in entries[:NUM_SAMPLES]]

# =========================
# FID / KID (MATCH NOTEBOOK)
# =========================

print("\n=== FID / KID ===")

if not os.path.exists(FID_STATS_PATH):
    print("Computing real dataset stats (one-time)...")
    mu_real, sigma_real = fid.get_folder_features(
        REAL_DIR,
        mode="clean",
        num_workers=0
    )
    np.savez(FID_STATS_PATH, mu=mu_real, sigma=sigma_real)
else:
    print("Loading cached stats...")
    data = np.load(FID_STATS_PATH)
    mu_real, sigma_real = data["mu"], data["sigma"]

mu_gen, sigma_gen = fid.get_folder_features(
    GEN_DIR,
    mode="clean",
    num_workers=0
)

fid_score = fid.frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
kid_score = fid.compute_kid(GEN_DIR, REAL_DIR, mode="clean")

print(f"FID: {fid_score:.4f}")
print(f"KID: {kid_score:.4f}")

# =========================
# CLIP SCORE (EXACT NOTEBOOK)
# =========================

print("\n=== CLIP Score ===")

clip_metric = CLIPScore(
    model_name_or_path="openai/clip-vit-large-patch14"
).to(DEVICE)

to_tensor = transforms.ToTensor()

clip_scores = []

for i in tqdm(range(NUM_SAMPLES), desc="CLIPScore"):
    img = Image.open(gen_images[i]).convert("RGB")

    img_tensor = (to_tensor(img) * 255).to(torch.uint8).unsqueeze(0).to(DEVICE)

    prompt = captions[i]

    score = clip_metric(img_tensor, [prompt])
    clip_scores.append(score.item())

    clip_metric.reset()

clip_score = float(np.mean(clip_scores))
print(f"CLIP Score: {clip_score:.4f}")

# =========================
# BLIP SCORE (EXACT NOTEBOOK)
# =========================

print("\n=== BLIP Score ===")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip_model = BlipForImageTextRetrieval.from_pretrained(
    "Salesforce/blip-itm-base-coco",
    torch_dtype=torch.float16
).to(DEVICE).eval()

itm_scores = []
cos_scores = []

for i in tqdm(range(NUM_SAMPLES), desc="BLIPScore"):
    img = Image.open(gen_images[i]).convert("RGB")
    text = captions[i][:200]  # truncate

    inputs = blip_processor(img, text, return_tensors="pt").to(DEVICE, torch.float16)

    with torch.no_grad():
        itm_output = blip_model(**inputs)[0]
        itm_prob = torch.softmax(itm_output, dim=1)[0, 1].item()

        cos_output = blip_model(**inputs, use_itm_head=False)[0]
        cos_score = cos_output.item()

    itm_scores.append(itm_prob)
    cos_scores.append(cos_score)

blip_itm = float(np.mean(itm_scores))
blip_cos = float(np.mean(cos_scores))

print(f"BLIP ITM: {blip_itm:.4f}")
print(f"BLIP Cosine: {blip_cos:.4f}")

# =========================
# LPIPS (DIVERSITY — EXACT NOTEBOOK)
# =========================

print("\n=== LPIPS (Diversity) ===")

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

random.seed(42)

# sample subset
sample_paths = random.sample(gen_images, min(200, len(gen_images)))

images = torch.stack([
    transform(Image.open(p).convert("RGB"))
    for p in sample_paths
])

pairs = random.sample(
    [(i, j) for i in range(len(images)) for j in range(i+1, len(images))],
    min(500, len(images)*(len(images)-1)//2)
)

scores = []

for i, j in tqdm(pairs, desc="LPIPS"):
    with torch.no_grad():
        d = lpips_model(
            images[i:i+1].to(DEVICE),
            images[j:j+1].to(DEVICE)
        )
    scores.append(d.item())

lpips_mean = float(np.mean(scores))
lpips_std = float(np.std(scores))

print(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")

# =========================
# SAVE RESULTS
# =========================

results = {
    "fid": fid_score,
    "kid": kid_score,
    "clip_score": clip_score,
    "blip_itm": blip_itm,
    "blip_cosine": blip_cos,
    "lpips_mean": lpips_mean,
    "lpips_std": lpips_std,
}

out_path = os.path.join(OUT_DIR, "metrics.json")

with open(out_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved results to:", out_path)