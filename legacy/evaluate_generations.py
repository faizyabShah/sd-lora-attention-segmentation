

import os
import json
import random
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from cleanfid import fid
from cleanfid.features import build_feature_extractor

from torchmetrics.multimodal import CLIPScore

from transformers import (
    BlipProcessor,
    BlipForImageTextRetrieval
)

import lpips


# =========================================================
# PATHS
# =========================================================

BASE_DIR = "./outputs-sdxl"

CAPTIONS_PATH = "./cityscapes_cropped/metadata.jsonl"

OUTPUT_DIR = "./outputs-sdxl-metrics"
FID_STATS_PATH = "./outputs-sdxl-metrics/cityscapes_fid_stats.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DATASET DIRECTORIES
# =========================================================

DATASETS = {
    "frozen": os.path.join(BASE_DIR, "cityscapes-gen-frozen"),
    "r8": os.path.join(BASE_DIR, "cityscapes-gen-lora-r8"),
    "r16": os.path.join(BASE_DIR, "cityscapes-gen-lora-r16"),
    "r32": os.path.join(BASE_DIR, "cityscapes-gen-lora-r32"),
    "r64": os.path.join(BASE_DIR, "cityscapes-gen-lora-r64"),
    "r128": os.path.join(BASE_DIR, "cityscapes-gen-lora-r128"),
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# =========================================================
# HELPERS
# =========================================================

def load_images(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ])


# =========================================================
# FIND MINIMUM COMMON IMAGE COUNT
# =========================================================

dataset_counts = {}

for name, path in DATASETS.items():
    imgs = load_images(path)
    dataset_counts[name] = len(imgs)

print("\n=== IMAGE COUNTS ===")

for k, v in dataset_counts.items():
    print(f"{k}: {v}")

NUM_SAMPLES = min(dataset_counts.values())

print(f"\nUsing first {NUM_SAMPLES} images from ALL datasets")


# =========================================================
# LOAD CAPTIONS
# =========================================================

with open(CAPTIONS_PATH, "r") as f:
    entries = [json.loads(line) for line in f][:NUM_SAMPLES]

captions = [e["text"] for e in entries]


# =========================================================
# PREPARE REAL IMAGE LIST
# =========================================================

# IMPORTANT:
# We use original cropped dataset images directly.
# clean-fid will internally resize to Inception resolution.

real_images = []

for e in entries:

    # metadata stores paths like:
    # images/000123.jpg

    path = os.path.join("./cityscapes_cropped", e["file_name"])

    real_images.append(path)

print(f"\nLoaded {len(real_images)} real images")


# =========================================================
# BUILD FEATURE EXTRACTOR
# =========================================================

print("\n=== Preparing FID extractor ===")

feature_extractor = build_feature_extractor(
    mode="clean",
    device=DEVICE
)


# =========================================================
# COMPUTE REAL STATS
# =========================================================

if not os.path.exists(FID_STATS_PATH):

    print("\nComputing REAL dataset statistics...")

    feats_real = fid.get_files_features(
        real_images,
        model=feature_extractor,
        num_workers=0,
        mode="clean"
    )

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)

    np.savez(
        FID_STATS_PATH,
        mu=mu_real,
        sigma=sigma_real
    )

else:

    print("\nLoading cached REAL statistics...")

    data = np.load(FID_STATS_PATH)

    mu_real = data["mu"]
    sigma_real = data["sigma"]


# =========================================================
# LOAD MODELS
# =========================================================

print("\n=== Loading evaluation models ===")

clip_metric = CLIPScore(
    model_name_or_path="openai/clip-vit-large-patch14"
).to(DEVICE)

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-itm-base-coco"
)

blip_model = BlipForImageTextRetrieval.from_pretrained(
    "Salesforce/blip-itm-base-coco",
    torch_dtype=torch.float16
).to(DEVICE).eval()

lpips_model = lpips.LPIPS(net="alex").to(DEVICE)


# =========================================================
# TRANSFORMS
# =========================================================

to_tensor = transforms.ToTensor()

lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# =========================================================
# MAIN EVALUATION LOOP
# =========================================================

all_results = []

for DATASET_NAME, GEN_DIR in DATASETS.items():

    print("\n\n===================================")
    print(f"Processing: {DATASET_NAME}")
    print("===================================")

    OUT_DIR = os.path.join(OUTPUT_DIR, DATASET_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    gen_images = load_images(GEN_DIR)[:NUM_SAMPLES]

    # =====================================================
    # FID
    # =====================================================

    print("\n=== FID ===")

    feats_gen = fid.get_files_features(
        gen_images,
        model=feature_extractor,
        num_workers=0,
        mode="clean"
    )

    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)

    fid_score = fid.frechet_distance(
        mu_real,
        sigma_real,
        mu_gen,
        sigma_gen
    )

    print(f"FID: {fid_score:.4f}")


    # =====================================================
    # CLIP SCORE
    # =====================================================

    print("\n=== CLIP Score ===")

    clip_scores = []

    for i in tqdm(range(NUM_SAMPLES), desc=f"CLIP {DATASET_NAME}"):

        img = Image.open(gen_images[i]).convert("RGB")

        img_tensor = (
            to_tensor(img) * 255
        ).to(torch.uint8).unsqueeze(0).to(DEVICE)

        score = clip_metric(
            img_tensor,
            [captions[i]]
        )

        clip_scores.append(score.item())

        clip_metric.reset()

    clip_score = float(np.mean(clip_scores))

    print(f"CLIP Score: {clip_score:.4f}")


    # =====================================================
    # BLIP SCORE
    # =====================================================

    print("\n=== BLIP Score ===")

    itm_scores = []
    cos_scores = []

    for i in tqdm(range(NUM_SAMPLES), desc=f"BLIP {DATASET_NAME}"):

        img = Image.open(gen_images[i]).convert("RGB")

        text = captions[i]

        inputs = blip_processor(
            img,
            text,
            return_tensors="pt"
        ).to(DEVICE, torch.float16)

        with torch.no_grad():

            itm_output = blip_model(**inputs)[0]

            itm_prob = torch.softmax(
                itm_output,
                dim=1
            )[0, 1].item()

            cos_output = blip_model(
                **inputs,
                use_itm_head=False
            )[0]

            cos_score = cos_output.item()

        itm_scores.append(itm_prob)
        cos_scores.append(cos_score)

    blip_itm = float(np.mean(itm_scores))
    blip_cos = float(np.mean(cos_scores))

    print(f"BLIP ITM: {blip_itm:.4f}")
    print(f"BLIP Cosine: {blip_cos:.4f}")


    # =====================================================
    # LPIPS
    # =====================================================

    print("\n=== LPIPS Diversity ===")

    random.seed(42)

    sample_paths = random.sample(
        gen_images,
        min(200, len(gen_images))
    )

    images = torch.stack([
        lpips_transform(
            Image.open(p).convert("RGB")
        )
        for p in sample_paths
    ])

    pairs = random.sample(
        [
            (i, j)
            for i in range(len(images))
            for j in range(i + 1, len(images))
        ],
        min(
            500,
            len(images)*(len(images)-1)//2
        )
    )

    scores = []

    for i, j in tqdm(pairs, desc=f"LPIPS {DATASET_NAME}"):

        with torch.no_grad():

            d = lpips_model(
                images[i:i+1].to(DEVICE),
                images[j:j+1].to(DEVICE)
            )

        scores.append(d.item())

    lpips_mean = float(np.mean(scores))
    lpips_std = float(np.std(scores))

    print(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results = {
        "dataset": DATASET_NAME,
        "num_samples": NUM_SAMPLES,
        "fid": fid_score,
        "clip_score": clip_score,
        "blip_itm": blip_itm,
        "blip_cosine": blip_cos,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
    }

    out_path = os.path.join(
        OUT_DIR,
        "metrics.json"
    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    all_results.append(results)

    print(f"\nSaved results to: {out_path}")


# =========================================================
# SAVE GLOBAL SUMMARY
# =========================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "all_metrics.json"
)

with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=4)

print("\n✅ ALL DONE")
print(f"Summary saved to: {summary_path}")