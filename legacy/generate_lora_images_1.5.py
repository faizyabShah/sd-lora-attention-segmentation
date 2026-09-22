import os
import json
import argparse
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rank",
    type=int,
    required=True,
    help="LoRA rank"
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Training seed used for this LoRA"
)

args = parser.parse_args()

RANK = args.rank
TRAIN_SEED = args.seed

# =========================
# PATHS
# =========================

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

CAPTIONS_PATH = "./cityscapes_cropped/metadata.jsonl"

# New training output structure:
# outputs/rank_64/seed_3/
LORA_PATH = f"./outputs/rank_{RANK}/seed_{TRAIN_SEED}"

# Keep generated images separate from trained weights
OUTPUT_DIR = (
    f"./outputs/generated/"
    f"rank_{RANK}/seed_{TRAIN_SEED}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# GENERATION SEED SETTINGS
# =========================

# Old generation used seed = i.
# Now we start from a different seed range.
GENERATION_SEED_BASE = 10000

# =========================
# LOAD PROMPTS
# =========================

print("=== Loading captions ===")

with open(CAPTIONS_PATH, "r") as f:
    entries = [json.loads(line) for line in f]

prompts = [e["text"] for e in entries]

print(f"Loaded {len(prompts)} prompts")
print(f"First prompt: {prompts[0]}")

# =========================
# CHECK LoRA PATH
# =========================

print("\n=== Configuration ===")

print(f"Rank: {RANK}")
print(f"Training seed: {TRAIN_SEED}")
print(f"LoRA path: {LORA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Generation seed base: {GENERATION_SEED_BASE}")

if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA directory does not exist: {LORA_PATH}"
    )

# =========================
# LOAD PIPELINE
# =========================

print("\n=== Loading model ===")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None
).to("cuda")

pipe.enable_attention_slicing()

# =========================
# LOAD LoRA
# =========================

print("\n=== Loading LoRA ===")

pipe.load_lora_weights(LORA_PATH)

print("LoRA weights loaded")

# =========================
# GENERATION
# =========================

print("\n=== Generating images ===")

for i, prompt in enumerate(tqdm(prompts, desc="Generating")):

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{i:05d}.png"
    )

    # Skip already-generated images
    if os.path.exists(out_path):
        continue

    # Different progression from old seed=i setup
    generation_seed = GENERATION_SEED_BASE + i

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(generation_seed)

    image = pipe(
        prompt,
        generator=generator,
        num_inference_steps=30
    ).images[0]

    image.save(out_path)

# =========================
# SAVE MAPPING
# =========================

mapping = []

for i, entry in enumerate(entries):

    generation_seed = GENERATION_SEED_BASE + i

    mapping.append({
        "image": f"{i:05d}.png",
        "prompt": entry["text"],
        "original_file": (
            entry["original_file"]
            if "original_file" in entry
            else entry["file_name"]
        ),
        "rank": RANK,
        "training_seed": TRAIN_SEED,
        "generation_seed": generation_seed
    })

with open(
    os.path.join(OUTPUT_DIR, "prompt_mapping.json"),
    "w"
) as f:
    json.dump(mapping, f, indent=2)

print(
    f"\nSaved mapping ({len(mapping)} entries)"
)

print("\n=== Generation complete ===")
print(f"Rank: {RANK}")
print(f"Training seed: {TRAIN_SEED}")
print(f"Images saved to: {OUTPUT_DIR}")