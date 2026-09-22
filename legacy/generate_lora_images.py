import os
import json
import argparse
from tqdm import tqdm

import torch
from diffusers import StableDiffusionXLPipeline

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

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

CAPTIONS_PATH = "./cityscapes_cropped/metadata.jsonl"
LORA_PATH = f"./outputs-sdxl/cityscapes-lora-r{RANK}"
OUTPUT_DIR = f"./outputs-sdxl/cityscapes-gen-lora-r{RANK}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# LOAD PIPELINE
# =========================

print("\n=== Loading SDXL model ===")

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 🔥 memory optimizations (important for SDXL)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# =========================
# LOAD LoRA
# =========================

pipe.load_lora_weights(LORA_PATH)

print("LoRA weights loaded")

# =========================
# GENERATION
# =========================

print("\n=== Generating images ===")

for i, prompt in enumerate(tqdm(prompts, desc="Generating")):

    out_path = os.path.join(OUTPUT_DIR, f"{i:05d}.png")

    if os.path.exists(out_path):
        continue

    generator = torch.Generator("cuda").manual_seed(i)

    image = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=30,
        guidance_scale=7.5   # SDXL default-ish
    ).images[0]

    image.save(out_path)

# =========================
# SAVE MAPPING
# =========================

mapping = []

for i, entry in enumerate(entries):
    mapping.append({
        "image": f"{i:05d}.png",
        "prompt": entry["text"],
        "original_file": entry["original_file"] if "original_file" in entry else entry["file_name"]
    })

with open(os.path.join(OUTPUT_DIR, "prompt_mapping.json"), "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\nSaved mapping ({len(mapping)} entries)")