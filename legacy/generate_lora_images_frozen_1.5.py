import os
import json
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline

# =========================
# PATHS
# =========================

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

CAPTIONS_PATH = "./cityscapes_cropped/metadata.jsonl"
OUTPUT_DIR = "./outputs/cityscapes-gen-frozen"

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
# LOAD SDXL (FROZEN)
# =========================

print("\n=== Loading SDXL model ===")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing()

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
    mapping.append({
        "image": f"{i:05d}.png",
        "prompt": entry["text"],
        "original_file": entry.get("original_file", entry["file_name"])
    })

with open(os.path.join(OUTPUT_DIR, "prompt_mapping.json"), "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\nSaved mapping ({len(mapping)} entries)")