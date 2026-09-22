import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="./leftImg8bit")
parser.add_argument("--dst", type=str, default="./cityscapes_cropped")
parser.add_argument("--save_every", type=int, default=100)
args = parser.parse_args()

SRC_DIR = args.src
DST_DIR = args.dst
SAVE_EVERY = args.save_every

IMAGES_DIR = os.path.join(DST_DIR, "images")
META_PATH = os.path.join(DST_DIR, "metadata.jsonl")

os.makedirs(IMAGES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD FLORENCE-2
# =========================

print("=== Loading Florence-2 Large ===")

model_id = "microsoft/Florence-2-large"

def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code = True
)

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa"  # use PyTorch's built-in efficient attention instead
    ).to(DEVICE).eval()

model.eval()

# =========================
# RESUME SUPPORT
# =========================

existing_entries = []
existing_files = set()

if os.path.exists(META_PATH):
    print("Resuming from existing metadata...")
    with open(META_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            existing_entries.append(item)
            existing_files.add(item["original_file"])

start_idx = len(existing_entries)
print(f"Starting from index: {start_idx}")

# =========================
# GET ALL IMAGE PATHS
# =========================

def get_all_images(root):
    paths = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)
            for file in os.listdir(city_dir):
                if file.endswith(".png"):
                    paths.append(os.path.join(city_dir, file))
    return sorted(paths)

all_images = get_all_images(SRC_DIR)
print(f"Total images found: {len(all_images)}")

# =========================
# CAPTION FUNCTION
# =========================

def generate_caption(image):
    prompt = "<DETAILED_CAPTION>"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return caption.strip()

# =========================
# MAIN LOOP
# =========================

buffer = []
counter = start_idx

for path in tqdm(all_images, desc="Processing"):

    if path in existing_files:
        continue

    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size  # expected 2048x1024

        # =========================
        # CENTER CROP 1024x1024
        # =========================

        crop_size = 1024
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        img = img.crop((left, top, right, bottom))

        # =========================
        # SAVE IMAGE
        # =========================

        filename = f"{counter:06d}.jpg"
        out_path = os.path.join(IMAGES_DIR, filename)
        img.save(out_path)

        # =========================
        # CAPTION
        # =========================

        caption = generate_caption(img)

        entry = {
            "file_name": f"images/{filename}",
            "text": caption,
            "original_file": path
        }

        buffer.append(entry)
        counter += 1

        # =========================
        # PERIODIC SAVE
        # =========================

        if len(buffer) >= SAVE_EVERY:
            with open(META_PATH, "a") as f:
                for item in buffer:
                    f.write(json.dumps(item) + "\n")
            buffer = []

    except Exception as e:
        print(f"Error processing {path}: {e}")

# =========================
# FINAL SAVE
# =========================

if buffer:
    with open(META_PATH, "a") as f:
        for item in buffer:
            f.write(json.dumps(item) + "\n")

print(f"\nDone. Total images processed: {counter}")