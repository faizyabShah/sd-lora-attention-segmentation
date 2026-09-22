import os
import json
import shutil
import subprocess
import argparse

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

RANK = args.rank
SEED = args.seed

# =========================
# PATHS (EDIT THESE ONLY)
# =========================

# =========================
# PATHS
# =========================

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
WORK_DIR = "./cityscapes_cropped"

TRAIN_SCRIPT = "train_text_to_image_lora.py"
OUTPUT_DIR = f"outputs/rank_{RANK}/seed_{SEED}"
LOG_DIR = f"outputs/logs/rank_{RANK}/seed_{SEED}"


# Verify
with open(os.path.join(WORK_DIR, "metadata.jsonl"), "r") as f:
    first = json.loads(f.readline())

test_path = os.path.join(WORK_DIR, first["file_name"])
print("Sample check:")
print("Path:", test_path)
print("Exists:", os.path.exists(test_path))
print("Caption:", first["text"][:80])

# =========================
# STEP 2: DOWNLOAD TRAIN SCRIPT
# =========================

print("\n=== Downloading training script ===")

if not os.path.exists(TRAIN_SCRIPT):
    subprocess.run([
        "wget",
        "https://raw.githubusercontent.com/huggingface/diffusers/v0.36.0/examples/text_to_image/train_text_to_image_lora.py",
        "-O",
        TRAIN_SCRIPT
    ], check=True)

print("Training script ready")

# =========================
# STEP 3: PATCH SCRIPT
# =========================



# =========================
# STEP 4: TRAINING
# =========================

print(f"\n=== Starting training (rank={RANK}) ===")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

command = [
    "accelerate", "launch",

    TRAIN_SCRIPT,

    "--pretrained_model_name_or_path", BASE_MODEL,
    "--train_data_dir", WORK_DIR,
    "--caption_column", "text",

    "--resolution", "512",

    "--train_batch_size", "2",
    "--gradient_accumulation_steps", "2",

    "--max_train_steps", "5000",
    "--learning_rate", "1e-4",

    "--lr_scheduler", "cosine",
    "--lr_warmup_steps", "200",
    "--snr_gamma", "5.0",

    "--rank", str(RANK),

    "--output_dir", OUTPUT_DIR,
    "--checkpointing_steps", "400",

    "--seed", str(SEED),
    "--logging_dir", LOG_DIR,

    "--mixed_precision", "fp16",

    "--gradient_checkpointing",

    # 🔥 IMPORTANT FIX
    "--dataloader_num_workers", "0",

]

print("\nCommand:")
print(" ".join(command))

subprocess.run(command, check=True)

print("\n=== Training complete ===")