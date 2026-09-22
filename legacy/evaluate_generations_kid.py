import os
import json
import shutil

from cleanfid import fid


# =========================================================
# PATHS
# =========================================================

BASE_DIR = "./outputs-sdxl"

OUTPUT_DIR = "./outputs-sdxl-metrics-kid"

REAL_DIR = "./cityscapes_cropped/images"

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
# FIND COMMON SAMPLE COUNT
# =========================================================

dataset_counts = {}

for name, path in DATASETS.items():

    imgs = load_images(path)

    dataset_counts[name] = len(imgs)

print("\n=== IMAGE COUNTS ===")

for k, v in dataset_counts.items():
    print(f"{k}: {v}")

NUM_SAMPLES = min(dataset_counts.values())

print(f"\nUsing first {NUM_SAMPLES} images from all datasets")


# =========================================================
# PREPARE ALIGNED REAL SUBSET
# =========================================================

REAL_SUBSET_DIR = os.path.join(
    OUTPUT_DIR,
    "real_subset"
)

os.makedirs(REAL_SUBSET_DIR, exist_ok=True)

print("\nPreparing aligned real subset...")

real_images = load_images(REAL_DIR)[:NUM_SAMPLES]

for i, src in enumerate(real_images):

    dst = os.path.join(
        REAL_SUBSET_DIR,
        f"{i:05d}.png"
    )

    if not os.path.exists(dst):
        shutil.copy(src, dst)

print(f"Prepared {len(os.listdir(REAL_SUBSET_DIR))} real images")


# =========================================================
# COMPUTE KID
# =========================================================

all_results = []

for DATASET_NAME, GEN_DIR in DATASETS.items():

    print("\n===================================")
    print(f"Processing: {DATASET_NAME}")
    print("===================================")

    kid_score = fid.compute_kid(
        GEN_DIR,
        REAL_SUBSET_DIR,
        mode="clean"
    )

    print(f"KID: {kid_score:.6f}")

    results = {
        "dataset": DATASET_NAME,
        "num_samples": NUM_SAMPLES,
        "kid": float(kid_score),
    }

    out_dir = os.path.join(
        OUTPUT_DIR,
        DATASET_NAME
    )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        "kid_metrics.json"
    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    all_results.append(results)

    print(f"Saved to: {out_path}")


# =========================================================
# SAVE GLOBAL SUMMARY
# =========================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "all_kid_metrics.json"
)

with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=4)

print("\n✅ DONE")
print(f"Summary saved to: {summary_path}")