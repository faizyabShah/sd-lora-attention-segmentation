"""Prepare data and launch a LoRA fine-tune of SD 1.5 on the Cityscapes crops.

This is a launcher, not a training loop: the actual training is HuggingFace's
examples/text_to_image/train_text_to_image_lora.py, vendored into scripts/ and
patched once. Running it here (rather than from the .slurm file) means dataset
assembly happens exactly once, before accelerate forks the worker processes.

What it does:
  1. vendors train_text_to_image_lora.py matching the installed diffusers, and
     patches it to cast LoRA params to fp32 after unet.add_adapter (the
     mixed-precision fix from the Kaggle notebook);
  2. assembles an imagefolder dataset dir -- one symlink per split into
     data/cityscapes_cropped/ plus a copy of metadata.jsonl, so the file_name
     fields (<split>/<city>/<file>.png) resolve;
  3. execs `accelerate launch --multi_gpu` on the vendored script.

RESOLUTION NOTE (verified against the vendored source, not assumed): diffusers
has no resize-only path. Its transform is always

    Resize(resolution) -> CenterCrop(resolution) if --center_crop else RandomCrop(resolution)

Resize(768) with an int arg matches the *smaller edge*, so a square 1024x1024
crop becomes exactly 768x768 and the following crop is a no-op either way. Both
branches were checked to be bit-identical to a plain resize on this data, so no
second crop happens and no edges are lost. 01_crop.py already took the central
square offline, which is the crop the Kaggle notebook was doing inline via
--center_crop on 2048x1024 originals.

--center_crop is passed by default anyway (harmless here, deterministic if a
non-square image ever slips in); --no-center-crop turns it off.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import diffusers

# ---------------------------------------------------------------- constants --
# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
SCRIPTS_DIR = Path(__file__).resolve().parent

IMAGES_ROOT = PROJECT_ROOT / "data" / "cityscapes_cropped"
CAPTIONS_FILE = PROJECT_ROOT / "data" / "florence2_detailedcaptions" / "metadata.jsonl"
DATASET_DIR = PROJECT_ROOT / "data" / "lora_train_dataset"

VENDORED_SCRIPT = SCRIPTS_DIR / "train_text_to_image_lora.py"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    "v{version}/examples/text_to_image/train_text_to_image_lora.py"
)

SPLITS = ("train", "val", "test", "train_extra")
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PATCH_ANCHOR = "unet.add_adapter(unet_lora_config)"
PATCH_MARKER = "param.data.to(torch.float32)"
PATCH_BODY = """    # Fix: cast LoRA params to float32 for mixed precision training
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
"""


def vendor_and_patch() -> Path:
    """Ensure a patched copy of the diffusers LoRA script exists in scripts/."""
    if VENDORED_SCRIPT.exists():
        text = VENDORED_SCRIPT.read_text()
        if PATCH_MARKER in text:
            print(f"using existing patched script: {VENDORED_SCRIPT}")
            return VENDORED_SCRIPT
        print(f"found unpatched {VENDORED_SCRIPT.name}, applying fp32 patch")
    else:
        url = UPSTREAM_URL.format(version=diffusers.__version__)
        print(f"downloading {url}")
        subprocess.run(["wget", "-q", url, "-O", str(VENDORED_SCRIPT)], check=True)
        text = VENDORED_SCRIPT.read_text()

    if PATCH_MARKER not in text:
        if PATCH_ANCHOR not in text:
            print(f"ERROR: patch anchor {PATCH_ANCHOR!r} not found -- upstream script "
                  "changed; patch by hand.", file=sys.stderr)
            sys.exit(1)
        out = []
        for line in text.splitlines(keepends=True):
            out.append(line)
            if PATCH_ANCHOR in line:
                out.append(PATCH_BODY)
        VENDORED_SCRIPT.write_text("".join(out))
        print("fp32 patch applied")

    return VENDORED_SCRIPT


def prepare_dataset() -> Path:
    """Build the imagefolder tree the diffusers loader expects.

    The diffusers script globs `<train_data_dir>/**`, and HuggingFace's resolver
    does NOT traverse symlinked *directories* -- linking whole splits yields a
    silently empty dataset. It does follow symlinked *files*, so we mirror the
    <split>/<city>/ directories for real and symlink each PNG individually.
    No image is copied; only inodes are spent.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any previous whole-split directory symlinks (the broken layout).
    for split in SPLITS:
        stale = DATASET_DIR / split
        if stale.is_symlink():
            stale.unlink()
            print(f"  removed stale directory symlink: {split}")

    linked = existing = 0
    for split in SPLITS:
        src_split = IMAGES_ROOT / split
        if not src_split.is_dir():
            print(f"  WARNING: split missing, skipping: {src_split}")
            continue
        for src in sorted(src_split.rglob("*.png")):
            dst = DATASET_DIR / src.relative_to(IMAGES_ROOT)
            if dst.is_symlink() or dst.exists():
                existing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            linked += 1
    print(f"  symlinked {linked} images ({existing} already present)")

    shutil.copy(CAPTIONS_FILE, DATASET_DIR / "metadata.jsonl")
    with (DATASET_DIR / "metadata.jsonl").open(encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"  metadata.jsonl: {total} image-caption pairs")

    # Verify through the very resolver the training script uses, so a layout
    # the loader cannot see fails here instead of training on nothing.
    from datasets.data_files import resolve_pattern

    found = sum(
        1 for f in resolve_pattern(f"{DATASET_DIR}/**", base_path="")
        if f.endswith(".png")
    )
    print(f"  loader sees {found} images")
    if found < total:
        print(f"ERROR: loader resolves only {found} of {total} images -- "
              "dataset layout is wrong, aborting before training.", file=sys.stderr)
        sys.exit(1)

    return DATASET_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, required=True, help="LoRA rank")
    ap.add_argument("--max-train-steps", type=int, required=True)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lr-warmup-steps", type=int, default=1200)
    ap.add_argument("--checkpointing-steps", type=int, default=3000)
    ap.add_argument("--num-processes", type=int, default=2)
    # 4 of the 8 allocated CPUs handle the PNG decode + 1024->768 resize in
    # parallel; at 0 that work serializes inside the training process.
    ap.add_argument("--dataloader-num-workers", type=int, default=4,
                    help="image loading/resize workers; 0 loads in-process")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--logging-dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    # On square 1024 crops this is bit-identical to omitting it, because
    # Resize(768) already produces exactly 768x768. Kept on by default as a
    # guard: if a non-square image ever entered the set, CenterCrop stays
    # deterministic where RandomCrop would silently vary between ranks.
    ap.add_argument("--center-crop", action=argparse.BooleanOptionalAction,
                    default=True, help="pass --center_crop to diffusers")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare everything and print the command, do not train")
    args = ap.parse_args()

    if not IMAGES_ROOT.is_dir():
        print(f"ERROR: crops not found: {IMAGES_ROOT}", file=sys.stderr)
        return 1
    if not CAPTIONS_FILE.is_file():
        print(f"ERROR: captions not found: {CAPTIONS_FILE}", file=sys.stderr)
        return 1

    # Both must be absolute: diffusers does Path(output_dir, logging_dir), so a
    # relative logging dir ends up nested inside the weights dir.
    output_dir = (args.output_dir or PROJECT_ROOT / "output" / "lora_weights"
                  / f"rank{args.rank}").resolve()
    logging_dir = (args.logging_dir or PROJECT_ROOT / "output" / "logs"
                   / f"tb_rank{args.rank}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== rank {args.rank}, {args.max_train_steps} steps, "
          f"{args.resolution}x{args.resolution} ===")
    script = vendor_and_patch()
    print("assembling dataset:")
    dataset_dir = prepare_dataset()

    cmd = [
        "accelerate", "launch",
        "--multi_gpu", f"--num_processes={args.num_processes}",
        "--mixed_precision=fp16",
        str(script),
        f"--pretrained_model_name_or_path={MODEL_ID}",
        f"--train_data_dir={dataset_dir}",
        "--caption_column=text",
        f"--resolution={args.resolution}",
        f"--dataloader_num_workers={args.dataloader_num_workers}",
        f"--train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--max_train_steps={args.max_train_steps}",
        f"--learning_rate={args.learning_rate}",
        "--max_grad_norm=1.0",
        "--lr_scheduler=cosine",
        f"--lr_warmup_steps={args.lr_warmup_steps}",
        "--snr_gamma=5.0",
        f"--rank={args.rank}",
        f"--output_dir={output_dir}",
        f"--checkpointing_steps={args.checkpointing_steps}",
        f"--seed={args.seed}",
        f"--logging_dir={logging_dir}",
        "--report_to=tensorboard",
        # Safe on a fresh run too: diffusers warns and starts from scratch when
        # no checkpoint exists, so the same command works for resubmits.
        "--resume_from_checkpoint=latest",
    ]
    if args.center_crop:
        cmd.append("--center_crop")

    print("\n" + " \\\n  ".join(cmd) + "\n", flush=True)
    if args.dry_run:
        print("--dry-run: not launching")
        return 0

    # exec so accelerate owns the process group and SLURM signals reach it.
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
