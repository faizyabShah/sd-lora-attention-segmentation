#!/usr/bin/env python3
"""Fetch the two required models; omit duplicate checkpoint formats and optional models."""
from pathlib import Path
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "outputs/setup/huggingface"))

from huggingface_hub import snapshot_download


def fetch(model, patterns):
    # The repository is on NFS, where flock reports ENOLCK on this machine.
    # Download with locks on local storage, then copy the completed snapshot.
    staging = Path(os.environ.get(
        "BRACE_DOWNLOAD_CACHE", str(Path(tempfile.gettempdir()) / f"brace-hf-{os.getuid()}")
    ))
    snapshot = Path(snapshot_download(model, allow_patterns=patterns,
                                     cache_dir=staging / "hub", max_workers=2))
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")))
    destination = cache / ("models--" + model.replace("/", "--"))
    shutil.copytree(snapshot, destination / "snapshots" / snapshot.name,
                    symlinks=False, dirs_exist_ok=True)
    (destination / "refs").mkdir(parents=True, exist_ok=True)
    (destination / "refs/main").write_text(snapshot.name)
    print(f"Cached {model} at {destination} (revision {snapshot.name})", flush=True)


if __name__ == "__main__":
    fetch(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        [
            "model_index.json", "scheduler/*.json", "tokenizer/*",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            "feature_extractor/*.json", "safety_checker/config.json",
            "safety_checker/model.safetensors",
        ],
    )
    fetch(
        "openai/clip-vit-base-patch16",
        ["*.json", "*.txt", "pytorch_model.bin"],
    )
