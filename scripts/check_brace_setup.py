#!/usr/bin/env python3
"""Read-only local readiness check; does not fetch weights or initialize CUDA models."""
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "outputs/setup/huggingface"))


def main():
    print(f"Repository: {ROOT}\nPython: {sys.version.split()[0]} ({sys.executable})")
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 required for the handoff environment")
    for requirement in (ROOT / "requirements.txt").read_text().splitlines():
        requirement = requirement.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==", 1)
        if name == "bitsandbytes":
            print("bitsandbytes: optional legacy 8-bit training extra; excluded from basic BRACE setup")
            continue
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        print(f"{name}: {actual} (expected {expected})")
        if actual.split("+", 1)[0] != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    # Imports run in a child so a binary-extension crash cannot hide other checks.
    check = subprocess.run([
        sys.executable, "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import torch, torchvision, transformers, denseCRF, peft; "
        "from diffusers import StableDiffusionPipeline; "
        "print('Core imports: OK; torch:', torch.__version__, 'CUDA build:', torch.version.cuda, "
        "'CUDA available:', torch.cuda.is_available()); "
        "assert torch.version.cuda == '11.8', 'Expected the CUDA 11.8 PyTorch build'",
        str(ROOT / "outputs/dependencies/simplecrf_py310"),
    ], cwd=ROOT, capture_output=True, text=True)
    print(check.stdout.strip())
    if check.returncode:
        failures.append("Core imports failed: " + check.stderr.strip())
    expected_paths = [
        "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv",
        "datasets/ovam_experiment_with_dataset/evaluation/coco_captions_sampled.csv",
        "datasets/ovam_experiment_with_dataset/evaluation/ovam/coco_captions/annotations",
        "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages",
        "datasets/voc2012/VOCdevkit/VOC2012/SegmentationClass",
        "datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
    ]
    for name in expected_paths:
        present = (ROOT / name).exists()
        print(f"{'OK' if present else 'MISSING'}: {name}")
        if not present:
            failures.append(name)
    manifest = ROOT / "outputs/segmentation/sg_real_atlas/manifest.json"
    data = json.loads(manifest.read_text())
    print("Real-photo manifest:", ", ".join(f"{k}: {len(data[k])}" for k in ("D", "F", "S") if k in data))
    try:
        from huggingface_hub import snapshot_download
        required_models = {
            "stable-diffusion-v1-5/stable-diffusion-v1-5": [
                "model_index.json", "scheduler/scheduler_config.json",
                "tokenizer/vocab.json", "tokenizer/merges.txt",
                "text_encoder/config.json", "text_encoder/model.safetensors",
                "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
                "feature_extractor/preprocessor_config.json",
                "safety_checker/config.json", "safety_checker/model.safetensors",
            ],
            "openai/clip-vit-base-patch16": ["config.json", "pytorch_model.bin",
                "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        }
        for model, required in required_models.items():
            try:
                path = snapshot_download(model, local_files_only=True)
                missing = [name for name in required if not (Path(path) / name).is_file()]
                print(f"Model cache {model}: {path} ({len(missing)} missing required files)")
                if missing:
                    failures.append(f"Model {model}: missing {missing}")
            except Exception as exc:
                failures.append(f"Model cache {model}: {exc}")
    except ImportError:
        failures.append("huggingface_hub unavailable; model cache not checked")
    for failure in failures:
        print("NEEDS ATTENTION:", failure)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
