"""LoRA training entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def build_train_command(config: dict) -> list[str]:
    """Build accelerate launch command from train config."""
    model = config["model"]
    training = config["training"]
    output = config["output"]

    return [
        "accelerate",
        "launch",
        training.get("script", "train_text_to_image_lora.py"),
        "--pretrained_model_name_or_path",
        model["base_model"],
        "--train_data_dir",
        training["train_data_dir"],
        "--output_dir",
        output["checkpoint_dir"],
        "--resolution",
        str(training.get("resolution", 512)),
        "--train_batch_size",
        str(training.get("batch_size", 4)),
        "--learning_rate",
        str(training.get("learning_rate", 1e-4)),
        "--max_train_steps",
        str(training.get("max_train_steps", 2000)),
        "--checkpointing_steps",
        str(training.get("checkpointing_steps", 500)),
        "--mixed_precision",
        str(training.get("mixed_precision", "fp16")),
        "--report_to",
        str(training.get("report_to", "tensorboard")),
    ]


def train(config_path: str | Path, dry_run: bool = False) -> list[str]:
    """Train a LoRA adapter using the configured accelerate workflow."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    command = build_train_command(config)
    if not dry_run:
        subprocess.run(command, check=True)
    return command


def main() -> None:
    """CLI entrypoint for LoRA training."""
    parser = argparse.ArgumentParser(description="Launch LoRA training using accelerate.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training yaml config.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    args = parser.parse_args()

    command = train(config_path=args.config, dry_run=args.dry_run)
    print(" ".join(command))


if __name__ == "__main__":
    main()
