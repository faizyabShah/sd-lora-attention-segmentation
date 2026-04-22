"""Script entrypoint to generate images."""

from __future__ import annotations

import argparse

from src.generation.generate import generate
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images using SD + LoRA.")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = generate(config_path=args.config)
    print(f"Generated {len(df)} images")


if __name__ == "__main__":
    main()
