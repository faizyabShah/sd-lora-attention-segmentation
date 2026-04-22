"""Script entrypoint to run captioning."""

from __future__ import annotations

import argparse

from src.captioning.florence import run_florence_captioning
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Florence caption generation over a dataset.")
    parser.add_argument("--images-dir", default="data/processed/images")
    parser.add_argument("--output-csv", default="data/captions/captions.csv")
    parser.add_argument("--model-id", default="microsoft/Florence-2-base")
    parser.add_argument("--prompt", default="<DETAILED_CAPTION>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = run_florence_captioning(
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        model_id=args.model_id,
        prompt=args.prompt,
    )
    print(f"Wrote {len(df)} captions to {args.output_csv}")


if __name__ == "__main__":
    main()
