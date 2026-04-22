"""Script entrypoint to prepare dataset artifacts."""

from __future__ import annotations

import argparse

from transformers import CLIPTokenizer

from src.dataset.filtering import filter_by_token_length
from src.dataset.formatter import create_metadata_jsonl
from src.dataset.loader import load_captions_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter captions and format metadata for LoRA training.")
    parser.add_argument("--captions-csv", default="data/captions/captions.csv")
    parser.add_argument("--output-dir", default="data/processed/lora_dataset")
    parser.add_argument("--max-tokens", type=int, default=77)
    args = parser.parse_args()

    df = load_captions_csv(args.captions_csv)
    if "image_path" not in df.columns:
        raise ValueError("captions.csv must contain an 'image_path' column.")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    filtered = filter_by_token_length(
        df,
        tokenizer=lambda text: len(tokenizer.encode(text, add_special_tokens=True)),
        max_tokens=args.max_tokens,
    )
    metadata_path, images_dir = create_metadata_jsonl(
        filtered,
        image_column="image_path",
        caption_column="caption",
        output_dir=args.output_dir,
        copy_images=True,
    )
    print(f"Filtered {len(df)} -> {len(filtered)} rows")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Copied images to: {images_dir}")


if __name__ == "__main__":
    main()
