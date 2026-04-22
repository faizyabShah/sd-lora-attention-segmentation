"""Dataset formatting utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def format_record(record: dict) -> dict:
    """Format a single record for downstream training."""
    return {"file_name": record["file_name"], "text": record["text"]}


def create_metadata_jsonl(
    df: pd.DataFrame,
    image_column: str,
    caption_column: str,
    output_dir: str | Path,
    copy_images: bool = True,
) -> tuple[Path, Path]:
    """Write metadata.jsonl for LoRA training and optionally copy images."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            src_path = Path(row[image_column])
            dst_name = src_path.name
            dst_path = images_dir / dst_name
            if copy_images:
                shutil.copy2(src_path, dst_path)
            payload = format_record({"file_name": dst_name, "text": str(row[caption_column])})
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return metadata_path, images_dir
