"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_image_paths(folder: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Recursively collect image files from a folder."""
    root = Path(folder)
    ext = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext]
    return sorted(paths)


def load_images_from_folder(
    folder: str | Path,
    max_images: int | None = None,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Load image paths from folder with optional cap."""
    images = get_image_paths(folder, extensions=extensions)
    if max_images is None:
        return images
    return images[:max_images]


def load_captions_csv(captions_path: str | Path) -> pd.DataFrame:
    """Load captions CSV and validate basic schema."""
    df = pd.read_csv(captions_path)
    if "caption" not in df.columns:
        raise ValueError("Captions file must include a 'caption' column.")
    return df
