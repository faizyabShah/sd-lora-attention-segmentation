"""I/O helpers."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def extract_zip(zip_path: str | Path, output_dir: str | Path) -> None:
    """Extract a ZIP file to an output directory."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def download_and_extract_zip(url: str, output_dir: str | Path, filename: str = "dataset.zip") -> Path:
    """Download a zip from Google Drive with gdown and extract it.

    This mirrors notebook behavior while keeping it reusable in scripts.
    """
    output_dir = ensure_dir(output_dir)
    zip_path = output_dir / filename

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("gdown is required for Google Drive downloads.") from exc

    gdown.download(url, str(zip_path), quiet=False)
    extract_zip(zip_path, output_dir)
    return zip_path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copy a file and return destination path."""
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    shutil.copy2(src, dst_path)
    return dst_path


def resolve_existing_file(path: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    """Resolve file path and validate existence."""
    file_path = Path(path)
    if not file_path.is_absolute() and base_dir is not None:
        file_path = Path(base_dir) / file_path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path
