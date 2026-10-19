"""Rendering utilities for the qualitative SG audit.

The experiment runners call this module only in ``--visual-dir`` mode.  It
keeps the numerical implementations in the original runners and standardizes
how their CA seeds, propagated SA scores, and final masks are displayed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def normalize_map(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = array.astype(np.float32, copy=False)
    low, high = float(np.nanmin(array)), float(np.nanmax(array))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def resize_map(value: torch.Tensor | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = normalize_map(value)
    if array.ndim == 1:
        side = math.isqrt(array.size)
        if side * side != array.size:
            raise ValueError(f"Cannot reshape non-square attention vector of length {array.size}")
        array = array.reshape(side, side)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D map or flat square vector, got shape {array.shape}")
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def best_thresholds(result_dir: Path, fallback: float = 0.45) -> dict[str, float]:
    """Recover each cell's full-VOC threshold directly from stored counts."""
    files = sorted((result_dir / "voc" / "shards").glob("*.npz"))
    files = [path for path in files if ".diagnostics." not in path.name]
    if not files:
        summary = result_dir / "summary.json"
        if not summary.is_file():
            return {}
        cells = json.loads(summary.read_text(encoding="utf-8")).get("cells", {})
        output = {}
        for name, values in cells.items():
            for key in ("voc_threshold", "voc_full_threshold"):
                if key in values:
                    output[name] = float(values[key])
                    break
        return output
    names: tuple[str, ...] | None = None
    thresholds: np.ndarray | None = None
    totals: np.ndarray | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            current_names = tuple(str(v) for v in data["configurations"].tolist())
            current_thresholds = np.asarray(data["thresholds"], dtype=np.float64)
            if names is None:
                names, thresholds = current_names, current_thresholds
                totals = np.asarray(data["counts"], dtype=np.int64).sum(axis=1)
            else:
                if current_names != names or not np.allclose(current_thresholds, thresholds):
                    raise RuntimeError(f"Incompatible shard settings in {path}")
                totals += np.asarray(data["counts"], dtype=np.int64).sum(axis=1)
    assert names is not None and thresholds is not None and totals is not None
    union = totals[:, :, :3].sum(axis=-1)
    iou = np.divide(totals[:, :, 0], union, out=np.zeros_like(union, dtype=np.float64), where=union > 0)
    return {name: float(thresholds[int(np.argmax(iou[i]))]) for i, name in enumerate(names)}


def released_image(row: Mapping[str, str], manifest_dir: Path) -> np.ndarray:
    path = manifest_dir / row["image"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"))


def _overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    # Compact magma-like ramp without requiring Matplotlib in the GPU env.
    x = np.clip(heat, 0, 1)
    knots = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float32)
    colors = np.asarray(((0, 0, 4), (81, 18, 124), (183, 55, 121),
                         (252, 137, 97), (252, 253, 191)), dtype=np.float32)
    colored = np.stack([np.interp(x, knots, colors[:, channel]) for channel in range(3)], axis=-1).astype(np.uint8)
    return np.clip((1.0 - alpha) * image.astype(np.float32) + alpha * colored, 0, 255).astype(np.uint8)


def _mask_rgb(mask: np.ndarray, truth: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = (240, 240, 240)
    edge = truth ^ ndimage.binary_erosion(truth)
    canvas[edge] = (0, 220, 255)
    return canvas


def _iou(mask: np.ndarray, truth: np.ndarray) -> float:
    union = np.logical_or(mask, truth).sum()
    return float(np.logical_and(mask, truth).sum() / union) if union else 1.0


def render_variant_pages(
    output_dir: Path,
    family: str,
    classname: str,
    image_id: str,
    original: np.ndarray,
    truth: np.ndarray,
    names: Sequence[str],
    ca_maps: Sequence[torch.Tensor | np.ndarray],
    score_maps: Sequence[torch.Tensor | np.ndarray],
    thresholds: Mapping[str, float],
    mask_overrides: Mapping[str, np.ndarray] | None = None,
    rows_per_page: int = 6,
) -> list[dict[str, object]]:
    """Write labeled four-column pages and a machine-readable sample index."""
    from PIL import ImageDraw, ImageFont

    if not (len(names) == len(ca_maps) == len(score_maps)):
        raise ValueError("names, CA maps and score maps must have identical length")
    output_dir = Path(output_dir) / family / f"{classname.replace(' ', '_')}__{image_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(v) for v in truth.shape)
    original = np.asarray(Image.fromarray(np.asarray(original, dtype=np.uint8)).resize((shape[1], shape[0]), Image.Resampling.LANCZOS))
    records: list[dict[str, object]] = []
    prepared = []
    for name, ca, score in zip(names, ca_maps, score_maps):
        ca_full, score_full = resize_map(ca, shape), resize_map(score, shape)
        threshold = float(thresholds.get(name, 0.45))
        mask = np.asarray(mask_overrides[name], dtype=bool) if mask_overrides and name in mask_overrides else score_full >= threshold
        sample_iou = _iou(mask, truth)
        prepared.append((name, ca_full, score_full, mask, threshold, sample_iou))

    pages = math.ceil(len(prepared) / rows_per_page)
    tile = 256
    title_height, variant_height = 28, 28
    row_height = variant_height + title_height + tile + 12
    page_width = 4 * tile
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = label_font = ImageFont.load_default()
    for page_index in range(pages):
        rows = prepared[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        header = 36
        canvas = Image.new("RGB", (page_width, header + row_height * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{family} — {classname} — {image_id} — page {page_index + 1}/{pages}",
                  fill="black", font=label_font)
        for row_index, (name, ca, score, mask, threshold, sample_iou) in enumerate(rows):
            panels = (
                ("Released VOC-Sim image", original),
                ("Cross-attention seed overlay", _overlay(original, ca)),
                (("No SA: CA seed repeated" if name.endswith("/seed") else "Self-attention result overlay"),
                 _overlay(original, score)),
                (f"Predicted map  t={threshold:.3f}  IoU={sample_iou:.3f}", _mask_rgb(mask, truth)),
            )
            y = header + row_index * row_height
            draw.rectangle((0, y, page_width, y + variant_height), fill=(235, 235, 235))
            draw.text((8, y + 5), name, fill="black", font=label_font)
            for col, (title, panel) in enumerate(panels):
                x = col * tile
                draw.text((x + 5, y + variant_height + 5), title, fill="black", font=title_font)
                image = Image.fromarray(panel).resize((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y + variant_height + title_height))
            records.append({
                "family": family, "class": classname, "image_id": image_id,
                "variant": name, "threshold": threshold, "sample_iou": sample_iou,
                "page": f"page_{page_index + 1:02d}.jpg", "row": row_index + 1,
            })
        canvas.save(output_dir / f"page_{page_index + 1:02d}.jpg", quality=91, optimize=True)
    (output_dir / "index.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records
