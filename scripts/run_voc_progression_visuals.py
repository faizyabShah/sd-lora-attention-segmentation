#!/usr/bin/env python3
"""Create balanced VOC-Sim panels showing the path from OVAM to our method."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_capture.live import generate_attention_sample
from src.segmentation.attention_aggregation import normalize_map
from src.segmentation.best_method import binary_iou, display_score, frozen_best_stages
from src.segmentation.ovam_full import ovam_full_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/voc_sim_progression_25"),
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--fixed-stage1-results",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"),
    )
    parser.add_argument(
        "--densecrf-site-packages",
        type=Path,
        default=Path("/home/shafa/miniconda3/envs/ovam_sdxl/lib/python3.10/site-packages"),
    )
    return parser.parse_args()


def select_rows(manifest: Path, count: int) -> list[dict[str, str]]:
    """Select one median example per class, then deterministic spread extras."""
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["annotated"].strip().lower() == "true"
        ]
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["classname"], []).append(row)
    if count < len(by_class):
        raise ValueError(f"count must be at least the {len(by_class)} classes")
    selected = [by_class[name][len(by_class[name]) // 2] for name in sorted(by_class)]
    selected_ids = {row["image_id"] for row in selected}
    remaining = sorted(
        (row for row in rows if row["image_id"] not in selected_ids),
        key=lambda row: row["image_id"],
    )
    needed = min(count, len(rows)) - len(selected)
    if needed:
        indices = np.linspace(0, len(remaining) - 1, needed).round().astype(int)
        selected.extend(remaining[index] for index in indices)
    return selected


def heatmap(value: torch.Tensor, size: int = 320) -> Image.Image:
    array = value.detach().float().cpu().clamp(0, 1).numpy()
    red = np.clip(2.0 * array, 0, 1)
    blue = np.clip(2.0 * (1.0 - array), 0, 1)
    green = np.clip(1.0 - np.abs(2.0 * array - 1.0), 0, 1) * 0.8
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def overlay(image: Image.Image, attention: Image.Image) -> Image.Image:
    return Image.blend(image.convert("RGB"), attention.convert("RGB"), 0.52)


def ground_truth_view(image: Image.Image, target: np.ndarray) -> Image.Image:
    mask = Image.fromarray((target.astype(np.uint8) * 150)).resize(
        image.size, Image.Resampling.NEAREST
    )
    return Image.composite(Image.new("RGB", image.size, (0, 220, 80)), image, mask)


def save_panel(path: Path, tiles: list[tuple[str, Image.Image]], columns: int = 3) -> None:
    tile_size = tiles[0][1].width
    title_height = 50
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + title_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, tile) in enumerate(tiles):
        x = index % columns * tile_size
        y = index // columns * (tile_size + title_height)
        wrapped = "\n".join(textwrap.wrap(label, width=38)[:2])
        draw.multiline_text((x + 7, y + 6), wrapped, fill="black", spacing=2)
        canvas.paste(tile.convert("RGB"), (x, y + title_height))
    temporary = path.with_suffix(".tmp.png")
    canvas.save(temporary, optimize=True)
    os.replace(temporary, path)


def captioned_image(tile: Image.Image, label: str) -> Image.Image:
    """Add a permanent label and IoU strip to one exported method image."""
    title_height = 50
    output = Image.new("RGB", (tile.width, tile.height + title_height), "white")
    draw = ImageDraw.Draw(output)
    wrapped = "\n".join(textwrap.wrap(label, width=38)[:2])
    draw.multiline_text((7, 6), wrapped, fill="black", spacing=2)
    output.paste(tile.convert("RGB"), (0, title_height))
    return output


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = select_rows(args.manifest, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    selection = [
        {
            "image_id": row["image_id"],
            "classname": row["classname"],
            "seed": int(row["seed"]),
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Existing selection differs from this invocation")
    else:
        atomic_json(selection_path, selection)

    records_path = args.output_dir / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    completed = {record["image_id"] for record in records}
    rankings = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"][
        "head_rankings"
    ]
    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    for index, row in enumerate(rows, start=1):
        if row["image_id"] in completed:
            continue
        print(f"[{index}/{len(rows)}] {row['image_id']} ({row['classname']})", flush=True)
        sample = generate_attention_sample(pipe, row, args.manifest.parent, args.storage_dtype)
        try:
            _, _, ovam_score, ovam_prediction = ovam_full_prediction(
                sample,
                args.densecrf_site_packages,
                self_attention_minimum=0.85,
                threshold=0.40,
            )
            stages = frozen_best_stages(sample, pipe.tokenizer, rankings, args.device)
            target = sample.load_target_mask()
            released = Image.open(args.manifest.parent / row["image"]).convert("RGB").resize(
                (320, 320), Image.Resampling.LANCZOS
            )
            sample_dir = args.output_dir / f"{index:02d}_{safe_name(row['image_id'])}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            released.save(sample_dir / "original.png", optimize=True)
            gt = ground_truth_view(released, target)
            gt.save(sample_dir / "ground_truth_overlay.png", optimize=True)

            methods: list[dict] = []
            ovam_iou = binary_iou(ovam_prediction, target)
            ovam_display = normalize_map(ovam_score.float(), "max").cpu()
            method_values = [("ovam", "OVAM", ovam_display, ovam_prediction, ovam_iou)]
            method_values.extend(
                (
                    stage.key,
                    stage.label,
                    display_score(stage.score, target.shape),
                    stage.prediction,
                    binary_iou(stage.prediction, target),
                )
                for stage in stages
            )
            attention_tiles = [("Released image", released), ("Ground truth", gt)]
            overlay_tiles = [("Released image", released), ("Ground truth", gt)]
            for order, (key, label, score, prediction, iou) in enumerate(method_values, start=1):
                attention = heatmap(score)
                mixed = overlay(released, attention)
                prefix = f"{order:02d}_{key}"
                attention_path = sample_dir / f"{prefix}_attention.png"
                overlay_path = sample_dir / f"{prefix}_overlay.png"
                title = f"{label} | IoU {iou:.3f}"
                captioned_image(attention, title).save(attention_path, optimize=True)
                captioned_image(mixed, title).save(overlay_path, optimize=True)
                attention_tiles.append((title, attention))
                overlay_tiles.append((title, mixed))
                methods.append(
                    {
                        "key": key,
                        "label": label,
                        "iou": iou,
                        "attention": str(attention_path.relative_to(args.output_dir)),
                        "overlay": str(overlay_path.relative_to(args.output_dir)),
                    }
                )
            save_panel(sample_dir / "journey_attention_maps.png", attention_tiles)
            save_panel(sample_dir / "journey_overlays.png", overlay_tiles)
            record = {
                "image_id": row["image_id"],
                "classname": row["classname"],
                "seed": int(row["seed"]),
                "prompt": row["prompt"],
                "source_image": row["image"],
                "source_annotation": row["annotation"],
                "methods": methods,
                "attention_panel": str(
                    (sample_dir / "journey_attention_maps.png").relative_to(args.output_dir)
                ),
                "overlay_panel": str(
                    (sample_dir / "journey_overlays.png").relative_to(args.output_dir)
                ),
            }
            records.append(record)
            completed.add(row["image_id"])
            atomic_json(records_path, records)
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    lines = [
        "# VOC-Sim progression examples",
        "",
        f"This directory contains **{len(records)} deterministic, class-balanced examples**: one median-index annotated sample from each of the 20 classes, followed by evenly spaced extras to reach {len(rows)}.",
        "",
        "Each sample directory has two complete journey panels and separate files for every method. `journey_attention_maps.png` shows standalone attention scores; `journey_overlays.png` places the same scores over the released VOC-Sim image. Every method tile states its foreground IoU against the released annotation.",
        "",
        "OVAM is the paper-correct VOC-Sim baseline (SA floor 0.85, threshold 0.40, DenseCRF). The remaining columns cumulatively construct our frozen best method: all CA layers, selected CA layers, selected CA heads, final 64×64 SA propagation, four propagation passes, late denoising calls, selected SA heads, and component cleanup.",
        "",
        "The displayed base is the released VOC-Sim image paired with the manual mask. Attention is regenerated live from the released prompt and seed, then discarded after each panel; no Q/K/V tensors are saved.",
        "",
        "| # | Class | Seed | OVAM IoU | Final IoU | Attention panel | Overlay panel |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for index, record in enumerate(records, start=1):
        values = {method["key"]: method["iou"] for method in record["methods"]}
        lines.append(
            f"| {index} | {record['classname']} | {record['seed']} | {values['ovam']:.3f} | "
            f"{values['final_cleanup']:.3f} | [{record['image_id']}]({record['attention_panel']}) | "
            f"[overlay]({record['overlay_panel']}) |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Created {len(records)} progression examples in {args.output_dir}")


if __name__ == "__main__":
    main()

