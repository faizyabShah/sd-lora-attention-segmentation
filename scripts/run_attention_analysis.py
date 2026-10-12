"""Collect compact explanatory evidence for the full VOC-Sim experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_streaming_voc_ablations import read_rows, save_checkpoint
from src.attention_capture.live import generate_attention_sample
from src.segmentation.attention_analysis import analyze_attention_sample
from src.segmentation.ovam_full import ovam_full_prediction
from src.segmentation.ovam_reference import load_released_ovam_components


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
        "--output-dir", type=Path, default=Path("outputs/analysis/voc_sim_attention_report")
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--panels-per-class", type=int, default=3)
    parser.add_argument(
        "--densecrf-site-packages",
        type=Path,
        default=Path("/home/shafa/miniconda3/envs/ovam_sdxl/lib/python3.10/site-packages"),
    )
    parser.add_argument(
        "--fixed-stage1-results",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"),
    )
    return parser.parse_args()


def _heatmap(value: torch.Tensor, size: int = 256) -> Image.Image:
    array = value.detach().float().cpu().clamp(0, 1).numpy()
    red = np.clip(2 * array, 0, 1)
    blue = np.clip(2 * (1 - array), 0, 1)
    green = np.clip(1 - np.abs(2 * array - 1), 0, 1) * 0.75
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)


def _overlay(image: Image.Image, heatmap: Image.Image, alpha: float = 0.52) -> Image.Image:
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)


def _mask_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 150)).resize(base.size, Image.Resampling.NEAREST)
    layer = Image.new("RGB", base.size, color)
    return Image.composite(layer, base, mask_image)


def save_visualizations(
    path: Path,
    original_path: Path,
    annotation_path: Path,
    display: dict[str, torch.Tensor],
) -> dict[str, dict[str, str]]:
    size = 256
    image = Image.open(original_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    annotation = np.any(np.asarray(Image.open(annotation_path).convert("RGB")) != 0, axis=-1)
    methods = [
        ("ovam_implemented", "Our full OVAM reproduction", display["ovam"]),
        ("top4_ca_seed", "Top-4 CA seed", display["top4_ca"]),
        ("sa_before_joint", "SA before joint tuning", display["focused"]),
        ("final_sa", "Final SA map", display["winner"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    tiles = [
        ("Released image", image),
        ("Ground truth", _mask_overlay(image, annotation, (0, 220, 80))),
    ]
    for key, label, value in methods:
        heatmap = _heatmap(value)
        overlay = _overlay(image, heatmap)
        heatmap_path = path.parent / f"{path.stem}_{key}_attention.png"
        overlay_path = path.parent / f"{path.stem}_{key}_overlay.png"
        heatmap.save(heatmap_path, optimize=True)
        overlay.save(overlay_path, optimize=True)
        artifacts[key] = {"attention": str(heatmap_path), "overlay": str(overlay_path)}
        tiles.extend(((f"{label}: attention", heatmap), (f"{label}: overlay", overlay)))
    label_height = 24
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (size * columns, (size + label_height) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, tile) in enumerate(tiles):
        x = index % columns * size
        y = index // columns * (size + label_height)
        canvas.paste(tile, (x, y + label_height))
        draw.text((x + 7, y + 6), label, fill="black")
    temporary = path.with_suffix(".tmp.png")
    canvas.save(temporary, optimize=True)
    temporary.replace(path)
    return artifacts


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    shard_dir = args.output_dir / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = shard_dir / "checkpoint.pt"
    rankings = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"][
        "head_rankings"
    ]
    signature = {
        "suite": "attention_explanation_live_full_ovam_highest_resolution_v4",
        "manifest": str(args.manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "fixed_stage1_results": str(args.fixed_stage1_results),
    }
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {"signature": signature, "records": []}
    completed = {record["image_id"] for record in state["records"]}
    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    if args.panels_per_class <= 0:
        panel_ids: set[str] = set()
    else:
        panel_indices = np.linspace(0, len(rows) - 1, min(args.panels_per_class, len(rows))).round()
        panel_ids = {rows[int(index)]["image_id"] for index in panel_indices}

    for row in rows:
        if row["image_id"] in completed:
            continue
        print(f"[{classname}] {row['image_id']} ({len(completed) + 1}/{len(rows)})", flush=True)
        sample = generate_attention_sample(pipe, row, args.manifest.parent, args.storage_dtype)
        try:
            released_ca, released_sa, released_ovam_map, released_ovam_prediction = (
                load_released_ovam_components(
                    row, args.manifest.parent
                )
            )
            live_ca, live_sa, implemented_ovam_map, implemented_ovam_prediction = (
                ovam_full_prediction(sample, args.densecrf_site_packages)
            )
            record, display = analyze_attention_sample(
                sample,
                pipe.tokenizer,
                rankings,
                implemented_ovam_map,
                implemented_ovam_prediction,
                released_ovam_map,
                released_ovam_prediction,
                device=args.device,
            )

            def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
                left = left.float().flatten()
                right = right.float().flatten()
                return float(torch.corrcoef(torch.stack((left, right)))[0, 1])
            record["ovam_reproduction_validation"] = {
                "cross_attention_correlation": correlation(live_ca, released_ca),
                "self_attention_correlation": correlation(live_sa, released_sa),
                "fused_map_correlation": correlation(
                    torch.nn.functional.interpolate(
                        implemented_ovam_map[None, None], size=(64, 64), mode="bilinear",
                        align_corners=False,
                    )[0, 0],
                    torch.nn.functional.interpolate(
                        released_ovam_map[None, None], size=(64, 64), mode="bilinear",
                        align_corners=False,
                    )[0, 0],
                ),
                "post_crf_mask_iou": float(
                    (implemented_ovam_prediction & released_ovam_prediction).sum()
                    / (implemented_ovam_prediction | released_ovam_prediction).sum().clamp_min(1)
                ),
                "live_sa_unique_values": int(torch.unique(live_sa).numel()),
                "released_sa_unique_values": int(torch.unique(released_sa).numel()),
            }
            record["seed"] = int(row["seed"])
            record["source_image"] = row["image"]
            record["source_annotation"] = row["annotation"]
            if row["image_id"] in panel_ids:
                panel_path = shard_dir / "panels" / f"{row['image_id']}.png"
                record["visualizations"] = save_visualizations(
                    panel_path,
                    args.manifest.parent / row["image"],
                    args.manifest.parent / row["annotation"],
                    display,
                )
                record["panel"] = str(panel_path)
            state["records"].append(record)
            completed.add(row["image_id"])
            save_checkpoint(checkpoint_path, state)
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()
    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed explanatory analysis for {len(completed)} {classname} images")


if __name__ == "__main__":
    main()
