"""Evaluate OVAM's released VOC-Sim masks against the released annotations.

This is the authoritative comparison for the paper pipeline because these masks
were produced by the authors with CA, high-resolution SA fusion, thresholding,
and DenseCRF.  It does not depend on reproducing their generation environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.metrics import BinarySegmentationMetrics


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
        "--mask-dir",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam_sa/voc_sim/mask"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/segmentation/ovam_paper_baseline/summary.json"),
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Label the result as token optimized (use with mask_optimized).",
    )
    return parser.parse_args()


def _binary_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    return np.any(image != 0, axis=-1)


def _mask_name(annotation: str) -> str:
    stem = Path(annotation).stem
    if not stem.endswith("_annotation"):
        raise ValueError(f"Unexpected annotation filename: {annotation}")
    return stem.removesuffix("_annotation") + "_ovamsa_crf_mask.png"


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["annotated"].lower() == "true"]

    metric = BinarySegmentationMetrics()
    per_image: list[dict[str, object]] = []
    missing: list[str] = []
    for row in rows:
        mask_path = args.mask_dir / _mask_name(row["annotation"])
        if not mask_path.is_file():
            missing.append(str(mask_path))
            continue
        target = _binary_image(args.manifest.parent / row["annotation"])
        prediction = _binary_image(mask_path)
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch for {row['image_id']}: {prediction.shape} != {target.shape}")
        metric.update(row["classname"], torch.from_numpy(prediction), target)
        intersection = int(np.count_nonzero(prediction & target))
        union = int(np.count_nonzero(prediction | target))
        per_image.append(
            {
                "image_id": row["image_id"],
                "classname": row["classname"],
                "iou": intersection / union if union else 1.0,
            }
        )

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks; first: {missing[0]}")
    result = metric.result()
    if result["images"] != len(rows):
        raise RuntimeError(f"Evaluated {result['images']} of {len(rows)} annotated images")
    payload = {
        "method": "ovam_token_optimized" if args.optimized else "ovam_paper_unoptimized",
        "pipeline": {
            "cross_attention": "OVAM ordinary prompt" if not args.optimized else "OVAM token optimized",
            "self_attention_fusion": True,
            "self_attention_min": 0.95 if args.optimized else 0.85,
            "threshold": 0.75 if args.optimized else 0.4,
            "dense_crf": True,
            "source": "authors' released masks",
        },
        **result,
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{payload['method']}: mIoU={result['miou']:.9f}, images={result['images']}")


if __name__ == "__main__":
    main()
