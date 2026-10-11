"""Evaluate the authors' released ordinary-prompt OVAM heatmaps on VOC-Sim."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/segmentation/ovam_paper_baseline/cross_attention_only_summary.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    metric = BinarySegmentationMetrics()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["annotated"].lower() == "true"]
    heatmap_dir = args.manifest.parent / "heatmaps"
    for row in rows:
        target_rgb = np.asarray(Image.open(args.manifest.parent / row["annotation"]).convert("RGB"))
        target = np.any(target_rgb != 0, axis=-1)
        heatmap_path = heatmap_dir / f"{Path(row['image']).stem}_heatmap_token0.npy"
        heatmap = torch.from_numpy(np.load(heatmap_path)[1]).float()[None, None]
        resized = torch.nn.functional.interpolate(heatmap, size=target.shape, mode="bilinear")[0, 0]
        prediction = resized / resized.max() > args.threshold
        metric.update(row["classname"], prediction, target)

    result = {
        "method": "ovam_released_cross_attention_only",
        "threshold": args.threshold,
        "source": "authors' released ordinary-prompt heatmaps, channel 1 of token0",
        **metric.result(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['method']}: mIoU={result['miou']:.9f}, images={result['images']}")


if __name__ == "__main__":
    main()
