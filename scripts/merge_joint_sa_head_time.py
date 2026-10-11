"""Merge all class shards from the joint SA head/timestep experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_streaming_voc_ablations import merge_metrics
from scripts.run_streaming_voc_ablations import finalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_joint_sa_head_time"),
    )
    parser.add_argument("--expected-shards", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_dir() and path.name.startswith("class_")
    )
    complete = [path for path in shards if (path / "COMPLETE").is_file()]
    if len(complete) != args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} complete shards, found {len(complete)}")
    metrics: dict = {}
    classes: list[str] = []
    completed_ids: set[str] = set()
    suite_metadata = None
    for path in complete:
        state = torch.load(path / "checkpoint.pt", map_location="cpu")
        classes.append(state["signature"]["classname"])
        overlap = completed_ids.intersection(state["completed"])
        if overlap:
            raise RuntimeError(f"Duplicate manifest IDs across shards: {sorted(overlap)[:5]}")
        completed_ids.update(state["completed"])
        merge_metrics(metrics, state["metrics"])
        if suite_metadata is None:
            suite_metadata = state["suite_metadata"]
        elif suite_metadata != state["suite_metadata"]:
            raise RuntimeError("Suite metadata differs between shards")
    results = finalized(metrics)
    settings = {
        "classes": sorted(classes),
        "images": len(completed_ids),
        "shards": len(complete),
        "suite_metadata": suite_metadata,
    }
    output_path = args.input_dir / "all_results.json.gz"
    temporary_gz = output_path.with_suffix(".tmp")
    with gzip.open(temporary_gz, "wt", encoding="utf-8") as handle:
        json.dump({"settings": settings, "results": results}, handle, sort_keys=True)
    os.replace(temporary_gz, output_path)
    winners = {}
    configuration_count = 0
    for experiment, configurations in sorted(results.items()):
        configuration_count += len(configurations)
        name = max(configurations, key=lambda key: configurations[key]["miou"])
        winners[experiment] = {"configuration": name, **configurations[name]}
    summary = {
        "settings": settings,
        "configuration_count": configuration_count,
        "best_by_experiment": winners,
    }
    temporary = args.input_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.input_dir / "summary.json")
    print(
        f"Merged {len(complete)} shards, {len(completed_ids)} images, and "
        f"{configuration_count} configurations into {args.input_dir}"
    )


if __name__ == "__main__":
    main()
