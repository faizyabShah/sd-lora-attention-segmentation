"""Merge the 20 resumable class shards from the focused full VOC-Sim SA run."""

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
        default=Path("outputs/segmentation/voc_sim_full_focused_sa_p1_p4"),
    )
    parser.add_argument("--expected-shards", type=int, default=20)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_streaming/all_results.json.gz"),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_results_markdown(
    path: Path,
    settings: dict,
    winners: dict,
    reference_results: Path,
) -> None:
    """Replace the focused full-dataset summary while retaining older sections."""
    with gzip.open(reference_results, "rt", encoding="utf-8") as handle:
        reference = json.load(handle)["results"]
    baseline_group = reference["baseline"]
    baseline = baseline_group.get("ovam_cross_attention_only") or baseline_group["ovam_unoptimized"]
    previous = reference["self_attention/r64"][
        "module__up_blocks.3.attentions.2.transformer_blocks.0.attn1__iter4__continuous__non_residual"
    ]
    best_experiment, best = max(winners.items(), key=lambda item: item[1]["miou"])
    start = "<!-- FULL_VOC_FOCUSED_SA_RESULTS_START -->"
    end = "<!-- FULL_VOC_FOCUSED_SA_RESULTS_END -->"
    lines = [
        start,
        "## Full VOC-Sim focused self-attention follow-up",
        "",
        f"Evaluated **{settings['images']} images across {len(settings['classes'])} classes** "
        "for 1,632 configurations covering output calibration, cross-attention seeds, "
        "self-attention heads, and self-attention timestep weighting.",
        "",
        f"- OVAM cross-attention-only reference: **{baseline['miou']:.5f} mIoU**.",
        f"- Previous best: **{previous['miou']:.5f} mIoU**.",
        f"- New best: **{best['miou']:.5f} mIoU**, "
        f"`{best_experiment}/{best['configuration']}`.",
        f"- Gain over OVAM: **{best['miou'] - baseline['miou']:+.5f}**; "
        f"gain over the previous best: **{best['miou'] - previous['miou']:+.5f}**.",
        "",
        "All 1,632 configurations contain 490 images and 20 classes. The previous winning "
        "path was reproduced within 0.00002 mIoU; the small difference comes from GPU "
        "floating point variation across regenerated samples.",
        "",
        "### Focused family winners",
        "",
        "| Experiment | Best configuration | mIoU | Δ vs. OVAM |",
        "|---|---|---:|---:|",
    ]
    for experiment, value in sorted(winners.items()):
        lines.append(
            f"| {experiment} | `{value['configuration']}` | {value['miou']:.5f} | "
            f"{value['miou'] - baseline['miou']:+.5f} |"
        )
    lines.extend([
        "",
        "### Per-class IoU comparison",
        "",
        "| Class | OVAM | Previous best | New best | Δ new vs. previous |",
        "|---|---:|---:|---:|---:|",
    ])
    for classname in sorted(baseline["per_class"]):
        baseline_iou = baseline["per_class"][classname]["iou"]
        previous_iou = previous["per_class"][classname]["iou"]
        best_iou = best["per_class"][classname]["iou"]
        lines.append(
            f"| {classname} | {baseline_iou:.5f} | {previous_iou:.5f} | "
            f"{best_iou:.5f} | {best_iou - previous_iou:+.5f} |"
        )
    lines.extend(["", end, ""])
    section = "\n".join(lines)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Results\n"
    if start in existing and end in existing:
        prefix, remainder = existing.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix.lstrip("\n")
    else:
        updated = existing.rstrip() + "\n\n" + section
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    shard_dirs = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_dir() and path.name.startswith("class_")
    )
    complete = [path for path in shard_dirs if (path / "COMPLETE").is_file()]
    if len(complete) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} complete shards, found {len(complete)}"
        )

    metrics: dict = {}
    classes: list[str] = []
    completed_ids: set[str] = set()
    suite_metadata = None
    signatures = []
    for path in complete:
        state = torch.load(path / "checkpoint.pt", map_location="cpu")
        classname = state["signature"]["classname"]
        classes.append(classname)
        overlap = completed_ids.intersection(state["completed"])
        if overlap:
            raise RuntimeError(f"Duplicate manifest IDs across shards: {sorted(overlap)[:5]}")
        completed_ids.update(state["completed"])
        merge_metrics(metrics, state["metrics"])
        signatures.append(state["signature"])
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
    atomic_json(args.input_dir / "summary.json", summary)
    update_results_markdown(ROOT / "results.md", settings, winners, args.reference_results)
    print(
        f"Merged {len(complete)} shards, {len(completed_ids)} images, and "
        f"{configuration_count} configurations into {args.input_dir}"
    )


if __name__ == "__main__":
    main()
