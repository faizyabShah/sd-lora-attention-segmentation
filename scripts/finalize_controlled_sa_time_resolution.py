#!/usr/bin/env python3
"""Add joint-grid diagnostics and settings to controlled SA package A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.controlled_cv import git_commit, load_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/sa_time_resolution"),
    )
    args = parser.parse_args()
    summary_path = args.package_dir / "summary.json"
    report_path = args.package_dir / "report.md"
    if not summary_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Run scripts/select_controlled_cv.py before finalizing")
    results = load_results(sorted((args.package_dir / "shards").glob("*.npz")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configurations = summary["configurations"]
    ranked = sorted(
        (row for name, row in configurations.items() if name.startswith("time_resolution/")),
        key=lambda row: -row["miou@0.45"],
    )
    reference_name = "time_resolution/all_00_30__r8_r16_r32_r64"
    reference = configurations[reference_name]
    summary["joint_time_resolution_diagnostic"] = {
        "selection_status": "interaction evidence only; not eligible for automatic freezing",
        "reference": reference_name,
        "reference_miou@0.45": reference["miou@0.45"],
        "best": ranked[0],
        "gain_over_reference": ranked[0]["miou@0.45"] - reference["miou@0.45"],
        "top_20": ranked[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.package_dir / "settings.json").write_text(
        json.dumps(
            {
                **results.settings,
                "shard_commits": list(results.commits),
                "selector_commit": summary.get("selector_commit"),
                "finalizer_commit": git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lines = [
        "",
        "## Joint SA time-resolution diagnostic",
        "",
        "This interaction grid is diagnostic and is not eligible for automatic freezing.",
        "",
        f"Inclusive reference: `{reference_name}` at {reference['miou@0.45']:.5f} mIoU.",
        "",
        "| Rank | Configuration | mIoU@0.45 | Gain over inclusive reference |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {rank} | `{row['configuration']}` | {row['miou@0.45']:.5f} | "
            f"{row['miou@0.45'] - reference['miou@0.45']:+.5f} |"
        )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Finalized controlled SA package in {args.package_dir}")


if __name__ == "__main__":
    main()

