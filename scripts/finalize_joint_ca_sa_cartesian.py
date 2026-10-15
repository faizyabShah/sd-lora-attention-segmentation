#!/usr/bin/env python3
"""Append matrix and class diagnostics to the joint CA x SA CV report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_joint_ca_sa_cartesian import (  # noqa: E402
    CA_CONFIGURATIONS,
    REFERENCE,
    SA_CHOICES,
    joint_name,
)
from src.segmentation import controlled_cv as cv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/joint_ca_sa_cartesian"),
    )
    args = parser.parse_args()
    package = args.package_dir
    results = cv.load_results(sorted((package / "shards").glob("*.npz")))
    folds = json.loads((package / "fold_results.json").read_text(encoding="utf-8"))
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))[
        "configurations"
    ]
    screen = folds["screens"]["Joint CA-SA"]
    frozen = screen["frozen"]

    lines = [
        "",
        "## Full CA × SA matrix (490-image in-sample diagnostic)",
        "",
        "Rows contain every CA-C cell; columns contain no-SA and every unique SA-C cell. "
        "Selection itself is cross-validated above.",
        "",
        "| CA configuration | " + " | ".join(f"SA {s}" for s in SA_CHOICES) + " |",
        "|---|" + "---:|" * len(SA_CHOICES),
    ]
    for ca_name in CA_CONFIGURATIONS:
        values = [summary[joint_name(ca_name, sa)]["miou@0.45"] for sa in SA_CHOICES]
        lines.append(f"| `{ca_name}` | " + " | ".join(f"{v:.4f}" for v in values) + " |")

    lines += [
        "",
        "## Best partner for each CA configuration (in-sample diagnostic)",
        "",
        "| CA | Best SA | mIoU@0.45 | Gain over that CA without SA |",
        "|---|---|---:|---:|",
    ]
    for ca_name in CA_CONFIGURATIONS:
        best_sa = max(SA_CHOICES, key=lambda s: summary[joint_name(ca_name, s)]["miou@0.45"])
        best = summary[joint_name(ca_name, best_sa)]["miou@0.45"]
        ca_only = summary[joint_name(ca_name, "none")]["miou@0.45"]
        lines.append(f"| `{ca_name}` | `{best_sa}` | {best:.4f} | {best-ca_only:+.4f} |")

    reference_classes = cv.class_ious(results, results.index(REFERENCE))
    if frozen == REFERENCE:
        comparison = max(
            (name for name in results.configurations if name != REFERENCE),
            key=lambda name: summary[name]["miou@0.45"],
        )
        comparison_label = "closest in-sample alternative (diagnostic only)"
    else:
        comparison = frozen
        comparison_label = "cross-validated joint selection"
    selected_classes = cv.class_ious(results, results.index(comparison))
    assessment = cv.assess(
        results, results.index(REFERENCE), results.index(comparison)
    )
    bootstrap = cv.paired_bootstrap(
        cv.image_ious(results, results.index(comparison)),
        cv.image_ious(results, results.index(REFERENCE)),
    )
    lines += [
        "",
        f"## Per-class IoU: {comparison_label}",
        "",
        f"`{comparison}` versus `{REFERENCE}`. Δ mIoU@0.45 "
        f"{assessment['delta']:+.4f} ({assessment['verdict']}); paired mean image-IoU "
        f"{bootstrap['mean']:+.4f} [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}].",
        "",
        "| Class | Sequential reference | Comparison | Δ |",
        "|---|---:|---:|---:|",
    ]
    for classname in sorted(reference_classes):
        before, after = reference_classes[classname], selected_classes[classname]
        lines.append(f"| {classname} | {before:.4f} | {after:.4f} | {after-before:+.4f} |")
    with (package / "report.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print((package / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
