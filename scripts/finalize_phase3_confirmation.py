#!/usr/bin/env python3
"""Add frozen-cell diagnostics to the completed P3-C report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_propagation_package import crossfit  # noqa: E402
from scripts.run_phase3_confirmation import MANDATORY, REFERENCE  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/phase3_confirmation"),
    )
    args = parser.parse_args()
    results = cv.load_results(sorted((args.package_dir / "shards").glob("*.npz")))
    selected = json.loads((args.package_dir / "fold_results.json").read_text(encoding="utf-8"))
    mandatory_screen = selected["screens"]["P3-C mandatory confirmation"]
    screen = selected["screens"]["P3-C expanded confirmation"]
    candidates = list(MANDATORY[1:])
    if screen["frozen"] != REFERENCE and screen["frozen"] not in candidates:
        candidates.append(screen["frozen"])
    incremental = None
    if screen["frozen"] != mandatory_screen["frozen"]:
        incremental = crossfit(results, screen["frozen"], mandatory_screen["frozen"])
        gain = incremental["oof_miou_selected"] - incremental["oof_miou_reference"]
        improved = sum(row["delta"] > 0 for row in incremental["per_class"].values())
        worst_class, worst = min(
            incremental["per_class"].items(), key=lambda pair: pair[1]["delta"]
        )
        eligible = (
            gain >= cv.PROMOTE
            or (
                gain >= cv.CONDITIONAL
                and improved >= cv.CONDITIONAL_MIN_CLASSES
                and worst["delta"] >= -cv.CONDITIONAL_MAX_CLASS_LOSS
            )
        )
        incremental["gain"] = gain
        incremental["classes_improved"] = improved
        incremental["worst_class"] = worst_class
        incremental["worst_class_delta"] = worst["delta"]
        incremental["eligible_under_r4"] = eligible
    diagnostics = {
        "reference": REFERENCE,
        "mandatory": list(MANDATORY),
        "mandatory_selection": mandatory_screen,
        "selected_pipeline": screen,
        "fixed_candidate_crossfit": {
            name: crossfit(results, name, REFERENCE) for name in candidates
        },
        "incremental_expanded_vs_mandatory": incremental,
    }
    (args.package_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    marker = "## Frozen mandatory-cell diagnostics"
    lines = ["", marker, "", "| Configuration | Reference OOF mIoU | Candidate OOF mIoU | Gain | Paired mean image-IoU [95% CI] | Better / worse |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in diagnostics["fixed_candidate_crossfit"].items():
        bootstrap = item["paired_bootstrap"]
        gain = item["oof_miou_selected"] - item["oof_miou_reference"]
        lines.append(
            f"| `{name}` | {item['oof_miou_reference']:.4f} | "
            f"{item['oof_miou_selected']:.4f} | {gain:+.4f} | "
            f"{bootstrap['mean']:+.4f} [{bootstrap['ci_low']:+.4f}, "
            f"{bootstrap['ci_high']:+.4f}] | {bootstrap['better']} / "
            f"{bootstrap['worse']} |"
        )
    if incremental is not None:
        bootstrap = incremental["paired_bootstrap"]
        decision = "eligible" if incremental["eligible_under_r4"] else "not promoted"
        lines += [
            "",
            "### Incremental expanded-candidate check",
            "",
            f"Against the mandatory winner, `{screen['frozen']}` changes OOF mIoU "
            f"from {incremental['oof_miou_reference']:.4f} to "
            f"{incremental['oof_miou_selected']:.4f} "
            f"({incremental['gain']:+.4f}); paired mean image-IoU "
            f"{bootstrap['mean']:+.4f} [{bootstrap['ci_low']:+.4f}, "
            f"{bootstrap['ci_high']:+.4f}]. It improves "
            f"{incremental['classes_improved']}/20 classes; the worst change is "
            f"`{incremental['worst_class']}` "
            f"{incremental['worst_class_delta']:+.4f}. Therefore it is **{decision}** "
            "under R4 as an addition to the mandatory system.",
        ]
    report_path = args.package_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
