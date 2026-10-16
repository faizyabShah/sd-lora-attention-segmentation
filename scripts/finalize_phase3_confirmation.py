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
    screen = selected["screens"]["P3-C expanded confirmation"]
    candidates = list(MANDATORY[1:])
    if screen["frozen"] != REFERENCE and screen["frozen"] not in candidates:
        candidates.append(screen["frozen"])
    diagnostics = {
        "reference": REFERENCE,
        "mandatory": list(MANDATORY),
        "selected_pipeline": screen,
        "fixed_candidate_crossfit": {
            name: crossfit(results, name, REFERENCE) for name in candidates
        },
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
    report_path = args.package_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
