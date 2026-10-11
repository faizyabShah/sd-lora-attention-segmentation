"""Write every full VOC-Sim ablation result to one Markdown document."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs/segmentation/voc_sim_full_streaming/all_results.json.gz"
DEFAULT_EXHAUSTIVE = (
    ROOT
    / "outputs/segmentation/voc_sim_full_streaming/exhaustive_layer_subsets/all_subsets.csv.gz"
)
DEFAULT_EXHAUSTIVE_SUMMARY = (
    ROOT
    / "outputs/segmentation/voc_sim_full_streaming/exhaustive_layer_subsets/summary.json"
)
DEFAULT_FOCUSED_RESULTS = (
    ROOT / "outputs/segmentation/voc_sim_full_focused_sa_p1_p4/all_results.json.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--exhaustive", type=Path, default=DEFAULT_EXHAUSTIVE)
    parser.add_argument("--exhaustive-summary", type=Path, default=DEFAULT_EXHAUSTIVE_SUMMARY)
    parser.add_argument("--focused-results", type=Path, default=DEFAULT_FOCUSED_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "ablations.md")
    return parser.parse_args()


def safe(value: object) -> str:
    """Make arbitrary labels safe inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def score(value: object) -> str:
    return f"{float(value):.6f}"


def write_regular(
    handle, results: dict, classes: list[str], heading_level: int = 2
) -> int:
    total = 0
    for experiment, configurations in sorted(results.items()):
        rows = sorted(
            configurations.items(),
            key=lambda item: (-float(item[1]["miou"]), item[0]),
        )
        handle.write(
            f"{'#' * heading_level} `{safe(experiment)}` "
            f"({len(rows):,} configurations)\n\n"
        )
        columns = [
            "Rank",
            "Configuration",
            "mIoU",
            "Mean image IoU",
            "Mean background IoU",
            *classes,
        ]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---:", "---", *(["---:"] * (len(columns) - 2))]) + "|\n")
        for rank, (name, result) in enumerate(rows, start=1):
            values = [
                str(rank),
                f"`{safe(name)}`",
                score(result["miou"]),
                score(result["mean_image_iou"]),
                score(result["mean_background_iou"]),
            ]
            values.extend(score(result["per_class"][name]["iou"]) for name in classes)
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n")
        total += len(rows)
    return total


def write_exhaustive(handle, path: Path, classes: list[str]) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows.sort(key=lambda row: (-float(row["miou"]), int(row["subset_mask"])))

    handle.write(f"## Exhaustive cross-attention layer subsets ({len(rows):,} configurations)\n\n")
    handle.write(
        "This table contains every nonempty subset of the 16 cross-attention layers. "
        "It is sorted by mIoU; layer indices are defined above.\n\n"
    )
    columns = [
        "Rank",
        "Mask",
        "Category",
        "Count",
        "Selected layers",
        "Omitted layers",
        "mIoU",
        *classes,
    ]
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("|" + "|".join(["---:", "---", "---", "---:", "---", "---", *(["---:"] * (len(columns) - 6))]) + "|\n")
    for rank, row in enumerate(rows, start=1):
        values = [
            str(rank),
            f"`{safe(row['subset_hex'])}`",
            safe(row["category"]),
            row["selected_count"],
            safe(row["selected_layer_indices"] or "none"),
            safe(row["omitted_layer_indices"] or "none"),
            score(row["miou"]),
        ]
        values.extend(score(row[f"iou_{name}"]) for name in classes)
        handle.write("| " + " | ".join(values) + " |\n")
    handle.write("\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    with gzip.open(args.results, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    focused_payload = None
    if args.focused_results.is_file():
        with gzip.open(args.focused_results, "rt", encoding="utf-8") as source:
            focused_payload = json.load(source)
    exhaustive_summary = json.loads(args.exhaustive_summary.read_text(encoding="utf-8"))
    classes = list(payload["settings"]["classes"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Full VOC-Sim ablation results\n\n")
        handle.write(
            f"This document contains every evaluated configuration for the full VOC-Sim "
            f"experiment: **{payload['settings']['images']} annotated images across "
            f"{len(classes)} classes**. Tables are sorted by foreground mIoU within each "
            "experiment family. No configuration has been omitted.\n\n"
        )
        handle.write(
            "For regular configurations, `mIoU` is the macro mean of the 20 class IoUs, "
            "`Mean image IoU` averages per-image foreground IoU, and `Mean background "
            "IoU` averages class-level background IoU. The class columns contain pooled "
            "foreground IoU. Raw TP, FP, FN, TN, image counts, and full-precision values "
            "remain in the machine-readable source files listed below.\n\n"
        )
        handle.write("## Source files\n\n")
        handle.write(f"- Regular results: `{args.results.relative_to(ROOT)}`\n")
        handle.write(f"- Exhaustive results: `{args.exhaustive.relative_to(ROOT)}`\n")
        handle.write(f"- Exhaustive summary: `{args.exhaustive_summary.relative_to(ROOT)}`\n\n")
        if focused_payload is not None:
            handle.write(f"- Focused priorities 1--4: `{args.focused_results.relative_to(ROOT)}`\n\n")
        handle.write("## Cross-attention layer index\n\n")
        handle.write("| Index | Module |\n|---:|---|\n")
        for index, module in sorted(
            exhaustive_summary["layer_index"].items(), key=lambda item: int(item[0])
        ):
            handle.write(f"| {index} | `{safe(module)}` |\n")
        handle.write("\n")
        regular_count = write_regular(handle, payload["results"], classes)
        focused_count = 0
        if focused_payload is not None:
            if focused_payload["settings"]["classes"] != classes:
                raise ValueError("Focused and original result files use different classes")
            if focused_payload["settings"]["images"] != payload["settings"]["images"]:
                raise ValueError("Focused and original result files use different image counts")
            handle.write("## Focused self-attention follow-up: Priorities 1--4\n\n")
            focused_count = write_regular(
                handle, focused_payload["results"], classes, heading_level=3
            )
        exhaustive_count = write_exhaustive(handle, args.exhaustive, classes)
        handle.write("## Completeness\n\n")
        handle.write(f"- Regular configurations: **{regular_count:,}**\n")
        handle.write(f"- Focused follow-up configurations: **{focused_count:,}**\n")
        handle.write(f"- Exhaustive layer subsets: **{exhaustive_count:,}**\n")
        handle.write(
            f"- Total result rows: **{regular_count + focused_count + exhaustive_count:,}**\n"
        )
    os.replace(temporary, args.output)
    print(
        f"Wrote {args.output} with {regular_count:,} regular, {focused_count:,} focused, "
        f"and {exhaustive_count:,} exhaustive results"
    )


if __name__ == "__main__":
    main()
