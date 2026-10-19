#!/usr/bin/env python3
"""Validate and index the one-image-per-class qualitative SG audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    "sg1_sg3": 76,
    "sg3_locked": 6,
    "sg5": 40,
    "sg8": 32,
    "sg9": 8,
    "sg10": 54,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/segmentation/sg_visual_audit"))
    args = parser.parse_args()
    root = args.output_dir
    grouped: dict[str, list[dict]] = defaultdict(list)
    records = []
    for path in sorted(root.glob("*/*/index.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        if not sample:
            raise RuntimeError(f"Empty visual index: {path}")
        family = sample[0]["family"]
        if len(sample) != EXPECTED.get(family):
            raise RuntimeError(f"{path}: expected {EXPECTED.get(family)} variants, found {len(sample)}")
        if any(row["family"] != family for row in sample):
            raise RuntimeError(f"Mixed families in {path}")
        grouped[family].append({"dir": path.parent, "records": sample})
        records.extend(sample)
    missing = {family: 20 - len(grouped.get(family, [])) for family in EXPECTED if len(grouped.get(family, [])) != 20}
    if missing:
        raise RuntimeError(f"Incomplete audit: {missing}")

    (root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    samples = sorted({(r["class"], r["image_id"]) for r in records})
    with (root / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "image_id"))
        writer.writerows(samples)

    lines = [
        "# SG qualitative audit", "",
        "This directory shows one deterministic VOC-Sim sample per class for every SG cell that was actually run.", "",
        "Every row contains the released VOC-Sim image, the CA seed overlaid on that image, the continuous SA-derived score overlaid on it, and the final predicted mask. The cyan outline in the mask panel is the released ground-truth boundary. Thresholds are the full-VOC thresholds selected independently for each cell.", "",
        "DenseCRF cells show the pre-CRF score in the SA panel and the post-CRF binary prediction in the final panel.", "",
        "## Coverage", "",
        "| Package | Variants per class | Classes |", "|---|---:|---:|",
    ]
    for family, count in EXPECTED.items():
        lines.append(f"| `{family}` | {count} | {len(grouped[family])} |")
    lines += ["", "## Pages", ""]
    for family in EXPECTED:
        lines += [f"### {family}", "", "| Class | Image | Pages |", "|---|---|---|"]
        for sample in sorted(grouped[family], key=lambda x: x["records"][0]["class"]):
            first = sample["records"][0]
            pages = sorted({row["page"] for row in sample["records"]})
            links = " ".join(f"[page {i + 1}]({sample['dir'].relative_to(root) / page})" for i, page in enumerate(pages))
            lines.append(f"| {first['class']} | `{first['image_id']}` | {links} |")
        lines.append("")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated {len(records)} variant/sample rows across {len(samples)} images; wrote {root / 'README.md'}")


if __name__ == "__main__":
    main()
