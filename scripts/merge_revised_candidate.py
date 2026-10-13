#!/usr/bin/env python3
"""Merge CA-10--30/all-SA-head candidate shards for one dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_revised_system_component_ablation import RAW_KEYS, aggregate
from scripts.run_revised_candidate import METHOD_KEY, METHOD_LABELS


OUTPUT_ROOT = Path("outputs/segmentation/ca10_30_all_sa_heads")
REFERENCE_ROOT = Path("outputs/segmentation/revised_system_component_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc_sim", "coco_cap"), default="voc_sim")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--method-key", default=METHOD_KEY)
    parser.add_argument("--method-label", default=METHOD_LABELS[METHOD_KEY])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.output_root / args.dataset
    shards = sorted(
        path
        for path in directory.glob("class_*")
        if path.is_dir() and (path / "COMPLETE").is_file()
    )
    if len(shards) != 20:
        raise RuntimeError(
            f"Expected 20 complete {args.dataset} shards, found {len(shards)}"
        )
    per_class: dict[str, dict] = {}
    common: dict | None = None
    for shard in shards:
        payload = json.loads((shard / "results.json").read_text(encoding="utf-8"))
        settings = payload["settings"]
        if settings["completed"] != settings["expected"]:
            raise RuntimeError(f"Incomplete shard: {shard}")
        classname = settings["classname"]
        if classname in per_class:
            raise RuntimeError(f"Duplicate class: {classname}")
        per_class[classname] = {
            key: payload["results"][args.method_key][key] for key in RAW_KEYS
        }
        comparable = {
            key: settings[key]
            for key in (
                "suite",
                "dataset",
                "manifest",
                "model",
                "storage_dtype",
                "model_dtype",
                "fixed_stage1_results",
                "ca_steps",
                "sa_heads",
                "fixed_settings",
            )
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise RuntimeError(f"Settings differ in {shard}")

    result = aggregate(per_class)
    reference = args.reference_root / args.dataset / "results.json"
    baseline = json.loads(reference.read_text(encoding="utf-8"))["results"]
    comparisons = {
        "revised_full_system": baseline["revised_full_system"]["miou"],
        "revised_all_sa_heads": baseline["all_sa_heads"]["miou"],
    }
    result["comparisons"] = comparisons
    result["change_from_revised_full"] = result["miou"] - comparisons["revised_full_system"]
    result["change_from_all_sa_heads"] = result["miou"] - comparisons["revised_all_sa_heads"]
    output = {
        "settings": {
            **(common or {}),
            "method_labels": {args.method_key: args.method_label},
        },
        "results": {args.method_key: result},
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (directory / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "iou"))
        writer.writerows(
            (name, value["iou"]) for name, value in result["per_class"].items()
        )
    ca_start, ca_end = (common or {}).get("ca_steps", ["?", "?"])
    report = [
        f"# CA calls {ca_start}--{ca_end} with all SA heads",
        "",
        "All other revised-system settings are fixed.",
        "",
        f"| Configuration | {args.dataset} mIoU | Change from revised full |",
        "|---|---:|---:|",
        f"| Revised full system | {comparisons['revised_full_system']:.4f} | +0.0000 |",
        f"| All SA heads, CA calls 0--30 | {comparisons['revised_all_sa_heads']:.4f} | "
        f"{comparisons['revised_all_sa_heads'] - comparisons['revised_full_system']:+.4f} |",
        f"| All SA heads, CA calls {ca_start}--{ca_end} | {result['miou']:.4f} | "
        f"{result['change_from_revised_full']:+.4f} |",
    ]
    (directory / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Preserve the path used by the already-submitted VOC-Sim merge.
    if args.dataset == "voc_sim" and args.output_root == OUTPUT_ROOT:
        (args.output_root / "report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
    print(f"Merged candidate results into {directory}")


if __name__ == "__main__":
    main()
