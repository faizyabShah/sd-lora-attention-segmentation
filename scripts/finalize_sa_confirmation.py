#!/usr/bin/env python3
"""Append SA-C class and component diagnostics after controlled selection."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.segmentation import controlled_cv as cv  # noqa: E402


def class_ious(results, configuration):
    counts = results.counts[results.index(configuration), :, cv.SELECTION_INDEX]
    classes = np.asarray(results.classes)
    output = {}
    for classname in sorted(set(results.classes)):
        pooled = counts[classes == classname].sum(axis=0)
        output[classname] = float(pooled[0] / max(pooled[:3].sum(), 1))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(
        "outputs/segmentation/controlled_attention_cv/sa_confirmation"))
    args = parser.parse_args(); package = args.package_dir
    results = cv.load_results(sorted((package / "shards").glob("*.npz")))
    folds = json.loads((package / "fold_results.json").read_text())
    screen = folds["screens"]["SA-C"]; frozen, reference = screen["frozen"], screen["reference"]
    ref_classes, selected_classes = class_ious(results, reference), class_ious(results, frozen)
    per_class = {c: {"reference": ref_classes[c], "selected": selected_classes[c],
                     "delta": selected_classes[c] - ref_classes[c]} for c in ref_classes}
    rows = json.loads((package / "summary.json").read_text())["configurations"]
    comparisons = {
        "T within T+L+H": ("C/T+L+H", "C/L+H"),
        "L within T+L+H": ("C/T+L+H", "C/T+H"),
        "H within T+L+H": ("C/T+L+H", "C/T+L"),
        "R within T+R+H": ("C/T+R+H", "C/T+H"),
    }
    effects = {label: {"with": a, "without": b,
                        "delta_miou@0.45": rows[a]["miou@0.45"] - rows[b]["miou@0.45"]}
               for label, (a, b) in comparisons.items()}
    (package / "diagnostics.json").write_text(json.dumps(
        {"selected": frozen, "reference": reference, "per_class": per_class,
         "component_effects": effects}, indent=2) + "\n")
    lines = ["", "## SA-C component effects (full 490, in-sample diagnostics)", "",
             "| Component comparison | With | Without | Δ mIoU@0.45 |",
             "|---|---|---|---:|"]
    for label, item in effects.items():
        lines.append(f"| {label} | `{item['with']}` | `{item['without']}` | {item['delta_miou@0.45']:+.4f} |")
    lines += ["", f"## Per-class IoU: selected `{frozen}` against `{reference}`", "",
              "| Class | Reference | Selected | Δ |", "|---|---:|---:|---:|"]
    for classname, item in sorted(per_class.items(), key=lambda pair: pair[1]["delta"]):
        lines.append(f"| {classname} | {item['reference']:.4f} | {item['selected']:.4f} | {item['delta']:+.4f} |")
    with (package / "report.md").open("a") as handle: handle.write("\n".join(lines) + "\n")
    print((package / "report.md").read_text())


if __name__ == "__main__": main()
