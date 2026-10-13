#!/usr/bin/env python3
"""Cross-validated selection for one controlled-attention package (protocol Section 5a).

Reads the package's per-image ``.npz`` shards and a screen specification, and
writes ``fold_results.json``, ``summary.json`` and ``report.md`` to the
package output directory. The specification is JSON::

    {"screens": [
        {"name": "CA-H",
         "reference": "heads/all",
         "candidates": ["heads/0", "heads/0_1", ...],      # or "prefix": "heads/"
         "inclusiveness": {"heads/0_1": 2, ...}}             # optional (Section 5 tie rule)
    ]}

Selection uses threshold 0.45 only; 0.35/0.55 and AP are reported as
calibration diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.controlled_cv import (  # noqa: E402
    cross_validate,
    git_commit,
    load_results,
    summary_row,
)


def screen_candidates(results, screen: dict) -> list[str]:
    if "candidates" in screen:
        missing = [c for c in screen["candidates"] if c not in results.configurations]
        if missing:
            raise KeyError(f"{screen['name']}: unknown configurations {missing[:5]}")
        return list(screen["candidates"])
    prefix = screen["prefix"]
    return [c for c in results.configurations if c.startswith(prefix) and c != screen["reference"]]


def report(results, screens: dict, rows: dict) -> str:
    lines = [
        "# Cross-validated selection",
        "",
        f"{len(results.image_ids)} images, {len(set(results.classes))} classes, "
        f"{len(results.configurations)} configurations; shards complete: "
        f"{sum(results.complete)}/{len(results.complete)}. Selection at threshold 0.45.",
        "",
        "| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | "
        "In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |",
        "|---|---|---|---:|---|---|---|---:|---|:---:|",
    ]
    for name, cv in screens.items():
        b = cv["paired_bootstrap_frozen_vs_reference"]
        lines.append(
            f"| {name} | `{cv['reference']}` | `{cv['vote_winner']}` ({cv['vote_winner_votes']}/5) | "
            f"{cv['oof_gain']:+.4f} | `{cv['frozen']}` | {cv['freeze_reason']} | "
            f"{cv['out_of_fold_miou_selection']:.4f} / {cv['out_of_fold_miou_reference']:.4f} | "
            f"{cv['in_sample_miou_frozen'] - cv['in_sample_miou_reference']:+.4f} | "
            f"{b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | "
            f"{'yes' if cv['calibration_dependent'] else 'no'} |"
        )
    for name, cv in screens.items():
        lines += ["", f"## {name}", "", "| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |",
                  "|---:|---|---|---|---|"]
        for f in cv["folds"]:
            lines.append(
                f"| {f['fold']} | `{f['chosen']}` | {f['reason']} | "
                f"{f['train_miou_chosen']:.4f} / {f['train_miou_reference']:.4f} | "
                f"{f['heldout_miou_chosen']:.4f} / {f['heldout_miou_reference']:.4f} |"
            )
        ranked = sorted((rows[c] for c in [cv["reference"], *cv["candidates"]]), key=lambda r: -r["miou@0.45"])
        lines += ["", "Top configurations, full 490 (in-sample):", "",
                  "| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |", "|---|---:|---:|---:|---:|"]
        for r in ranked[:10]:
            lines.append(f"| `{r['configuration']}` | {r['miou@0.35']:.4f} | {r['miou@0.45']:.4f} | "
                         f"{r['miou@0.55']:.4f} | {r['mean_ap']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, required=True, help="directory holding the shard .npz files")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    shards = sorted(args.package_dir.glob("**/*.npz"))
    results = load_results(shards)
    if not all(results.complete) and not args.allow_incomplete:
        raise SystemExit(f"{sum(not c for c in results.complete)} shard(s) are incomplete")
    spec = json.loads(args.spec.read_text())
    rows = {name: summary_row(results, i) for i, name in enumerate(results.configurations)}
    screens = {}
    for screen in spec["screens"]:
        screens[screen["name"]] = cross_validate(
            results, screen["reference"], screen_candidates(results, screen), screen.get("inclusiveness")
        )
    meta = {"selector_commit": git_commit(), "shard_commits": list(results.commits),
            "images": len(results.image_ids), "configurations": len(results.configurations)}
    (args.package_dir / "fold_results.json").write_text(json.dumps({**meta, "screens": screens}, indent=2) + "\n")
    (args.package_dir / "summary.json").write_text(json.dumps({**meta, "configurations": rows}, indent=2) + "\n")
    (args.package_dir / "report.md").write_text(report(results, screens, rows))
    print((args.package_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
