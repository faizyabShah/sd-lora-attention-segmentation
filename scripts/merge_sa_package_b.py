#!/usr/bin/env python3
"""Merge SA package B: cross-validated SA-L and SA-H, head report, exhaustive search, operator check.

Reads ``<package>/shards/*.npz`` written by ``run_sa_package_b.py`` and
``<package>/exhaustive_raw/*.npy``; writes ``fold_results.json``,
``summary.json``, ``report.md``, ``head_report.json`` and ``exhaustive_cv.json``.
The operator check (``OP/AT_x`` = ``y = A^T x``) and the CA seed alone are
reported against the SA reference; neither is a selection candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_ca_package_b import HEADS, exhaustive_cv, head_report  # noqa: E402
from scripts.select_controlled_cv import report as selector_report  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

SA_REFERENCE = "SL/all_16"
HEAD_REFERENCE = "SH/" + "_".join(map(str, range(HEADS)))


def compare(results: cv.Results, name: str, reference: str) -> dict:
    a, b = results.index(name), results.index(reference)
    return {
        "configuration": name, "reference": reference,
        **{f"miou@{t:.2f}": cv.miou(results, a, i) for i, t in enumerate(results.thresholds)},
        **{f"reference_miou@{t:.2f}": cv.miou(results, b, i) for i, t in enumerate(results.thresholds)},
        "mean_ap": float(cv.np.nanmean(results.ap[a])), "reference_mean_ap": float(cv.np.nanmean(results.ap[b])),
        "paired_bootstrap": cv.paired_bootstrap(cv.image_ious(results, a), cv.image_ious(results, b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/segmentation/controlled_attention_cv/sa_layer_head"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    package = args.package_dir
    results = cv.load_results(sorted((package / "shards").glob("*.npz")))
    if (len(results.image_ids) != 490 or not all(results.complete)) and not args.allow_incomplete:
        raise SystemExit(f"{len(results.image_ids)} images, {sum(results.complete)}/{len(results.complete)} shards complete")
    if not results.settings["ca_seed"].get("frozen") and not args.allow_incomplete:
        raise SystemExit("The CA seed of these shards is provisional; official SA results need the frozen CA seed")
    inclusiveness = json.loads((package / "configurations.json").read_text())["inclusiveness"]
    screens = {
        "SA-L": cv.cross_validate(results, SA_REFERENCE,
                                  [c for c in results.configurations if c.startswith("SL/") and c != SA_REFERENCE], inclusiveness),
        "SA-H": cv.cross_validate(results, HEAD_REFERENCE,
                                  [c for c in results.configurations if c.startswith("SH/") and c != HEAD_REFERENCE], inclusiveness),
    }
    checks = {"ca_seed_alone": compare(results, "SEED/ca_seed", SA_REFERENCE),
              "operator_AT_x": compare(results, "OP/AT_x", SA_REFERENCE)}
    rows = {name: cv.summary_row(results, i) for i, name in enumerate(results.configurations)}
    heads = head_report(results, screens["SA-H"], prefix="SH")
    raw = package / "exhaustive_raw"
    exhaustive = exhaustive_cv(results, raw, modules_key="sa_modules") if raw.exists() and any(raw.iterdir()) else None
    meta = {"merge_commit": cv.git_commit(), "shard_commits": list(results.commits),
            "shard_environments": list(results.environments), "ca_seed": results.settings["ca_seed"],
            "images": len(results.image_ids), "configurations": len(results.configurations)}
    (package / "fold_results.json").write_text(json.dumps({**meta, "screens": screens, "checks": checks}, indent=2) + "\n")
    (package / "summary.json").write_text(json.dumps({**meta, "configurations": rows}, indent=2) + "\n")
    (package / "head_report.json").write_text(json.dumps(heads, indent=2) + "\n")
    if exhaustive is not None:
        (package / "exhaustive_cv.json").write_text(json.dumps(exhaustive, indent=2) + "\n")

    seed = results.settings["ca_seed"]
    lines = [selector_report(results, screens, rows).replace("# Cross-validated selection", "# SA package B: SA-L, SA-H, operator check"),
             f"CA seed: `{seed['name']}` (frozen: {seed['frozen']}).", "", "## Seed alone and operator check (against the SA reference, in-sample)", "",
             "| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP | Paired Δ image IoU vs SA reference [95% CI] |", "|---|---:|---:|---:|---:|---|"]
    ref = compare(results, SA_REFERENCE, SA_REFERENCE)
    for label, c in (("SA reference: all modules, heads, calls (y = A x)", ref), ("CA seed alone", checks["ca_seed_alone"]),
                     ("Operator check: y = Aᵀx", checks["operator_AT_x"])):
        b = c["paired_bootstrap"]
        lines.append(f"| {label} | {c['miou@0.35']:.4f} | {c['miou@0.45']:.4f} | {c['miou@0.55']:.4f} | {c['mean_ap']:.4f} | "
                     f"{b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    lines += ["", "## SA-H head report", "", "| Head | Alone | Leave this head out | Folds retaining it |", "|---:|---:|---:|---:|"]
    for h in range(HEADS):
        lines.append(f"| {h} | {heads['single_head_miou'][h]:.4f} | {heads['leave_one_out_miou'][h]:.4f} | {heads['folds_retaining_each_head'][h]}/5 |")
    lines += ["", f"All 8 heads: {heads['reference_miou']:.4f}. Frozen subset `{heads['frozen_subset']}` ranks {heads['frozen_rank_among_255']} of 255 "
              f"in-sample; best in-sample `{heads['best_in_sample_subset']}` ({heads['best_in_sample_miou']:.4f}).", "",
              "| Round | Removed | Heads left | mIoU |", "|---:|---:|---|---:|"]
    for r, step in enumerate(heads["greedy_backward_path"]):
        lines.append(f"| {r} | {step['removed'] if step['removed'] is not None else '—'} | {' '.join(map(str, step['heads']))} | {step['miou']:.4f} |")
    if exhaustive is not None:
        lines += ["", "## Exhaustive SA module subsets (exploratory, not eligible for freezing)", "",
                  f"Out-of-fold selection {exhaustive['out_of_fold_miou_selection']:.4f} vs all-16 {exhaustive['out_of_fold_miou_reference']:.4f} "
                  f"(gain {exhaustive['oof_gain']:+.4f}); fold votes {exhaustive['selection_counts']}."]
    (package / "report.md").write_text("\n".join(lines) + "\n")
    print((package / "report.md").read_text())


if __name__ == "__main__":
    main()
