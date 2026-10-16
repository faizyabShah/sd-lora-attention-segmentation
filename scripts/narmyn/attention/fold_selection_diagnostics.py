#!/usr/bin/env python3
"""Registry R9 diagnostics for a package selected with ``scripts/select_propagation_cv.py``.

For every screen in ``fold_results.json``: the pooled out-of-fold counts of the
selection and of the reference (each at its fold-chosen threshold), the paired
bootstrap of per-image IoU, the change at fixed thresholds 0.35/0.45/0.55,
per-class changes at the frozen setting's full-data threshold, and, for each
candidate that was never chosen, the training-fold check that excluded it.
Writes ``diagnostics.json`` next to ``fold_results.json``.

    python scripts/narmyn/attention/fold_selection_diagnostics.py --package-dir <package>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402


def class_ious(counts: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    return {c: float(counts[classes == c, 0].sum() / counts[classes == c, :3].sum()) for c in sorted(set(classes))}


def pooled(counts, classes) -> float:
    return float(np.mean(list(class_ious(counts, classes).values())))


def image_iou(counts: np.ndarray) -> np.ndarray:
    return counts[:, 0] / np.maximum(counts[:, :3].sum(axis=1), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    results = cv.load_results(sorted((args.package_dir / "shards").glob("*.npz")))
    committed = json.loads((args.package_dir / "fold_results.json").read_text())
    thresholds = list(results.thresholds)
    classes = np.asarray(results.classes)
    out = {}
    for name, screen in committed["screens"].items():
        reference = screen["reference"]
        sel = np.zeros((len(results.image_ids), 4), np.int64)
        ref = np.zeros_like(sel)
        for row in screen["folds"]:
            held = results.folds == row["fold"]
            sel[held] = results.counts[results.index(row["chosen"]), held, thresholds.index(row["chosen_threshold"])]
            ref[held] = results.counts[results.index(reference), held, thresholds.index(row["reference_threshold"])]
        frozen = screen["frozen"]
        t_full = thresholds.index(screen["full_threshold_frozen"])
        t_ref = thresholds.index(screen["full_threshold_reference"])
        a = class_ious(results.counts[results.index(frozen), :, t_full], classes)
        b = class_ious(results.counts[results.index(reference), :, t_ref], classes)
        excluded = {}
        for candidate in screen["candidates"]:
            if any(row["chosen"] == candidate for row in screen["folds"]):
                continue
            per_fold = []
            for k in sorted(set(results.folds.tolist())):
                train = results.folds != k
                best_c = max(range(len(thresholds)), key=lambda t: pooled(results.counts[results.index(candidate), train, t], classes[train]))
                best_r = max(range(len(thresholds)), key=lambda t: pooled(results.counts[results.index(reference), train, t], classes[train]))
                cc = class_ious(results.counts[results.index(candidate), train, best_c], classes[train])
                rr = class_ious(results.counts[results.index(reference), train, best_r], classes[train])
                delta = {c: cc[c] - rr[c] for c in cc}
                worst = min(delta, key=delta.get)
                per_fold.append({"fold": k, "gain": float(np.mean(list(cc.values())) - np.mean(list(rr.values()))),
                                 "classes_improved": int(sum(v > 0 for v in delta.values())),
                                 "worst_class": worst, "worst_class_delta": float(delta[worst])})
            excluded[candidate] = per_fold
        out[name] = {
            "frozen": frozen, "reference": reference,
            "oof_miou_reference": pooled(ref, classes), "oof_miou_selection": pooled(sel, classes),
            "oof_paired_bootstrap": cv.paired_bootstrap(image_iou(sel), image_iou(ref)),
            "delta_miou_fixed": {f"{t:.2f}": pooled(results.counts[results.index(frozen), :, thresholds.index(t)], classes)
                                 - pooled(results.counts[results.index(reference), :, thresholds.index(t)], classes)
                                 for t in (0.35, 0.45, 0.55)},
            "per_class_delta_full": {c: a[c] - b[c] for c in a},
            "classes_improved_full": int(sum(a[c] > b[c] for c in a)),
            "never_chosen_training_fold_checks": excluded,
        }
    meta = {"images": len(results.image_ids), "commits": list(results.commits), "environments": list(results.environments)}
    (args.package_dir / "diagnostics.json").write_text(json.dumps({**meta, "screens": out}, indent=2) + "\n")
    for name, d in out.items():
        b = d["oof_paired_bootstrap"]
        print(f"{name}: {d['frozen']} OOF {d['oof_miou_reference']:.4f} -> {d['oof_miou_selection']:.4f}; "
              f"paired {b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] {b['better']}/{b['worse']}; "
              f"fixed {', '.join(f'{k}: {v:+.4f}' for k, v in d['delta_miou_fixed'].items())}; "
              f"classes improved {d['classes_improved_full']}/20")


if __name__ == "__main__":
    main()
