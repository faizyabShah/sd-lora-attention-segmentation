#!/usr/bin/env python3
"""Fold-wise method and threshold selection for the propagation package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as base  # noqa: E402


def class_ious(results, configuration, threshold_index, mask):
    counts = results.counts[configuration, mask, threshold_index]
    classes = np.asarray(results.classes)[mask]
    output = {}
    for classname in sorted(set(classes)):
        pooled = counts[classes == classname, :3].sum(axis=0)
        output[classname] = float(pooled[0] / max(pooled.sum(), 1))
    return output


def miou(results, configuration, threshold_index, mask):
    return float(np.mean(list(class_ious(results, configuration, threshold_index, mask).values())))


def best_threshold(results, configuration, mask, threshold_indices=None):
    indices = list(range(len(results.thresholds))) if threshold_indices is None else list(threshold_indices)
    if not indices:
        raise ValueError("At least one eligible threshold is required")
    scores = {i: miou(results, configuration, i, mask) for i in indices}
    index = max(
        indices,
        key=lambda i: (scores[i], -abs(results.thresholds[i] - 0.45), -results.thresholds[i]),
    )
    return index, scores[index]


def assess(results, reference, candidate, ref_t, cand_t, mask):
    ref = class_ious(results, reference, ref_t, mask)
    cand = class_ious(results, candidate, cand_t, mask)
    deltas = {name: cand[name] - ref[name] for name in ref}
    delta = float(np.mean(list(deltas.values())))
    improved, worst = sum(value > 0 for value in deltas.values()), min(deltas.values())
    if delta >= base.PROMOTE:
        verdict = "promote"
    elif (
        delta >= base.CONDITIONAL
        and improved >= base.CONDITIONAL_MIN_CLASSES
        and worst >= -base.CONDITIONAL_MAX_CLASS_LOSS
    ):
        verdict = "conditional"
    else:
        verdict = "marginal"
    return {
        "delta": delta,
        "classes_improved": improved,
        "worst_class_delta": worst,
        "verdict": verdict,
    }


def pooled_miou(counts, classes):
    values = []
    classes = np.asarray(classes)
    for classname in sorted(set(classes)):
        pooled = counts[classes == classname, :3].sum(axis=0)
        values.append(pooled[0] / max(pooled.sum(), 1))
    return float(np.mean(values))


def image_ious(counts):
    union = counts[:, 0] + counts[:, 1] + counts[:, 2]
    return np.where(union > 0, counts[:, 0] / np.maximum(union, 1), 1.0)


def run_screen(results, reference_name, candidates, inclusiveness, threshold_indices=None):
    # Optional preregistered per-cell grids; the default reproduces all existing screens.
    allowed = threshold_indices or {}
    def choose_threshold(index, mask):
        return best_threshold(results, index, mask, allowed.get(results.configurations[index]))
    reference = results.index(reference_name)
    folds = []
    oof = np.zeros((len(results.image_ids), 4), dtype=np.int64)
    oof_ref = np.zeros_like(oof)
    for fold in range(base.FOLD_COUNT):
        held = results.folds == fold
        train = ~held
        ref_t, ref_train = choose_threshold(reference, train)
        assessments = []
        for name in candidates:
            index = results.index(name)
            cand_t, cand_train = choose_threshold(index, train)
            item = assess(results, reference, index, ref_t, cand_t, train)
            assessments.append(
                {"name": name, "index": index, "threshold_index": cand_t,
                 "train_miou": cand_train, **item}
            )
        eligible = [item for item in assessments if item["verdict"] != "marginal"]
        if eligible:
            best = max(item["delta"] for item in eligible)
            near = [item for item in eligible if item["delta"] >= best - base.TIE_MARGIN]
            chosen = max(
                near,
                key=lambda item: (inclusiveness.get(item["name"], 0), item["delta"]),
            )
            chosen_name, chosen_index, chosen_t, reason = (
                chosen["name"], chosen["index"], chosen["threshold_index"], chosen["verdict"]
            )
        else:
            chosen_name, chosen_index, chosen_t, reason = reference_name, reference, ref_t, "no eligible candidate"
        oof[held] = results.counts[chosen_index, held, chosen_t]
        oof_ref[held] = results.counts[reference, held, ref_t]
        folds.append(
            {
                "fold": fold,
                "chosen": chosen_name,
                "chosen_threshold": results.thresholds[chosen_t],
                "reference_threshold": results.thresholds[ref_t],
                "reason": reason,
                "train_miou_chosen": miou(results, chosen_index, chosen_t, train),
                "train_miou_reference": ref_train,
                "heldout_miou_chosen": miou(results, chosen_index, chosen_t, held),
                "heldout_miou_reference": miou(results, reference, ref_t, held),
            }
        )
    votes = {}
    for row in folds:
        votes[row["chosen"]] = votes.get(row["chosen"], 0) + 1
    winner, winner_votes = max(votes.items(), key=lambda item: (item[1], item[0] == reference_name))
    selection_miou = pooled_miou(oof, results.classes)
    reference_miou = pooled_miou(oof_ref, results.classes)
    frozen = (
        winner
        if winner != reference_name
        and winner_votes >= base.FREEZE_MIN_FOLDS
        and selection_miou > reference_miou
        else reference_name
    )
    if winner == reference_name:
        why = "reference won the fold vote"
    elif winner_votes < base.FREEZE_MIN_FOLDS:
        why = f"fewer than {base.FREEZE_MIN_FOLDS} fold votes"
    elif selection_miou <= reference_miou:
        why = "OOF guard failed"
    else:
        why = "fold vote and OOF guard passed"
    full_mask = np.ones(len(results.image_ids), dtype=bool)
    frozen_index = results.index(frozen)
    frozen_t, frozen_full = choose_threshold(frozen_index, full_mask)
    reference_t, reference_full = choose_threshold(reference, full_mask)
    return {
        "reference": reference_name,
        "candidates": candidates,
        "folds": folds,
        "selection_counts": votes,
        "vote_winner": winner,
        "vote_winner_votes": winner_votes,
        "frozen": frozen,
        "freeze_reason": why,
        "oof_miou_selection": selection_miou,
        "oof_miou_reference": reference_miou,
        "oof_gain": selection_miou - reference_miou,
        "full_threshold_frozen": results.thresholds[frozen_t],
        "full_threshold_reference": results.thresholds[reference_t],
        "full_miou_frozen": frozen_full,
        "full_miou_reference": reference_full,
        "paired_oof": base.paired_bootstrap(image_ious(oof), image_ious(oof_ref)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    results = base.load_results(sorted((args.package_dir / "shards").glob("*.npz")))
    if not all(results.complete):
        raise SystemExit("Propagation shards are incomplete")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    reference = spec["reference"]
    screens = {}
    for screen in spec["screens"]:
        candidates = screen.get("candidates")
        if candidates is None:
            candidates = [
                name for name in results.configurations if name.startswith(screen["prefix"])
            ]
        missing = [name for name in candidates if name not in results.configurations]
        if missing:
            raise ValueError(f"{screen['name']}: missing configurations: {missing}")
        screens[screen["name"]] = run_screen(
            results, reference, candidates, screen.get("inclusiveness", {})
        )

    fixed_index = results.thresholds.index(0.45)
    full = np.ones(len(results.image_ids), dtype=bool)
    fixed_rows = {
        name: miou(results, index, fixed_index, full)
        for index, name in enumerate(results.configurations)
    }
    output = {
        "images": len(results.image_ids),
        "classes": len(set(results.classes)),
        "configurations": len(results.configurations),
        "thresholds": list(results.thresholds),
        "screens": screens,
        "fixed_miou@0.45": fixed_rows,
    }
    (args.package_dir / "fold_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Controlled propagation package",
        "",
        f"{len(results.image_ids)} images, {len(set(results.classes))} classes, "
        f"{len(results.configurations)} configurations, {len(results.thresholds)} thresholds.",
        "Method and threshold are selected on four folds and evaluated on the held-out fold.",
        "",
        "| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Full threshold | Full mIoU | Why |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for name, item in screens.items():
        lines.append(
            f"| {name} | `{item['vote_winner']}` | {item['vote_winner_votes']}/5 | "
            f"{item['oof_miou_reference']:.4f} → {item['oof_miou_selection']:.4f} | "
            f"{item['oof_gain']:+.4f} | `{item['frozen']}` | "
            f"{item['full_threshold_frozen']:.3f} | {item['full_miou_frozen']:.4f} | "
            f"{item['freeze_reason']} |"
        )
    for name, item in screens.items():
        lines += ["", f"## {name}", "", "| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |",
                  "|---:|---|---:|---:|---:|---:|---|"]
        for row in item["folds"]:
            lines.append(
                f"| {row['fold']} | `{row['chosen']}` | {row['chosen_threshold']:.3f} | "
                f"{row['reference_threshold']:.3f} | {row['train_miou_chosen']:.4f}/"
                f"{row['train_miou_reference']:.4f} | {row['heldout_miou_chosen']:.4f}/"
                f"{row['heldout_miou_reference']:.4f} | {row['reason']} |"
            )
        ranked = sorted(
            ((configuration, fixed_rows[configuration]) for configuration in item["candidates"]),
            key=lambda pair: -pair[1],
        )
        lines += ["", "Top fixed-threshold configurations:", "", "| Configuration | mIoU@0.45 |", "|---|---:|"]
        for configuration, score in ranked[:8]:
            lines.append(f"| `{configuration}` | {score:.4f} |")
    (args.package_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.package_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
