#!/usr/bin/env python3
"""Freeze B2a supports and seeded, fold-safe B2b support choices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "dining-table", "dog", "horse", "motorbike", "person",
               "potted-plant", "sheep", "sofa", "train", "tv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/token_optimization_b2_plan.json"))
    parser.add_argument("--selection-seed", type=int, default=20261017)
    args = parser.parse_args()
    train = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim_train"
    b2a = []
    for classname in VOC_CLASSES:
        key = classname
        for annotation in sorted((train / "annotations").glob(f"sd-15_{key}_*.png")):
            seed = int(annotation.stem.rsplit("_", 1)[1])
            b2a.append({"id": annotation.stem, "classname": classname,
                        "prompt": f"A photograph of a {classname.replace('-', ' ')}", "seed": seed,
                        "annotation": str(annotation.relative_to(ROOT))})

    manifest = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["image_id"]: row for row in csv.DictReader(handle)
                if row["annotated"].lower() == "true"}
    with (ROOT / "configs/voc_sim_folds.csv").open(newline="", encoding="utf-8") as handle:
        folds = {row["image_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    rng = np.random.default_rng(args.selection_seed)
    b2b = []
    for heldout in range(5):
        for classname in VOC_CLASSES:
            eligible = sorted(image_id for image_id, row in rows.items()
                              if row["classname"] == classname and folds[image_id] != heldout)
            chosen = eligible[int(rng.integers(len(eligible)))]
            row = rows[chosen]
            b2b.append({"id": f"fold{heldout}_{classname.replace(' ', '-')}",
                        "heldout_fold": heldout, "classname": classname,
                        "support_image_id": chosen, "support_fold": folds[chosen],
                        "prompt": row["prompt"], "seed": int(row["seed"]),
                        "annotation": str((manifest.parent / row["annotation"]).relative_to(ROOT))})
    payload = {
        "version": 1, "selection_seed": args.selection_seed,
        "recipe": {"model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "steps": 30,
                   "epochs": 800, "optimizer": "SGD", "initial_lr": 100.0,
                   "scheduler": "StepLR", "step_size": 120, "gamma": 0.7,
                   "loss": "CrossEntropyLoss", "normalization": "global min-max",
                   "attention": "OVAM linear DAAM, unconditional queries, no middle block"},
        "b2a": b2a, "b2b": b2b,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(b2a)} B2a and {len(b2b)} B2b tasks to {args.output}")


if __name__ == "__main__":
    main()

