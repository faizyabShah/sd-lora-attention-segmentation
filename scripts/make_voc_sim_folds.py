#!/usr/bin/env python3
"""Write configs/voc_sim_folds.csv: 5 class-stratified folds of the annotated VOC-Sim images.

Protocol Section 2. The assignment depends only on the manifest and the fixed
seed, so re-running this script reproduces the committed file exactly; use
``--check`` to verify that.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.controlled_cv import FOLD_COUNT, FOLD_SEED, make_folds, read_folds, write_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--seed", type=int, default=FOLD_SEED)
    parser.add_argument("--check", action="store_true", help="verify the existing file instead of writing")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        assignments = make_folds(csv.DictReader(handle), FOLD_COUNT, args.seed)
    if args.check:
        existing = read_folds(args.output)
        expected = {str(r["image_id"]): int(r["fold"]) for r in assignments}
        if existing != expected:
            raise SystemExit(f"{args.output} does not match the manifest and seed {args.seed}")
        print(f"{args.output} matches ({len(existing)} images)")
        return
    write_folds(args.output, assignments)
    sizes = Counter(int(r["fold"]) for r in assignments)
    per_class = Counter((str(r["classname"]), int(r["fold"])) for r in assignments)
    spread = max(per_class.values()) - min(per_class.values())
    print(f"Wrote {len(assignments)} images to {args.output}; fold sizes {dict(sorted(sizes.items()))}; "
          f"per-class per-fold counts range {min(per_class.values())}-{max(per_class.values())} (spread {spread})")


if __name__ == "__main__":
    main()
