#!/usr/bin/env python3
"""Token confirmation T-C (phase 3 registry Section 2 / Section 4 step 4).

Selects the few-shot pipeline from the token package's committed shards (no
new generation): the conditional released token on REF, P1, R2 and P1+R2, with
the P3-C mandatory structure and tie order. Writes ``fold_results.json`` and
``diagnostics.json`` to ``outputs/segmentation/controlled_attention_cv/token_confirmation/``.

    python scripts/narmyn/attention/token_confirmation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

SPEC = ROOT / "configs" / "controlled_token_confirmation.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "token_confirmation"


def main() -> None:
    spec = json.loads(SPEC.read_text())
    results = cv.load_results(sorted((ROOT / spec["source_package"] / "shards").glob("*.npz")))
    thresholds = list(results.thresholds)
    classes = np.asarray(results.classes)

    def pooled(counts, mask):
        return float(np.mean([counts[classes[mask] == c, 0].sum() / counts[classes[mask] == c, :3].sum() for c in sorted(set(classes))]))

    def crossfit(config, reference):
        """Fixed-candidate cross-fit: each cell's threshold chosen on the training folds, scored on the held-out fold."""
        out = {}
        for name in (config, reference):
            i, counts = results.index(name), np.zeros((len(classes), 4), np.int64)
            for k in sorted(set(results.folds.tolist())):
                train, held = results.folds != k, results.folds == k
                t = max(range(len(thresholds)), key=lambda t: (pooled(results.counts[i, train, t], train),
                                                                 -abs(thresholds[t] - 0.45), -thresholds[t]))
                counts[held] = results.counts[i, held, t]
            out[name] = counts
        iou = lambda c: c[:, 0] / np.maximum(c[:, :3].sum(axis=1), 1)  # noqa: E731
        everyone = np.ones(len(classes), bool)
        a = {c: out[config][classes == c, 0].sum() / out[config][classes == c, :3].sum() for c in sorted(set(classes))}
        b = {c: out[reference][classes == c, 0].sum() / out[reference][classes == c, :3].sum() for c in sorted(set(classes))}
        return {"configuration": config, "reference": reference, "oof_miou_reference": pooled(out[reference], everyone),
                "oof_miou_configuration": pooled(out[config], everyone),
                "paired_bootstrap": cv.paired_bootstrap(iou(out[config]), iou(out[reference])),
                "per_class_delta": {c: float(a[c] - b[c]) for c in a}}

    screens = {s["name"]: run_screen(results, spec["reference"], s["candidates"], s.get("inclusiveness", {})) for s in spec["screens"]}
    diagnostics = [crossfit(c, r) for c, r in (("TC/R2", "TC/REF"), ("TC/P1+R2", "TC/REF"), ("TC/P1+R2", "TC/R2"),
                                               ("TC/R2", "W/P1+R2"), ("TC/P1+R2", "W/P1+R2"))]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"source_commits": list(results.commits), "images": len(results.image_ids), "spec": str(SPEC.relative_to(ROOT))}
    (OUT / "fold_results.json").write_text(json.dumps({**meta, "screens": screens}, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps({**meta, "crossfit": diagnostics}, indent=2) + "\n")
    for name, s in screens.items():
        print(f"{name}: {s['frozen']} {s['vote_winner_votes']}/5, OOF {s['oof_miou_reference']:.4f} -> {s['oof_miou_selection']:.4f}")
    for d in diagnostics:
        b = d["paired_bootstrap"]
        print(f"  {d['configuration']} vs {d['reference']}: {d['oof_miou_reference']:.4f} -> {d['oof_miou_configuration']:.4f}, "
              f"paired {b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
