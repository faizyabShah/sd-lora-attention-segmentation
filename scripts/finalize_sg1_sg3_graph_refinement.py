#!/usr/bin/env python3
"""Select SG1/SG3 on VOC folds and report untouched COCO transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

OUT = ROOT / "outputs/segmentation/controlled_attention_cv/sg1_sg3_graph_refinement"


def load(dataset: str) -> cv.Results:
    results = cv.load_results(sorted((OUT / dataset / "shards").glob("*.npz")))
    if not all(results.complete) or len(set(results.classes)) != 20:
        raise SystemExit(f"{dataset}: incomplete package")
    return results


def main() -> None:
    voc, coco = load("voc"), load("coco")
    screens = {}
    for attribution in ("W", "T"):
        for family in ("SG1", "SG3"):
            candidates = [n for n in voc.configurations if n.startswith(f"{attribution}/{family}/")]
            inclusiveness = {n: (2 if "_r0" in n else 1) + (1 if n.endswith("_n1") else 0) for n in candidates}
            screens[f"{attribution}-{family}"] = run_screen(
                voc, f"{attribution}/REF", candidates, inclusiveness
            )
    full_voc = np.ones(len(voc.image_ids), dtype=bool)
    coco_classes = np.asarray(coco.classes)
    transfer = {}
    for key, screen in screens.items():
        reference, frozen = screen["reference"], screen["frozen"]
        ref_t = best_threshold(voc, voc.index(reference), full_voc)[0]
        frozen_t = best_threshold(voc, voc.index(frozen), full_voc)[0]
        a = coco.counts[coco.index(frozen), :, frozen_t]
        b = coco.counts[coco.index(reference), :, ref_t]
        transfer[key] = {
            "frozen": frozen, "reference": reference,
            "coco_miou_frozen": pooled_miou(a, coco_classes),
            "coco_miou_reference": pooled_miou(b, coco_classes),
            "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b)),
        }
    t45 = voc.thresholds.index(0.45)
    diagnostics = {}
    for dataset, results in (("voc", voc), ("coco", coco)):
        classes = np.asarray(results.classes)
        diagnostics[dataset] = {}
        for name in results.configurations:
            counts = results.counts[results.index(name), :, t45, :3].sum(axis=0)
            tp, fp, fn = (int(v) for v in counts)
            diagnostics[dataset][name] = {
                "miou@0.45": pooled_miou(results.counts[results.index(name), :, t45], classes),
                "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            }
    output = {"protocol": "Section 68", "voc_images": len(voc.image_ids),
              "coco_images": len(coco.image_ids), "screens": screens,
              "coco_transfer": transfer, "diagnostics": diagnostics}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# SG1/SG3 independent graph screen", "",
             f"VOC-Sim {len(voc.image_ids)} images; COCO-Cap {len(coco.image_ids)} images.", "",
             "| Screen | Winner | Votes | OOF reference → selection | Gain | Frozen | COCO frozen/reference |",
             "|---|---|---:|---:|---:|---|---:|"]
    for key, screen in screens.items():
        tr = transfer[key]
        lines.append(f"| {key} | `{screen['vote_winner']}` | {screen['vote_winner_votes']}/5 | "
                     f"{screen['oof_miou_reference']:.4f} → {screen['oof_miou_selection']:.4f} | "
                     f"{screen['oof_gain']:+.4f} | `{screen['frozen']}` | "
                     f"{tr['coco_miou_frozen']:.4f}/{tr['coco_miou_reference']:.4f} |")
    lines += ["", "Recall and precision for every cell are in `summary.json`. DenseCRF, SG5 interaction, and spatial AN1 diagnostics are deliberately deferred to the locked confirmation after owner review."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
