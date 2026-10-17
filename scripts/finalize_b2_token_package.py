#!/usr/bin/env python3
"""Create compact B2a/S4/B2b VOC and COCO result reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402


def score(results, name, mask=None):
    index = results.index(name)
    return cv.miou(results, index, 0, mask), cv.class_ious(results, index, 0, mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/token_learning_b2"))
    parser.add_argument("--plan", type=Path, default=Path("configs/token_optimization_b2_plan.json"))
    args = parser.parse_args()
    voc = cv.load_results((args.root / "voc_evaluation/shards").glob("*.npz"))
    if len(voc.image_ids) != 490 or not all(voc.complete):
        raise RuntimeError(f"VOC evaluation incomplete: {len(voc.image_ids)} images, complete={voc.complete}")
    rows = {}
    classes = np.asarray(voc.classes)
    for name in voc.configurations:
        mask = classes != "tv" if name in ("A3", "A13", "A23") else None
        value, per_class = score(voc, name, mask)
        rows[name] = {"miou": value, "classes": per_class,
                      "images": int(len(classes) if mask is None else mask.sum())}
    one = np.mean([rows[name]["miou"] for name in ("A1", "A2", "A3")])
    two = np.mean([rows[name]["miou"] for name in ("A12", "A13", "A23")])
    s4 = {"one_shot_mean": float(one), "one_shot_range": [min(rows[n]["miou"] for n in ("A1", "A2", "A3")),
                                                            max(rows[n]["miou"] for n in ("A1", "A2", "A3"))],
          "two_shot_mean": float(two), "two_shot_range": [min(rows[n]["miou"] for n in ("A12", "A13", "A23")),
                                                            max(rows[n]["miou"] for n in ("A12", "A13", "A23"))],
          "all_available": rows["AALL"]["miou"],
          "note": "A3/A13/A23 exclude tv; all_available uses three supports except tv, which has two."}
    rel_i = voc.index("REL")
    diagnostics = {
        "AALL_vs_REL": cv.paired_bootstrap(cv.image_ious(voc, voc.index("AALL"), 0), cv.image_ious(voc, rel_i, 0)),
        "B2B_vs_REL": cv.paired_bootstrap(cv.image_ious(voc, voc.index("B2B"), 0), cv.image_ious(voc, rel_i, 0)),
    }
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    vector_rows = []
    released_dir = Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim_train/tokens")
    for task in plan["b2a"]:
        ours_path = args.root / "b2a/tokens" / f"{task['id']}.npz"
        matches = sorted(p for p in released_dir.glob(f"{task['id']}_*.npy") if not p.stem.endswith("_losses"))
        if not matches:
            continue
        with np.load(ours_path, allow_pickle=False) as data:
            ours = data["embedding"]
        theirs = np.load(matches[0])
        a, b = ours[1], theirs[1]
        vector_rows.append({"id": task["id"], "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                            "l2": float(np.linalg.norm(a - b))})
    output = {"voc": {"images": len(voc.image_ids), "threshold": 0.475,
                       "configurations": rows, "s4": s4, "paired": diagnostics},
              "b2a_vector_reproduction": {
                  "matched": len(vector_rows),
                  "mean_cosine": float(np.mean([r["cosine"] for r in vector_rows])),
                  "mean_l2": float(np.mean([r["l2"] for r in vector_rows])),
                  "per_support": vector_rows,
              }}
    coco_dir = args.root / "coco_evaluation/shards"
    coco_paths = list(coco_dir.glob("*.npz"))
    if coco_paths:
        coco = cv.load_results(coco_paths)
        if len(coco.image_ids) != 707 or not all(coco.complete):
            raise RuntimeError(f"COCO evaluation incomplete: {len(coco.image_ids)} images")
        output["coco"] = {"images": len(coco.image_ids), "threshold": 0.475,
                          "configurations": {name: {"miou": score(coco, name)[0],
                                                     "classes": score(coco, name)[1]}
                                             for name in coco.configurations},
                          "paired_AALL_vs_REL": cv.paired_bootstrap(
                              cv.image_ious(coco, coco.index("AALL"), 0),
                              cv.image_ious(coco, coco.index("REL"), 0))}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# B2 learned-token package", "", "Frozen pipeline: conditional token CA, one SA pass, DenseCRF, threshold 0.475.", "",
             "## VOC-Sim", "", "| Configuration | mIoU | Images |", "|---|---:|---:|"]
    lines += [f"| `{name}` | {item['miou']:.4f} | {item['images']} |" for name, item in rows.items()]
    lines += ["", "### S4 shot scaling", "", "| Supports | mIoU | Range |", "|---|---:|---:|",
              f"| 1 | {one:.4f} | {s4['one_shot_range'][0]:.4f}–{s4['one_shot_range'][1]:.4f} |",
              f"| 2 | {two:.4f} | {s4['two_shot_range'][0]:.4f}–{s4['two_shot_range'][1]:.4f} |",
              f"| all available | {s4['all_available']:.4f} | — |",
              "", f"B2a vector comparison covers {len(vector_rows)} released supports: mean cosine "
              f"{output['b2a_vector_reproduction']['mean_cosine']:.4f}."]
    if "coco" in output:
        lines += ["", "## COCO-Cap transfer", "", "| Configuration | mIoU |", "|---|---:|"]
        lines += [f"| `{name}` | {item['miou']:.4f} |" for name, item in output["coco"]["configurations"].items()]
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print((args.root / "report.md").read_text())


if __name__ == "__main__":
    main()
