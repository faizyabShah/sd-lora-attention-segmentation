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


def comparison(rows, candidate, reference="REL"):
    delta = {name: rows[candidate]["classes"][name] - rows[reference]["classes"][name]
             for name in rows[reference]["classes"]}
    return {
        "miou_gain": rows[candidate]["miou"] - rows[reference]["miou"],
        "classes_improved": sum(value > 0 for value in delta.values()),
        "per_class_delta": delta,
    }


def optimization_stats(root: Path, track: str) -> dict:
    records = []
    for path in sorted((root / track / "tokens").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            records.append({"id": path.stem, "initial_loss": float(data["losses"][0]),
                            "best_loss": float(data["best_loss"]),
                            "best_epoch": int(data["best_epoch"])})
    return {
        "tokens": len(records),
        "mean_initial_loss": float(np.mean([r["initial_loss"] for r in records])),
        "mean_best_loss": float(np.mean([r["best_loss"] for r in records])),
        "best_loss_range": [min(r["best_loss"] for r in records), max(r["best_loss"] for r in records)],
        "median_best_epoch": float(np.median([r["best_epoch"] for r in records])),
        "best_epoch_range": [min(r["best_epoch"] for r in records), max(r["best_epoch"] for r in records)],
    }


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
    # Expected performance at each shot count: average all valid support
    # choices within a class, then macro-average the same 20 classes. ``tv``
    # has only two supports, so it contributes A1/A2 at one shot and A12 at
    # two shots. The displayed choice ranges use the 19 classes common to all
    # three choices, preventing the missing third tv support from changing the
    # class population being compared.
    one_names, two_names = ("A1", "A2", "A3"), ("A12", "A13", "A23")
    class_names = sorted(set(voc.classes))
    one = np.mean([np.mean([rows[name]["classes"][classname]
                            for name in (one_names[:2] if classname == "tv" else one_names)])
                   for classname in class_names])
    two = np.mean([np.mean([rows[name]["classes"][classname]
                            for name in (("A12",) if classname == "tv" else two_names)])
                   for classname in class_names])
    common = np.asarray(voc.classes) != "tv"
    one_choice = [cv.miou(voc, voc.index(name), 0, common) for name in one_names]
    two_choice = [cv.miou(voc, voc.index(name), 0, common) for name in two_names]
    s4 = {"one_shot_expected_miou": float(one), "one_shot_common19_range": [min(one_choice), max(one_choice)],
          "two_shot_expected_miou": float(two), "two_shot_common19_range": [min(two_choice), max(two_choice)],
          "all_available": rows["AALL"]["miou"],
          "note": "Expected mIoU averages valid choices within each class, then all 20 classes. Choice ranges use the common 19 classes. All-available uses three supports except tv, which has two."}
    rel_i = voc.index("REL")
    diagnostics = {
        "AALL_vs_REL": cv.paired_bootstrap(cv.image_ious(voc, voc.index("AALL"), 0), cv.image_ious(voc, rel_i, 0)),
        "B2B_vs_REL": cv.paired_bootstrap(cv.image_ious(voc, voc.index("B2B"), 0), cv.image_ious(voc, rel_i, 0)),
    }
    comparisons = {name: comparison(rows, name) for name in ("AALL", "B2B")}
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
                  "median_cosine": float(np.median([r["cosine"] for r in vector_rows])),
                  "cosine_range": [min(r["cosine"] for r in vector_rows),
                                   max(r["cosine"] for r in vector_rows)],
                  "mean_l2": float(np.mean([r["l2"] for r in vector_rows])),
                  "per_support": vector_rows,
              },
              "optimization": {track: optimization_stats(args.root, track) for track in ("b2a", "b2b")}}
    output["voc"]["comparisons"] = comparisons
    coco_dir = args.root / "coco_evaluation/shards"
    coco_paths = list(coco_dir.glob("*.npz"))
    if coco_paths:
        coco = cv.load_results(coco_paths)
        if len(coco.image_ids) != 707 or not all(coco.complete):
            raise RuntimeError(f"COCO evaluation incomplete: {len(coco.image_ids)} images")
        coco_rows = {name: {"miou": score(coco, name)[0], "classes": score(coco, name)[1]}
                     for name in coco.configurations}
        output["coco"] = {"images": len(coco.image_ids), "threshold": 0.475,
                          "configurations": coco_rows,
                          "comparison": comparison(coco_rows, "AALL"),
                          "paired_AALL_vs_REL": cv.paired_bootstrap(
                              cv.image_ious(coco, coco.index("AALL"), 0),
                              cv.image_ious(coco, coco.index("REL"), 0))}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# B2 learned-token package", "", "Frozen pipeline: conditional token CA, one SA pass, DenseCRF, threshold 0.475.", "",
             "## VOC-Sim", "", "| Configuration | mIoU | Images |", "|---|---:|---:|"]
    lines += [f"| `{name}` | {item['miou']:.4f} | {item['images']} |" for name, item in rows.items()]
    lines += ["", "### S4 shot scaling", "", "| Supports | mIoU | Range |", "|---|---:|---:|",
              f"| 1 | {one:.4f} | {s4['one_shot_common19_range'][0]:.4f}–{s4['one_shot_common19_range'][1]:.4f} |",
              f"| 2 | {two:.4f} | {s4['two_shot_common19_range'][0]:.4f}–{s4['two_shot_common19_range'][1]:.4f} |",
              f"| all available | {s4['all_available']:.4f} | — |",
              "", f"B2a vector comparison covers {len(vector_rows)} released supports: mean cosine "
              f"{output['b2a_vector_reproduction']['mean_cosine']:.4f}."]
    lines += ["", "### Primary VOC comparisons", "",
              "| Comparison | mIoU change | Improved classes | Paired mean image-IoU change [95% CI] | Better/worse images |",
              "|---|---:|---:|---:|---:|"]
    for name in ("AALL", "B2B"):
        comp, paired = comparisons[name], diagnostics[f"{name}_vs_REL"]
        lines.append(f"| {name} vs REL | {comp['miou_gain']:+.4f} | {comp['classes_improved']}/20 | "
                     f"{paired['mean']:+.4f} [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] | "
                     f"{paired['better']}/{paired['worse']} |")
    lines += ["", "| VOC class | REL | AALL | Δ | B2B | Δ |", "|---|---:|---:|---:|---:|---:|"]
    for classname in class_names:
        lines.append(f"| {classname} | {rows['REL']['classes'][classname]:.4f} | "
                     f"{rows['AALL']['classes'][classname]:.4f} | "
                     f"{comparisons['AALL']['per_class_delta'][classname]:+.4f} | "
                     f"{rows['B2B']['classes'][classname]:.4f} | "
                     f"{comparisons['B2B']['per_class_delta'][classname]:+.4f} |")
    if "coco" in output:
        lines += ["", "## COCO-Cap transfer", "", "| Configuration | mIoU |", "|---|---:|"]
        lines += [f"| `{name}` | {item['miou']:.4f} |" for name, item in output["coco"]["configurations"].items()]
        comp, paired = output["coco"]["comparison"], output["coco"]["paired_AALL_vs_REL"]
        lines += ["", f"AALL changes mIoU by {comp['miou_gain']:+.4f} and improves {comp['classes_improved']}/20 classes. "
                  f"Paired mean image-IoU change: {paired['mean']:+.4f} "
                  f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]; "
                  f"{paired['better']} better, {paired['worse']} worse, {paired['ties']} ties.",
                  "", "| COCO class | REL | AALL | Δ |", "|---|---:|---:|---:|"]
        for classname in sorted(output["coco"]["configurations"]["REL"]["classes"]):
            lines.append(f"| {classname} | {output['coco']['configurations']['REL']['classes'][classname]:.4f} | "
                         f"{output['coco']['configurations']['AALL']['classes'][classname]:.4f} | "
                         f"{comp['per_class_delta'][classname]:+.4f} |")
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print((args.root / "report.md").read_text())


if __name__ == "__main__":
    main()
