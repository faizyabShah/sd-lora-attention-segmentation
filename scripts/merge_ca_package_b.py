#!/usr/bin/env python3
"""Merge CA package B: cross-validated selection, head report, exhaustive search, figures.

Reads ``<package>/shards/*.npz`` (CA-L, CA-H, CA-A per image) and
``<package>/exhaustive_raw/*.npy`` (65,535 module subsets at 0.45), applies the
protocol's Section 5/5a procedure, and writes ``fold_results.json``,
``summary.json``, ``report.md``, ``head_report.json``, ``exhaustive_cv.json``
and ``figures/`` in the package directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_controlled_cv import report as selector_report  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

HEADS = 8
FULL_MASK = (1 << 16) - 1


def head_name(subset) -> str:
    return "H/" + "_".join(str(h) for h in sorted(subset))


def head_report(results: cv.Results, cv_heads: dict) -> dict:
    """Section 4 required head report (in-sample unless stated)."""
    score = lambda subset: cv.miou(results, results.index(head_name(subset)))  # noqa: E731
    full = tuple(range(HEADS))
    singles = {h: score((h,)) for h in full}
    leave_one = {h: score(tuple(x for x in full if x != h)) for h in full}

    backward, current = [{"heads": list(full), "removed": None, "miou": score(full)}], list(full)
    while len(current) > 1:
        options = {h: score(tuple(x for x in current if x != h)) for h in current}
        drop = max(options, key=options.get)
        current.remove(drop)
        backward.append({"heads": list(current), "removed": drop, "miou": options[drop]})

    first = max(singles, key=singles.get)
    forward, current = [{"heads": [first], "added": first, "miou": singles[first]}], [first]
    while len(current) < HEADS:
        options = {h: score(tuple(sorted(current + [h]))) for h in full if h not in current}
        add = max(options, key=options.get)
        current.append(add)
        forward.append({"heads": sorted(current), "added": add, "miou": options[add]})

    ranking = sorted(
        (c for c in results.configurations if c.startswith("H/")),
        key=lambda c: -cv.miou(results, results.index(c)),
    )
    retained = {h: 0 for h in full}
    for fold in cv_heads["folds"]:
        for h in map(int, fold["chosen"].split("/")[1].split("_")):
            retained[h] += 1
    frozen = cv_heads["frozen"]
    return {
        "note": "in-sample (all 490 images) except fold retention",
        "reference_miou": score(full),
        "single_head_miou": singles,
        "leave_one_out_miou": leave_one,
        "greedy_backward_path": backward,
        "greedy_forward_path": forward,
        "frozen_subset": frozen,
        "frozen_rank_among_255": ranking.index(frozen) + 1,
        "best_in_sample_subset": ranking[0],
        "best_in_sample_miou": cv.miou(results, results.index(ranking[0])),
        "folds_retaining_each_head": retained,
    }


def exhaustive_cv(results: cv.Results, raw_dir: Path) -> dict:
    """Section 5a on all 65,535 module subsets (exploratory; not eligible for freezing)."""
    ids, classes, folds = results.image_ids, np.array(results.classes), results.folds
    names = sorted(set(classes))
    class_index = np.array([names.index(c) for c in classes])
    sums = None  # [fold, class, subset, (tp, fp, fn)]
    for i, image_id in enumerate(ids):
        counts = np.load(raw_dir / f"{image_id}.npy").astype(np.int64)[:, :3]
        if sums is None:
            sums = np.zeros((cv.FOLD_COUNT, len(names), counts.shape[0], 3), dtype=np.int64)
        sums[folds[i], class_index[i]] += counts
    subsets = sums.shape[2]
    popcount = np.array([bin(m).count("1") for m in range(1, subsets + 1)])
    ref = FULL_MASK - 1

    def class_iou(block):  # block [class, subset, 3]
        union = block.sum(axis=-1)
        return np.where(union > 0, block[..., 0] / np.maximum(union, 1), 1.0)

    total = sums.sum(axis=0)
    per_fold = []
    oof_ref = total[:, ref]
    chosen_per_fold = []
    for k in range(cv.FOLD_COUNT):
        train = class_iou(total - sums[k])
        delta_c = train - train[:, ref : ref + 1]
        delta = delta_c.mean(axis=0)
        improved = (delta_c > 0).sum(axis=0)
        worst = delta_c.min(axis=0)
        eligible = (delta >= cv.PROMOTE) | (
            (delta >= cv.CONDITIONAL) & (improved >= cv.CONDITIONAL_MIN_CLASSES) & (worst >= -cv.CONDITIONAL_MAX_CLASS_LOSS)
        )
        eligible[ref] = False
        if eligible.any():
            best = delta[eligible].max()
            near = np.flatnonzero(eligible & (delta >= best - cv.TIE_MARGIN))
            chosen = near[np.lexsort((-delta[near], -popcount[near]))[0]]
        else:
            chosen = ref
        chosen_per_fold.append(int(chosen))
        held = class_iou(sums[k])
        per_fold.append({
            "fold": k, "chosen_mask": hex(chosen + 1),
            "chosen_modules": [results.settings["modules"][b] for b in range(16) if (chosen + 1) >> b & 1],
            "train_delta": float(delta[chosen]), "heldout_miou_chosen": float(held[:, chosen].mean()),
            "heldout_miou_reference": float(held[:, ref].mean()),
        })
    oof_counts = sum(sums[k][:, chosen_per_fold[k]] for k in range(cv.FOLD_COUNT))
    oof_ref_counts = oof_ref
    oof_sel = float(class_iou(oof_counts[:, None])[:, 0].mean())
    oof_reference = float(class_iou(oof_ref_counts[:, None])[:, 0].mean())
    votes = {}
    for c in chosen_per_fold:
        votes[hex(c + 1)] = votes.get(hex(c + 1), 0) + 1
    in_sample = class_iou(total).mean(axis=0)
    top = np.argsort(-in_sample)[:10]
    return {
        "note": "exploratory diagnostic (protocol Section 4 CA-L): the exhaustive maximum is not eligible "
                "for automatic selection; threshold 0.45 only",
        "subsets": int(subsets),
        "reference_mask": hex(FULL_MASK),
        "folds": per_fold,
        "selection_counts": votes,
        "out_of_fold_miou_selection": oof_sel,
        "out_of_fold_miou_reference": oof_reference,
        "oof_gain": oof_sel - oof_reference,
        "in_sample_miou_reference": float(in_sample[ref]),
        "in_sample_top10": [
            {"mask": hex(m + 1), "modules": int(popcount[m]), "miou": float(in_sample[m])} for m in top
        ],
    }


def render_figures(package: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    out = package / "figures"
    out.mkdir(exist_ok=True)
    for path in sorted((package / "figure_maps").glob("*.npz")):
        data = np.load(path)
        modules = [str(m) for m in data["modules"]]
        maps = data["module_maps"].astype(np.float32)
        lo, hi = float(maps.min()), float(maps.max())

        def resolution(name: str) -> int:
            if name.startswith("mid_block"):
                return 8
            level = int(name.split(".")[1])
            return {0: 64, 1: 32, 2: 16}[level] if name.startswith("down") else {3: 64, 2: 32, 1: 16}[level]

        rows = [64, 32, 16, 8]
        fig, axes = plt.subplots(len(rows), 6, figsize=(15, 10.5))
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].imshow(data["image"]); axes[0, 0].set_title("generated image", fontsize=8)
        axes[1, 0].imshow(data["target"], cmap="gray"); axes[1, 0].set_title("annotation", fontsize=8)
        for r, res in enumerate(rows):
            members = sorted(i for i, m in enumerate(modules) if resolution(m) == res)
            for c, i in enumerate(members):
                ax = axes[r, c + 1]
                ax.imshow(maps[i], vmin=lo, vmax=hi, cmap="viridis")
                label = modules[i].replace(".transformer_blocks.0.attn2", "").replace("_blocks", "").replace("attentions.", "a")
                ax.set_title(f"{i:02d} {label} ({res}×{res})", fontsize=7)
        fig.suptitle(f"{path.stem}: head-averaged CA per module, calls 0–30, common colour scale", fontsize=10)
        fig.tight_layout()
        target = out / f"{path.stem}_ca_modules.png"
        fig.savefig(target, dpi=110)
        plt.close(fig)
        written.append(str(target))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    package = args.package_dir
    results = cv.load_results(sorted((package / "shards").glob("*.npz")))
    if (len(results.image_ids) != 490 or not all(results.complete)) and not args.allow_incomplete:
        raise SystemExit(f"{len(results.image_ids)} images, {sum(results.complete)}/{len(results.complete)} shards complete")
    meta = json.loads((package / "configurations.json").read_text())
    inclusiveness = meta["inclusiveness"]
    reference = results.settings["reference"]
    screens = {
        "CA-L": cv.cross_validate(results, reference["L"],
                                  [c for c in results.configurations if c.startswith("L/") and c != reference["L"]],
                                  inclusiveness),
        "CA-H": cv.cross_validate(results, reference["H"],
                                  [c for c in results.configurations if c.startswith("H/") and c != reference["H"]],
                                  inclusiveness),
        "CA-A": cv.cross_validate(results, reference["A"],
                                  [c for c in results.configurations if c.startswith("A/") and c != reference["A"]]),
    }
    rows = {name: cv.summary_row(results, i) for i, name in enumerate(results.configurations)}
    heads = head_report(results, screens["CA-H"])
    exhaustive = exhaustive_cv(results, package / "exhaustive_raw") if (package / "exhaustive_raw").exists() else None
    figures = render_figures(package)
    meta_out = {"merge_commit": cv.git_commit(), "shard_commits": list(results.commits),
                "images": len(results.image_ids), "configurations": len(results.configurations)}
    (package / "fold_results.json").write_text(json.dumps({**meta_out, "screens": screens}, indent=2) + "\n")
    (package / "summary.json").write_text(json.dumps({**meta_out, "configurations": rows}, indent=2) + "\n")
    (package / "head_report.json").write_text(json.dumps(heads, indent=2) + "\n")
    if exhaustive is not None:
        (package / "exhaustive_cv.json").write_text(json.dumps(exhaustive, indent=2) + "\n")

    lines = [selector_report(results, screens, rows).replace("# Cross-validated selection", "# CA package B: CA-L, CA-H, CA-A"), "## CA-H head report", ""]
    lines += ["| Head | Alone | Leave this head out | Folds retaining it |", "|---:|---:|---:|---:|"]
    for h in range(HEADS):
        lines.append(f"| {h} | {heads['single_head_miou'][h]:.4f} | {heads['leave_one_out_miou'][h]:.4f} | "
                     f"{heads['folds_retaining_each_head'][h]}/5 |")
    lines += ["", f"All 8 heads: {heads['reference_miou']:.4f}. Frozen subset `{heads['frozen_subset']}` ranks "
              f"{heads['frozen_rank_among_255']} of 255 in-sample; best in-sample subset `{heads['best_in_sample_subset']}` "
              f"({heads['best_in_sample_miou']:.4f}).", "", "Greedy backward elimination (in-sample):", "",
              "| Round | Removed | Heads left | mIoU |", "|---:|---:|---|---:|"]
    for r, step in enumerate(heads["greedy_backward_path"]):
        lines.append(f"| {r} | {step['removed'] if step['removed'] is not None else '—'} | "
                     f"{' '.join(map(str, step['heads']))} | {step['miou']:.4f} |")
    lines += ["", "Greedy forward selection (in-sample):", "", "| Round | Added | Heads | mIoU |", "|---:|---:|---|---:|"]
    for r, step in enumerate(heads["greedy_forward_path"]):
        lines.append(f"| {r} | {step['added']} | {' '.join(map(str, step['heads']))} | {step['miou']:.4f} |")
    if exhaustive is not None:
        lines += ["", "## Exhaustive module subsets (exploratory, not eligible for freezing)", "",
                  f"Out-of-fold mIoU of the selection procedure {exhaustive['out_of_fold_miou_selection']:.4f} vs "
                  f"all-16 reference {exhaustive['out_of_fold_miou_reference']:.4f} (gain {exhaustive['oof_gain']:+.4f}); "
                  f"fold votes {exhaustive['selection_counts']}.", "",
                  "| Fold | Chosen mask | Modules | Train Δ | Held-out mIoU chosen / ref |", "|---:|---|---:|---:|---|"]
        for f in exhaustive["folds"]:
            lines.append(f"| {f['fold']} | `{f['chosen_mask']}` | {len(f['chosen_modules'])} | {f['train_delta']:+.4f} | "
                         f"{f['heldout_miou_chosen']:.4f} / {f['heldout_miou_reference']:.4f} |")
        lines += ["", "In-sample top 5: " + ", ".join(f"`{t['mask']}` ({t['modules']} modules, {t['miou']:.4f})"
                                                     for t in exhaustive["in_sample_top10"][:5]) + "."]
    if figures:
        lines += ["", "## Figures", ""] + [f"- `{Path(f).relative_to(package)}`" for f in figures]
    (package / "report.md").write_text("\n".join(lines) + "\n")
    print((package / "report.md").read_text())


if __name__ == "__main__":
    main()
