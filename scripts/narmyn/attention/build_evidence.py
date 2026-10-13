#!/usr/bin/env python3
"""Build the evidence package: every experiment, every configuration, and the figures.

Reads the committed result files under outputs/segmentation (plus, if present,
the local exhaustive_raw counts of CA package B) and writes
docs/narmyn/attention/evidence/:

  ledger.csv                    one row per experiment
  all_configurations.csv.gz     one row per evaluated configuration, all experiments
  controlled_ca_package_b.csv   the 426 controlled configurations with folds and calibration
  tables/*.md, tables/*.tex     the tables used in EVIDENCE.md and the report
  figures/*.png, figures/*.pdf  the figures used in EVIDENCE.md and the report

Nothing is re-run; every number comes from a committed result file.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import gzip
import itertools
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402

SEG = ROOT / "outputs" / "segmentation"
OUT = ROOT / "docs" / "narmyn" / "attention" / "evidence"
PKG_B = SEG / "controlled_attention_cv" / "ca_layer_head_attribution"

# Reference palette (dataviz skill, light mode; static figures for print).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")  # worse (red) - neutral gray - better (blue)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def leaves(node, path=()):
    if isinstance(node, dict):
        if "miou" in node and isinstance(node["miou"], (int, float)):
            yield path, node
            return
        for key, value in node.items():
            yield from leaves(value, path + (key,))


def shard_mious(folder: Path) -> tuple[dict[str, float], int, int]:
    """mIoU per configuration from per-class shard results (pooled class IoU, macro mean)."""
    shards = sorted(folder.glob("class_*/results.json"))
    complete = sum((s.parent / "COMPLETE").exists() for s in shards)
    per = defaultdict(dict)
    for shard in shards:
        for name, value in load_json(shard)["results"].items():
            per[name][shard.parent.name] = value["iou"]
    return {k: float(np.mean(list(v.values()))) for k, v in per.items()}, len(shards), complete


# --------------------------------------------------------------------------- #
# Experiment ledger (tier B: exploratory, pre-protocol; tier A: controlled)
# --------------------------------------------------------------------------- #
DEV = "VOC-Sim dev subset (99 images, 4 classes; inside the 490)"
VOC = "VOC-Sim (490 images, 20 classes)"
COCO = "COCO-Cap (707 images, 20 classes)"
EXPERIMENTS = [
    # id, title, dataset, status, varied, fixed, source (relative to outputs/segmentation), family depth, role
    ("B1", "CA time windows and resolutions", DEV, "exploratory, in-sample",
     "denoising-call windows; CA resolutions", "conditional CA, min-max, threshold 0.40",
     "voc_sim_100_ablations/all_results.json", 3, "first look at time and resolution"),
    ("B2", "Single and leave-out CA layers", DEV, "exploratory, in-sample", "CA layers", "as B1",
     "voc_sim_100_cross_attention_layers/results.json", 1, "layer screening"),
    ("B3", "Joint time x resolution", DEV, "exploratory, in-sample", "windows x resolutions", "as B1",
     "voc_sim_100_joint_time_resolution/results.json", 1, "interaction check"),
    ("B4", "Exhaustive CA layer subsets", DEV, "exploratory, in-sample", "all 65,535 subsets of 16 CA layers",
     "as B1", "voc_sim_100_exhaustive_layer_subsets/summary.json", None,
     "chose BEST_12 = layers {0,2,5,6,7,9,10,11,12,13,14,15}"),
    ("B5", "Threshold x normalization", DEV, "exploratory, in-sample",
     "threshold 0.10-0.90; max / min-max / percentile", "BEST_12, all-16, OVAM-15 layers",
     "voc_sim_100_threshold_normalization/results.json", 1, "chose percentile 1-99 and threshold 0.45"),
    ("B6", "Stage-1 suite (heads, time-layer, fusion, cleanup)", DEV, "exploratory, in-sample",
     "head subsets, time x layer, fusion weights, morphology", "BEST_12, percentile, 0.45",
     "voc_sim_100_stage1_suite/results.json", 2, "chose the fixed top-4 CA heads per layer"),
    ("B7", "Self-attention screens and progressive SA", DEV, "exploratory, in-sample",
     "SA resolution sequences, passes, residual, seed", "B6 CA seed", "voc_sim_100_self_attention", None,
     "first evidence that SA propagation helps"),
    ("B8", "Full streaming ablations", VOC, "exploratory, in-sample (choices from B1-B7 frozen)",
     "26 families: layers, resolutions, time, thresholds, stage-1, SA at 8/16/32/64, progressive SA",
     "dev-set choices", "voc_sim_full_streaming/all_results.json.gz", 2,
     "SA at U3.2 (64x64), 4 passes: 0.7777"),
    ("B9", "Focused SA follow-up", VOC, "exploratory, in-sample",
     "CA seed, seed operator, SA heads, SA time, output calibration, cleanup", "B8 SA module",
     "voc_sim_full_focused_sa_p1_p4/all_results.json.gz", 3, "top-4-head seed, 2 passes, t=0.45: 0.7922"),
    ("B10", "Joint SA head x time x passes x threshold", VOC, "exploratory, in-sample",
     "passes 1-4, 13 SA head sets, 15 time weightings, 5 thresholds, cleanup", "top-4 seed, U3.2",
     "voc_sim_full_joint_sa_head_time/all_results.json.gz", 3,
     "final method: 4 passes, heads {0,1,2,4,5,6}, calls 15-30, t=0.45, 1% cleanup: 0.7997"),
    ("B11", "Component ablation of the final method", f"{VOC}; {COCO}", "exploratory (COCO untuned)",
     "remove one component at a time", "final method (B10)", "final_component_ablation", "shards",
     "SA propagation is the largest component"),
    ("B12", "COCO-Cap generalization of the final method", COCO, "untuned transfer",
     "OVAM vs frozen final method", "final method (B10)", "coco_cap_generalization/results.json", 1,
     "0.5719 -> 0.6080"),
    ("B13", "Revised system (1 pass, all 16 CA layers) and its ablation", f"{VOC}; {COCO}",
     "exploratory; informed by COCO-Cap", "remove one component at a time", "1 pass, all layers",
     "revised_system_component_ablation", "shards", "revised system: VOC 0.7824, COCO 0.6322"),
    ("B14", "CA call windows with all SA heads", f"{VOC}; {COCO}", "exploratory",
     "CA calls 10-30 / 5-30", "revised system", "ca10_30_all_sa_heads|ca5_30_selected_ca_all_sa_heads", "shards",
     "no gain over all calls"),
    ("B15", "Clean CA-only screens (pre-sync)", VOC, "exploratory, in-sample; audited, superseded by A1",
     "time, resolution, time x resolution, layers, 255 head subsets, exhaustive layers",
     "conditional, all pieces, min-max, 0.45", "voc_sim_clean_ca_ablations/all_results.json.gz", 2,
     "ratified as correct; re-run with folds as A1"),
    ("B16", "OVAM released masks", VOC, "reference (authors' masks)", "unoptimized / token-optimized / CA-only",
     "authors' renders", "ovam_paper_baseline", None, "0.7036 / 0.8247 / 0.6616"),
    ("B17", "OVAM optimized token and DenseCRF on the final method", f"{VOC}; {COCO}",
     "exploratory; nothing selected", "class word vs optimized token; DenseCRF; 1 vs 4 passes; CA selection",
     "earlier final method", "narmyn_token_crf", None, "token + DenseCRF: VOC 0.8525, COCO 0.6612"),
    ("A1", "Controlled CA package B: CA-L, CA-H, CA-A", VOC, "controlled, 5-fold cross-validated",
     "159 layer sets, 255 head subsets, 12 attribution conventions; 65,535 exhaustive (exploratory)",
     "protocol v1.2 Section 3", "controlled_attention_cv/ca_layer_head_attribution", None,
     "CA-L drop modules 10-11; CA-H drop head 1; CA-A keep reference (CA-C pending)"),
    ("A2", "Controlled CA package A: CA-T, CA-R", VOC, "controlled, 5-fold cross-validated (Codex)",
     "calls, resolutions, joint", "protocol v1.2 Section 3", "controlled_attention_cv/ca_time_resolution", None,
     "pending"),
]


def family_of(path: tuple) -> str:
    body = [p for p in path if p not in ("results", "experiments")]
    return "/".join(body[:-1])


def jsonl_mious(folder: Path) -> dict[str, float]:
    """Pooled-class mIoU per configuration from per-image JSONL records (tp/fp/fn)."""
    counts = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=np.int64)))
    for path in sorted(folder.glob("class_*.jsonl")):
        for line in path.open():
            record = json.loads(line)
            for name, v in record["results"].items():
                counts[name][record["classname"]] += (v["tp"], v["fp"], v["fn"])
    return {name: float(np.mean([c[0] / c.sum() if c.sum() else 1.0 for c in per.values()])) for name, per in counts.items()}


def collect_rows() -> tuple[list[dict], list[dict]]:
    """All configurations of tier-B experiments, and one summary per experiment."""
    rows, ledger = [], []
    for eid, title, dataset, status, varied, fixed, source, depth, role in EXPERIMENTS:
        if source.startswith("controlled_attention_cv"):
            ledger.append({"id": eid, "experiment": title, "dataset": dataset, "status": status, "varied": varied,
                           "fixed": fixed, "source": f"outputs/segmentation/{source}", "decision_or_role": role})
            continue
        entry = {"id": eid, "experiment": title, "dataset": dataset, "status": status, "varied": varied,
                 "fixed": fixed, "source": f"outputs/segmentation/{source}", "decision_or_role": role}
        found = []
        if depth == "shards":
            for part in source.split("|"):
                for ds in ("voc_sim", "coco_cap"):
                    folder = SEG / part / ds
                    if not folder.exists():
                        continue
                    mious, shards, complete = shard_mious(folder)
                    for name, value in mious.items():
                        found.append({"family": f"{part}/{ds}", "configuration": name, "images": "",
                                      "miou": value, "mean_image_iou": "",
                                      "note": "" if complete == 20 else f"partial: {complete}/20 classes"})
        elif source.endswith((".json", ".json.gz")):
            data = load_json(SEG / source)
            for path, value in leaves(data):
                if path and path[0] in ("best", "best_overall", "best_per_layer_configuration", "settings", "run"):
                    continue
                found.append({"family": family_of(path), "configuration": path[-1],
                              "images": value.get("images", ""), "miou": value["miou"],
                              "mean_image_iou": value.get("mean_image_iou", ""), "note": ""})
        elif source == "voc_sim_100_self_attention":
            for folder in sorted((SEG / source).iterdir()):
                for path, value in leaves(load_json(folder / "results.json")):
                    if path and path[0] in ("best", "settings"):
                        continue
                    found.append({"family": "/".join([folder.name, family_of(path)]).strip("/"), "configuration": path[-1],
                                  "images": value.get("images", ""), "miou": value["miou"],
                                  "mean_image_iou": value.get("mean_image_iou", ""), "note": ""})
        elif source == "ovam_paper_baseline":
            for path in sorted((SEG / source).glob("*.json")):
                value = load_json(path)
                found.append({"family": "", "configuration": value.get("method", path.stem), "images": value["images"],
                              "miou": value["miou"], "mean_image_iou": value.get("mean_image_iou", ""), "note": "authors' masks"})
        elif source == "narmyn_token_crf":
            for ds, images in (("voc_sim", 490), ("coco_cap", 707)):
                for name, value in jsonl_mious(SEG / source / ds).items():
                    found.append({"family": ds, "configuration": name, "images": images, "miou": value,
                                  "mean_image_iou": "", "note": ""})
        for row in found:
            rows.append({"experiment_id": eid, "dataset": dataset, **row})
        if found:
            best = max((r for r in found if not r["family"].startswith("caption_subgroups")), key=lambda r: r["miou"])
            entry.update({"configurations": len(found), "best_configuration": f"{best['family']}/{best['configuration']}".strip("/"),
                          "best_miou": round(best["miou"], 4)})
        ledger.append(entry)
    return rows, ledger


# --------------------------------------------------------------------------- #
# Matplotlib style
# --------------------------------------------------------------------------- #
def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal",
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })
    return plt


def save(fig, name: str, written: list):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    written.append(name)


def short_module(name: str) -> str:
    if name.startswith("mid_block"):
        return "M"
    part = name.split(".")
    return f"{'D' if part[0] == 'down_blocks' else 'U'}{part[1]}.{part[3]}"


def module_resolution(name: str) -> int:
    if name.startswith("mid_block"):
        return 8
    level = int(name.split(".")[1])
    return {0: 64, 1: 32, 2: 16}[level] if name.startswith("down") else {3: 64, 2: 32, 1: 16}[level]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" if isinstance(rows[0][i], (int, float)) else "---"
                                                          for i in range(len(headers))) + "|"]
    for row in rows:
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


def tex_escape(value) -> str:
    text = f"{value:.4f}" if isinstance(value, float) else str(value)
    text = text.replace("\\", r"\textbackslash{}")
    for a, b in (("{", r"\{"), ("}", r"\}"), ("_", r"\_\allowbreak{}"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("/", r"/\allowbreak{}")):
        text = text.replace(a, b)
    for a, b in (("→", r"$\rightarrow$"), ("×", r"$\times$"), ("Δ", r"$\Delta$"), ("≥", r"$\geq$"), ("²", r"$^2$"),
                 ("—", "---"), ("–", "--"), ("−", "$-$")):
        text = text.replace(a, b)
    return text


def tex_table(headers, rows, caption, label, long=False) -> str:
    """Numeric columns right-aligned; text columns wrapped in widths that fit a 17 cm text block."""
    numeric = [isinstance(rows[0][i], (int, float)) for i in range(len(headers))]
    lengths = [max(len(str(h)), *(len(f"{r[i]:.4f}" if isinstance(r[i], float) else str(r[i])) for r in rows))
               for i, h in enumerate(headers)]
    numeric_width = [max(1.1, 0.17 * l) if n else 0 for n, l in zip(numeric, lengths)]
    available = 15.4 - sum(numeric_width) - 0.21 * len(headers)
    weights = [0 if n else max(l, 6) ** 0.7 for n, l in zip(numeric, lengths)]
    total = sum(weights) or 1.0
    spec = []
    for n, w, l in zip(numeric, weights, lengths):
        if n:
            spec.append("r")
        else:
            width = max(1.4, min(available * w / total, 0.19 * l + 0.4))
            spec.append(rf">{{\raggedright\arraybackslash}}p{{{width:.2f}cm}}")
    align = "".join(spec)
    head = " & ".join(tex_escape(h) for h in headers) + r" \\"
    body = "\n".join(" & ".join(tex_escape(v) for v in row) + r" \\" for row in rows)
    if long:
        return ("{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
                f"\\begin{{longtable}}{{{align}}}\n\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n"
                f"\\midrule\n\\endfirsthead\n\\toprule\n{head}\n\\midrule\n\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}}}\n")
    return (f"\\begin{{table}}[H]\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\n\\end{{table}}\n")


def write_table(name, headers, rows, caption, long=False) -> str:
    (OUT / "tables" / f"{name}.md").write_text(f"**{caption}**\n\n" + md_table(headers, rows) + "\n")
    (OUT / "tables" / f"{name}.tex").write_text(tex_table(headers, rows, caption, f"tab:{name}", long))
    return md_table(headers, rows)


# --------------------------------------------------------------------------- #
# Controlled package B
# --------------------------------------------------------------------------- #
def package_b(plt, figures: list) -> dict:
    results = cv.load_results(sorted((PKG_B / "shards").glob("*.npz")))
    folds = load_json(PKG_B / "fold_results.json")["screens"]
    heads = load_json(PKG_B / "head_report.json")
    configs = load_json(PKG_B / "configurations.json")
    modules = configs["modules"]
    names = results.configurations
    fold_ids = sorted(set(results.folds.tolist()))

    # Full per-configuration table with per-fold held-out mIoU and calibration.
    rows = []
    for c, name in enumerate(names):
        row = cv.summary_row(results, c)
        per_fold = [cv.miou(results, c, mask=results.folds == k) for k in fold_ids]
        rows.append({"configuration": name, "screen": {"L": "CA-L", "H": "CA-H", "A": "CA-A"}[name[0]],
                     "miou@0.35": row["miou@0.35"], "miou@0.45": row["miou@0.45"], "miou@0.55": row["miou@0.55"],
                     "mean_image_iou@0.45": row["mean_image_iou@0.45"], "mean_ap": row["mean_ap"],
                     **{f"fold{k}_miou": v for k, v in zip(fold_ids, per_fold)},
                     "fold_std": float(np.std(per_fold))})
    with (OUT / "controlled_ca_package_b.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by = {r["configuration"]: r for r in rows}
    ref = by["L/all_16"]["miou@0.45"]

    out = {"reference": ref, "screens": folds, "heads": heads}

    # CV summary table.
    summary = []
    for screen, s in folds.items():
        b = s["paired_bootstrap_frozen_vs_reference"]
        summary.append([screen, s["reference"], f"{s['vote_winner']} ({s['vote_winner_votes']}/5)",
                        s["out_of_fold_miou_selection"], s["out_of_fold_miou_reference"], s["oof_gain"],
                        f"{b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
                        "yes" if s["calibration_dependent"] else "no", s["freeze_reason"]])
    out["cv_table"] = write_table(
        "a1_cv_summary",
        ["Screen", "Reference", "Vote winner (votes)", "OOF mIoU selection", "OOF mIoU reference", "OOF gain",
         "Paired Δ image IoU [95% CI]", "Calibration-dependent", "Outcome"],
        summary, "Controlled CA package B: cross-validated selection (VOC-Sim, 5 folds, threshold 0.45)")

    # CA-L table: structured, single, leave-one; plus top leave-two.
    def lrow(name, label):
        r = by[name]
        return [label, r["miou@0.45"], r["miou@0.45"] - ref, r["miou@0.35"], r["miou@0.55"], r["mean_ap"], r["fold_std"]]
    lrows = [lrow(n, n[2:]) for n in ("L/all_16", "L/non_middle_15", "L/down_only", "L/middle_only", "L/up_only",
                                      "L/down_middle", "L/middle_up")]
    lrows += [lrow(f"L/single_{i:02d}", f"only {i:02d} {short_module(m)} ({module_resolution(m)}²)") for i, m in enumerate(modules)]
    lrows += [lrow(f"L/leave1_{i:02d}", f"without {i:02d} {short_module(m)}") for i, m in enumerate(modules)]
    headers = ["Configuration", "mIoU@0.45", "Δ vs all 16", "mIoU@0.35", "mIoU@0.55", "Mean AP", "SD across folds"]
    out["cal_table"] = write_table("a1_ca_l", headers, lrows, "CA-L: structured groups, single modules and leave-one-out (in-sample mIoU; all 159 configurations in controlled_ca_package_b.csv)", long=True)
    pairs = sorted((n for n in names if n.startswith("L/leave2_")), key=lambda n: -by[n]["miou@0.45"])[:10]
    out["leave2_table"] = write_table(
        "a1_ca_l_leave2", headers,
        [lrow(n, "without " + " & ".join(f"{int(x):02d} {short_module(modules[int(x)])}" for x in n.split("_")[1:])) for n in pairs],
        "CA-L: the ten best leave-two-out configurations (in-sample)")

    # CA-H tables.
    hrows = [[h, heads["single_head_miou"][str(h)], heads["leave_one_out_miou"][str(h)],
              heads["leave_one_out_miou"][str(h)] - heads["reference_miou"], f"{heads['folds_retaining_each_head'][str(h)]}/5"]
             for h in range(8)]
    out["head_table"] = write_table(
        "a1_ca_h_heads", ["Head", "Alone", "All but this head", "Δ vs all 8", "Folds retaining it"], hrows,
        "CA-H: each head alone and leave-one-out (in-sample mIoU at 0.45), and fold retention")
    path_rows = [[i, s["removed"] if s["removed"] is not None else "—", " ".join(map(str, s["heads"])), s["miou"]]
                 for i, s in enumerate(heads["greedy_backward_path"])]
    out["greedy_table"] = write_table("a1_ca_h_greedy", ["Round", "Removed", "Heads left", "mIoU@0.45"], path_rows,
                                      "CA-H: greedy backward elimination (in-sample)")
    top_h = sorted((n for n in names if n.startswith("H/")), key=lambda n: -by[n]["miou@0.45"])[:15]
    out["top_heads_table"] = write_table(
        "a1_ca_h_top", ["Head subset", "Heads", "mIoU@0.45", "Δ vs all 8", "mIoU@0.35", "mIoU@0.55", "Mean AP"],
        [[n[2:].replace("_", " "), len(n[2:].split("_")), by[n]["miou@0.45"], by[n]["miou@0.45"] - ref,
          by[n]["miou@0.35"], by[n]["miou@0.55"], by[n]["mean_ap"]] for n in top_h],
        "CA-H: the 15 best of 255 head subsets (in-sample; all 255 in controlled_ca_package_b.csv)")

    # CA-A table.
    arows = []
    for n in (x for x in names if x.startswith("A/")):
        q, d, p = n[2:].split("_")
        arows.append(["conditional" if q == "cond" else "unconditional", d, p, by[n]["miou@0.45"], by[n]["miou@0.45"] - ref,
                      by[n]["miou@0.35"], by[n]["miou@0.55"], by[n]["mean_ap"]])
    arows.sort(key=lambda r: -r[3])
    out["attr_table"] = write_table(
        "a1_ca_a", ["Query branch", "Softmax", "Class pieces", "mIoU@0.45", "Δ vs reference", "mIoU@0.35", "mIoU@0.55", "Mean AP"],
        arows, "CA-A: the 12 attribution conventions (in-sample; reference = conditional, padded, all)")

    # Exhaustive.
    exhaustive = load_json(PKG_B / "exhaustive_cv.json") if (PKG_B / "exhaustive_cv.json").exists() else None
    out["exhaustive"] = exhaustive
    ex_scores = None
    raw = PKG_B / "exhaustive_raw"
    if raw.exists() and len(list(raw.glob("*.npy"))) == len(results.image_ids):
        classes = sorted(set(results.classes))
        sums = np.zeros((len(classes), 65535, 3), dtype=np.int64)
        for image_id, classname in zip(results.image_ids, results.classes):
            sums[classes.index(classname)] += np.load(raw / f"{image_id}.npy").astype(np.int64)[:, :3]
        union = sums.sum(-1)
        ex_scores = np.where(union > 0, sums[..., 0] / np.maximum(union, 1), 1.0).mean(axis=0)
        np.save(OUT / "exhaustive_in_sample_miou.npy", ex_scores.astype(np.float32))

    # ---------------- figures ----------------
    # F-A1: single modules and leave-one-out, grouped by resolution.
    order = sorted(range(16), key=lambda i: (-module_resolution(modules[i]), modules[i]))
    labels = [f"{short_module(modules[i])}\n{module_resolution(modules[i])}²" for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    single = [by[f"L/single_{i:02d}"]["miou@0.45"] for i in order]
    axes[0].bar(range(16), single, width=0.6, color=BLUE)
    axes[0].axhline(ref, color=INK2, linewidth=1)
    axes[0].text(15.4, ref + 0.01, f"all 16 = {ref:.3f}", ha="right", va="bottom", color=INK2, fontsize=8)
    axes[0].set_xticks(range(16), labels, fontsize=7)
    axes[0].set_ylabel("mIoU at 0.45")
    axes[0].set_title("Each CA module alone")
    delta = [by[f"L/leave1_{i:02d}"]["miou@0.45"] - ref for i in order]
    axes[1].bar(range(16), delta, width=0.6, color=[BLUE if d > 0 else ORANGE for d in delta])
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[1].set_xticks(range(16), labels, fontsize=7)
    axes[1].set_ylabel("Δ mIoU vs all 16")
    axes[1].set_title("All modules except one (blue = removing it helps)")
    fig.suptitle("CA-L, VOC-Sim, in-sample (Section 3 condition)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "a1_ca_l_single_leave1", figures)
    plt.close(fig)

    # F-A2: leave-two-out Δ heatmap.
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    grid = np.full((16, 16), np.nan)
    for a, b in itertools.combinations(range(16), 2):
        grid[a, b] = grid[b, a] = by[f"L/leave2_{a:02d}_{b:02d}"]["miou@0.45"] - ref
    for i in range(16):
        grid[i, i] = by[f"L/leave1_{i:02d}"]["miou@0.45"] - ref
    grid = grid[np.ix_(order, order)]
    limit = float(np.nanmax(np.abs(grid)))
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit))
    ax.grid(False)
    ax.set_xticks(range(16), [l.replace("\n", " ") for l in labels], rotation=90, fontsize=7)
    ax.set_yticks(range(16), [l.replace("\n", " ") for l in labels], fontsize=7)
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.scatter([best[1]], [best[0]], s=60, facecolors="none", edgecolors=INK, linewidths=1.5)
    bar = fig.colorbar(image, ax=ax, fraction=0.046)
    bar.set_label("Δ mIoU vs all 16 (blue = better)")
    ax.set_title("CA-L: remove two modules (diagonal = remove one); circle = best pair")
    fig.tight_layout()
    save(fig, "a1_ca_l_leave2_heatmap", figures)
    plt.close(fig)

    # F-A3: heads alone / leave-one-out.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(range(8), [heads["single_head_miou"][str(h)] for h in range(8)], width=0.55, color=BLUE)
    axes[0].axhline(heads["reference_miou"], color=INK2, linewidth=1)
    axes[0].text(7.4, heads["reference_miou"] + 0.01, f"all 8 = {heads['reference_miou']:.3f}", ha="right", color=INK2, fontsize=8)
    axes[0].set_xticks(range(8), [f"head {h}" for h in range(8)])
    axes[0].set_ylabel("mIoU at 0.45")
    axes[0].set_title("Each head alone (same head index in every module)")
    d = [heads["leave_one_out_miou"][str(h)] - heads["reference_miou"] for h in range(8)]
    axes[1].bar(range(8), d, width=0.55, color=[BLUE if v > 0 else ORANGE for v in d])
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[1].set_xticks(range(8), [f"head {h}" for h in range(8)])
    axes[1].set_ylabel("Δ mIoU vs all 8")
    axes[1].set_title("All heads except one (blue = removing it helps)")
    fig.tight_layout()
    save(fig, "a1_ca_h_single_leave1", figures)
    plt.close(fig)

    # F-A4: all 255 subsets by size, greedy paths.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rng = np.random.default_rng(0)
    for n in (x for x in names if x.startswith("H/")):
        size = len(n[2:].split("_"))
        axes[0].scatter(size + rng.uniform(-0.18, 0.18), by[n]["miou@0.45"], s=10, color=AXIS, linewidths=0)
    chosen = folds["CA-H"]["frozen"]
    for n, colour, label in ((folds["CA-H"]["reference"], ORANGE, "all 8 heads"), (chosen, BLUE, "chosen: without head 1")):
        size = len(n[2:].split("_"))
        axes[0].scatter(size, by[n]["miou@0.45"], s=46, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
    axes[0].set_xlabel("Number of heads in the subset")
    axes[0].set_ylabel("mIoU at 0.45")
    axes[0].set_title("All 255 head subsets")
    axes[0].legend(loc="lower right")
    back = heads["greedy_backward_path"]
    fwd = heads["greedy_forward_path"]
    axes[1].plot([len(s["heads"]) for s in back], [s["miou"] for s in back], color=BLUE, marker="o", markersize=5, label="backward: remove worst")
    axes[1].plot([len(s["heads"]) for s in fwd], [s["miou"] for s in fwd], color=ORANGE, marker="o", markersize=5, label="forward: add best")
    for s in back[1:3]:
        axes[1].annotate(f"−{s['removed']}", (len(s["heads"]), s["miou"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=INK2)
    axes[1].set_xlabel("Number of heads")
    axes[1].set_ylabel("mIoU at 0.45")
    axes[1].set_title("Greedy head paths")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "a1_ca_h_subsets_greedy", figures)
    plt.close(fig)

    # F-A5: CA-A grid.
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    cells = [["cond", "padded"], ["cond", "unpadded"], ["uncond", "padded"], ["uncond", "unpadded"]]
    matrix = np.array([[by[f"A/{q}_{d}_{p}"]["miou@0.45"] - ref for p in ("all", "first", "last")] for q, d in cells])
    lim = float(np.abs(matrix).max())
    ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), ["all pieces", "first piece", "last piece"])
    ax.set_yticks(range(4), [f"{'conditional' if q == 'cond' else 'unconditional'} query, {d}" for q, d in cells])
    ax.set_title("CA-A: Δ mIoU vs reference (conditional, padded, all pieces)")
    fig.tight_layout()
    save(fig, "a1_ca_a_grid", figures)
    plt.close(fig)

    # F-A6: calibration of the promoted changes.
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for n, colour, label in (("L/all_16", INK2, "reference (all 16, all heads)"),
                             (folds["CA-L"]["frozen"], BLUE, "CA-L: without modules 10, 11"),
                             (folds["CA-H"]["frozen"], ORANGE, "CA-H: without head 1")):
        ys = [by[n][f"miou@{t:.2f}"] for t in cv.THRESHOLDS]
        ax.plot(cv.THRESHOLDS, ys, color=colour, marker="o", markersize=5, label=label)
    ax.axvline(cv.SELECTION_THRESHOLD, color=AXIS, linewidth=1)
    ax.text(cv.SELECTION_THRESHOLD, ax.get_ylim()[0], " selection threshold", color=MUTED, fontsize=7, va="bottom")
    ax.set_xticks(cv.THRESHOLDS)
    ax.set_xlabel("Mask threshold")
    ax.set_ylabel("mIoU")
    ax.set_title("Calibration check: gains appear only at ≥ 0.45")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "a1_calibration", figures)
    plt.close(fig)

    # F-A7: held-out folds.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, (screen, s) in zip(axes, folds.items()):
        xs = [f["fold"] for f in s["folds"]]
        ax.plot(xs, [f["heldout_miou_reference"] for f in s["folds"]], color=ORANGE, marker="o", markersize=5, label="reference")
        ax.plot(xs, [f["heldout_miou_chosen"] for f in s["folds"]], color=BLUE, marker="o", markersize=5, label="chosen on other 4 folds")
        ax.set_xticks(xs, [f"fold {x}" for x in xs])
        ax.set_title(f"{screen}: OOF gain {s['oof_gain']:+.4f}")
    axes[0].set_ylabel("Held-out mIoU at 0.45")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "a1_heldout_folds", figures)
    plt.close(fig)

    # F-A8: exhaustive distribution.
    if ex_scores is not None:
        popcount = np.array([bin(m).count("1") for m in range(1, 65536)])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
        axes[0].hist(ex_scores, bins=80, color=BLUE)
        axes[0].axvline(ex_scores[65534], color=ORANGE, linewidth=1.5)
        axes[0].text(ex_scores[65534], axes[0].get_ylim()[1] * 0.92, " all 16", color=INK2, fontsize=8)
        axes[0].set_xlabel("In-sample mIoU at 0.45")
        axes[0].set_ylabel("Number of subsets")
        axes[0].set_title("All 65,535 CA module subsets")
        best_per = [ex_scores[popcount == k].max() for k in range(1, 17)]
        median_per = [np.median(ex_scores[popcount == k]) for k in range(1, 17)]
        axes[1].plot(range(1, 17), best_per, color=BLUE, marker="o", markersize=4, label="best subset of this size")
        axes[1].plot(range(1, 17), median_per, color=ORANGE, marker="o", markersize=4, label="median subset")
        axes[1].set_xlabel("Number of modules")
        axes[1].set_ylabel("In-sample mIoU at 0.45")
        axes[1].set_title("Exploratory only: maxima are optimistic")
        axes[1].legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        save(fig, "a1_exhaustive", figures)
        plt.close(fig)

    for png in sorted((PKG_B / "figures").glob("*_ca_modules.png")):
        shutil.copy(png, OUT / "figures" / png.name)
        figures.append(png.stem)
    return out


# --------------------------------------------------------------------------- #
# Tier B figures and tables
# --------------------------------------------------------------------------- #
def tier_b(plt, rows: list[dict], figures: list) -> dict:
    out = {}
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["experiment_id"]].append(r)

    # Family winners for the big VOC experiments.
    fam_rows = []
    for eid in ("B8", "B9", "B10", "B15"):
        fam = defaultdict(list)
        for r in by_exp[eid]:
            fam[r["family"]].append(r)
        for name, items in sorted(fam.items()):
            best = max(items, key=lambda r: r["miou"])
            fam_rows.append([eid, name or "—", len(items), best["configuration"][:70], best["miou"]])
    out["family_table"] = write_table("b_family_winners", ["Exp.", "Family", "Configurations", "Best configuration", "Best mIoU"],
                                      fam_rows, "Exploratory VOC-Sim experiments: best configuration per family (in-sample, 490 images)", long=True)

    # Progression (from committed result files and attention_analysis.md).
    released = {k: load_json(SEG / "ovam_paper_baseline" / f)["miou"] for k, f in
                (("unopt", "summary.json"), ("opt", "optimized_summary.json"), ("ca", "cross_attention_only_summary.json"))}
    def pick(eid, family, config=None):
        items = [r for r in by_exp[eid] if r["family"] == family and (config is None or r["configuration"] == config)]
        if not items:
            raise KeyError(f"{eid}: {family}/{config} not found")
        return max(r["miou"] for r in items)
    token = {ds: {r["configuration"]: r["miou"] for r in by_exp["B17"] if r["family"] == ds} for ds in ("voc_sim", "coco_cap")}
    voc_steps = [
        ("OVAM CA only (released heatmaps)", released["ca"], "B16"),
        ("OVAM CA only (regenerated)", pick("B8", "baseline", "ovam_unoptimized"), "B8"),
        ("OVAM, implemented (CA×SA, DenseCRF)", 0.6934, "attention_analysis.md"),
        ("OVAM, released masks", released["unopt"], "B16"),
        ("+ SA propagation at U3.2, 4 passes", pick("B8", "self_attention/r64"), "B8"),
        ("+ top-4-head CA seed, 2 passes, t=0.45", pick("B9", "focused/output_calibration", "top4_heads_percentile__r64__iter2__percentile_01_99__t0.45"), "B9"),
        ("+ heads {0,1,2,4,5,6}, calls 15–30, 4 passes, 1% cleanup", pick("B10", "joint/joint_postprocessing", "iter4__leave_3_7__trailing_15_30__t0.45__remove_fraction_0.01"), "B10"),
        ("Revised: 1 pass, all 16 CA layers", pick("B13", "revised_system_component_ablation/voc_sim", "revised_full_system"), "B13"),
        ("OVAM token-optimized, released masks", released["opt"], "B16"),
        ("Earlier final + optimized token + DenseCRF (1 pass)", token["voc_sim"]["opt_top4/sa_iter1+crf"], "B17"),
    ]
    coco_steps = [
        ("OVAM, implemented (paper COCO parameters)", token["coco_cap"]["ovam_implemented"], "B17"),
        ("Final method, frozen from VOC", pick("B12", "", "frozen_best_voc"), "B12"),
        ("Revised: 1 pass, all 16 CA layers", pick("B13", "revised_system_component_ablation/coco_cap", "revised_full_system"), "B13"),
        ("Earlier final + optimized token + DenseCRF (1 pass)", token["coco_cap"]["opt_top4/sa_iter1+crf"], "B17"),
        ("OVAM token-optimized (paper Table 1)", 0.692, "OVAM paper"),
    ]
    out["progression"] = {"voc": voc_steps, "coco": coco_steps}
    out["progression_table"] = write_table(
        "b_progression", ["Dataset", "Method", "mIoU", "Evidence"],
        [["VOC-Sim", a, b, c] for a, b, c in voc_steps] + [["COCO-Cap", a, b, c] for a, b, c in coco_steps],
        "How the method developed, with the baselines it is compared against")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [10, 5]})
    for ax, steps, title in ((axes[0], voc_steps, "VOC-Sim (490 images; tuned here)"), (axes[1], coco_steps, "COCO-Cap (707 images; never tuned)")):
        y = np.arange(len(steps))[::-1]
        for yi, (label, value, _) in zip(y, steps):
            colour = ORANGE if "OVAM" in label else BLUE
            ax.hlines(yi, 0.5, value, color=GRID, linewidth=1)
            ax.scatter(value, yi, s=64, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.text(value + 0.006, yi, f"{value:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [s[0] for s in steps], fontsize=8, color=INK2)
        ax.set_xlim(0.5, max(s[1] for s in steps) + 0.05)
        ax.set_xlabel("mIoU")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=7, label="OVAM baselines"),
                        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="our pipeline")],
               loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "b_progression", figures)
    plt.close(fig)

    # Joint suite heatmap: passes x time window (heads leave_3_7, t=0.45).
    joint = {r["configuration"]: r["miou"] for r in by_exp["B10"] if r["family"] == "joint/joint_head_time_threshold"}
    times = ["all", "trailing_05_30", "trailing_10_30", "trailing_15_30", "steps_10_20", "steps_10_25", "steps_15_20",
             "step_14", "step_15", "step_16", "late_linear", "late_exp_0.05", "late_exp_0.10", "sqrt_snr", "log1p_snr"]
    times = [t for t in times if f"iter4__leave_3_7__{t}__t0.45" in joint]
    matrix = np.array([[joint[f"iter{i}__leave_3_7__{t}__t0.45"] for t in times] for i in (1, 2, 3, 4)])
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(matrix, cmap=seq, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(matrix):
        level = (v - matrix.min()) / max(matrix.max() - matrix.min(), 1e-9)
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=SURFACE if level > 0.45 else INK)
    ax.set_xticks(range(len(times)), [t.replace("_", " ") for t in times], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(4), [f"{i} pass{'es' if i > 1 else ''}" for i in (1, 2, 3, 4)])
    fig.colorbar(im, ax=ax, fraction=0.025).set_label("mIoU")
    ax.set_title("B10 (exploratory): SA passes × denoising-call weighting; SA heads {0,1,2,4,5,6}, threshold 0.45")
    fig.tight_layout()
    save(fig, "b_joint_passes_time", figures)
    plt.close(fig)

    # Head sets and thresholds in B10.
    head_sets = sorted({k.split("__")[1] for k in joint})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    vals = [joint.get(f"iter4__{h}__trailing_15_30__t0.45", np.nan) for h in head_sets]
    axes[0].scatter(range(len(head_sets)), vals, s=48, color=[BLUE if h == "leave_3_7" else MUTED for h in head_sets],
                    edgecolors=SURFACE, linewidths=1.5, zorder=3)
    best_i = int(np.nanargmax(vals))
    axes[0].annotate("chosen: without 3, 7", (best_i, vals[best_i]), textcoords="offset points", xytext=(9, -3), ha="left", fontsize=7, color=INK2)
    axes[0].set_xticks(range(len(head_sets)), [h.replace("_", " ") for h in head_sets], rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel("mIoU")
    axes[0].set_title("SA head sets (4 passes, calls 15–30, t=0.45)")
    for colour, i in ((BLUE, 4), (ORANGE, 1)):
        ts = (0.35, 0.40, 0.45, 0.50, 0.55)
        axes[1].plot(ts, [joint[f"iter{i}__leave_3_7__trailing_15_30__t{t:.2f}"] for t in ts], color=colour, marker="o", markersize=5,
                     label=f"{i} pass{'es' if i > 1 else ''}")
    axes[1].set_xlabel("Mask threshold")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Threshold (heads {0,1,2,4,5,6}, calls 15–30)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "b_joint_heads_threshold", figures)
    plt.close(fig)

    # Component ablations (final and revised), VOC and COCO.
    comp = []
    for eid, family, label in (("B11", "final_component_ablation", "final (4 passes, 12 layers)"), ("B13", "revised_system_component_ablation", "revised (1 pass, 16 layers)")):
        for ds in ("voc_sim", "coco_cap"):
            items = {r["configuration"]: r["miou"] for r in by_exp[eid] if r["family"] == f"{family}/{ds}"}
            full = items.get("full_system", items.get("revised_full_system"))
            for k, v in items.items():
                comp.append([label, "VOC-Sim" if ds == "voc_sim" else "COCO-Cap", k.replace("_", " "), v, v - full])
    out["component_table"] = write_table("b_component_ablations", ["System", "Dataset", "Change", "mIoU", "Δ vs full"], comp,
                                         "Remove-one-component ablations of the final and revised systems", long=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    final_rows = [r for r in comp if r[0].startswith("final") and r[2] != "full system"]
    changes = [r[2] for r in final_rows if r[1] == "VOC-Sim"]
    for ax, ds in zip(axes, ("VOC-Sim", "COCO-Cap")):
        d = [next(r[4] for r in final_rows if r[1] == ds and r[2] == c) for c in changes]
        y = np.arange(len(changes))[::-1]
        ax.barh(y, d, height=0.55, color=[BLUE if v > 0 else ORANGE for v in d])
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, changes, fontsize=8)
        ax.set_xlabel("Δ mIoU when this component is removed or replaced")
        ax.set_title(f"{ds}: final method (orange = component helps)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "b_component_ablation", figures)
    plt.close(fig)

    # Token / DenseCRF.
    tok = []
    for ds in ("voc_sim", "coco_cap"):
        t = token[ds]
        for key in ("ovam_implemented", "plain_top4/sa_iter1", "plain_top4/sa_iter1+crf", "opt_top4/sa_iter1",
                    "opt_top4/sa_iter1+crf", "opt_all16/sa_iter1+crf", "plain_top4/sa_iter4", "opt_top4/sa_iter4+crf"):
            tok.append(["VOC-Sim" if ds == "voc_sim" else "COCO-Cap", key, t[key]])
    tok += [["VOC-Sim", "released OVAM token-optimized masks", released["opt"]], ["COCO-Cap", "OVAM token-optimized (paper)", 0.692]]
    out["token_table"] = write_table("b_token_crf", ["Dataset", "Configuration", "mIoU"], tok,
                                     "OVAM optimized token and DenseCRF on the earlier final method (exploratory)")
    return out


# --------------------------------------------------------------------------- #
# Evidence document (Markdown) and LaTeX section
# --------------------------------------------------------------------------- #
DECISIONS = [
    # setting, value in the earlier final method, where chosen, images used to choose, evidence, controlled status
    ("CA layers", "12 of 16 (drop 1, 3, 4, 8)", "B4 exhaustive search", "99 dev images", "ledger B4; Fig. a1_exhaustive",
     "A1 CA-L (CV): keep all but modules 10, 11 instead; awaiting CA-C"),
    ("CA heads", "fixed top-4 per layer", "B6 stage-1 suite", "99 dev images", "ledger B6",
     "A1 CA-H (CV): drop head 1 in every module; awaiting CA-C"),
    ("CA attribution", "OVAM: unconditional query, unpadded softmax, first piece", "inherited from OVAM", "—", "Table a1_ca_a",
     "A1 CA-A (CV): reference (conditional, padded, all pieces) kept; OVAM's cell is −0.0156 CA-only"),
    ("CA calls", "all 31 (0–30)", "B1/B8 windows; B14", "99 dev / 490", "ledger B1, B8, B14", "A2 CA-T (Codex), pending"),
    ("CA normalization, threshold", "percentile 1–99, ≥ 0.45", "B5", "99 dev images", "ledger B5", "protocol fixes min–max, 0.45 for the CA phase"),
    ("SA module", "U3.2 (up_blocks.3.attentions.2 attn1, 64×64)", "B8 SA screens", "490 (in-sample)", "ledger B8", "SA-L (Claude) after CA freeze"),
    ("SA passes", "4 (revised: 1)", "B10; revised from B11/B13", "490; 1-pass informed by COCO-Cap", "Fig. b_joint_passes_time", "after SA freeze"),
    ("SA heads", "{0,1,2,4,5,6}", "B10", "490 (in-sample)", "Fig. b_joint_heads_threshold", "SA-H (Claude) after CA freeze"),
    ("SA calls", "15–30", "B10", "490 (in-sample)", "Fig. b_joint_passes_time", "SA-T (Codex) after CA freeze"),
    ("Mask threshold", "0.45", "B5, B9, B10", "99 dev / 490", "Fig. b_joint_heads_threshold", "fixed at 0.45; 0.35/0.55 as diagnostics"),
    ("Cleanup", "remove components < 1%", "B10", "490 (in-sample)", "Table b_component_ablations", "deferred phase"),
    ("DenseCRF", "not in the final method", "B17 (nothing selected)", "—", "Table b_token_crf", "deferred phase"),
    ("Optimized token", "not in the final method", "B17 (nothing selected; tokens trained by OVAM on separate images)", "—", "Table b_token_crf", "deferred phase"),
]


def compose_report(b: dict | None, tb: dict) -> None:
    def fig(name, caption):
        return f"![{caption}](figures/{name}.png)\n\n*{caption}*\n"

    decision_rows = [list(d) for d in DECISIONS]
    decisions = write_table("decision_trail", ["Setting", "Value used so far", "Chosen in", "Images used to choose", "Evidence", "Controlled status"],
                            decision_rows, "Where each setting of the method came from")
    ledger = (OUT / "tables" / "ledger.md").read_text().split("\n", 2)[2]
    lines = [
        "# Evidence: every experiment and how each setting was reached",
        "",
        "Generated by `scripts/narmyn/attention/build_evidence.py` from the committed result files; nothing is re-run. "
        "mIoU is the 20-class macro mean of pooled foreground IoU unless stated.",
        "",
        "**How to read it.** Experiments are in two tiers.",
        "",
        "- **Tier B (B1–B17), exploratory.** Settings were chosen on the same images they are reported on "
        "(the 99 dev images are part of the 490). These numbers show how the method was found, not how well it generalizes.",
        "- **Tier A (A1, A2), controlled.** Protocol v1.2: one axis at a time from a fixed reference, settings chosen on "
        "four folds and scored on the fifth, frozen only with 4 of 5 fold votes and an out-of-fold gain. "
        "Only A1 (CA layers, heads, attribution) is complete; nothing is frozen until the combined CA confirmation.",
        "",
        "Full data: `all_configurations.csv.gz` (one row per configuration, all experiments), "
        "`controlled_ca_package_b.csv` (the 426 controlled configurations with per-fold held-out mIoU, thresholds 0.35/0.45/0.55 and AP), "
        "`ledger.csv`, `tables/*.md|tex`, `figures/*.png|pdf`.",
        "",
        "## 1. Every experiment",
        "",
        ledger,
        "## 2. Where each setting came from",
        "",
        decisions,
        "",
        "## 3. Development of the method and its baselines",
        "",
        fig("b_progression", "Fig. 1. Each step of the method and the baselines. VOC-Sim values are in-sample; COCO-Cap was never used to choose the earlier final method."),
        tb["progression_table"],
        "",
    ]
    if b is not None:
        lines += [
            "## 4. Controlled CA experiments (tier A1, cross-validated)",
            "",
            "Fixed condition (protocol Section 3): CA only; calls 0–30; conditional query; softmax over 77 padded tokens; "
            "mean over all class-word pieces; every module-call weighted equally; min–max at 64×64; bilinear resize; "
            "foreground at ≥ 0.45. One axis varies at a time.",
            "",
            "### 4.1 Selection result",
            "",
            b["cv_table"],
            "",
            fig("a1_heldout_folds", "Fig. 2. Held-out mIoU of the setting chosen on the other four folds, against the reference, per fold."),
            fig("a1_calibration", "Fig. 3. Calibration check. Both promoted changes help at 0.45 and 0.55 but not at 0.35, and slightly lower AP: they make the map more selective rather than ranking pixels better."),
            "### 4.2 CA layers (CA-L, 159 configurations)",
            "",
            fig("a1_ca_l_single_leave1", "Fig. 4. Each CA module alone (left) and all modules except one (right)."),
            fig("a1_ca_l_leave2_heatmap", "Fig. 5. All 120 leave-two-out configurations; the diagonal is leave-one-out."),
            b["leave2_table"],
            "",
            "The single-module and leave-one-out values for all 16 modules are in `tables/a1_ca_l.md`.",
            "",
            fig("a1_exhaustive", "Fig. 6. All 65,535 module subsets (exploratory; the maximum is not eligible for selection)."),
            "### 4.3 CA heads (CA-H, 255 subsets)",
            "",
            b["head_table"],
            "",
            fig("a1_ca_h_single_leave1", "Fig. 7. Each head alone and all heads except one (the same head index in every module)."),
            fig("a1_ca_h_subsets_greedy", "Fig. 8. All 255 head subsets by size, and the greedy removal and addition paths."),
            b["greedy_table"],
            "",
            b["top_heads_table"],
            "",
            "### 4.4 CA attribution convention (CA-A, 12 configurations)",
            "",
            b["attr_table"],
            "",
            fig("a1_ca_a_grid", "Fig. 9. Δ mIoU of every convention against the reference. The earlier final method used OVAM's cell (unconditional, unpadded, first piece)."),
            "### 4.5 What the cross-attention maps look like",
            "",
            "Head-averaged CA of every module, calls 0–30, upsampled to 64×64 on a common colour scale, one row per resolution. "
            "The four images were fixed by index before any result was seen.",
            "",
        ]
        for name in ("ovam-voc-sim-0", "ovam-voc-sim-125", "ovam-voc-sim-250", "ovam-voc-sim-350"):
            lines.append(fig(f"{name}_ca_modules", f"Fig. CA maps, {name}."))
    lines += [
        "## 5. Exploratory SA experiments (tier B)",
        "",
        fig("b_joint_passes_time", "Fig. 10. B10: SA passes × denoising-call weighting (in-sample)."),
        fig("b_joint_heads_threshold", "Fig. 11. B10: SA head subsets and the mask threshold (in-sample)."),
        fig("b_component_ablation", "Fig. 12. Removing one component of the earlier final method, on VOC-Sim and on the untuned COCO-Cap."),
        "Best configuration of every family in the large VOC-Sim experiments: `tables/b_family_winners.md`. "
        "Component ablations of the final and revised systems: `tables/b_component_ablations.md`.",
        "",
        "## 6. Optimized token and DenseCRF (exploratory, nothing selected)",
        "",
        tb["token_table"],
        "",
        "## 7. Limits of this evidence",
        "",
        "- Tier B settings were chosen on the reported images; only COCO-Cap is untouched by the earlier final method, and the revised one-pass system was informed by COCO-Cap.",
        "- Tier A is complete only for CA layers, heads and attribution. CA calls and resolutions (A2), the combined CA confirmation, and all SA screens are still to run under the protocol.",
        "- Both promoted CA changes are calibration-dependent (Fig. 3).",
        "- Released OVAM masks were made on the authors' renders; all other rows use regenerated images scored on the same annotations.",
        "",
    ]
    (OUT / "EVIDENCE.md").write_text("\n".join(lines))

    # LaTeX section for Overleaf.
    tex = [r"% Generated by scripts/narmyn/attention/build_evidence.py. Preamble: \usepackage{array,booktabs,longtable,float,graphicx} (array before longtable).",
           r"\section{Evidence: experiments and how each setting was reached}", "",
           r"\input{evidence/tables/ledger.tex}", r"\input{evidence/tables/decision_trail.tex}"]
    captions = {
        "b_progression": ("Each step of the method and the baselines. VOC-Sim values are in-sample; COCO-Cap was never used to choose the earlier final method.", 1.0),
        "a1_heldout_folds": ("Controlled CA screens: held-out mIoU of the setting chosen on the other four folds, against the reference, per fold.", 1.0),
        "a1_calibration": ("Calibration check: both promoted CA changes help at thresholds 0.45 and 0.55 but not at 0.35.", 0.6),
        "a1_ca_l_single_leave1": ("CA-L: each CA module alone (left) and all modules except one (right).", 1.0),
        "a1_ca_l_leave2_heatmap": ("CA-L: all 120 leave-two-out configurations; the diagonal is leave-one-out.", 0.7),
        "a1_exhaustive": ("All 65,535 CA module subsets (exploratory; the maximum is not eligible for selection).", 1.0),
        "a1_ca_h_single_leave1": ("CA-H: each head alone and all heads except one (the same head index in every module).", 1.0),
        "a1_ca_h_subsets_greedy": ("CA-H: all 255 head subsets by size, and the greedy removal and addition paths.", 1.0),
        "a1_ca_a_grid": ("CA-A: change in mIoU of each attribution convention against the reference.", 0.65),
        "b_joint_passes_time": ("Exploratory B10: SA passes by denoising-call weighting (in-sample).", 1.0),
        "b_joint_heads_threshold": ("Exploratory B10: SA head subsets and the mask threshold (in-sample).", 1.0),
        "b_component_ablation": ("Removing one component of the earlier final method, on VOC-Sim and on the untuned COCO-Cap.", 1.0),
    }
    figs = ["b_progression"]
    if b is not None:
        tex += [r"\input{evidence/tables/a1_cv_summary.tex}"]
        figs += ["a1_heldout_folds", "a1_calibration", "a1_ca_l_single_leave1", "a1_ca_l_leave2_heatmap", "a1_exhaustive",
                 "a1_ca_h_single_leave1", "a1_ca_h_subsets_greedy", "a1_ca_a_grid"]
        tex += [rf"\input{{evidence/tables/{t}.tex}}" for t in ("a1_ca_l_leave2", "a1_ca_h_heads", "a1_ca_h_greedy", "a1_ca_h_top", "a1_ca_a")]
    figs += ["b_joint_passes_time", "b_joint_heads_threshold", "b_component_ablation"]
    for name in figs:
        caption, width = captions[name]
        tex += [r"\begin{figure}[H]\centering", rf"\includegraphics[width={width}\linewidth]{{evidence/figures/{name}.pdf}}",
                rf"\caption{{{tex_escape(caption)}}}\label{{fig:{name}}}", r"\end{figure}"]
    if b is not None:
        for name in ("ovam-voc-sim-0", "ovam-voc-sim-125", "ovam-voc-sim-250", "ovam-voc-sim-350"):
            tex += [r"\begin{figure}[H]\centering", rf"\includegraphics[width=0.85\linewidth]{{evidence/figures/{name}_ca_modules.png}}",
                    rf"\caption{{Head-averaged cross-attention of every module for {tex_escape(name)}, calls 0--30, common colour scale, one row per resolution.}}", r"\end{figure}"]
    tex += [r"\input{evidence/tables/b_progression.tex}", r"\input{evidence/tables/b_token_crf.tex}",
            r"\input{evidence/tables/b_component_ablations.tex}", r"\input{evidence/tables/b_family_winners.tex}"]
    (OUT / "evidence_section.tex").write_text("\n".join(tex) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    plt = style()
    figures: list[str] = []
    rows, ledger = collect_rows()

    # Controlled rows go into the global table too.
    a_rows = []
    if (PKG_B / "shards").exists():
        b = package_b(plt, figures)
        with (OUT / "controlled_ca_package_b.csv").open() as handle:
            for r in csv.DictReader(handle):
                a_rows.append({"experiment_id": "A1", "dataset": VOC, "family": r["screen"], "configuration": r["configuration"],
                               "images": 490, "miou": float(r["miou@0.45"]), "mean_image_iou": float(r["mean_image_iou@0.45"]),
                               "note": "controlled; per-fold and calibration columns in controlled_ca_package_b.csv"})
        entry = next(e for e in ledger if e["id"] == "A1")
        entry.update({"configurations": len(a_rows) + 65535, "best_configuration": "see CV summary (CA-C pending)",
                      "best_miou": round(max(r["miou"] for r in a_rows), 4)})
    else:
        b = None
    all_rows = rows + a_rows
    with gzip.open(OUT / "all_configurations.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "dataset", "family", "configuration", "images",
                                                    "miou", "mean_image_iou", "note"])
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "ledger.csv").open("w", newline="") as handle:
        fields = ["id", "experiment", "dataset", "status", "varied", "fixed", "configurations", "best_configuration",
                  "best_miou", "decision_or_role", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger)
    ledger_rows = [[e["id"], e["experiment"], e["dataset"], e["status"], e.get("configurations", "—"),
                    e.get("best_miou", "—"), e["decision_or_role"]] for e in ledger]
    write_table("ledger", ["ID", "Experiment", "Dataset", "Status", "Configurations", "Best mIoU", "Decision / role"],
                ledger_rows, "Every experiment run so far", long=True)
    tb = tier_b(plt, rows, figures)
    compose_report(b, tb)
    json.dump({"figures": figures, "configurations": len(all_rows), "experiments": len(ledger)},
              open(OUT / "build_summary.json", "w"), indent=2)
    print(f"{len(ledger)} experiments, {len(all_rows)} configuration rows (+65,535 exhaustive subsets), {len(figures)} figures")


if __name__ == "__main__":
    main()
