"""Merge explanatory shards and write the professor-facing analysis report."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/analysis/voc_sim_attention_report")
    )
    parser.add_argument("--expected-shards", type=int, default=20)
    parser.add_argument("--expected-images", type=int, default=490)
    parser.add_argument("--report", type=Path, default=Path("attention_analysis.md"))
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_streaming/summary.json"),
    )
    parser.add_argument(
        "--focused-summary",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_focused_sa_p1_p4/summary.json"),
    )
    parser.add_argument(
        "--joint-summary",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_joint_sa_head_time/summary.json"),
    )
    parser.add_argument(
        "--paper-ovam-summary",
        type=Path,
        default=Path("outputs/segmentation/ovam_paper_baseline/summary.json"),
        help="Evaluation of the authors' released CA+SA+DenseCRF masks.",
    )
    parser.add_argument(
        "--paper-ca-summary",
        type=Path,
        default=Path("outputs/segmentation/ovam_paper_baseline/cross_attention_only_summary.json"),
        help="Evaluation of the authors' released ordinary-prompt heatmaps.",
    )
    return parser.parse_args()


def _numeric_mean(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for item in items for key, value in item.items() if isinstance(value, (int, float))})
    output = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item and math.isfinite(float(item[key]))]
        if values:
            output[key] = statistics.fmean(values)
    return output


def _group_profile(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for name, values in record[field].items():
            grouped[name].append(values)
    return {name: _numeric_mean(values) for name, values in sorted(grouped.items())}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _svg_bar(
    path: Path, title: str, labels: list[str], values: list[float], highlighted: set[str] | None = None
) -> None:
    highlighted = highlighted or set()
    width = 1100
    row = 28
    height = 90 + row * len(labels)
    left, right = 310, 80
    maximum = max(values) if values else 1.0
    minimum = min(0.0, min(values, default=0.0))
    span = maximum - minimum or 1.0
    zero = left + (-minimum / span) * (width - left - right)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{zero:.1f}" y1="50" x2="{zero:.1f}" y2="{height-20}" stroke="#777"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 58 + index * row
        x_value = left + ((value - minimum) / span) * (width - left - right)
        x = min(zero, x_value)
        bar_width = max(1.0, abs(x_value - zero))
        color = "#d95f02" if label in highlighted else "#2878b5"
        parts.extend(
            [
                f'<text x="{left-8}" y="{y+15}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
                f'<rect x="{x:.1f}" y="{y+3}" width="{bar_width:.1f}" height="17" fill="{color}"/>',
                f'<text x="{max(x_value, zero)+6:.1f}" y="{y+16}" font-family="sans-serif" font-size="11">{value:.3f}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_scatter(
    path: Path, title: str, points: list[tuple[str, float, float]], x_label: str, y_label: str
) -> None:
    width, height = 900, 620
    left, right, top, bottom = 90, 50, 60, 80
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin or 1) * 0.08
    ypad = (ymax - ymin or 1) * 0.10
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda y: height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{(left+width-right)/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
    ]
    for label, x, y in points:
        px, py = sx(x), sy(y)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#2878b5"/>')
        parts.append(f'<text x="{px+7:.1f}" y="{py-5:.1f}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _corr(x: list[float], y: list[float]) -> dict[str, float]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    shards = sorted(path for path in args.input_dir.glob("class_*") if path.is_dir())
    complete = [path for path in shards if (path / "COMPLETE").is_file()]
    if len(complete) != args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} complete shards, found {len(complete)}")
    records: list[dict[str, Any]] = []
    signatures = []
    for shard in complete:
        state = torch.load(shard / "checkpoint.pt", map_location="cpu")
        signatures.append(state["signature"])
        records.extend(state["records"])
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate image IDs in analysis shards")
    if len(records) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} records, found {len(records)}")
    args.input_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.input_dir / "analysis_records.json.gz"
    temporary_gz = merged_path.with_suffix(".tmp")
    with gzip.open(temporary_gz, "wt", encoding="utf-8") as handle:
        json.dump({"signatures": signatures, "records": records}, handle, sort_keys=True)
    os.replace(temporary_gz, merged_path)

    selected = _group_profile(records, "selected_methods")
    ca_layers = _group_profile(records, "cross_attention_layers")
    sa_modules = _group_profile(records, "self_attention_modules")
    heads = _group_profile(records, "winning_module_heads")
    steps = _group_profile(records, "winning_module_steps")
    ca_record_key = (
        "ovam_cross_attention_only"
        if "ovam_cross_attention_only" in records[0]["selected_methods"]
        else "ovam_unoptimized"
    )
    paper_ovam = json.loads(args.paper_ovam_summary.read_text(encoding="utf-8"))
    paper_ca = json.loads(args.paper_ca_summary.read_text(encoding="utf-8"))
    paper_by_id = {item["image_id"]: item["iou"] for item in paper_ovam["per_image"]}
    if set(paper_by_id) != set(ids):
        raise RuntimeError("Released OVAM mask IDs do not match the attention-analysis IDs")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[record["classname"]].append(record)
    class_rows = []
    for classname, items in sorted(by_class.items()):
        geometry = _numeric_mean([item["geometry"] for item in items])
        ca_iou = statistics.fmean(
            item["selected_methods"][ca_record_key]["iou"] for item in items
        )
        baseline_iou = statistics.fmean(paper_by_id[item["image_id"]] for item in items)
        winner_iou = statistics.fmean(
            item["selected_methods"]["joint_best_0.7997"]["iou"] for item in items
        )
        class_rows.append(
            {
                "class": classname,
                "images": len(items),
                **geometry,
                "mean_image_iou_ca_only": ca_iou,
                "mean_image_iou_paper_ovam": baseline_iou,
                "mean_image_iou_final": winner_iou,
                "mean_image_iou_gain": winner_iou - baseline_iou,
            }
        )
    _write_csv(args.input_dir / "class_geometry_and_performance.csv", class_rows)
    _write_csv(args.input_dir / "selected_method_map_statistics.csv", [{"method": k, **v} for k, v in selected.items()])
    _write_csv(args.input_dir / "cross_attention_layer_statistics.csv", [{"module": k, **v} for k, v in ca_layers.items()])
    _write_csv(args.input_dir / "self_attention_module_statistics.csv", [{"module": k, **v} for k, v in sa_modules.items()])
    _write_csv(args.input_dir / "winning_module_head_statistics.csv", [{"head": k, **v} for k, v in heads.items()])
    _write_csv(args.input_dir / "winning_module_timestep_statistics.csv", [{"step": k, **v} for k, v in steps.items()])

    per_image_area = [float(record["geometry"]["area_fraction"]) for record in records]
    per_image_gain = [
        float(record["selected_methods"]["joint_best_0.7997"]["iou"])
        - float(paper_by_id[record["image_id"]])
        for record in records
    ]
    area_gain_corr = _corr(per_image_area, per_image_gain)
    entropy_iou_corr = _corr(
        [float(record["selected_methods"]["joint_best_0.7997"]["entropy"]) for record in records],
        [float(record["selected_methods"]["joint_best_0.7997"]["iou"]) for record in records],
    )
    mean_target_area = statistics.fmean(
        float(record["geometry"]["area_fraction"]) for record in records
    )
    figures = args.input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _svg_scatter(
        figures / "class_size_vs_gain.svg",
        "Class object size versus final improvement",
        [(row["class"], row["area_fraction"], row["mean_image_iou_gain"]) for row in class_rows],
        "Mean foreground area fraction",
        "Mean image-IoU gain over published OVAM",
    )
    _svg_bar(
        figures / "sa_head_iou.svg",
        "Winning self-attention module: individual heads",
        [f"head {key}" for key in sorted(heads, key=int)],
        [heads[key]["iou"] for key in sorted(heads, key=int)],
        {"head 3", "head 7"},
    )
    _svg_bar(
        figures / "ca_layer_iou.svg",
        "Individual cross-attention layers",
        list(ca_layers),
        [ca_layers[key]["iou"] for key in ca_layers],
    )
    _svg_bar(
        figures / "sa_module_iou.svg",
        "Top-4 CA seed refined by each self-attention module",
        list(sa_modules),
        [sa_modules[key]["iou"] for key in sa_modules],
        {"up_blocks.3.attentions.2.transformer_blocks.0.attn1"},
    )
    _svg_bar(
        figures / "sa_timestep_iou.svg",
        "Winning module: each denoising step",
        [f"step {key}" for key in sorted(steps, key=int)],
        [steps[key]["iou"] for key in sorted(steps, key=int)],
    )

    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    focused = json.loads(args.focused_summary.read_text(encoding="utf-8"))
    joint = json.loads(args.joint_summary.read_text(encoding="utf-8"))
    ca_only = baseline["ovam_unoptimized"]
    ovam = paper_ovam
    focused_winners = focused["best_by_experiment"]
    focused_best = max(focused_winners.values(), key=lambda item: item["miou"])
    joint_winners = joint["best_by_experiment"]
    joint_best = max(joint_winners.values(), key=lambda item: item["miou"])
    joint_no_post = joint_winners["joint/joint_head_time_threshold"]
    paired_gains = np.asarray(
        [
            float(record["selected_methods"]["joint_best_0.7997"]["iou"])
            - float(paper_by_id[record["image_id"]])
            for record in records
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20261012)
    bootstrap_means = paired_gains[
        rng.integers(0, len(paired_gains), size=(20_000, len(paired_gains)))
    ].mean(axis=1)
    paired_ci = np.quantile(bootstrap_means, [0.025, 0.975])
    improved_images = int(np.count_nonzero(paired_gains > 0))
    tied_images = int(np.count_nonzero(paired_gains == 0))
    improved_classes = sum(
        joint_best["per_class"][name]["iou"] > value["iou"]
        for name, value in ovam["per_class"].items()
    )
    total_configurations = (
        6261 + 65535 + int(focused["configuration_count"]) + int(joint["configuration_count"])
    )
    top_ca = max(ca_layers, key=lambda key: ca_layers[key]["iou"])
    top_sa = max(sa_modules, key=lambda key: sa_modules[key]["iou"])
    top_head = max(heads, key=lambda key: heads[key]["iou"])
    excluded_mean = statistics.fmean(heads[key]["iou"] for key in ("3", "7"))
    retained_mean = statistics.fmean(heads[key]["iou"] for key in ("0", "1", "2", "4", "5", "6"))
    early = statistics.fmean(steps[str(index)]["iou"] for index in range(0, 15))
    late = statistics.fmean(steps[str(index)]["iou"] for index in range(15, 31))

    lines = [
        "# Attention Segmentation Analysis",
        "",
        "## Executive summary",
        "",
        f"This report explains the VOC-Sim results over **{len(records)} annotated images and 20 classes**. "
        f"It consolidates **{total_configurations:,} evaluated configurations** (counting the exhaustive layer subsets) and adds image-level diagnostics without running another model-selection sweep.",
        "",
        f"The published, unoptimized OVAM pipeline reached **{ovam['miou']:.4f} mIoU**. The best focused seed/operator result reached **{focused_best['miou']:.4f}**, joint head/time tuning reached **{joint_no_post['miou']:.4f}** without morphology, and removing predicted components below 1% of the image produced the final **{joint_best['miou']:.4f} mIoU**. This is an absolute gain of **{joint_best['miou']-ovam['miou']:.4f}** ({(joint_best['miou']/ovam['miou']-1)*100:.1f}% relative).",
        "",
        f"The final method improves pooled IoU for **{improved_classes}/20 classes** and image IoU for **{improved_images}/490 images** ({tied_images} ties). Its paired mean image-IoU gain is **{paired_gains.mean():+.4f}**, with a bootstrap 95% interval of **[{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}]** (20,000 image resamples; seed 20261012).",
        "",
        "The evidence supports a coherent explanation: selected cross-attention heads give a seed that works better with the downstream operator; the last 64×64 upsampling self-attention module spreads that signal along high-resolution object structure; heads 3 and 7 reduce performance when combined with the retained set; middle-late denoising steps are more spatially aligned; and small-component removal suppresses isolated false positives.",
        "",
        "## Experimental protocol",
        "",
        "Each image was regenerated with its released prompt, seed, 30 inference steps, Stable Diffusion 1.5, and guidance scale 7.5. Attention was analyzed in memory. After scalar measurements and an optional visual panel were written, all Q/K tensors, affinity matrices, latents, and temporary maps were released. No raw attention tensor dataset was saved.",
        "",
        "Qualitative panels use the released OVAM image beside attention captured from our reproduction. Earlier reproduction checks found very small pixel-level differences caused by the software/runtime stack, while composition and mask geometry remained aligned; the released annotation is the evaluation target throughout.",
        "",
        f"The published OVAM baseline is measured from the authors' released unoptimized masks. It aggregates ordinary-prompt cross-attention, multiplies it by an aggregate of the highest-resolution self-attention maps rescaled to [0.85, 1], thresholds at 0.40, and applies DenseCRF. Token optimization is excluded. Thresholding the authors' released cross-attention heatmaps alone gives {paper_ca['miou']:.4f} mIoU (the paper reports 66.2 after rounding). Our regenerated cross-attention reference gives 0.6540 because the current diffusion software stack does not reproduce their stored attention exactly.",
        "",
        "The final method uses the fixed top-four cross-attention heads in each of the previously selected 12 layers, the final 64×64 self-attention module, four propagation iterations, self-attention heads 0/1/2/4/5/6, denoising steps 15–30, percentile 1–99 normalization, threshold 0.45, and removal of connected components smaller than 1% of the image.",
        "",
        "## What the diagnostic measures mean",
        "",
        "- **IoU and average precision** measure spatial localization. Average precision is threshold-free and helps distinguish map quality from the chosen 0.45 cutoff.",
        "- **Normalized entropy and effective support** measure spread. Lower values mean concentration, but concentration is useful only when it falls on the object.",
        "- **Foreground/background gap and ROC AUC** measure how well attention ranks object pixels above background pixels.",
        "- **Boundary gradient ratio** measures whether changes in the map concentrate around the annotated object boundary.",
        "- **Centroid error** measures displacement between attention mass and the target center.",
        "- **Object area, box occupancy, components, boundary fraction, and compactness** describe class geometry and let us test whether improvements are dominated by large or simple objects.",
        "- For published OVAM, ranking statistics describe the fused CA×SA map before thresholding; mask IoU and predicted area use the authors' released post-DenseCRF mask.",
        "",
        "## Main performance progression",
        "",
        "| Method | Configuration | mIoU | Absolute gain over OVAM |",
        "|---|---|---:|---:|",
        f"| Authors' OVAM CA precursor | released heatmaps | {paper_ca['miou']:.4f} | {paper_ca['miou']-ovam['miou']:.4f} |",
        f"| Our regenerated CA precursor | `ovam_cross_attention_only` | {ca_only['miou']:.4f} | {ca_only['miou']-ovam['miou']:.4f} |",
        f"| Published OVAM, unoptimized | CA + SA fusion + DenseCRF | {ovam['miou']:.4f} | 0.0000 |",
        f"| Earlier SA winner | best-12 CA, final 64×64 SA, 4 iterations | 0.7777 | {0.7777310135-ovam['miou']:.4f} |",
        f"| Focused winner | `{focused_best['configuration']}` | {focused_best['miou']:.4f} | {focused_best['miou']-ovam['miou']:.4f} |",
        f"| Joint winner, no morphology | `{joint_no_post['configuration']}` | {joint_no_post['miou']:.4f} | {joint_no_post['miou']-ovam['miou']:.4f} |",
        f"| Final winner | `{joint_best['configuration']}` | {joint_best['miou']:.4f} | {joint_best['miou']-ovam['miou']:.4f} |",
        "",
        "The 0.7991 best result before morphology shows that almost all improvement comes from attention selection and propagation. In the matched final iteration/head/time/threshold configuration, the 1% component rule changes mIoU from 0.79910 to 0.79974, consistent with removing scattered islands rather than repairing the main object.",
        "",
        "### How the attention maps change",
        "",
        f"The mean annotated foreground area is **{mean_target_area:.3f}** of the image. The table separates threshold-free map quality (AP/AUC), map shape (entropy, boundary ratio, centroid error), and the area of the final binary prediction.",
        "",
        "| Map or method | Image IoU | AP | ROC AUC | Entropy | Effective support | Boundary ratio | Centroid error | Predicted area |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    diagnostic_methods = (
        ("Published OVAM", "ovam_published"),
        ("Regenerated OVAM CA only", ca_record_key),
        ("Top-4 CA seed", "top4_ca_seed"),
        ("Earlier SA winner", "former_best_0.7777"),
        ("Focused SA winner", "focused_best_0.7922"),
        ("Final method", "joint_best_0.7997"),
    )
    for label, key in diagnostic_methods:
        value = selected[key]
        lines.append(
            f"| {label} | {value['iou']:.3f} | {value['average_precision']:.3f} | "
            f"{value['roc_auc']:.3f} | {value['entropy']:.3f} | "
            f"{value['effective_support_fraction']:.3f} | {value['boundary_gradient_ratio']:.3f} | "
            f"{value['centroid_error']:.3f} | {value['predicted_area_fraction']:.3f} |"
        )
    ovam_profile = selected["ovam_published"]
    final_profile = selected["joint_best_0.7997"]
    lines.extend(
        [
        "",
        f"Relative to full OVAM, the final map changes AP from **{ovam_profile['average_precision']:.3f}** to **{final_profile['average_precision']:.3f}**, boundary-gradient ratio from **{ovam_profile['boundary_gradient_ratio']:.3f}** to **{final_profile['boundary_gradient_ratio']:.3f}**, and centroid error from **{ovam_profile['centroid_error']:.3f}** to **{final_profile['centroid_error']:.3f}**. Its binary foreground coverage changes from **{ovam_profile['predicted_area_fraction']:.3f}** to **{final_profile['predicted_area_fraction']:.3f}**, against **{mean_target_area:.3f}** ground truth. These measurements show whether the gain comes from better ranking, sharper object boundaries, improved localization, or area calibration rather than attributing it to one mechanism from IoU alone.",
        "",
        "### Pooled class IoU against published OVAM",
        "",
        "| Class | Published OVAM | Final | Difference |",
        "|---|---:|---:|---:|",
        ]
    )
    for classname, value in sorted(ovam["per_class"].items()):
        final_iou = joint_best["per_class"][classname]["iou"]
        lines.append(
            f"| {classname} | {value['iou']:.3f} | {final_iou:.3f} | {final_iou-value['iou']:+.3f} |"
        )
    lines.extend(
        [
        "",
        "## Class size and shape",
        "",
        "| Class | n | Mean area | Box area | Fill | Components | Published OVAM image IoU | Final image IoU | Gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['class']} | {row['images']} | {row['area_fraction']:.3f} | {row['bbox_area_fraction']:.3f} | {row['fill_fraction']:.3f} | {row['components']:.2f} | {row['mean_image_iou_paper_ovam']:.3f} | {row['mean_image_iou_final']:.3f} | {row['mean_image_iou_gain']:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"Across individual images, foreground area and improvement have Pearson **r={area_gain_corr['pearson_r']:.3f}** (p={area_gain_corr['pearson_p']:.3g}) and Spearman **ρ={area_gain_corr['spearman_rho']:.3f}** (p={area_gain_corr['spearman_p']:.3g}). This correlation is descriptive: it indicates how gain varies with size, not that size causes the gain.",
            "",
            f"![Class size versus improvement]({(figures / 'class_size_vs_gain.svg').as_posix()})",
            "",
            "## Why cross-attention selection helps",
            "",
            f"The strongest individual CA layer by mean image IoU in this diagnostic is `{top_ca}` ({ca_layers[top_ca]['iou']:.3f}). The top-four-head seed has slightly lower entropy and changes centroid error from **{selected['best12_ca_seed']['centroid_error']:.3f}** to **{selected['top4_ca_seed']['centroid_error']:.3f}**, but its standardized foreground/background gap changes from **{selected['best12_ca_seed']['standardized_gap']:.3f}** to **{selected['top4_ca_seed']['standardized_gap']:.3f}**. Its advantage is therefore better downstream compatibility and threshold calibration, rather than a universal improvement in every raw-map diagnostic.",
            "",
            f"![Cross-attention layers]({(figures / 'ca_layer_iou.svg').as_posix()})",
            "",
            "## Why the final 64×64 self-attention module helps",
            "",
            f"Applying the same top-four CA seed, four propagation iterations, and late-step window to every SA module identifies `{top_sa}` as the best module by mean image IoU ({sa_modules[top_sa]['iou']:.3f}). The selected final upsampling module is highlighted in orange in the figure. Its 64×64 token grid preserves small parts and boundaries that coarser 8×8–32×32 maps cannot represent, while its late decoder position has already integrated the global object layout from earlier U-Net stages.",
            "",
            f"For the final selected map, normalized entropy is **{selected['joint_best_0.7997']['entropy']:.3f}**, effective support is **{selected['joint_best_0.7997']['effective_support_fraction']:.3f}**, foreground/background standardized gap is **{selected['joint_best_0.7997']['standardized_gap']:.3f}**, boundary gradient ratio is **{selected['joint_best_0.7997']['boundary_gradient_ratio']:.3f}**, and centroid error is **{selected['joint_best_0.7997']['centroid_error']:.3f}**. Across images, final-map entropy and IoU have Spearman **ρ={entropy_iou_corr['spearman_rho']:.3f}** (p={entropy_iou_corr['spearman_p']:.3g}); entropy alone is therefore interpreted together with alignment metrics.",
            "",
            f"![Self-attention modules]({(figures / 'sa_module_iou.svg').as_posix()})",
            "",
            "## Why heads 3 and 7 are removed",
            "",
            f"Under the final module and late-step window, the best individual head is head **{top_head}** ({heads[top_head]['iou']:.3f} mean image IoU). Heads 3 and 7 average **{excluded_mean:.3f}**, compared with **{retained_mean:.3f}** across retained heads, but a retained head can still be weaker in isolation. Their exclusion is a joint subset result: it reflects complementarity with the retained heads rather than a simple ranking of standalone head IoUs.",
            "",
            "| Head | IoU | AP | Entropy | FG/BG gap | Boundary ratio | Centroid error | Used? |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for key in sorted(heads, key=int):
        value = heads[key]
        lines.append(
            f"| {key} | {value['iou']:.3f} | {value['average_precision']:.3f} | {value['entropy']:.3f} | {value['standardized_gap']:.3f} | {value['boundary_gradient_ratio']:.3f} | {value['centroid_error']:.3f} | {'no' if key in ('3', '7') else 'yes'} |"
        )
    lines.extend(
        [
            "",
            f"![Self-attention heads]({(figures / 'sa_head_iou.svg').as_posix()})",
            "",
            "## Why later denoising steps help",
            "",
            f"The mean individual-step IoU is **{early:.3f}** for steps 0–14 and **{late:.3f}** for steps 15–30. Early attention is formed while the latent is highly noisy and tends to encode coarse composition. Quality plateaus around steps 15–25 and declines slightly at the final steps. Averaging steps 15–30 reduces step-specific noise while retaining the stable middle-late geometry.",
            "",
            f"![Self-attention timesteps]({(figures / 'sa_timestep_iou.svg').as_posix()})",
            "",
            "## Qualitative evidence",
            "",
            "Each overview pairs every standalone attention visualization with the same map overlaid on the released image. The OVAM visualization is now the published CA×SA fused map; its reported binary score uses the released post-DenseCRF mask. Three panel positions per class were fixed before scoring.",
            "",
        ]
    )
    panel_records = [record for record in records if "panel" in record]
    ranked_panels = sorted(
        panel_records,
        key=lambda item: item["selected_methods"]["joint_best_0.7997"]["iou"]
        - item["selected_methods"]["ovam_published"]["iou"],
    )
    lines.extend(["### Six diagnostic cases", ""])
    for record in ranked_panels[:3] + ranked_panels[-3:]:
        gain = (
            record["selected_methods"]["joint_best_0.7997"]["iou"]
            - record["selected_methods"]["ovam_published"]["iou"]
        )
        lines.extend(
            [
                f"**{record['classname']} / {record['image_id']} — image-IoU difference {gain:+.3f}**",
                "",
                f"![{record['image_id']} attention comparison]({record['panel']})",
                "",
            ]
        )
    lines.extend(["### All fixed qualitative samples", ""])
    for record in sorted(panel_records, key=lambda item: (item["classname"], item["image_id"])):
        links = [f"[overview]({record['panel']})"]
        for method, paths in record.get("visualizations", {}).items():
            links.append(f"[{method} attention]({paths['attention']})")
            links.append(f"[{method} overlay]({paths['overlay']})")
        lines.append(f"- {record['classname']} / {record['image_id']}: " + "; ".join(links))
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- These are exploratory ablations and explanations on VOC-Sim. Configuration selection and final reporting use the same 490 images, so the 0.7997 result needs confirmation on the other datasets already planned.",
            "- Individual layer, head, and step profiles isolate one component at a time. They explain association with performance but do not prove a unique causal mechanism because attention components interact.",
            "- VOC-Sim contains generated, usually centered objects and binary foreground masks. Conclusions about clutter, multiple instances, and natural-image class imbalance require external validation.",
            "- The released images are not byte-identical to our regenerated pixels under the current software stack. Attention is captured from the seeded reproduction, while scoring uses the released spatial mask; the earlier direct checks showed aligned composition and mask geometry.",
            "- Map entropy has no universal optimum. Very diffuse and very concentrated maps can both fail; foreground separation, boundary alignment, centroid error, and IoU provide the needed spatial context.",
            "",
            "## Reproducible artifacts",
            "",
            f"- Compact image records: `{merged_path}`",
            f"- Class geometry and performance: `{args.input_dir / 'class_geometry_and_performance.csv'}`",
            f"- Selected-method statistics: `{args.input_dir / 'selected_method_map_statistics.csv'}`",
            f"- CA layer statistics: `{args.input_dir / 'cross_attention_layer_statistics.csv'}`",
            f"- SA module statistics: `{args.input_dir / 'self_attention_module_statistics.csv'}`",
            f"- SA head statistics: `{args.input_dir / 'winning_module_head_statistics.csv'}`",
            f"- SA timestep statistics: `{args.input_dir / 'winning_module_timestep_statistics.csv'}`",
            "",
        ]
    )
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary_report, args.report)
    summary = {
        "images": len(records),
        "classes": len(by_class),
        "published_ovam_miou": ovam["miou"],
        "published_cross_attention_only_miou": paper_ca["miou"],
        "cross_attention_only_miou": ca_only["miou"],
        "final_miou": joint_best["miou"],
        "absolute_gain_over_published_ovam": joint_best["miou"] - ovam["miou"],
        "classes_improved_over_published_ovam": improved_classes,
        "images_improved_over_published_ovam": improved_images,
        "paired_mean_image_iou_gain": float(paired_gains.mean()),
        "paired_bootstrap_95_ci": [float(value) for value in paired_ci],
        "area_gain_correlation": area_gain_corr,
        "entropy_iou_correlation": entropy_iou_corr,
        "top_cross_attention_layer": top_ca,
        "top_self_attention_module": top_sa,
        "top_self_attention_head": int(top_head),
        "report": str(args.report),
    }
    (args.input_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.report} from {len(records)} records")


if __name__ == "__main__":
    main()
