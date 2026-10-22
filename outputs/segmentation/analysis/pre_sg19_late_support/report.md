# Pre-SG19 late support diagnostic

Middle-added pixels require raw M>c, CA background at 0.45 and middle foreground at the listed threshold. No segmentation method or cutoff is selected. Feature direction and reliability bins are defined on VOC only.

## Primary threshold 0.425

| Feature | Direction | VOC TP/FP ROC | COCO TP/FP ROC | COCO TP vs >16 FP | COCO TP vs separate FP |
|---|---|---:|---:|---:|---:|
| late_score | higher | 0.586 | 0.675 | 0.688 | 0.675 |
| late_minus_ca | higher | 0.535 | 0.564 | 0.561 | 0.567 |
| late_minus_middle | lower | 0.580 | 0.548 | 0.551 | 0.615 |
| retention_ratio | lower | 0.566 | 0.533 | 0.537 | 0.598 |
| late_up_consensus | higher | 0.556 | 0.530 | 0.511 | 0.516 |
| late_vs_middle_consensus | lower | 0.578 | 0.540 | 0.538 | 0.600 |
| late_median | higher | 0.585 | 0.676 | 0.689 | 0.677 |
| late_trimmed_mean | higher | 0.584 | 0.676 | 0.688 | 0.676 |
| late_std | lower | 0.566 | 0.460 | 0.420 | 0.444 |
| late_mad | lower | 0.542 | 0.464 | 0.431 | 0.442 |

Exact ROC and PR AUC for all four contrasts and both thresholds are in `auc_table.csv`. Bin counts, TP/FP ratios, distant/separate FP rates and direct L>c/L>M splits are in `results.json`.

## VOC-selected per-class feature

| Class | Feature | VOC ROC | COCO ROC |
|---|---|---:|---:|
| person | late_minus_middle | 0.695 | 0.481 |
| chair | late_trimmed_mean | 0.700 | 0.683 |
| bird | late_mad | 0.621 | 0.480 |
| bicycle | late_score | 0.622 | 0.726 |
| cow | late_up_consensus | 0.668 | 0.577 |
| dining table | late_score | 0.769 | 0.864 |
| tv | late_minus_middle | 0.684 | 0.575 |

Reliability plot: `late_support_tp_fp_ratio.png`. Temporary candidate-pixel feature caches are deleted after successful finalization; only reduced reports remain.
