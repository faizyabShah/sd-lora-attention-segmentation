# Pre-SG18 / Pre-SG21 analysis for ChatGPT

Diagnostic only: frozen W CA, module-15 SA, 31 calls × 8 heads, threshold 0.45; no method or threshold was selected.
VOC 490 images; COCO 707 images.

## Strongest VOC temporal rankings

Both raw and inverse ROC are supplied in the CSV. This table ranks the better VOC direction; COCO columns remain raw so direction can be checked.

| feature | VOC ROC | COCO ROC | COCO inverse ROC | COCO TP vs >16 FP | COCO useful/harmful |
| --- | --- | --- | --- | --- | --- |
| middle_up_fraction | 0.617 | 0.617 | 0.383 | 0.611 | 0.517 |
| middle_down_fraction | 0.617 | 0.383 | 0.617 | 0.389 | 0.483 |
| temporal_median | 0.604 | 0.675 | 0.325 | 0.685 | 0.587 |
| temporal_iqr | 0.596 | 0.518 | 0.482 | 0.548 | 0.547 |
| longest_up_run | 0.595 | 0.587 | 0.413 | 0.576 | 0.493 |
| temporal_mean | 0.593 | 0.657 | 0.343 | 0.666 | 0.576 |
| late_down_fraction | 0.592 | 0.392 | 0.608 | 0.397 | 0.489 |
| late_up_fraction | 0.592 | 0.608 | 0.392 | 0.603 | 0.511 |
| temporal_mad | 0.590 | 0.506 | 0.494 | 0.536 | 0.539 |
| time_down_fraction | 0.587 | 0.415 | 0.585 | 0.424 | 0.508 |
| time_up_fraction | 0.587 | 0.585 | 0.415 | 0.576 | 0.492 |
| longest_down_run | 0.584 | 0.414 | 0.586 | 0.420 | 0.507 |

## Fixed stage maps versus CA

| dataset | stage | tp_add | fp_add | tp_fp_ratio | fp_gt16_fraction | fp_separate_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| voc | early | 6180356 | 4563531 | 1.354 | 0.643 | 0.008 |
| voc | middle | 5387965 | 1541872 | 3.494 | 0.456 | 0.035 |
| voc | late | 5061759 | 1567839 | 3.228 | 0.438 | 0.042 |
| coco | early | 9118524 | 17908709 | 0.509 | 0.804 | 0.041 |
| coco | middle | 7751504 | 8241630 | 0.941 | 0.776 | 0.112 |
| coco | late | 6766342 | 7233430 | 0.935 | 0.774 | 0.144 |

## Cross-dataset cell stability

| metric | n | rho | p |
| --- | --- | --- | --- |
| tp_fp_ratio | 248 | 0.902 | 0.000 |
| useful_change_fraction | 248 | 0.738 | 0.000 |
| fp_gt16_fraction | 248 | 0.902 | 0.000 |
| fp_separate_fraction | 248 | 0.666 | 0.000 |

## Structural correlations

| dataset | descriptor | outcome | n | rho |
| --- | --- | --- | --- | --- |
| coco | local_mass_r1 | fp_separate_fraction | 248 | 0.949 |
| coco | diagonal_mass | fp_separate_fraction | 248 | 0.948 |
| coco | local_mass_r2 | fp_separate_fraction | 248 | 0.946 |
| coco | top10_mass | fp_separate_fraction | 248 | 0.945 |
| coco | top5_mass | fp_separate_fraction | 248 | 0.945 |
| coco | top1_mass | fp_separate_fraction | 248 | 0.945 |
| coco | entropy | tp_fp_ratio | 248 | -0.944 |
| coco | entropy | fp_separate_fraction | 248 | -0.941 |
| coco | local_mass_r4 | fp_separate_fraction | 248 | 0.940 |
| coco | top10_mass | fp_gt16_fraction | 248 | -0.939 |
| coco | median_row_entropy | fp_separate_fraction | 248 | -0.939 |
| coco | top5_mass | fp_gt16_fraction | 248 | -0.938 |
| coco | local_mass_r4 | tp_fp_ratio | 248 | 0.937 |
| coco | entropy | fp_gt16_fraction | 248 | 0.936 |
| coco | median_row_entropy | tp_fp_ratio | 248 | -0.935 |

## Interpretation limits

- Observation maps are each normalized independently before applying the fixed diagnostic threshold. Their event totals measure each observation against the same CA prediction; they do not sum to the ordinary mean effect.
- Stage maps use the fixed early/middle/late windows and are scored the same way.
- TP/FP ratios can be undefined when FP=0. Null AUC denotes a missing class.
- Histogram AUC uses 512 fixed bins; correlated pixels make these descriptive rankings rather than independent-sample significance tests.
- GT is used only for event labels, distance, connected components, and diagnostics.

Full tables: `results.json`, `temporal_feature_auc.csv`, `head_call_metrics.csv`, `head_summary.csv`, `head_stage_summary.csv`, `call_summary.csv`, `structural_correlations.csv`, and PNG heatmaps/curves.
