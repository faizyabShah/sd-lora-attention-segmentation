# Pre-SG19 late support diagnostic

## Main findings

- All 490 VOC and 707 COCO images were evaluated. Candidate middle-added pixels: 6,837,692 VOC (5,225,499 TP; 1,612,193 FP) and 12,671,265 COCO (6,481,988 TP; 6,189,277 FP). This is a diagnostic of candidate pixels, not a new segmentation result.
- Raw late score distinguishes TP from FP with pixel-pooled ROC-AUC 0.586 on VOC and 0.675 on COCO. Late median and 20% trimmed mean are essentially equivalent (VOC 0.585/0.584; COCO 0.676/0.676). At the 0.45 control threshold, raw-score ROC-AUC is 0.571 VOC and 0.678 COCO.
- VOC-defined raw late-score deciles have TP/FP ratios 2.71 in the bottom bin and 9.57 in the top bin; on COCO, using the same edges, the ratios rise monotonically from 0.55 to 7.68. The VOC lower bins are not fully monotonic.
- The more specifically temporal features transfer less well: `L-c` gives 0.535 VOC / 0.564 COCO ROC-AUC; the VOC-oriented `L-M` gives 0.580 / 0.548; retention ratio gives 0.566 / 0.533; late upward consensus gives 0.556 / 0.530. Raw score may reflect general semantic confidence rather than uniquely useful late support; middle-score and CA-score controls have not yet been evaluated.
- `L>c` holds for 96.8% of VOC and 96.1% of COCO candidate pixels, so its sign is not a useful veto. For `L>M`, TP rates are 0.691 VOC and 0.464 COCO; for `L<M`, they are 0.802 and 0.538. Late strengthening is less TP enriched than late weakening, opposite a simple reinforcement story.
- VOC person is a caution: its best VOC feature is lower `L-M` (ROC-AUC 0.695), but that orientation transfers at only 0.481 on COCO. Chair transfers 0.700 to 0.683; bicycle 0.622 to 0.726. Class results are heterogeneous.
- Verdict for the preregistered go/no-go: **mixed, not a clear go**. The COCO raw-score AUC exceeds 0.65 and reliability bins are encouraging; VOC overall AUC is below 0.65, and the distinctive late-change/consensus signals are weak. No late veto threshold, SG19 mask, or CRF was tested.
- The diagnostic required replay because individual late observations were not cached. Mean absolute replay-versus-SG18 map differences were 0.00078/0.00096 for VOC middle/late and 0.00115/0.00137 for COCO. Some individual-image mean differences reached about 0.03; this is a reproducibility limitation when comparing against the earlier SG18 run.

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

## How to read the diagnostic

ROC-AUC is pixel pooled, not image macro averaged. PR-AUC depends on the TP prevalence of each contrast; use the TP/FP support counts in `auc_table.csv` when comparing it. Feature direction is defined once on VOC. Higher support bins should have higher TP/FP ratios on both datasets if a late veto is plausible. No SG19 mask or cutoff was tested.

## Direct sign splits at the frozen 0.425 threshold


### VOC

| Feature | Sign | Candidate fraction | TP rate | FP >16 rate | Separate FP rate |
|---|---|---:|---:|---:|---:|
| late_minus_ca | positive | 0.9684 | 0.7668 | 0.1125 | 0.0096 |
| late_minus_ca | negative | 0.0316 | 0.6863 | 0.1158 | 0.0045 |
| late_minus_ca | zero | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| late_minus_middle | positive | 0.3377 | 0.6910 | 0.1368 | 0.0148 |
| late_minus_middle | negative | 0.6623 | 0.8015 | 0.1003 | 0.0066 |
| late_minus_middle | zero | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### COCO

| Feature | Sign | Candidate fraction | TP rate | FP >16 rate | Separate FP rate |
|---|---|---:|---:|---:|---:|
| late_minus_ca | positive | 0.9606 | 0.5148 | 0.3846 | 0.0510 |
| late_minus_ca | negative | 0.0394 | 0.4329 | 0.3665 | 0.0411 |
| late_minus_ca | zero | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| late_minus_middle | positive | 0.3629 | 0.4644 | 0.4179 | 0.0698 |
| late_minus_middle | negative | 0.6371 | 0.5384 | 0.3645 | 0.0396 |
| late_minus_middle | zero | 0.0000 | 0.4545 | 0.5455 | 0.0909 |

## Files

- `results.json`: full exact AUC/PR-AUC, direct sign splits, class transfer, bin edges and source-map drift.
- `auc_table.csv`: every feature, contrast, dataset and threshold.
- `reliability_bins.csv`: transferred VOC bins and TP/FP ratios.
- `late_support_tp_fp_ratio.png`: VOC/COCO ratio curves.
- `report.md`: compact top-line table.

Question for ChatGPT: Is any late-support signal strong and transferable enough to justify a locked SG19 middle-proposal plus late-veto experiment? Consider ROC-AUC, PR-AUC prevalence, the VOC-to-COCO bin trend, and the person-class tradeoff.
