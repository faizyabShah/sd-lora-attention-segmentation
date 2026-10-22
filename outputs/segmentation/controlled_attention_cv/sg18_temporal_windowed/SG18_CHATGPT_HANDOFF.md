# SG18 handoff for ChatGPT

## Experiment and completion

Locked, training-free SG18 temporal-windowed **one-pass** self-attention propagation. Same frozen class-word CA seed, SA module 15, all 8 heads, 31 diffusion calls (0–30), independent min-max normalization and no CRF. W/REF averages all 31 calls; W/T1 uses calls 10–20; W/T2 uses 10–30; W/T3 uses 6–30. W/P1 is a two-pass external benchmark and was ineligible for SG18 selection. No optimized tokens or training.

Production Slurm job 235290 completed successfully. Coverage: 490/490 VOC-Sim and 707/707 COCO-Cap images; eight VOC visual sheets and timing completed. The five-fold selection and thresholds used VOC only; COCO is frozen transfer. 20,000 paired bootstrap resamples for mean-image IoU intervals.

## VOC selection

- Fold winners: W/T1, W/T1, W/T1, W/T1, W/T1 (5/5 votes).
- Selected OOF mIoU 0.793916 versus W/REF 0.787280; gain +0.006636. W/T1 promoted.
- Full-VOC thresholds: 0.425 for REF and T1/T2/T3; 0.450 for P1. The five held-out fold thresholds can differ and are in selection.json.

## Main results

Values are class-macro mIoU. Parentheses show delta versus W/REF in that dataset. P1 is an external benchmark, not a SG18 candidate.

| Method | Forced VOC OOF | Full VOC | Frozen COCO | VOC precision / recall | COCO precision / recall |
|---|---:|---:|---:|---:|---:|
| W/REF | 0.7873 | 0.7886 (+0.0000) | 0.6172 (+0.0000) | 0.8529 / 0.9063 | 0.6600 / 0.8698 |
| W/T1 | 0.7939 | 0.7959 (+0.0073) | 0.6285 (+0.0113) | 0.8646 / 0.9016 | 0.6755 / 0.8670 |
| W/T2 | 0.7936 | 0.7936 (+0.0050) | 0.6240 (+0.0068) | 0.8616 / 0.9017 | 0.6763 / 0.8576 |
| W/T3 | 0.7916 | 0.7935 (+0.0048) | 0.6227 (+0.0056) | 0.8604 / 0.9033 | 0.6708 / 0.8631 |
| W/P1 | 0.7997 | 0.8006 (+0.0120) | 0.6245 (+0.0073) | 0.8673 / 0.9047 | 0.6624 / 0.8764 |

## Paired image-level differences versus REF

These are mean-image IoU differences (different estimand from class-macro mIoU) and bootstrap 95% intervals.

| Dataset | Method | Mean difference | 95% interval | Better / worse / tie images |
|---|---|---:|---:|---:|
| VOC | W/T1 | +0.0074 | [+0.0051, +0.0099] | 358 / 132 / 0 |
| VOC | W/T2 | +0.0048 | [+0.0023, +0.0074] | 327 / 163 / 0 |
| VOC | W/T3 | +0.0048 | [+0.0032, +0.0066] | 358 / 132 / 0 |
| VOC | W/P1 | +0.0144 | [+0.0108, +0.0179] | 370 / 120 / 0 |
| COCO | W/T1 | +0.0105 | [+0.0086, +0.0123] | 521 / 185 / 1 |
| COCO | W/T2 | +0.0085 | [+0.0057, +0.0113] | 493 / 213 / 1 |
| COCO | W/T3 | +0.0064 | [+0.0049, +0.0080] | 529 / 177 / 1 |
| COCO | W/P1 | +0.0139 | [+0.0109, +0.0171] | 464 / 242 / 1 |

## Mechanism for promoted W/T1

At the VOC-frozen method thresholds, with CA mask threshold 0.45, the values below compare CA-relative additions made by T1 and REF. Counts pool pixels across images; they are descriptive and must not be read as independent sample counts. Fixed-0.45 diagnostics for all methods are in results.json.

| Dataset | REF-added TP retained by T1 | REF-added FP removed by T1 | FP removed / FP introduced in direct REF comparison | TP lost / TP gained in direct REF comparison |
|---|---:|---:|---:|---:|
| VOC | 95.2% | 18.8% | 453,720 / 95,469 | 298,077 / 162,450 |
| COCO | 96.9% | 12.9% | 1,404,320 / 341,586 | 268,324 / 205,218 |

The selected window tends to reduce false positives while losing some true positives. On VOC it raised precision from 0.8529 to 0.8646 and lowered recall from 0.9063 to 0.9016; on COCO precision rose 0.6600 to 0.6755 and recall fell 0.8698 to 0.8670. This is consistent with the mIoU gain, but the pooled pixel counts do not establish a causal mechanism by themselves.

## Per-class IoU differences versus REF

Each entry is `method IoU (delta versus REF)`. VOC and COCO have slightly different class spelling for dining table/potted plant.

### VOC

| Class | REF | T1 | T2 | T3 | P1 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.835 | 0.838 (+0.002) | 0.837 (+0.002) | 0.839 (+0.004) | 0.867 (+0.031) |
| bicycle | 0.737 | 0.752 (+0.014) | 0.744 (+0.006) | 0.742 (+0.005) | 0.748 (+0.011) |
| bird | 0.840 | 0.854 (+0.014) | 0.850 (+0.009) | 0.850 (+0.010) | 0.839 (-0.001) |
| boat | 0.812 | 0.830 (+0.017) | 0.831 (+0.019) | 0.826 (+0.014) | 0.836 (+0.024) |
| bottle | 0.856 | 0.870 (+0.014) | 0.866 (+0.010) | 0.861 (+0.005) | 0.864 (+0.008) |
| bus | 0.909 | 0.916 (+0.007) | 0.916 (+0.007) | 0.916 (+0.007) | 0.905 (-0.004) |
| car | 0.825 | 0.836 (+0.011) | 0.835 (+0.009) | 0.836 (+0.010) | 0.826 (+0.000) |
| cat | 0.813 | 0.805 (-0.009) | 0.803 (-0.010) | 0.807 (-0.006) | 0.826 (+0.013) |
| chair | 0.629 | 0.662 (+0.033) | 0.650 (+0.021) | 0.645 (+0.016) | 0.709 (+0.080) |
| cow | 0.843 | 0.855 (+0.012) | 0.857 (+0.015) | 0.852 (+0.009) | 0.844 (+0.001) |
| dining-table | 0.453 | 0.465 (+0.012) | 0.464 (+0.012) | 0.462 (+0.009) | 0.449 (-0.004) |
| dog | 0.903 | 0.907 (+0.004) | 0.906 (+0.004) | 0.905 (+0.002) | 0.902 (-0.001) |
| horse | 0.866 | 0.875 (+0.009) | 0.872 (+0.006) | 0.873 (+0.007) | 0.875 (+0.009) |
| motorbike | 0.840 | 0.851 (+0.011) | 0.852 (+0.012) | 0.849 (+0.009) | 0.851 (+0.011) |
| person | 0.706 | 0.678 (-0.028) | 0.672 (-0.034) | 0.687 (-0.019) | 0.734 (+0.028) |
| potted-plant | 0.731 | 0.732 (+0.001) | 0.724 (-0.007) | 0.728 (-0.003) | 0.749 (+0.018) |
| sheep | 0.800 | 0.797 (-0.003) | 0.804 (+0.003) | 0.803 (+0.003) | 0.799 (-0.001) |
| sofa | 0.937 | 0.939 (+0.002) | 0.936 (-0.001) | 0.938 (+0.001) | 0.948 (+0.011) |
| train | 0.643 | 0.666 (+0.022) | 0.665 (+0.022) | 0.660 (+0.017) | 0.663 (+0.019) |
| tv | 0.793 | 0.792 (-0.001) | 0.787 (-0.006) | 0.790 (-0.003) | 0.779 (-0.014) |

### COCO

| Class | REF | T1 | T2 | T3 | P1 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.564 | 0.610 (+0.046) | 0.605 (+0.041) | 0.586 (+0.022) | 0.587 (+0.023) |
| bicycle | 0.581 | 0.608 (+0.027) | 0.604 (+0.023) | 0.596 (+0.015) | 0.569 (-0.012) |
| bird | 0.713 | 0.717 (+0.004) | 0.727 (+0.014) | 0.718 (+0.005) | 0.722 (+0.010) |
| boat | 0.517 | 0.537 (+0.019) | 0.541 (+0.024) | 0.534 (+0.017) | 0.517 (-0.000) |
| bottle | 0.457 | 0.466 (+0.009) | 0.470 (+0.013) | 0.466 (+0.010) | 0.464 (+0.007) |
| bus | 0.920 | 0.926 (+0.006) | 0.928 (+0.008) | 0.927 (+0.007) | 0.905 (-0.015) |
| car | 0.568 | 0.571 (+0.003) | 0.565 (-0.003) | 0.572 (+0.004) | 0.558 (-0.010) |
| cat | 0.829 | 0.833 (+0.004) | 0.827 (-0.002) | 0.829 (+0.000) | 0.846 (+0.018) |
| chair | 0.419 | 0.431 (+0.012) | 0.421 (+0.002) | 0.421 (+0.002) | 0.436 (+0.017) |
| cow | 0.801 | 0.816 (+0.015) | 0.811 (+0.010) | 0.808 (+0.007) | 0.814 (+0.014) |
| dining table | 0.374 | 0.363 (-0.011) | 0.352 (-0.022) | 0.361 (-0.014) | 0.386 (+0.012) |
| dog | 0.745 | 0.749 (+0.004) | 0.751 (+0.005) | 0.749 (+0.004) | 0.743 (-0.002) |
| horse | 0.635 | 0.654 (+0.019) | 0.658 (+0.022) | 0.651 (+0.016) | 0.645 (+0.010) |
| motorbike | 0.618 | 0.633 (+0.015) | 0.633 (+0.015) | 0.629 (+0.011) | 0.618 (+0.001) |
| person | 0.219 | 0.227 (+0.008) | 0.210 (-0.009) | 0.216 (-0.003) | 0.235 (+0.016) |
| potted plant | 0.643 | 0.653 (+0.010) | 0.649 (+0.006) | 0.648 (+0.005) | 0.670 (+0.027) |
| sheep | 0.846 | 0.867 (+0.021) | 0.867 (+0.022) | 0.862 (+0.016) | 0.863 (+0.017) |
| sofa | 0.590 | 0.601 (+0.011) | 0.595 (+0.004) | 0.593 (+0.003) | 0.602 (+0.012) |
| train | 0.608 | 0.617 (+0.010) | 0.618 (+0.010) | 0.616 (+0.008) | 0.614 (+0.007) |
| tv | 0.697 | 0.691 (-0.006) | 0.650 (-0.047) | 0.673 (-0.024) | 0.692 (-0.005) |

## Timing and limits

All three timing methods run the same UNet and capture the same attentions; the window only changes post-capture aggregation. Three separate live generations per method, one image. Median end-to-end and post-capture aggregation times:

| Method | End-to-end seconds | Aggregation seconds |
|---|---:|---:|
| W/REF | 16.064 | 0.287 |
| W/T1 | 16.445 | 0.090 |
| W/P1 | 17.077 | 0.343 |

Do not claim an inference speedup; capture dominates and only three timing repeats were run. T1 is the best SG18 one-pass option by VOC selection and COCO transfer, but the two-pass P1 remains higher on full VOC (0.8006 versus 0.7959). T1 is higher on COCO (0.6285 versus 0.6245). This is specific to the W class-word pipeline and these two datasets.

## Files in this package

- `selection.json`: fold choices, thresholds, forced OOF scores, selection guard.
- `results.json`: full metrics, per-class values, paired bootstrap and mechanism at frozen and fixed thresholds.
- `per_image_selected.csv`: image-level outcomes for independent analysis.
- `timing.json`: three trials per timing method.
- `visuals/`: eight visual sheets, including largest gain/loss examples.
- `completed.json`: output coverage.
- `report.md`: compact auto-generated summary.

Questions for ChatGPT: Does the T1 result warrant follow-up? How would you interpret its precision/recall and per-class trade-offs, particularly VOC person (negative) versus COCO person (positive)? What single controlled follow-up would clarify the mechanism while keeping VOC-only selection and COCO untouched?
