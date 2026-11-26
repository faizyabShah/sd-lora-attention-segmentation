# Map analysis part 1: where does frozen BRACE lose? (exploratory diagnostics)

Stored paper capture (VOC-Sim 490, COCO-Cap 707). Ground truth is used only to measure. BRACE mask = V3 → τ 0.55 → DenseCRF R2. Per-image IoU means are class-macro (mean over classes of the per-image mean) unless stated; the paper's pooled class-macro mIoU is given for reference.

## VOC-Sim (490 images)

### 1a. Bounds

| what | per-image IoU (class-macro) | pooled class-macro mIoU |
|---|---:|---:|
| BRACE @ τ 0.55 + R2 | 0.835 | 0.830 |
| BRACE, OOF τ + R2 | — | 0.830 |
| oracle per-image τ + R2 (grid 0.30–0.90) | 0.896 | 0.902 |
| oracle per-image τ, raw V3 (grid) | 0.888 | — |
| perfect ranking of seed (best of all thresholds, raw) | 0.776 | — |
| perfect ranking of P1 (best of all thresholds, raw) | 0.884 | — |
| perfect ranking of V0 (best of all thresholds, raw) | 0.894 | — |
| perfect ranking of V3 (best of all thresholds, raw) | 0.894 | — |
| perfect ranking of C16 (best of all thresholds, raw) | 0.893 | — |

AUROC (pixel ranking quality), class-macro: seed 0.953, P1 0.978, V0 0.980, V3 0.981, C16 0.981

### 1b. Error anatomy of the BRACE mask (fractions of the object area, mean over images)

| subset | n | IoU | FN boundary | FN interior | FP boundary | FP far | instances (mean) | predicted components (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 490 | 0.835 | 0.015 | 0.049 | 0.051 | 0.173 | 3.64 | 39.62 |
| area quintile 1 (0.02–0.17) | 98 | 0.750 | 0.023 | 0.034 | 0.090 | 0.573 | 2.67 | 48.20 |
| area quintile 2 (0.17–0.27) | 98 | 0.820 | 0.014 | 0.027 | 0.070 | 0.178 | 2.82 | 46.78 |
| area quintile 3 (0.27–0.38) | 98 | 0.883 | 0.011 | 0.016 | 0.047 | 0.066 | 2.60 | 44.16 |
| area quintile 4 (0.38–0.57) | 98 | 0.894 | 0.014 | 0.037 | 0.034 | 0.033 | 3.41 | 35.71 |
| area quintile 5 (0.57–0.98) | 98 | 0.827 | 0.013 | 0.133 | 0.016 | 0.017 | 6.71 | 23.27 |
| multi-instance (≥2 components) | 194 | 0.793 | 0.017 | 0.064 | 0.063 | 0.227 | 7.68 | 47.47 |

Per class (mean IoU, prec, rec, FN interior, FP far):

| class | n | IoU | prec | rec | FN interior | FP far | worst failure type |
|---|---:|---:|---:|---:|---:|---:|---|
| aeroplane | 24 | 0.891 | 0.937 | 0.949 | 0.033 | 0.023 | — (0) |
| bicycle | 25 | 0.794 | 0.816 | 0.968 | 0.022 | 0.108 | — (0) |
| bird | 25 | 0.909 | 0.942 | 0.964 | 0.014 | 0.027 | — (0) |
| boat | 25 | 0.820 | 0.879 | 0.927 | 0.048 | 0.103 | — (0) |
| bottle | 25 | 0.890 | 0.936 | 0.948 | 0.046 | 0.041 | part only (under-segment) (1) |
| bus | 25 | 0.907 | 0.925 | 0.981 | 0.013 | 0.052 | — (0) |
| car | 23 | 0.857 | 0.914 | 0.936 | 0.055 | 0.056 | — (0) |
| cat | 25 | 0.865 | 0.977 | 0.886 | 0.099 | 0.007 | — (0) |
| chair | 24 | 0.799 | 0.823 | 0.969 | 0.012 | 0.090 | — (0) |
| cow | 25 | 0.915 | 0.975 | 0.937 | 0.047 | 0.004 | part only (under-segment) (1) |
| dining-table | 25 | 0.400 | 0.409 | 0.971 | 0.020 | 2.205 | bleed (over-segment) (18) |
| dog | 25 | 0.932 | 0.969 | 0.962 | 0.025 | 0.008 | — (0) |
| horse | 23 | 0.903 | 0.942 | 0.957 | 0.032 | 0.035 | — (0) |
| motorbike | 25 | 0.871 | 0.895 | 0.971 | 0.020 | 0.058 | — (0) |
| person | 24 | 0.824 | 0.931 | 0.888 | 0.101 | 0.066 | — (0) |
| potted-plant | 25 | 0.746 | 0.926 | 0.801 | 0.151 | 0.042 | part only (under-segment) (2) |
| sheep | 25 | 0.828 | 0.973 | 0.854 | 0.128 | 0.011 | part only (under-segment) (3) |
| sofa | 25 | 0.950 | 0.970 | 0.980 | 0.014 | 0.014 | — (0) |
| train | 25 | 0.739 | 0.758 | 0.964 | 0.023 | 0.436 | bleed (over-segment) (2) |
| tv | 22 | 0.867 | 0.958 | 0.903 | 0.089 | 0.031 | part only (under-segment) (1) |

Failure types (IoU < 0.5): bleed (over-segment) 20, mixed 2, ok 460, part only (under-segment) 8

### 1c. Worst 10% (49 images)

Contact sheet `fig/part1_worst_voc.jpg` (per image: image | truth | V3 | BRACE mask; in IoU order, row-major, two per row).

| rank | image | class | IoU | prec | rec | type |
|---:|---|---|---:|---:|---:|---|
| 1 | ovam-voc-sim-261 | dining-table | 0.12 | 0.12 | 1.00 | bleed (over-segment) |
| 2 | ovam-voc-sim-260 | dining-table | 0.14 | 0.14 | 1.00 | bleed (over-segment) |
| 3 | ovam-voc-sim-274 | dining-table | 0.14 | 0.15 | 0.74 | mixed |
| 4 | ovam-voc-sim-463 | train | 0.14 | 0.15 | 0.77 | mixed |
| 5 | ovam-voc-sim-267 | dining-table | 0.15 | 0.15 | 1.00 | bleed (over-segment) |
| 6 | ovam-voc-sim-251 | dining-table | 0.17 | 0.17 | 1.00 | bleed (over-segment) |
| 7 | ovam-voc-sim-404 | sheep | 0.20 | 1.00 | 0.20 | part only (under-segment) |
| 8 | ovam-voc-sim-273 | dining-table | 0.21 | 0.21 | 0.99 | bleed (over-segment) |
| 9 | ovam-voc-sim-266 | dining-table | 0.23 | 0.23 | 0.99 | bleed (over-segment) |
| 10 | ovam-voc-sim-257 | dining-table | 0.24 | 0.24 | 0.98 | bleed (over-segment) |
| 11 | ovam-voc-sim-255 | dining-table | 0.27 | 0.27 | 1.00 | bleed (over-segment) |
| 12 | ovam-voc-sim-423 | sheep | 0.30 | 1.00 | 0.30 | part only (under-segment) |
| 13 | ovam-voc-sim-268 | dining-table | 0.31 | 0.31 | 0.98 | bleed (over-segment) |
| 14 | ovam-voc-sim-265 | dining-table | 0.31 | 0.31 | 0.95 | bleed (over-segment) |
| 15 | ovam-voc-sim-263 | dining-table | 0.32 | 0.32 | 0.93 | bleed (over-segment) |
| 16 | ovam-voc-sim-250 | dining-table | 0.33 | 0.33 | 1.00 | bleed (over-segment) |
| 17 | ovam-voc-sim-272 | dining-table | 0.36 | 0.36 | 1.00 | bleed (over-segment) |
| 18 | ovam-voc-sim-269 | dining-table | 0.36 | 0.37 | 0.98 | bleed (over-segment) |
| 19 | ovam-voc-sim-381 | potted-plant | 0.37 | 0.99 | 0.37 | part only (under-segment) |
| 20 | ovam-voc-sim-468 | train | 0.37 | 0.37 | 1.00 | bleed (over-segment) |
| 21 | ovam-voc-sim-424 | sheep | 0.38 | 1.00 | 0.38 | part only (under-segment) |
| 22 | ovam-voc-sim-392 | potted-plant | 0.42 | 0.98 | 0.42 | part only (under-segment) |
| 23 | ovam-voc-sim-256 | dining-table | 0.45 | 0.45 | 1.00 | bleed (over-segment) |
| 24 | ovam-voc-sim-254 | dining-table | 0.45 | 0.45 | 1.00 | bleed (over-segment) |
| 25 | ovam-voc-sim-492 | tv | 0.46 | 0.92 | 0.48 | part only (under-segment) |
| 26 | ovam-voc-sim-101 | bottle | 0.46 | 0.88 | 0.50 | part only (under-segment) |
| 27 | ovam-voc-sim-459 | train | 0.47 | 0.47 | 1.00 | bleed (over-segment) |
| 28 | ovam-voc-sim-271 | dining-table | 0.47 | 0.47 | 1.00 | bleed (over-segment) |
| 29 | ovam-voc-sim-225 | cow | 0.48 | 0.96 | 0.49 | part only (under-segment) |
| 30 | ovam-voc-sim-252 | dining-table | 0.50 | 0.50 | 1.00 | bleed (over-segment) |
| 31 | ovam-voc-sim-482 | tv | 0.51 | 0.97 | 0.52 | ok |
| 32 | ovam-voc-sim-313 | horse | 0.51 | 0.97 | 0.52 | ok |
| 33 | ovam-voc-sim-203 | chair | 0.52 | 0.54 | 0.95 | ok |
| 34 | ovam-voc-sim-456 | train | 0.53 | 0.55 | 0.92 | ok |
| 35 | ovam-voc-sim-365 | person | 0.53 | 1.00 | 0.53 | ok |
| 36 | ovam-voc-sim-94 | boat | 0.53 | 0.54 | 0.98 | ok |
| 37 | ovam-voc-sim-377 | potted-plant | 0.54 | 0.98 | 0.55 | ok |
| 38 | ovam-voc-sim-262 | dining-table | 0.55 | 0.55 | 1.00 | ok |
| 39 | ovam-voc-sim-102 | bottle | 0.55 | 0.55 | 1.00 | ok |
| 40 | ovam-voc-sim-399 | potted-plant | 0.56 | 0.95 | 0.58 | ok |
| 41 | ovam-voc-sim-183 | cat | 0.56 | 1.00 | 0.56 | ok |
| 42 | ovam-voc-sim-172 | car | 0.57 | 1.00 | 0.57 | ok |
| 43 | ovam-voc-sim-202 | chair | 0.58 | 0.58 | 1.00 | ok |
| 44 | ovam-voc-sim-471 | train | 0.58 | 0.58 | 1.00 | ok |
| 45 | ovam-voc-sim-216 | chair | 0.58 | 0.62 | 0.89 | ok |
| 46 | ovam-voc-sim-84 | boat | 0.58 | 0.64 | 0.86 | ok |
| 47 | ovam-voc-sim-370 | person | 0.58 | 0.59 | 0.99 | ok |
| 48 | ovam-voc-sim-189 | cat | 0.58 | 1.00 | 0.58 | ok |
| 49 | ovam-voc-sim-374 | person | 0.58 | 0.59 | 0.99 | ok |

### 1d. Calibration and threshold

Foreground fraction per V3 value bin (all pixels pooled):

| V3 bin | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg fraction | 0.01 | 0.02 | 0.05 | 0.12 | 0.22 | 0.47 | 0.68 | 0.83 | 0.94 | 0.98 |
| share of pixels | 0.300 | 0.131 | 0.071 | 0.047 | 0.042 | 0.044 | 0.055 | 0.072 | 0.108 | 0.131 |

Per-image best τ (+R2): mean 0.575, sd 0.140; share at the grid edges (0.30 / 0.90): 0.06 / 0.02.

| label-free statistic | Spearman ρ with best τ | Spearman ρ with true area |
|---|---:|---:|
| a_hat | -0.000 | +0.735 |
| v3_mean | -0.014 | +0.808 |
| v3_frac55 | -0.061 | +0.851 |
| v3_entropy | +0.004 | +0.407 |
| otsu | +0.165 | +0.288 |

Otsu threshold on V3 then R2 (label-free, exploratory): per-image IoU (class-macro) 0.828 vs fixed 0.55 0.835.

### 1e. Background-word maps at BRACE's false positives

Images with ≥ 50 FP and ≥ 50 TP pixels: 484. AUROC of max background-word map, FP vs TP pixels inside the mask: mean 0.365 (median 0.361; share > 0.7: 0.01).

Dominant background word at FP pixels: wall 0.11, floor 0.23, ground 0.20, sky 0.09, grass 0.07, tree 0.06, building 0.05, road 0.20

## COCO-Cap (707 images)

### 1a. Bounds

| what | per-image IoU (class-macro) | pooled class-macro mIoU |
|---|---:|---:|
| BRACE @ τ 0.55 + R2 | 0.706 | 0.696 |
| BRACE, OOF τ + R2 | — | 0.707 |
| oracle per-image τ + R2 (grid 0.30–0.90) | 0.788 | 0.793 |
| oracle per-image τ, raw V3 (grid) | 0.781 | — |
| perfect ranking of seed (best of all thresholds, raw) | 0.678 | — |
| perfect ranking of P1 (best of all thresholds, raw) | 0.788 | — |
| perfect ranking of V0 (best of all thresholds, raw) | 0.796 | — |
| perfect ranking of V3 (best of all thresholds, raw) | 0.799 | — |
| perfect ranking of C16 (best of all thresholds, raw) | 0.804 | — |

AUROC (pixel ranking quality), class-macro: seed 0.907, P1 0.928, V0 0.928, V3 0.927, C16 0.929

### 1b. Error anatomy of the BRACE mask (fractions of the object area, mean over images)

| subset | n | IoU | FN boundary | FN interior | FP boundary | FP far | instances (mean) | predicted components (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 707 | 0.721 | 0.026 | 0.058 | 0.072 | 1.968 | 2.22 | 30.65 |
| area quintile 1 (0.00–0.08) | 142 | 0.528 | 0.051 | 0.052 | 0.134 | 8.722 | 1.65 | 28.85 |
| area quintile 2 (0.08–0.14) | 141 | 0.688 | 0.030 | 0.048 | 0.080 | 0.571 | 2.34 | 27.64 |
| area quintile 3 (0.14–0.21) | 141 | 0.770 | 0.023 | 0.043 | 0.066 | 0.281 | 2.36 | 34.12 |
| area quintile 4 (0.21–0.35) | 141 | 0.813 | 0.016 | 0.035 | 0.051 | 0.172 | 2.15 | 31.00 |
| area quintile 5 (0.35–1.00) | 142 | 0.806 | 0.011 | 0.114 | 0.030 | 0.060 | 2.58 | 31.67 |
| multi-instance (≥2 components) | 277 | 0.667 | 0.029 | 0.051 | 0.100 | 1.652 | 4.10 | 38.36 |

Per class (mean IoU, prec, rec, FN interior, FP far):

| class | n | IoU | prec | rec | FN interior | FP far | worst failure type |
|---|---:|---:|---:|---:|---:|---:|---|
| aeroplane | 49 | 0.819 | 0.854 | 0.959 | 0.021 | 0.337 | part only (under-segment) (2) |
| bicycle | 46 | 0.621 | 0.643 | 0.956 | 0.032 | 2.440 | bleed (over-segment) (9) |
| bird | 48 | 0.840 | 0.916 | 0.919 | 0.028 | 0.510 | bleed (over-segment) (2) |
| boat | 32 | 0.593 | 0.612 | 0.972 | 0.012 | 1.733 | bleed (over-segment) (10) |
| bottle | 21 | 0.524 | 0.600 | 0.748 | 0.184 | 2.705 | mixed (4) |
| bus | 37 | 0.906 | 0.921 | 0.983 | 0.009 | 0.070 | bleed (over-segment) (1) |
| car | 28 | 0.750 | 0.772 | 0.971 | 0.019 | 3.983 | bleed (over-segment) (4) |
| cat | 46 | 0.844 | 0.897 | 0.915 | 0.061 | 0.087 | miss (wrong region) (1) |
| chair | 20 | 0.534 | 0.556 | 0.967 | 0.010 | 1.720 | bleed (over-segment) (10) |
| cow | 48 | 0.874 | 0.927 | 0.941 | 0.031 | 0.065 | bleed (over-segment) (1) |
| dining table | 14 | 0.473 | 0.999 | 0.474 | 0.522 | 0.000 | part only (under-segment) (7) |
| dog | 44 | 0.745 | 0.815 | 0.861 | 0.102 | 0.375 | bleed (over-segment) (3) |
| horse | 38 | 0.750 | 0.782 | 0.955 | 0.016 | 0.341 | bleed (over-segment) (6) |
| motorbike | 41 | 0.672 | 0.706 | 0.942 | 0.040 | 1.707 | bleed (over-segment) (8) |
| person | 52 | 0.378 | 0.484 | 0.753 | 0.193 | 13.948 | bleed (over-segment) (19) |
| potted plant | 14 | 0.713 | 0.728 | 0.967 | 0.020 | 4.200 | bleed (over-segment) (2) |
| sheep | 38 | 0.854 | 0.921 | 0.925 | 0.036 | 0.115 | bleed (over-segment) (2) |
| sofa | 34 | 0.674 | 0.686 | 0.977 | 0.014 | 1.794 | bleed (over-segment) (10) |
| train | 33 | 0.748 | 0.780 | 0.956 | 0.032 | 0.417 | bleed (over-segment) (3) |
| tv | 24 | 0.818 | 0.854 | 0.958 | 0.021 | 0.315 | bleed (over-segment) (2) |

Failure types (IoU < 0.5): bleed (over-segment) 97, miss (wrong region) 12, mixed 18, ok 565, part only (under-segment) 15

### 1c. Worst 10% (70 images)

Contact sheet `fig/part1_worst_coco.jpg` (per image: image | truth | V3 | BRACE mask; in IoU order, row-major, two per row).

| rank | image | class | IoU | prec | rec | type |
|---:|---|---|---:|---:|---:|---|
| 1 | ovam-coco-734544-person-914583522 | person | 0.00 | 0.00 | 0.00 | miss (wrong region) |
| 2 | ovam-coco-522350-bottle-3441128440 | bottle | 0.00 | 0.00 | 0.00 | miss (wrong region) |
| 3 | ovam-coco-97815-cat-844304551 | cat | 0.00 | 0.00 | 0.00 | miss (wrong region) |
| 4 | ovam-coco-675802-person-1958248880 | person | 0.00 | 0.00 | 0.01 | miss (wrong region) |
| 5 | ovam-coco-145346-person-113665514 | person | 0.00 | 0.00 | 0.96 | bleed (over-segment) |
| 6 | ovam-coco-404680-person-3197351734 | person | 0.01 | 0.01 | 0.61 | mixed |
| 7 | ovam-coco-24807-car-3022797403 | car | 0.01 | 0.01 | 1.00 | bleed (over-segment) |
| 8 | ovam-coco-249777-bicycle-261583150 | bicycle | 0.01 | 0.01 | 1.00 | bleed (over-segment) |
| 9 | ovam-coco-757365-motorbike-2638166565 | motorbike | 0.02 | 0.02 | 1.00 | bleed (over-segment) |
| 10 | ovam-coco-541868-bottle-1316857654 | bottle | 0.02 | 0.02 | 0.44 | mixed |
| 11 | ovam-coco-576973-dog-70278584 | dog | 0.02 | 0.05 | 0.04 | miss (wrong region) |
| 12 | ovam-coco-591621-sofa-363884250 | sofa | 0.02 | 0.02 | 1.00 | bleed (over-segment) |
| 13 | ovam-coco-324696-potted_plant-496844947 | potted plant | 0.03 | 0.03 | 1.00 | bleed (over-segment) |
| 14 | ovam-coco-542882-bottle-1538395797 | bottle | 0.03 | 0.03 | 0.49 | mixed |
| 15 | ovam-coco-73382-dog-3660910269 | dog | 0.03 | 0.08 | 0.05 | miss (wrong region) |
| 16 | ovam-coco-433542-person-2942988204 | person | 0.04 | 0.04 | 0.87 | bleed (over-segment) |
| 17 | ovam-coco-234773-person-3194216459 | person | 0.04 | 0.47 | 0.05 | miss (wrong region) |
| 18 | ovam-coco-360186-potted_plant-1385232648 | potted plant | 0.04 | 0.04 | 1.00 | bleed (over-segment) |
| 19 | ovam-coco-536299-person-3274602369 | person | 0.05 | 0.05 | 1.00 | bleed (over-segment) |
| 20 | ovam-coco-740547-bottle-3314758851 | bottle | 0.05 | 0.07 | 0.19 | miss (wrong region) |
| 21 | ovam-coco-114875-person-1775210319 | person | 0.05 | 0.05 | 0.97 | bleed (over-segment) |
| 22 | ovam-coco-645843-person-1762850619 | person | 0.06 | 0.06 | 0.97 | bleed (over-segment) |
| 23 | ovam-coco-197889-person-574893829 | person | 0.07 | 0.07 | 0.85 | bleed (over-segment) |
| 24 | ovam-coco-381851-chair-1731222400 | chair | 0.07 | 0.07 | 1.00 | bleed (over-segment) |
| 25 | ovam-coco-279301-bird-1402631476 | bird | 0.07 | 0.07 | 1.00 | bleed (over-segment) |
| 26 | ovam-coco-180769-person-1255158167 | person | 0.08 | 0.16 | 0.15 | miss (wrong region) |
| 27 | ovam-coco-581412-bird-1326868124 | bird | 0.08 | 0.08 | 0.96 | bleed (over-segment) |
| 28 | ovam-coco-329348-boat-2372929146 | boat | 0.08 | 0.08 | 1.00 | bleed (over-segment) |
| 29 | ovam-coco-644270-boat-1802700531 | boat | 0.08 | 0.08 | 0.99 | bleed (over-segment) |
| 30 | ovam-coco-416788-person-3383211596 | person | 0.09 | 0.09 | 0.96 | bleed (over-segment) |
| 31 | ovam-coco-201836-person-3302986691 | person | 0.09 | 0.09 | 0.95 | bleed (over-segment) |
| 32 | ovam-coco-160187-person-578773487 | person | 0.09 | 1.00 | 0.09 | miss (wrong region) |
| 33 | ovam-coco-288189-aeroplane-872826842 | aeroplane | 0.09 | 0.09 | 1.00 | bleed (over-segment) |
| 34 | ovam-coco-311405-person-2150012512 | person | 0.09 | 0.10 | 0.73 | mixed |
| 35 | ovam-coco-324251-boat-981775646 | boat | 0.09 | 0.09 | 1.00 | bleed (over-segment) |
| 36 | ovam-coco-498659-person-2178227007 | person | 0.10 | 0.10 | 0.54 | mixed |
| 37 | ovam-coco-72785-bottle-3924405499 | bottle | 0.10 | 0.10 | 0.68 | mixed |
| 38 | ovam-coco-118723-person-1820020555 | person | 0.10 | 0.10 | 1.00 | bleed (over-segment) |
| 39 | ovam-coco-773650-dog-4016172450 | dog | 0.11 | 0.12 | 0.59 | mixed |
| 40 | ovam-coco-313589-person-4210581391 | person | 0.12 | 0.12 | 1.00 | bleed (over-segment) |
| 41 | ovam-coco-108624-boat-1271200875 | boat | 0.12 | 0.12 | 1.00 | bleed (over-segment) |
| 42 | ovam-coco-591374-person-1032945820 | person | 0.12 | 0.13 | 0.84 | bleed (over-segment) |
| 43 | ovam-coco-48438-person-375522931 | person | 0.13 | 0.13 | 0.91 | bleed (over-segment) |
| 44 | ovam-coco-453219-train-279470692 | train | 0.13 | 0.13 | 1.00 | bleed (over-segment) |
| 45 | ovam-coco-512313-bicycle-2120171997 | bicycle | 0.14 | 0.14 | 0.94 | bleed (over-segment) |
| 46 | ovam-coco-738670-dog-2813656683 | dog | 0.14 | 0.14 | 1.00 | bleed (over-segment) |
| 47 | ovam-coco-173888-person-1324672332 | person | 0.16 | 0.16 | 1.00 | bleed (over-segment) |
| 48 | ovam-coco-507464-person-1617331032 | person | 0.17 | 0.17 | 1.00 | bleed (over-segment) |
| 49 | ovam-coco-413566-boat-2911578622 | boat | 0.18 | 0.18 | 1.00 | bleed (over-segment) |
| 50 | ovam-coco-344034-dining_table-4156308419 | dining table | 0.18 | 1.00 | 0.18 | miss (wrong region) |
| 51 | ovam-coco-189764-tv-3296850089 | tv | 0.19 | 0.19 | 1.00 | bleed (over-segment) |
| 52 | ovam-coco-727328-dining_table-689783086 | dining table | 0.20 | 1.00 | 0.20 | miss (wrong region) |
| 53 | ovam-coco-262359-person-681928534 | person | 0.20 | 0.20 | 1.00 | bleed (over-segment) |
| 54 | ovam-coco-572233-person-1010932185 | person | 0.21 | 0.21 | 0.95 | bleed (over-segment) |
| 55 | ovam-coco-663080-chair-271245509 | chair | 0.21 | 0.21 | 1.00 | bleed (over-segment) |
| 56 | ovam-coco-566695-dog-1071257360 | dog | 0.21 | 0.91 | 0.21 | part only (under-segment) |
| 57 | ovam-coco-470035-person-3542956483 | person | 0.21 | 0.29 | 0.42 | mixed |
| 58 | ovam-coco-501131-motorbike-554896328 | motorbike | 0.22 | 0.23 | 0.85 | bleed (over-segment) |
| 59 | ovam-coco-409433-chair-875541453 | chair | 0.23 | 0.23 | 1.00 | bleed (over-segment) |
| 60 | ovam-coco-61662-chair-4115572993 | chair | 0.23 | 0.23 | 0.99 | bleed (over-segment) |
| 61 | ovam-coco-357874-aeroplane-3643544608 | aeroplane | 0.23 | 0.23 | 0.95 | bleed (over-segment) |
| 62 | ovam-coco-445253-bicycle-3511618244 | bicycle | 0.23 | 0.26 | 0.74 | mixed |
| 63 | ovam-coco-739619-sofa-2088687870 | sofa | 0.24 | 0.24 | 1.00 | bleed (over-segment) |
| 64 | ovam-coco-428356-dining_table-2467085659 | dining table | 0.25 | 1.00 | 0.25 | part only (under-segment) |
| 65 | ovam-coco-617889-sofa-2384264643 | sofa | 0.25 | 0.25 | 0.99 | bleed (over-segment) |
| 66 | ovam-coco-667979-bicycle-2142282840 | bicycle | 0.25 | 0.25 | 0.96 | bleed (over-segment) |
| 67 | ovam-coco-449734-person-516411791 | person | 0.25 | 0.48 | 0.35 | mixed |
| 68 | ovam-coco-418199-chair-2511252585 | chair | 0.25 | 0.26 | 1.00 | bleed (over-segment) |
| 69 | ovam-coco-48448-person-811552420 | person | 0.26 | 1.00 | 0.26 | part only (under-segment) |
| 70 | ovam-coco-729216-chair-367164266 | chair | 0.26 | 0.26 | 0.98 | bleed (over-segment) |

### 1d. Calibration and threshold

Foreground fraction per V3 value bin (all pixels pooled):

| V3 bin | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg fraction | 0.02 | 0.02 | 0.03 | 0.06 | 0.11 | 0.25 | 0.48 | 0.68 | 0.82 | 0.90 |
| share of pixels | 0.301 | 0.167 | 0.103 | 0.073 | 0.053 | 0.041 | 0.043 | 0.054 | 0.078 | 0.087 |

Per-image best τ (+R2): mean 0.594, sd 0.145; share at the grid edges (0.30 / 0.90): 0.06 / 0.02.

| label-free statistic | Spearman ρ with best τ | Spearman ρ with true area |
|---|---:|---:|
| a_hat | +0.043 | +0.715 |
| v3_mean | +0.219 | +0.612 |
| v3_frac55 | +0.200 | +0.692 |
| v3_entropy | +0.238 | +0.407 |
| otsu | +0.103 | +0.433 |

Otsu threshold on V3 then R2 (label-free, exploratory): per-image IoU (class-macro) 0.663 vs fixed 0.55 0.706.

### 1e. Background-word maps at BRACE's false positives

Images with ≥ 50 FP and ≥ 50 TP pixels: 679. AUROC of max background-word map, FP vs TP pixels inside the mask: mean 0.489 (median 0.491; share > 0.7: 0.14).

Dominant background word at FP pixels: wall 0.11, floor 0.21, ground 0.08, sky 0.04, grass 0.12, tree 0.20, building 0.11, road 0.13

