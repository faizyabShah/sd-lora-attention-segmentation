# SA package B: SA-L, SA-H, operator check

490 images, 20 classes, 416 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| SA-L | `SL/all_16` | `SL/single_15` (5/5) | +0.0331 | `SL/single_15` | fold vote and OOF guard passed | 0.7884 / 0.7553 | +0.0331 | +0.0417 [+0.0365, +0.0469] | no |
| SA-H | `SH/0_1_2_3_4_5_6_7` | `SH/1_4_6` (5/5) | +0.0057 | `SH/1_4_6` | fold vote and OOF guard passed | 0.7610 / 0.7553 | +0.0057 | +0.0065 [+0.0056, +0.0074] | no |

## SA-L

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `SL/single_15` | promote | 0.7890 / 0.7576 | 0.7891 / 0.7464 |
| 1 | `SL/single_15` | promote | 0.7822 / 0.7485 | 0.8116 / 0.7782 |
| 2 | `SL/single_15` | promote | 0.7896 / 0.7567 | 0.7869 / 0.7517 |
| 3 | `SL/single_15` | promote | 0.7906 / 0.7576 | 0.7823 / 0.7452 |
| 4 | `SL/single_15` | promote | 0.7908 / 0.7556 | 0.7752 / 0.7479 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `SL/single_15` | 0.7689 | 0.7884 | 0.7531 | 0.9559 |
| `SL/single_12` | 0.7375 | 0.7698 | 0.7532 | 0.9557 |
| `SL/leave2_04_08` | 0.7275 | 0.7589 | 0.7450 | 0.9545 |
| `SL/leave2_05_08` | 0.7271 | 0.7584 | 0.7447 | 0.9544 |
| `SL/leave2_04_05` | 0.7258 | 0.7583 | 0.7458 | 0.9539 |
| `SL/leave2_06_08` | 0.7274 | 0.7583 | 0.7440 | 0.9544 |
| `SL/leave2_08_09` | 0.7271 | 0.7582 | 0.7441 | 0.9544 |
| `SL/leave2_04_06` | 0.7262 | 0.7582 | 0.7451 | 0.9539 |
| `SL/leave2_04_09` | 0.7260 | 0.7582 | 0.7453 | 0.9539 |
| `SL/leave2_08_10` | 0.7231 | 0.7580 | 0.7485 | 0.9542 |

## SA-H

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `SH/1_4_6` | conditional | 0.7631 / 0.7576 | 0.7528 / 0.7464 |
| 1 | `SH/1_4_6` | conditional | 0.7540 / 0.7485 | 0.7846 / 0.7782 |
| 2 | `SH/1_4_6` | conditional | 0.7625 / 0.7567 | 0.7571 / 0.7517 |
| 3 | `SH/1_4_6` | conditional | 0.7635 / 0.7576 | 0.7505 / 0.7452 |
| 4 | `SH/1_4_6` | conditional | 0.7613 / 0.7556 | 0.7536 / 0.7479 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `SH/4` | 0.7301 | 0.7625 | 0.7473 | 0.9564 |
| `SH/4_6` | 0.7288 | 0.7623 | 0.7498 | 0.9550 |
| `SH/1_4_6` | 0.7294 | 0.7610 | 0.7467 | 0.9552 |
| `SH/1_4` | 0.7302 | 0.7606 | 0.7446 | 0.9561 |
| `SH/6` | 0.7255 | 0.7605 | 0.7499 | 0.9523 |
| `SH/0_4_6` | 0.7238 | 0.7604 | 0.7522 | 0.9544 |
| `SH/0_1_4_6` | 0.7258 | 0.7600 | 0.7494 | 0.9547 |
| `SH/4_6_7` | 0.7278 | 0.7597 | 0.7463 | 0.9544 |
| `SH/0_1_4` | 0.7249 | 0.7594 | 0.7490 | 0.9553 |
| `SH/1_6` | 0.7281 | 0.7594 | 0.7453 | 0.9539 |

CA seed: `frozen_ca_r_8_16_64` (frozen: True).

## Seed alone and operator check (against the SA reference, in-sample)

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP | Paired Δ image IoU vs SA reference [95% CI] |
|---|---:|---:|---:|---:|---|
| SA reference: all modules, heads, calls (y = A x) | 0.7212 | 0.7553 | 0.7463 | 0.9535 | +0.0000 [+0.0000, +0.0000] |
| CA seed alone | 0.7067 | 0.6956 | 0.5877 | 0.9131 | -0.0566 [-0.0670, -0.0465] |
| Operator check: y = Aᵀx | 0.5914 | 0.4741 | 0.3273 | 0.9179 | -0.2521 [-0.2801, -0.2241] |

## SA-H head report

| Head | Alone | Leave this head out | Folds retaining it |
|---:|---:|---:|---:|
| 0 | 0.7541 | 0.7552 | 0/5 |
| 1 | 0.7577 | 0.7548 | 5/5 |
| 2 | 0.7512 | 0.7558 | 0/5 |
| 3 | 0.7486 | 0.7561 | 0/5 |
| 4 | 0.7625 | 0.7540 | 5/5 |
| 5 | 0.7460 | 0.7567 | 0/5 |
| 6 | 0.7605 | 0.7544 | 5/5 |
| 7 | 0.7513 | 0.7555 | 0/5 |

All 8 heads: 0.7553. Frozen subset `SH/1_4_6` ranks 3 of 255 in-sample; best in-sample `SH/4` (0.7625).

| Round | Removed | Heads left | mIoU |
|---:|---:|---|---:|
| 0 | — | 0 1 2 3 4 5 6 7 | 0.7553 |
| 1 | 5 | 0 1 2 3 4 6 7 | 0.7567 |
| 2 | 3 | 0 1 2 4 6 7 | 0.7577 |
| 3 | 2 | 0 1 4 6 7 | 0.7589 |
| 4 | 7 | 0 1 4 6 | 0.7600 |
| 5 | 0 | 1 4 6 | 0.7610 |
| 6 | 1 | 4 6 | 0.7623 |
| 7 | 6 | 4 | 0.7625 |

## Exhaustive SA module subsets (exploratory, not eligible for freezing)

Out-of-fold selection 0.7884 vs all-16 0.7553 (gain +0.0331); fold votes {'0x8000': 5}.
