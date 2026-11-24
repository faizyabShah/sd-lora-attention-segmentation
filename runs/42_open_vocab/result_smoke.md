# Run 42: the low-resolution consensus gate on 60 COCO classes never used in development (`PREREG.md`)

**SMOKE: evaluated on COCO train photos outside every fit set; profile sizes 2 / 3 / 6. Numbers are not results.**

**Confirmatory for the unseen-class question (O2); profile size (O4) descriptive.** Evaluation: smoke set, 118 items (40 photos); primary items (class area ≥ 1%): 80, of which 38 unseen-class items over 25 classes and 42 seen-class items over 17 classes. These val2017 photos were evaluated before for the 20 VOC classes only (runs 40, 41). Pipeline: uniform cross-attention seed, γ = 1, fine-only propagation (V0), raw map at the threshold chosen on 30 seen-class COCO train items. Profiles from captioned COCO train photos (no labels). Class-macro mIoU; 95% photo-bootstrap intervals (50 samples, paired).


## 1. Primary contrasts (primary items, raw V0, profile size 3)

| # | Contrast | Classes | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| U1 | MK4-LR − no gate | unseen (25) | +0.024 [-0.001, +0.045] | not resolved |
| U2 | MK4-LR − MK4 | unseen (25) | -0.019 [-0.028, +0.001] | not resolved |
| S1 | MK4-LR − no gate | seen (17) | +0.086 [+0.045, +0.141] | **helps** |
| S2 | MK4-LR − MK4 | seen (17) | +0.066 [+0.003, +0.133] | **helps** |

- The gate generalises to unseen classes (U1 CI above 0): **no**.
- LR's advantage over MK4 generalises (U2 CI above 0): **no**.
- Unseen / seen gain over no gate: 0.28.

## 2. mIoU, primary items (profile size 20)

| Classes | Map | Read-out | none | MK4 | MK4-LR |
|---|---|---|---:|---:|---:|
| unseen | seed | raw, seen-class thr (primary for V0) | 0.410 | 0.431 | 0.439 |
| unseen | V0 | raw, seen-class thr (primary for V0) | 0.483 | 0.526 | 0.507 |
| unseen | V0 | raw, 0.55 | 0.470 | 0.519 | 0.502 |
| unseen | V0 | + DenseCRF, seen-class thr | 0.470 | 0.514 | 0.499 |
| unseen | V0 | + DenseCRF, 0.55 | 0.457 | 0.507 | 0.495 |
| seen | seed | raw, seen-class thr (primary for V0) | 0.436 | 0.427 | 0.514 |
| seen | V0 | raw, seen-class thr (primary for V0) | 0.492 | 0.512 | 0.578 |
| seen | V0 | raw, 0.55 | 0.471 | 0.498 | 0.578 |
| seen | V0 | + DenseCRF, seen-class thr | 0.473 | 0.496 | 0.543 |
| seen | V0 | + DenseCRF, 0.55 | 0.460 | 0.486 | 0.551 |

Seen classes with run 33's profiles (`C_all`, 490 photos), same capture, raw V0: MK4 0.556, MK4-LR 0.590; this run's size-20 profiles: MK4 0.512, MK4-LR 0.578.

## 3. Profile size (O4; primary items, raw V0)

| Classes | Positive photos per class | MK4-LR − none | MK4 − none | MK4-LR − MK4 | MK4-LR mIoU |
|---|---|---|---|---|---:|
| unseen | 2 | +0.019 [-0.011, +0.038] | -0.021 [-0.073, +0.012] | +0.039 [+0.025, +0.068] | 0.501 |
| unseen | 3 | +0.024 [-0.001, +0.045] | +0.043 [+0.007, +0.060] | -0.019 [-0.028, +0.001] | 0.507 |
| unseen | 6 | +0.030 [+0.002, +0.047] | +0.050 [+0.013, +0.071] | -0.020 [-0.027, -0.003] | 0.513 |
| seen | 2 | +0.073 [+0.033, +0.113] | -0.046 [-0.137, +0.005] | +0.119 [+0.072, +0.208] | 0.565 |
| seen | 3 | +0.086 [+0.045, +0.141] | +0.020 [-0.037, +0.076] | +0.066 [+0.003, +0.133] | 0.578 |
| seen | 6 | +0.102 [+0.053, +0.152] | +0.043 [-0.039, +0.123] | +0.059 [-0.013, +0.146] | 0.594 |

Classes with fewer positive photos than the largest size: none.

## 4. Secondary rules (profile size 20)

| Contrast | Classes | seed only (raw) | all items, raw V0 | + DenseCRF V0 | seen, person excluded |
|---|---|---|---|---|---|
| MK4-LR − none | unseen | +0.029 [-0.003, +0.051] | +0.019 [-0.005, +0.031] | +0.029 [-0.001, +0.056] | — |
| MK4-LR − none | seen | +0.078 [+0.052, +0.121] | +0.079 [+0.039, +0.113] | +0.071 [+0.045, +0.120] | +0.082 [+0.035, +0.118] |
| MK4-LR − MK4 | unseen | +0.009 [-0.004, +0.035] | -0.016 [-0.028, +0.008] | -0.015 [-0.025, +0.011] | — |
| MK4-LR − MK4 | seen | +0.087 [+0.049, +0.130] | +0.083 [+0.014, +0.113] | +0.047 [-0.016, +0.121] | +0.065 [+0.009, +0.134] |
| MK4 − none | unseen | +0.020 [-0.013, +0.039] | +0.035 [+0.000, +0.042] | +0.044 [+0.010, +0.065] | — |
| MK4 − none | seen | -0.009 [-0.039, +0.021] | -0.004 [-0.032, +0.053] | +0.024 [-0.034, +0.088] | +0.017 [-0.037, +0.072] |

Raw V0 at every threshold, unseen classes (profile size 20; † = 95% CI excludes 0):

| Contrast | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MK4-LR − none | +0.068† | +0.070† | +0.077† | +0.075† | +0.073† | +0.059† | +0.052† | +0.046† | +0.041† | +0.036† | +0.032 | +0.029 | +0.023 | +0.022 | +0.016 | +0.002 | +0.001 |
| MK4-LR − MK4 | +0.019† | +0.014† | +0.016† | +0.013† | +0.011† | +0.008 | +0.005 | +0.001 | -0.003 | -0.010 | -0.017† | -0.019 | -0.014 | -0.007 | -0.000 | -0.001 | +0.004 |

## 5. Per class (primary items, raw V0, profile size 20)

| Class | seen | items | positive photos | none | MK4 | MK4-LR | LR − none | LR − MK4 | modules without profile (mean) |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | ✓ | 1 | 6 | 0.628 | 0.667 | 0.804 | +0.176 | +0.137 | 0.0 |
| bird | ✓ | 2 | 6 | 0.401 | 0.747 | 0.456 | +0.054 | -0.291 | 0.0 |
| boat | ✓ | 1 | 6 | 0.697 | 0.703 | 0.688 | -0.009 | -0.015 | 0.0 |
| bus | ✓ | 2 | 6 | 0.584 | 0.406 | 0.633 | +0.049 | +0.227 | 0.0 |
| car | ✓ | 3 | 6 | 0.186 | 0.230 | 0.458 | +0.272 | +0.227 | 0.0 |
| cat | ✓ | 2 | 6 | 0.839 | 0.832 | 0.831 | -0.008 | -0.001 | 0.0 |
| chair | ✓ | 1 | 6 | 0.336 | 0.525 | 0.437 | +0.100 | -0.088 | 0.0 |
| cow | ✓ | 1 | 6 | 0.797 | 0.785 | 0.789 | -0.008 | +0.004 | 0.0 |
| dining table | ✓ | 4 | 6 | 0.135 | 0.258 | 0.152 | +0.018 | -0.105 | 0.0 |
| dog | ✓ | 1 | 6 | 0.912 | 0.921 | 0.894 | -0.017 | -0.027 | 0.0 |
| horse | ✓ | 1 | 6 | 0.527 | 0.547 | 0.560 | +0.033 | +0.013 | 0.0 |
| motorbike | ✓ | 1 | 6 | 0.130 | 0.045 | 0.127 | -0.003 | +0.082 | 0.0 |
| person | ✓ | 15 | 6 | 0.233 | 0.311 | 0.392 | +0.159 | +0.081 | 0.0 |
| potted plant | ✓ | 1 | 6 | 0.599 | 0.530 | 0.821 | +0.222 | +0.291 | 0.0 |
| sofa | ✓ | 2 | 6 | 0.307 | 0.222 | 0.579 | +0.273 | +0.358 | 0.0 |
| train | ✓ | 3 | 6 | 0.615 | 0.580 | 0.585 | -0.030 | +0.005 | 0.0 |
| tv | ✓ | 1 | 6 | 0.438 | 0.401 | 0.621 | +0.182 | +0.219 | 0.0 |
| truck |  | 1 | 6 | 0.065 | 0.061 | 0.069 | +0.004 | +0.008 | 0.0 |
| traffic light |  | 1 | 6 | 0.012 | 0.063 | 0.062 | +0.050 | -0.001 | 0.0 |
| stop sign |  | 1 | 6 | 0.572 | 0.916 | 0.787 | +0.216 | -0.129 | 0.0 |
| bench |  | 3 | 6 | 0.698 | 0.720 | 0.734 | +0.035 | +0.014 | 0.0 |
| giraffe |  | 2 | 6 | 0.650 | 0.652 | 0.645 | -0.005 | -0.007 | 0.0 |
| umbrella |  | 2 | 6 | 0.420 | 0.538 | 0.487 | +0.067 | -0.051 | 0.0 |
| suitcase |  | 1 | 6 | 0.418 | 0.439 | 0.423 | +0.005 | -0.015 | 0.0 |
| tennis racket |  | 1 | 6 | 0.770 | 0.772 | 0.755 | -0.016 | -0.017 | 0.0 |
| cup |  | 1 | 6 | 0.651 | 0.648 | 0.663 | +0.012 | +0.015 | 0.0 |
| knife |  | 1 | 6 | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | 0.0 |
| banana |  | 1 | 6 | 0.855 | 0.864 | 0.866 | +0.012 | +0.002 | 0.0 |
| sandwich |  | 2 | 6 | 0.890 | 0.804 | 0.627 | -0.263 | -0.177 | 0.0 |
| broccoli |  | 2 | 6 | 0.492 | 0.748 | 0.741 | +0.249 | -0.007 | 0.0 |
| donut |  | 1 | 6 | 0.710 | 0.717 | 0.699 | -0.011 | -0.019 | 0.0 |
| cake |  | 3 | 6 | 0.631 | 0.643 | 0.669 | +0.037 | +0.026 | 0.0 |
| bed |  | 2 | 6 | 0.903 | 0.894 | 0.924 | +0.021 | +0.030 | 0.0 |
| mouse |  | 1 | 6 | 0.836 | 0.882 | 0.867 | +0.031 | -0.015 | 0.0 |
| keyboard |  | 1 | 6 | 0.388 | 0.385 | 0.388 | +0.000 | +0.003 | 0.0 |
| microwave |  | 1 | 6 | 0.041 | 0.039 | 0.038 | -0.003 | -0.001 | 0.0 |
| oven |  | 2 | 6 | 0.086 | 0.089 | 0.117 | +0.032 | +0.028 | 0.0 |
| sink |  | 1 | 6 | 0.239 | 0.232 | 0.254 | +0.015 | +0.022 | 0.0 |
| refrigerator |  | 2 | 6 | 0.148 | 0.352 | 0.230 | +0.082 | -0.123 | 0.0 |
| book |  | 3 | 6 | 0.118 | 0.132 | 0.123 | +0.004 | -0.009 | 0.0 |
| clock |  | 1 | 6 | 0.705 | 0.728 | 0.682 | -0.024 | -0.047 | 0.0 |
| teddy bear |  | 1 | 6 | 0.775 | 0.835 | 0.825 | +0.049 | -0.010 | 0.0 |

- seen: MK4-LR beats / trails no gate in 11 / 6 classes; beats / trails MK4 in 11 / 6.

- unseen: MK4-LR beats / trails no gate in 18 / 6 classes; beats / trails MK4 in 9 / 15.

BRACE stays frozen; nothing here is a promotion.

