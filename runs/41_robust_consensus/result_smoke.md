# Run 41: median and trimmed low-resolution consensus vs MK4-LR, end to end (`PREREG.md`)

**SMOKE: evaluation sets replaced by training sets (conf, cval → cfit; vval → vtrain). Numbers are not results.**

**Development on VOC train, frozen confirmation on a new COCO train2017 set.** Confirmation set: 24 items (24 photos); primary items (class area ≥ 1%): 24 (24 photos). COCO thresholds from 24 run 40 threshold-fit items; VOC thresholds and γ from 24 VOC train items. Source C profiles, SA weights on. Class-macro mIoU; 95% photo-bootstrap intervals (50 samples, paired).


## 1. Development on VOC train (PREREG §4)

Selected γ (VOC train, raw maps):

| Map | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---:|---:|---:|---:|---:|
| seed only | 0.25 | 2.0 | 1.5 | 2.0 | 2.0 |
| fine-only (V0) | 0.75 | 2.0 | 2.0 | 2.0 | 2.0 |
| capped hierarchy (V3) | 1.0 | 2.0 | 2.0 | 2.0 | 2.0 |

VOC train mIoU, + DenseCRF at the VOC-train threshold, selected γ (in-sample):

| Map | none | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---:|---:|---:|---:|---:|---:|
| fine-only (V0) | 0.552 | 0.561 | 0.627 | 0.633 | 0.632 | 0.635 |
| capped hierarchy (V3) | 0.549 | 0.589 | 0.606 | 0.613 | 0.610 | 0.613 |

| Candidate | d = mean over V0, V3 of (candidate − MK4-LR) [95% CI] |
|---|---|
| LR-med | +0.006 [-0.005, +0.009] |
| LR-trim | +0.004 [-0.000, +0.010] |
| LR without module 2 | +0.007 [+0.000, +0.012] |

- Winner by the fixed rule: **LR-trim**.
- Development: **fails** (the winner's CI is not above 0).
- Items missing one of the six low-resolution gates (D / T fall back to the mean): VOC train 0, confirmation 0.

## 2. Primary contrasts (new COCO set, primary items, SA weights on, selected γ, + DenseCRF at the COCO-fit threshold)

| # | Contrast | Map | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | LR-trim − MK4-LR | capped hierarchy (V3) | +0.003 [-0.001, +0.006] | not resolved |
| P2 | LR-trim − MK4-LR | fine-only (V0) | -0.008 [-0.031, -0.000] | **hurts** |

- LR-trim over MK4-LR: **not confirmed (failed development)**; P1 and P2 are descriptive (development passed, P1 or P2 helps, neither hurts).

## 3. mIoU, new COCO set, primary items (seed only: raw, COCO-fit threshold; V0 / V3: + DenseCRF)

| γ | Map | none | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---|---:|---:|---:|---:|---:|---:|
| selected | seed only | 0.477 | 0.487 | 0.537 | 0.532 | 0.538 | 0.542 |
| selected | fine-only (V0) | 0.544 | 0.570 | 0.624 | 0.620 | 0.617 | 0.623 |
| selected | capped hierarchy (V3) | 0.557 | 0.576 | 0.597 | 0.600 | 0.600 | 0.575 |
| 1 | seed only | 0.477 | 0.485 | 0.526 | 0.527 | 0.527 | 0.530 |
| 1 | fine-only (V0) | 0.544 | 0.572 | 0.608 | 0.615 | 0.609 | 0.619 |
| 1 | capped hierarchy (V3) | 0.557 | 0.576 | 0.586 | 0.581 | 0.590 | 0.575 |

## 4. Paired differences, new COCO set, primary items

| γ | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | LR-trim − none |
|---|---|---|---|---|---|---|
| selected | seed only | -0.005 [-0.017, +0.005] | +0.000 [-0.007, +0.010] | +0.004 [-0.004, +0.012] | +0.050 [+0.025, +0.089] | +0.061 [+0.033, +0.104] |
| selected | fine-only (V0) | -0.004 [-0.019, +0.003] | -0.008 [-0.031, -0.000] | -0.001 [-0.022, +0.013] | +0.054 [+0.013, +0.095] | +0.072 [+0.021, +0.112] |
| selected | capped hierarchy (V3) | +0.003 [-0.003, +0.008] | +0.003 [-0.001, +0.006] | -0.022 [-0.030, -0.003] | +0.020 [+0.005, +0.036] | +0.043 [+0.008, +0.072] |
| 1 | seed only | +0.001 [-0.006, +0.006] | +0.002 [-0.001, +0.005] | +0.004 [+0.000, +0.009] | +0.040 [+0.021, +0.061] | +0.050 [+0.037, +0.075] |
| 1 | fine-only (V0) | +0.007 [-0.006, +0.026] | +0.001 [-0.008, +0.012] | +0.011 [+0.000, +0.020] | +0.036 [-0.012, +0.073] | +0.064 [+0.019, +0.101] |
| 1 | capped hierarchy (V3) | -0.006 [-0.010, +0.002] | +0.004 [+0.001, +0.007] | -0.011 [-0.018, -0.002] | +0.010 [-0.001, +0.025] | +0.033 [-0.000, +0.066] |

Hierarchy (V3 − V0) per gate, selected γ, + DenseCRF:

| Gate | V3 − V0 |
|---|---|
| none | +0.013 [-0.007, +0.043] |
| MK4 | +0.006 [-0.028, +0.056] |
| MK4-LR | -0.027 [-0.050, +0.007] |
| LR-med | -0.021 [-0.044, +0.024] |
| LR-trim | -0.017 [-0.039, +0.030] |
| LR without module 2 | -0.049 [-0.069, +0.001] |

## 5. Person excluded, and all items (new COCO set, selected γ, + DenseCRF)

| Contrast | Map | person excluded | all items (incl. < 1%) |
|---|---|---|---|
| LR-med − MK4-LR | capped hierarchy (V3) | +0.002 [-0.004, +0.008] | +0.003 [-0.003, +0.008] |
| LR-med − MK4-LR | fine-only (V0) | -0.005 [-0.017, +0.003] | -0.004 [-0.019, +0.003] |
| LR-trim − MK4-LR | capped hierarchy (V3) | +0.004 [-0.000, +0.006] | +0.003 [-0.001, +0.006] |
| LR-trim − MK4-LR | fine-only (V0) | -0.008 [-0.029, +0.000] | -0.008 [-0.031, -0.000] |
| LR without module 2 − MK4-LR | capped hierarchy (V3) | -0.021 [-0.033, -0.004] | -0.022 [-0.030, -0.003] |
| LR without module 2 − MK4-LR | fine-only (V0) | +0.002 [-0.024, +0.018] | -0.001 [-0.022, +0.013] |
| MK4-LR − MK4 | capped hierarchy (V3) | +0.022 [+0.003, +0.031] | +0.020 [+0.005, +0.036] |
| MK4-LR − MK4 | fine-only (V0) | +0.056 [+0.011, +0.093] | +0.054 [+0.013, +0.095] |
| LR-trim − none | capped hierarchy (V3) | +0.049 [+0.009, +0.081] | +0.043 [+0.008, +0.072] |
| LR-trim − none | fine-only (V0) | +0.079 [+0.015, +0.112] | +0.072 [+0.021, +0.112] |

## 6. Threshold rules (new COCO set, primary items, selected γ)

| Map | Gate | + CRF, COCO-fit thr (primary) | + CRF, 0.55 | raw, COCO-fit thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.544 | 0.552 | 0.556 | 0.554 |
| fine-only (V0) | MK4 | 0.570 | 0.575 | 0.567 | 0.566 |
| fine-only (V0) | MK4-LR | 0.624 | 0.624 | 0.615 | 0.615 |
| fine-only (V0) | LR-med | 0.620 | 0.610 | 0.628 | 0.623 |
| fine-only (V0) | LR-trim | 0.617 | 0.617 | 0.623 | 0.623 |
| fine-only (V0) | LR without module 2 | 0.623 | 0.620 | 0.613 | 0.610 |
| capped hierarchy (V3) | none | 0.557 | 0.488 | 0.562 | 0.464 |
| capped hierarchy (V3) | MK4 | 0.576 | 0.489 | 0.567 | 0.473 |
| capped hierarchy (V3) | MK4-LR | 0.597 | 0.566 | 0.601 | 0.554 |
| capped hierarchy (V3) | LR-med | 0.600 | 0.584 | 0.609 | 0.580 |
| capped hierarchy (V3) | LR-trim | 0.600 | 0.578 | 0.607 | 0.567 |
| capped hierarchy (V3) | LR without module 2 | 0.575 | 0.561 | 0.585 | 0.551 |

Raw maps, candidate − MK4-LR at every threshold (selected γ; † = 95% CI excludes 0):

| Contrast | Map | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR-med − MK4-LR | seed only | -0.006 | -0.006 | -0.003 | +0.000 | +0.003 | +0.006 | +0.009 | +0.010 | +0.009 | +0.009 | +0.007 | +0.005 | +0.006 | +0.007 | +0.007 | +0.007 | +0.012 |
| LR-med − MK4-LR | fine-only (V0) | +0.023† | +0.020† | +0.019† | +0.020† | +0.023† | +0.026† | +0.031† | +0.031† | +0.027† | +0.018 | +0.008 | -0.002 | -0.012† | -0.021† | -0.029† | -0.034† | -0.037† |
| LR-med − MK4-LR | capped hierarchy (V3) | +0.006† | +0.006† | +0.008† | +0.008 | +0.011† | +0.011 | +0.013 | +0.017 | +0.020 | +0.026† | +0.026† | +0.021† | +0.015† | +0.011† | +0.005† | +0.002 | -0.004 |
| LR-trim − MK4-LR | seed only | +0.002 | -0.002 | -0.002 | -0.001 | -0.000 | -0.001 | -0.001 | -0.001 | -0.001 | -0.000 | -0.000 | -0.001 | -0.004 | -0.006 | -0.006 | -0.005 | -0.001 |
| LR-trim − MK4-LR | fine-only (V0) | +0.012† | +0.009† | +0.007† | +0.006† | +0.005† | +0.008† | +0.011† | +0.012† | +0.012† | +0.010 | +0.008 | +0.004 | +0.002 | -0.002 | -0.009† | -0.012† | -0.014† |
| LR-trim − MK4-LR | capped hierarchy (V3) | +0.005† | +0.004† | +0.006† | +0.006† | +0.006† | +0.006† | +0.005† | +0.006† | +0.008† | +0.010† | +0.012† | +0.013† | +0.010† | +0.007† | +0.005† | +0.003 | -0.001 |
| LR without module 2 − MK4-LR | seed only | +0.003 | +0.004 | +0.005 | +0.003 | -0.000 | -0.003 | -0.004 | -0.004 | -0.006 | -0.005 | -0.005 | -0.006 | -0.005 | -0.004 | -0.004 | -0.003 | +0.001 |
| LR without module 2 − MK4-LR | fine-only (V0) | +0.007 | +0.001 | -0.001 | -0.001 | -0.000 | +0.001 | +0.000 | -0.003 | -0.009 | -0.010 | -0.005 | +0.001 | +0.004 | +0.004 | +0.004 | +0.004 | -0.002 |
| LR without module 2 − MK4-LR | capped hierarchy (V3) | +0.006† | +0.006 | +0.008 | +0.009 | +0.012 | +0.008 | +0.001 | +0.001 | +0.001 | -0.000 | -0.003 | -0.009 | -0.013† | -0.015† | -0.017† | -0.013 | -0.011 |

## 7. Per class IoU (new COCO set, primary items, selected γ, + DenseCRF)

| Class | items | MK4 V3 | MK4-LR V3 | LR-trim V3 | MK4 V0 | MK4-LR V0 | LR-trim V0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 1 | 0.517 | 0.516 | 0.515 | 0.490 | 0.493 | 0.501 |
| boat | 1 | 0.621 | 0.629 | 0.627 | 0.625 | 0.628 | 0.622 |
| bottle | 4 | 0.359 | 0.378 | 0.375 | 0.236 | 0.551 | 0.553 |
| bus | 1 | 0.762 | 0.796 | 0.797 | 0.789 | 0.815 | 0.817 |
| car | 2 | 0.741 | 0.761 | 0.759 | 0.696 | 0.785 | 0.714 |
| cat | 1 | 0.897 | 0.884 | 0.883 | 0.910 | 0.898 | 0.899 |
| chair | 4 | 0.129 | 0.176 | 0.178 | 0.141 | 0.216 | 0.208 |
| cow | 2 | 0.603 | 0.609 | 0.617 | 0.643 | 0.646 | 0.643 |
| dog | 2 | 0.423 | 0.570 | 0.594 | 0.706 | 0.677 | 0.675 |
| motorbike | 1 | 0.549 | 0.561 | 0.561 | 0.603 | 0.594 | 0.595 |
| person | 1 | 0.771 | 0.770 | 0.765 | 0.749 | 0.775 | 0.768 |
| potted plant | 1 | 0.614 | 0.575 | 0.594 | 0.695 | 0.649 | 0.625 |
| sofa | 1 | 0.048 | 0.038 | 0.044 | 0.077 | 0.078 | 0.080 |
| train | 1 | 0.752 | 0.824 | 0.825 | 0.491 | 0.834 | 0.828 |
| tv | 1 | 0.856 | 0.861 | 0.863 | 0.698 | 0.723 | 0.720 |

Classes where LR-trim beats / trails MK4-LR: V3: 9 / 6; V0: 6 / 9.

Mean gate value at γ = 1 (primary items): MK4 0.496, MK4-LR 0.420, LR-med 0.408, LR-trim 0.413, LR without module 2 0.412.

## 8. Reuse sets (secondary; both evaluated before, so agreement only)

Selected γ, + DenseCRF. COCO val2017 (run 40 set): primary items, COCO-fit thresholds. VOC val: all items, VOC-train thresholds.

| Set | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | MK4-LR mIoU |
|---|---|---|---|---|---|---:|
| COCO val2017 | capped hierarchy (V3) | +0.003 [-0.003, +0.008] | +0.003 [-0.001, +0.006] | -0.022 [-0.030, -0.003] | +0.020 [+0.005, +0.036] | 0.597 |
| COCO val2017 | fine-only (V0) | -0.004 [-0.019, +0.003] | -0.008 [-0.031, -0.000] | -0.001 [-0.022, +0.013] | +0.054 [+0.013, +0.095] | 0.624 |
| VOC val | capped hierarchy (V3) | +0.007 [-0.005, +0.011] | +0.004 [-0.002, +0.018] | +0.007 [-0.003, +0.013] | +0.017 [-0.022, +0.047] | 0.606 |
| VOC val | fine-only (V0) | +0.005 [-0.005, +0.008] | +0.004 [-0.003, +0.006] | +0.008 [+0.002, +0.013] | +0.066 [+0.003, +0.124] | 0.627 |

BRACE stays frozen; any change to the paper's method is a separate owner decision.

