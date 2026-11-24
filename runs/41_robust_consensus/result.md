# Run 41: median and trimmed low-resolution consensus vs MK4-LR, end to end (`PREREG.md`)

**Development on VOC train, frozen confirmation on a new COCO train2017 set.** Confirmation set: 5875 items (3500 photos); primary items (class area ≥ 1%): 4791 (3240 photos). COCO thresholds from 1200 run 40 threshold-fit items; VOC thresholds and γ from 2170 VOC train items. Source C profiles, SA weights on. Class-macro mIoU; 95% photo-bootstrap intervals (2000 samples, paired).


## 1. Development on VOC train (PREREG §4)

Selected γ (VOC train, raw maps):

| Map | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---:|---:|---:|---:|---:|
| seed only | 0.25 | 1.5 | 1.5 | 1.5 | 1.5 |
| fine-only (V0) | 0.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| capped hierarchy (V3) | 0.75 | 1.5 | 1.0 | 1.0 | 1.0 |

VOC train mIoU, + DenseCRF at the VOC-train threshold, selected γ (in-sample):

| Map | none | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---:|---:|---:|---:|---:|---:|
| fine-only (V0) | 0.630 | 0.660 | 0.698 | 0.699 | 0.700 | 0.705 |
| capped hierarchy (V3) | 0.636 | 0.668 | 0.685 | 0.687 | 0.686 | 0.690 |

| Candidate | d = mean over V0, V3 of (candidate − MK4-LR) [95% CI] |
|---|---|
| LR-med | +0.001 [-0.002, +0.004] |
| LR-trim | +0.001 [-0.001, +0.004] |
| LR without module 2 | +0.006 [+0.003, +0.008] |

- Winner by the fixed rule: **LR-trim**.
- Development: **fails** (the winner's CI is not above 0).
- Items missing one of the six low-resolution gates (D / T fall back to the mean): VOC train 0, confirmation 0.

## 2. Primary contrasts (new COCO set, primary items, SA weights on, selected γ, + DenseCRF at the COCO-fit threshold)

| # | Contrast | Map | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | LR-trim − MK4-LR | capped hierarchy (V3) | -0.005 [-0.007, -0.003] | **hurts** |
| P2 | LR-trim − MK4-LR | fine-only (V0) | +0.001 [-0.001, +0.002] | not resolved |

- LR-trim over MK4-LR: **not confirmed (failed development)**; P1 and P2 are descriptive (rule: development passes, and P1 or P2 helps, and neither hurts).

## 3. mIoU, new COCO set, primary items (seed only: raw, COCO-fit threshold; V0 / V3: + DenseCRF)

| γ | Map | none | MK4 | MK4-LR | LR-med | LR-trim | LR without module 2 |
|---|---|---:|---:|---:|---:|---:|---:|
| selected | seed only | 0.467 | 0.490 | 0.533 | 0.533 | 0.534 | 0.540 |
| selected | fine-only (V0) | 0.508 | 0.536 | 0.561 | 0.562 | 0.562 | 0.569 |
| selected | capped hierarchy (V3) | 0.488 | 0.520 | 0.539 | 0.536 | 0.534 | 0.537 |
| 1 | seed only | 0.467 | 0.487 | 0.529 | 0.531 | 0.531 | 0.536 |
| 1 | fine-only (V0) | 0.508 | 0.531 | 0.557 | 0.559 | 0.558 | 0.564 |
| 1 | capped hierarchy (V3) | 0.488 | 0.519 | 0.532 | 0.536 | 0.534 | 0.537 |

## 4. Paired differences, new COCO set, primary items

| γ | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | LR-trim − none |
|---|---|---|---|---|---|---|
| selected | seed only | -0.001 [-0.003, +0.001] | +0.001 [-0.001, +0.002] | +0.007 [+0.005, +0.009] | +0.043 [+0.037, +0.050] | +0.067 [+0.059, +0.075] |
| selected | fine-only (V0) | +0.001 [-0.002, +0.003] | +0.001 [-0.001, +0.002] | +0.007 [+0.005, +0.010] | +0.025 [+0.019, +0.031] | +0.054 [+0.044, +0.063] |
| selected | capped hierarchy (V3) | -0.003 [-0.005, -0.001] | -0.005 [-0.007, -0.003] | -0.002 [-0.004, +0.000] | +0.019 [+0.013, +0.024] | +0.045 [+0.037, +0.053] |
| 1 | seed only | +0.002 [-0.000, +0.003] | +0.002 [+0.001, +0.003] | +0.007 [+0.006, +0.008] | +0.042 [+0.038, +0.046] | +0.064 [+0.057, +0.070] |
| 1 | fine-only (V0) | +0.002 [-0.001, +0.004] | +0.001 [-0.000, +0.002] | +0.006 [+0.004, +0.008] | +0.026 [+0.020, +0.032] | +0.051 [+0.041, +0.059] |
| 1 | capped hierarchy (V3) | +0.004 [+0.002, +0.006] | +0.001 [+0.000, +0.003] | +0.005 [+0.003, +0.006] | +0.013 [+0.009, +0.018] | +0.045 [+0.037, +0.053] |

Hierarchy (V3 − V0) per gate, selected γ, + DenseCRF:

| Gate | V3 − V0 |
|---|---|
| none | -0.019 [-0.026, -0.013] |
| MK4 | -0.016 [-0.022, -0.009] |
| MK4-LR | -0.022 [-0.028, -0.017] |
| LR-med | -0.026 [-0.033, -0.020] |
| LR-trim | -0.028 [-0.034, -0.023] |
| LR without module 2 | -0.032 [-0.038, -0.026] |

## 5. Person excluded, and all items (new COCO set, selected γ, + DenseCRF)

| Contrast | Map | person excluded | all items (incl. < 1%) |
|---|---|---|---|
| LR-med − MK4-LR | capped hierarchy (V3) | -0.003 [-0.005, -0.001] | -0.003 [-0.005, -0.000] |
| LR-med − MK4-LR | fine-only (V0) | +0.001 [-0.002, +0.004] | +0.000 [-0.004, +0.004] |
| LR-trim − MK4-LR | capped hierarchy (V3) | -0.005 [-0.007, -0.003] | -0.002 [-0.004, +0.000] |
| LR-trim − MK4-LR | fine-only (V0) | +0.000 [-0.001, +0.002] | -0.000 [-0.003, +0.002] |
| LR without module 2 − MK4-LR | capped hierarchy (V3) | -0.002 [-0.005, +0.001] | +0.002 [-0.001, +0.004] |
| LR without module 2 − MK4-LR | fine-only (V0) | +0.006 [+0.004, +0.009] | +0.008 [+0.005, +0.012] |
| MK4-LR − MK4 | capped hierarchy (V3) | +0.014 [+0.009, +0.019] | +0.024 [+0.018, +0.030] |
| MK4-LR − MK4 | fine-only (V0) | +0.018 [+0.011, +0.025] | +0.026 [+0.020, +0.033] |
| LR-trim − none | capped hierarchy (V3) | +0.037 [+0.028, +0.046] | +0.038 [+0.030, +0.046] |
| LR-trim − none | fine-only (V0) | +0.042 [+0.031, +0.052] | +0.045 [+0.035, +0.055] |

## 6. Threshold rules (new COCO set, primary items, selected γ)

| Map | Gate | + CRF, COCO-fit thr (primary) | + CRF, 0.55 | raw, COCO-fit thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.508 | 0.508 | 0.512 | 0.512 |
| fine-only (V0) | MK4 | 0.536 | 0.536 | 0.543 | 0.543 |
| fine-only (V0) | MK4-LR | 0.561 | 0.553 | 0.567 | 0.563 |
| fine-only (V0) | LR-med | 0.562 | 0.551 | 0.568 | 0.562 |
| fine-only (V0) | LR-trim | 0.562 | 0.552 | 0.568 | 0.562 |
| fine-only (V0) | LR without module 2 | 0.569 | 0.557 | 0.575 | 0.568 |
| capped hierarchy (V3) | none | 0.488 | 0.467 | 0.494 | 0.466 |
| capped hierarchy (V3) | MK4 | 0.520 | 0.506 | 0.526 | 0.506 |
| capped hierarchy (V3) | MK4-LR | 0.539 | 0.532 | 0.545 | 0.533 |
| capped hierarchy (V3) | LR-med | 0.536 | 0.526 | 0.541 | 0.526 |
| capped hierarchy (V3) | LR-trim | 0.534 | 0.525 | 0.540 | 0.525 |
| capped hierarchy (V3) | LR without module 2 | 0.537 | 0.530 | 0.544 | 0.531 |

Raw maps, candidate − MK4-LR at every threshold (selected γ; † = 95% CI excludes 0):

| Contrast | Map | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR-med − MK4-LR | seed only | -0.000 | -0.002 | -0.003† | -0.004† | -0.005† | -0.006† | -0.006† | -0.007† | -0.007† | -0.008† | -0.008† | -0.009† | -0.008† | -0.008† | -0.007† | -0.007† | -0.006† |
| LR-med − MK4-LR | fine-only (V0) | +0.008† | +0.008† | +0.007† | +0.007† | +0.006† | +0.004† | +0.003† | +0.003† | +0.001 | +0.000 | -0.001 | -0.002 | -0.003† | -0.004† | -0.005† | -0.005† | -0.005† |
| LR-med − MK4-LR | capped hierarchy (V3) | -0.010† | -0.010† | -0.011† | -0.011† | -0.011† | -0.011† | -0.010† | -0.010† | -0.009† | -0.007† | -0.006† | -0.005† | -0.005† | -0.003† | -0.001 | +0.001 | +0.002† |
| LR-trim − MK4-LR | seed only | +0.001 | +0.000 | -0.001 | -0.001 | -0.002† | -0.003† | -0.003† | -0.003† | -0.003† | -0.003† | -0.004† | -0.004† | -0.004† | -0.004† | -0.004† | -0.004† | -0.003† |
| LR-trim − MK4-LR | fine-only (V0) | +0.005† | +0.005† | +0.005† | +0.005† | +0.004† | +0.003† | +0.003† | +0.002† | +0.001 | -0.000 | -0.001 | -0.002† | -0.002† | -0.003† | -0.003† | -0.004† | -0.004† |
| LR-trim − MK4-LR | capped hierarchy (V3) | -0.011† | -0.012† | -0.013† | -0.013† | -0.013† | -0.013† | -0.012† | -0.012† | -0.011† | -0.009† | -0.008† | -0.006† | -0.006† | -0.004† | -0.002 | +0.000 | +0.002 |
| LR without module 2 − MK4-LR | seed only | +0.007† | +0.006† | +0.005† | +0.004† | +0.004† | +0.003† | +0.003† | +0.002† | +0.002 | +0.001 | +0.000 | -0.000 | -0.000 | +0.000 | -0.000 | -0.000 | -0.000 |
| LR without module 2 − MK4-LR | fine-only (V0) | +0.012† | +0.012† | +0.012† | +0.011† | +0.011† | +0.010† | +0.010† | +0.009† | +0.008† | +0.007† | +0.005† | +0.003† | +0.002 | +0.001 | +0.001 | +0.001 | +0.000 |
| LR without module 2 − MK4-LR | capped hierarchy (V3) | -0.008† | -0.008† | -0.008† | -0.008† | -0.008† | -0.007† | -0.006† | -0.006† | -0.005† | -0.003 | -0.002 | -0.001 | -0.001 | -0.000 | +0.002 | +0.004† | +0.005† |

## 7. Per class IoU (new COCO set, primary items, selected γ, + DenseCRF)

| Class | items | MK4 V3 | MK4-LR V3 | LR-trim V3 | MK4 V0 | MK4-LR V0 | LR-trim V0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 89 | 0.657 | 0.669 | 0.660 | 0.666 | 0.669 | 0.673 |
| bicycle | 66 | 0.373 | 0.367 | 0.362 | 0.393 | 0.389 | 0.386 |
| bird | 73 | 0.516 | 0.539 | 0.537 | 0.551 | 0.598 | 0.594 |
| boat | 85 | 0.529 | 0.558 | 0.555 | 0.547 | 0.556 | 0.549 |
| bottle | 158 | 0.290 | 0.314 | 0.307 | 0.317 | 0.387 | 0.396 |
| bus | 114 | 0.760 | 0.761 | 0.753 | 0.756 | 0.739 | 0.735 |
| car | 273 | 0.276 | 0.322 | 0.309 | 0.296 | 0.350 | 0.353 |
| cat | 141 | 0.765 | 0.765 | 0.752 | 0.802 | 0.794 | 0.791 |
| chair | 377 | 0.195 | 0.236 | 0.234 | 0.219 | 0.276 | 0.280 |
| cow | 70 | 0.745 | 0.762 | 0.753 | 0.769 | 0.778 | 0.772 |
| dining table | 398 | 0.308 | 0.345 | 0.335 | 0.287 | 0.362 | 0.362 |
| dog | 122 | 0.672 | 0.681 | 0.677 | 0.695 | 0.700 | 0.703 |
| horse | 92 | 0.712 | 0.727 | 0.715 | 0.720 | 0.755 | 0.755 |
| motorbike | 109 | 0.584 | 0.606 | 0.602 | 0.609 | 0.627 | 0.626 |
| person | 2039 | 0.525 | 0.632 | 0.622 | 0.480 | 0.646 | 0.651 |
| potted plant | 120 | 0.218 | 0.231 | 0.236 | 0.259 | 0.238 | 0.235 |
| sheep | 48 | 0.662 | 0.652 | 0.648 | 0.699 | 0.660 | 0.657 |
| sofa | 173 | 0.527 | 0.540 | 0.545 | 0.565 | 0.588 | 0.596 |
| train | 111 | 0.650 | 0.654 | 0.653 | 0.639 | 0.645 | 0.648 |
| tv | 133 | 0.445 | 0.416 | 0.419 | 0.458 | 0.471 | 0.475 |

Classes where LR-trim beats / trails MK4-LR: V3: 3 / 17; V0: 10 / 10.

Mean gate value at γ = 1 (primary items): MK4 0.483, MK4-LR 0.418, LR-med 0.410, LR-trim 0.412, LR without module 2 0.409.

## 8. Reuse sets (secondary; both evaluated before, so agreement only)

Selected γ, + DenseCRF. COCO val2017 (run 40 set): primary items, COCO-fit thresholds. VOC val: all items, VOC-train thresholds.

| Set | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | MK4-LR mIoU |
|---|---|---|---|---|---|---:|
| COCO val2017 | capped hierarchy (V3) | -0.004 [-0.006, -0.001] | -0.006 [-0.008, -0.003] | -0.003 [-0.006, -0.000] | +0.025 [+0.021, +0.029] | 0.546 |
| COCO val2017 | fine-only (V0) | -0.002 [-0.005, +0.001] | -0.001 [-0.003, +0.002] | +0.005 [+0.002, +0.008] | +0.036 [+0.030, +0.041] | 0.572 |
| VOC val | capped hierarchy (V3) | -0.001 [-0.005, +0.004] | -0.002 [-0.006, +0.002] | +0.002 [-0.003, +0.006] | +0.021 [+0.014, +0.028] | 0.668 |
| VOC val | fine-only (V0) | -0.001 [-0.006, +0.004] | -0.000 [-0.003, +0.003] | +0.004 [+0.001, +0.008] | +0.038 [+0.028, +0.048] | 0.684 |

BRACE stays frozen; any change to the paper's method is a separate owner decision.

