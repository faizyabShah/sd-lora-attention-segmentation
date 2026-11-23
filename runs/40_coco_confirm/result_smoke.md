# Run 40: low-resolution consensus gate on untouched real COCO val2017 photos (`PREREG.md`)

**Confirmatory in intent; method frozen before capture.** COCO val2017 minus the Section 136 set: 41 items (27 photos); primary items (class area ≥ 1%): 35 (25 photos). Thresholds fitted on 40 COCO train2017 items. Profiles: run 33 source C (490 COCO train photos, captions). Gate strength γ = 1 (primary). Class-macro mIoU; 95% photo-bootstrap intervals (50 samples, paired).


## 1. Primary contrasts (primary items, SA weights on, γ = 1, + DenseCRF at the train threshold)

| # | Contrast | Map | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| C1 | MK4-LR − MK4 | capped hierarchy (V3) | +0.034 [-0.023, +0.084] | not resolved |
| C2 | MK4-LR − MK4 | fine-only (V0) | +0.065 [+0.008, +0.149] | **helps** |
| C3 | MK4-LR − none | capped hierarchy (V3) | +0.023 [-0.042, +0.089] | not resolved |
| C4 | MK4-LR − none | fine-only (V0) | +0.116 [+0.023, +0.181] | **helps** |

- MK4-LR over MK4: **confirmed** (C1 or C2 helps, neither hurts).
- MK4-LR over no gate: **confirmed** (C3 or C4 helps, neither hurts).

## 2. mIoU, primary items (seed only: raw, train threshold; V0 / V3: + DenseCRF, train threshold)

| SA weights | γ | Map | none | MK4 | MK4-LR | MK4-RW |
|:-:|---|---|---:|---:|---:|---:|
|  | 1 | seed only | 0.323 | 0.344 | 0.420 | 0.420 |
|  | 1 | fine-only (V0) | 0.286 | 0.356 | 0.407 | 0.409 |
|  | 1 | capped hierarchy (V3) | 0.322 | 0.457 | 0.426 | 0.419 |
| ✓ | 1 | seed only | 0.364 | 0.345 | 0.445 | 0.460 |
| ✓ | 1 | fine-only (V0) | 0.295 | 0.346 | 0.411 | 0.412 |
| ✓ | 1 | capped hierarchy (V3) | 0.398 | 0.387 | 0.421 | 0.424 |
| ✓ | run 37 selected (MK4 0.25, MK4-LR 1.5, MK4-RW 1.5) | seed only | 0.364 | 0.408 | 0.443 | 0.460 |
| ✓ | run 37 selected (MK4 0.5, MK4-LR 1.5, MK4-RW 1.5) | fine-only (V0) | 0.295 | 0.343 | 0.418 | 0.417 |
| ✓ | run 37 selected (MK4 0.75, MK4-LR 1.5, MK4-RW 1.5) | capped hierarchy (V3) | 0.398 | 0.407 | 0.424 | 0.430 |

## 3. Paired differences, primary items

| SA weights | γ | Map | MK4 − none | MK4-LR − MK4 | MK4-RW − MK4 | MK4-LR − none |
|:-:|---|---|---|---|---|---|
|  | 1 | seed only | +0.021 [-0.050, +0.095] | +0.075 [+0.041, +0.105] | +0.076 [+0.041, +0.105] | +0.096 [+0.053, +0.149] |
|  | 1 | fine-only (V0) | +0.070 [-0.037, +0.152] | +0.051 [+0.021, +0.100] | +0.053 [+0.024, +0.101] | +0.121 [+0.047, +0.197] |
|  | 1 | capped hierarchy (V3) | +0.135 [+0.084, +0.170] | -0.031 [-0.053, +0.006] | -0.038 [-0.067, +0.003] | +0.104 [+0.054, +0.152] |
| ✓ | 1 | seed only | -0.019 [-0.078, +0.069] | +0.100 [+0.046, +0.145] | +0.115 [+0.063, +0.163] | +0.081 [+0.043, +0.129] |
| ✓ | 1 | fine-only (V0) | +0.051 [-0.058, +0.134] | +0.065 [+0.008, +0.149] | +0.066 [+0.008, +0.149] | +0.116 [+0.023, +0.181] |
| ✓ | 1 | capped hierarchy (V3) | -0.011 [-0.099, +0.092] | +0.034 [-0.023, +0.084] | +0.037 [-0.020, +0.094] | +0.023 [-0.042, +0.089] |
| ✓ | selected | seed only | +0.045 [+0.010, +0.071] | +0.034 [+0.011, +0.086] | +0.052 [+0.027, +0.106] | +0.079 [+0.037, +0.139] |
| ✓ | selected | fine-only (V0) | +0.047 [-0.047, +0.126] | +0.076 [+0.028, +0.130] | +0.075 [+0.026, +0.131] | +0.123 [+0.034, +0.202] |
| ✓ | selected | capped hierarchy (V3) | +0.009 [-0.068, +0.088] | +0.017 [-0.015, +0.052] | +0.023 [-0.007, +0.059] | +0.026 [-0.046, +0.100] |

Hierarchy (V3 − V0) and SA weights (on − off) with the gate fixed, γ = 1, primary rule:

| Gate | hierarchy, SA weights on | hierarchy, uniform | SA weights, V3 | SA weights, V0 |
|---|---|---|---|---|
| none | +0.103 [+0.006, +0.167] | +0.036 [-0.015, +0.074] | +0.077 [+0.014, +0.117] | +0.010 [-0.004, +0.025] |
| MK4 | +0.041 [+0.029, +0.062] | +0.101 [+0.039, +0.160] | -0.069 [-0.119, -0.001] | -0.010 [-0.025, +0.004] |
| MK4-LR | +0.010 [-0.018, +0.030] | +0.019 [-0.013, +0.039] | -0.005 [-0.023, +0.014] | +0.004 [-0.034, +0.040] |
| MK4-RW | +0.012 [-0.014, +0.028] | +0.010 [-0.019, +0.030] | +0.005 [-0.009, +0.012] | +0.003 [-0.035, +0.037] |

## 4. Person excluded, and all items (SA weights on, γ = 1, + DenseCRF)

| Contrast | Map | person excluded | all items (incl. < 1%) |
|---|---|---|---|
| MK4-LR − MK4 | capped hierarchy (V3) | +0.036 [-0.024, +0.078] | +0.054 [-0.005, +0.088] |
| MK4-LR − MK4 | fine-only (V0) | +0.060 [-0.005, +0.142] | +0.076 [+0.013, +0.110] |
| MK4-LR − none | capped hierarchy (V3) | +0.027 [-0.045, +0.084] | +0.014 [-0.028, +0.064] |
| MK4-LR − none | fine-only (V0) | +0.110 [+0.017, +0.159] | +0.083 [+0.030, +0.147] |
| MK4 − none | capped hierarchy (V3) | -0.009 [-0.099, +0.096] | -0.039 [-0.087, +0.031] |
| MK4 − none | fine-only (V0) | +0.050 [-0.072, +0.141] | +0.007 [-0.058, +0.087] |
| MK4-RW − MK4 | capped hierarchy (V3) | +0.038 [-0.024, +0.091] | +0.057 [-0.003, +0.090] |
| MK4-RW − MK4 | fine-only (V0) | +0.060 [-0.004, +0.142] | +0.088 [+0.018, +0.118] |

## 5. Other threshold rules (primary items, SA weights on, γ = 1)

| Map | Gate | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.295 | 0.332 | 0.356 | 0.377 |
| fine-only (V0) | MK4 | 0.346 | 0.363 | 0.355 | 0.370 |
| fine-only (V0) | MK4-LR | 0.411 | 0.412 | 0.449 | 0.430 |
| fine-only (V0) | MK4-RW | 0.412 | 0.415 | 0.450 | 0.432 |
| capped hierarchy (V3) | none | 0.398 | 0.412 | 0.407 | 0.400 |
| capped hierarchy (V3) | MK4 | 0.387 | 0.458 | 0.400 | 0.462 |
| capped hierarchy (V3) | MK4-LR | 0.421 | 0.462 | 0.443 | 0.462 |
| capped hierarchy (V3) | MK4-RW | 0.424 | 0.460 | 0.451 | 0.464 |

Raw maps, difference at every threshold (primary items, SA weights on, γ = 1; † = 95% CI excludes 0):

| Contrast | Map | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MK4-LR − MK4 | seed only | +0.087† | +0.089† | +0.089† | +0.088† | +0.086† | +0.079† | +0.073† | +0.063 | +0.051 | +0.034 | +0.011 | -0.009 | -0.025 | -0.035 | -0.036 | -0.035 | -0.034 |
| MK4-LR − MK4 | fine-only (V0) | +0.022 | +0.045† | +0.066 | +0.075 | +0.082 | +0.084 | +0.088† | +0.084† | +0.078† | +0.070† | +0.060† | +0.048 | +0.042 | +0.036 | +0.031 | +0.039† | +0.060† |
| MK4-LR − MK4 | capped hierarchy (V3) | +0.036† | +0.028† | +0.021 | +0.018 | +0.011 | +0.008 | +0.009 | +0.008 | +0.005 | +0.001 | -0.000 | +0.003 | +0.012 | +0.020 | +0.025 | +0.027 | +0.030 |
| MK4-LR − none | seed only | +0.100† | +0.097† | +0.091† | +0.081† | +0.073† | +0.059† | +0.045† | +0.033† | +0.020 | +0.008 | +0.003 | +0.000 | -0.004 | -0.008 | -0.007 | -0.009 | -0.010 |
| MK4-LR − none | fine-only (V0) | +0.054† | +0.054† | +0.053† | +0.053† | +0.050† | +0.048† | +0.053† | +0.055† | +0.058† | +0.061† | +0.054† | +0.040 | +0.041 | +0.033 | +0.021 | +0.024 | +0.032 |
| MK4-LR − none | capped hierarchy (V3) | +0.016† | +0.019† | +0.025† | +0.031† | +0.036† | +0.046† | +0.057† | +0.064† | +0.069† | +0.070† | +0.063† | +0.057† | +0.053† | +0.049† | +0.050† | +0.049† | +0.042† |

## 6. Per class IoU (primary items, SA weights on, γ = 1, + DenseCRF)

| Class | items | none V3 | MK4 V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boat | 1 | 0.039 | 0.065 | 0.040 | 0.039 | 0.029 | 0.072 | 0.048 | 0.051 |
| bottle | 1 | 0.333 | 0.106 | 0.107 | 0.107 | 0.106 | 0.101 | 0.110 | 0.107 |
| bus | 2 | 0.914 | 0.878 | 0.891 | 0.901 | 0.848 | 0.857 | 0.906 | 0.905 |
| car | 1 | 0.324 | 0.823 | 0.773 | 0.727 | 0.434 | 0.845 | 0.610 | 0.616 |
| cat | 1 | 0.935 | 0.881 | 0.898 | 0.925 | 0.372 | 0.824 | 0.943 | 0.943 |
| chair | 3 | 0.000 | 0.018 | 0.069 | 0.080 | 0.007 | 0.016 | 0.058 | 0.059 |
| dining table | 4 | 0.394 | 0.354 | 0.399 | 0.430 | 0.137 | 0.143 | 0.462 | 0.466 |
| person | 18 | 0.408 | 0.378 | 0.398 | 0.402 | 0.256 | 0.311 | 0.424 | 0.431 |
| potted plant | 1 | 0.077 | 0.055 | 0.057 | 0.057 | 0.005 | 0.062 | 0.052 | 0.040 |
| train | 1 | 0.539 | 0.147 | 0.502 | 0.517 | 0.466 | 0.000 | 0.452 | 0.449 |
| tv | 2 | 0.417 | 0.557 | 0.503 | 0.478 | 0.590 | 0.576 | 0.457 | 0.465 |

Classes where MK4-LR beats / trails MK4: V3: 8 / 3; V0: 7 / 4.

Mean gate value at γ = 1 (primary items): MK4 0.505, MK4-LR 0.466, MK4-RW 0.458.

BRACE stays frozen; any change to the paper's method is a separate owner decision.

