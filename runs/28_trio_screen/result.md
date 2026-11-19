# Three training-free ideas: COCO-Cap screen (`PREREG.md`)

`scripts/jc_trio.py`. **Exploratory screen on COCO-Cap** (owner's choice; COCO-Cap is not untuned transfer for these ideas). One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; Δ at 0.55 with paired image-bootstrap 95% interval (2000). *own best* = best threshold on COCO-Cap itself (in-sample).

## Data and checks

- 707 images, 20 classes. Recomputed vs runtime BRACE seed: max |Δ| 6.2e-04.
- Images with ≥ 1 caption competitor (K1/K0 differ from A): 208; mean competitors 0.32.
- Absent-class gate: mean 0.594; modules without a profile 0 of 7777.

## With DenseCRF (primary)

| Arm | Seed @0.55 | Seed own best | Hierarchy @0.55 | Hierarchy own best | Hierarchy @0.55, competitor images |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.527 | 0.629 (0.400) | 0.708 | 0.719 (0.625) | 0.688 |
| X1 SDXL seed | 0.343 | 0.444 (0.425) | 0.497 | 0.497 (0.525) | 0.467 |
| X2 √(BRACE·SDXL) seed | 0.559 | 0.619 (0.450) | 0.684 | 0.702 (0.650) | 0.675 |
| N1 absent-class-calibrated seed | 0.511 | 0.668 (0.350) | 0.731 | 0.740 (0.625) | 0.705 |
| K1 competition inside hierarchy | 0.527 | 0.629 (0.400) | 0.663 | 0.666 (0.575) | 0.569 |
| K0 competition on final maps | 0.527 | 0.629 (0.400) | 0.641 | 0.663 (0.625) | 0.528 |

## Raw (secondary)

| Arm | Seed @0.55 | Seed own best | Hierarchy @0.55 | Hierarchy own best | Hierarchy @0.55, competitor images |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.510 | 0.591 (0.400) | 0.693 | 0.709 (0.625) | 0.667 |
| X1 SDXL seed | 0.350 | 0.423 (0.425) | 0.489 | 0.490 (0.575) | 0.452 |
| X2 √(BRACE·SDXL) seed | 0.539 | 0.586 (0.450) | 0.668 | 0.693 (0.650) | 0.654 |
| N1 absent-class-calibrated seed | 0.503 | 0.628 (0.350) | 0.717 | 0.731 (0.625) | 0.687 |
| K1 competition inside hierarchy | 0.510 | 0.591 (0.400) | 0.650 | 0.654 (0.600) | 0.558 |
| K0 competition on final maps | 0.510 | 0.591 (0.400) | 0.629 | 0.655 (0.650) | 0.511 |

## Registered comparisons (hierarchy map + DenseCRF, 0.55)

| ID | Comparison | Images | Δ [95% CI] | Promising? |
|---|---|---|---|---|
| T1 | X2 − A | all (707) | -0.025 [-0.037, -0.013] | no |
| T2 | N1 − A | all (707) | +0.022 [+0.011, +0.035] | yes |
| T3 | K1 − A | all (707) | -0.046 [-0.062, -0.028] | no |
| T3b | K1 − A | competitor images (208) | -0.119 [-0.174, -0.085] | no |
| T4 | K1 − K0 | competitor images (208) | +0.041 [+0.006, +0.067] | yes |
| — | X1 − A (descriptive) | all (707) | -0.211 [-0.236, -0.181] | no |
| — | K0 − A (descriptive) | competitor images (208) | -0.160 [-0.202, -0.128] | no |

Promising = interval above zero; licenses a VOC-Sim development check only.

Seed + DenseCRF @0.55 vs A: X1 -0.184 [-0.208, -0.160]; X2 +0.032 [+0.015, +0.048]; N1 -0.016 [-0.025, -0.005]

Per class, hierarchy + DenseCRF @0.55 (A / X1 / X2 / N1 / K1 / K0): aeroplane 0.80 / 0.46 / 0.73 / 0.86 / 0.80 / 0.80, bicycle 0.63 / 0.32 / 0.61 / 0.67 / 0.56 / 0.49, bird 0.78 / 0.74 / 0.76 / 0.78 / 0.75 / 0.72, boat 0.59 / 0.37 / 0.56 / 0.63 / 0.55 / 0.54, bottle 0.57 / 0.11 / 0.55 / 0.63 / 0.62 / 0.51, bus 0.92 / 0.72 / 0.92 / 0.92 / 0.89 / 0.90, car 0.76 / 0.50 / 0.69 / 0.79 / 0.67 / 0.64, cat 0.92 / 0.91 / 0.92 / 0.92 / 0.90 / 0.91, chair 0.48 / 0.16 / 0.48 / 0.53 / 0.38 / 0.35, cow 0.87 / 0.80 / 0.86 / 0.89 / 0.86 / 0.83, dining table 0.48 / 0.60 / 0.50 / 0.53 / 0.48 / 0.48, dog 0.81 / 0.71 / 0.78 / 0.83 / 0.74 / 0.73, horse 0.72 / 0.68 / 0.71 / 0.78 / 0.72 / 0.65, motorbike 0.71 / 0.56 / 0.69 / 0.76 / 0.66 / 0.55, person 0.28 / 0.20 / 0.27 / 0.40 / 0.28 / 0.27, potted plant 0.74 / 0.45 / 0.76 / 0.69 / 0.74 / 0.74, sheep 0.87 / 0.81 / 0.86 / 0.86 / 0.82 / 0.77, sofa 0.70 / 0.18 / 0.59 / 0.75 / 0.64 / 0.62, train 0.75 / 0.50 / 0.71 / 0.70 / 0.51 / 0.62, tv 0.79 / 0.16 / 0.71 / 0.71 / 0.68 / 0.70

