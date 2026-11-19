# Three training-free ideas: COCO-Cap screen (`PREREG.md`)

`scripts/jc_trio.py`. **Exploratory screen on COCO-Cap** (owner's choice; COCO-Cap is not untuned transfer for these ideas). One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; Δ at 0.55 with paired image-bootstrap 95% interval (50). *own best* = best threshold on COCO-Cap itself (in-sample).

## Data and checks

- 8 images, 6 classes. Recomputed vs runtime BRACE seed: max |Δ| 2.6e-04.
- Images with ≥ 1 caption competitor (K1/K0 differ from A): 2; mean competitors 0.25.
- Absent-class gate: mean 0.634; modules without a profile 33 of 88.

## With DenseCRF (primary)

| Arm | Seed @0.55 | Seed own best | Hierarchy @0.55 | Hierarchy own best | Hierarchy @0.55, competitor images |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.539 | 0.646 (0.350) | 0.749 | 0.789 (0.500) | 0.618 |
| X1 SDXL seed | 0.311 | 0.622 (0.300) | 0.649 | 0.748 (0.400) | 0.497 |
| X2 √(BRACE·SDXL) seed | 0.504 | 0.684 (0.350) | 0.689 | 0.800 (0.450) | 0.491 |
| N1 absent-class-calibrated seed | 0.491 | 0.731 (0.300) | 0.836 | 0.852 (0.425) | 0.772 |
| K1 competition inside hierarchy | 0.539 | 0.646 (0.350) | 0.748 | 0.787 (0.450) | 0.614 |
| K0 competition on final maps | 0.539 | 0.646 (0.350) | 0.730 | 0.773 (0.450) | 0.559 |

## Raw (secondary)

| Arm | Seed @0.55 | Seed own best | Hierarchy @0.55 | Hierarchy own best | Hierarchy @0.55, competitor images |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.493 | 0.576 (0.325) | 0.721 | 0.748 (0.500) | 0.597 |
| X1 SDXL seed | 0.315 | 0.554 (0.325) | 0.630 | 0.682 (0.400) | 0.480 |
| X2 √(BRACE·SDXL) seed | 0.468 | 0.615 (0.325) | 0.670 | 0.754 (0.475) | 0.477 |
| N1 absent-class-calibrated seed | 0.470 | 0.658 (0.300) | 0.808 | 0.809 (0.575) | 0.746 |
| K1 competition inside hierarchy | 0.493 | 0.576 (0.325) | 0.719 | 0.747 (0.500) | 0.589 |
| K0 competition on final maps | 0.493 | 0.576 (0.325) | 0.705 | 0.732 (0.500) | 0.548 |

## Registered comparisons (hierarchy map + DenseCRF, 0.55)

| ID | Comparison | Images | Δ [95% CI] | Promising? |
|---|---|---|---|---|
| T1 | X2 − A | all (8) | -0.060 [-0.112, +0.012] | no |
| T2 | N1 − A | all (8) | +0.087 [+0.008, +0.168] | yes |
| T3 | K1 − A | all (8) | -0.001 [-0.003, +0.002] | no |
| T3b | K1 − A | competitor images (2) | -0.004 [-0.013, +0.005] | no |
| T4 | K1 − K0 | competitor images (2) | +0.055 [-0.012, +0.122] | no |
| — | X1 − A (descriptive) | all (8) | -0.100 [-0.259, +0.088] | no |
| — | K0 − A (descriptive) | competitor images (2) | -0.059 [-0.117, -0.001] | no |

Promising = interval above zero; licenses a VOC-Sim development check only.

Seed + DenseCRF @0.55 vs A: X1 -0.228 [-0.319, -0.136]; X2 -0.035 [-0.077, +0.000]; N1 -0.048 [-0.076, +0.014]

Per class, hierarchy + DenseCRF @0.55 (A / X1 / X2 / N1 / K1 / K0): aeroplane 0.88 / 0.71 / 0.88 / 0.88 / 0.88 / 0.88, bird 0.82 / 0.87 / 0.82 / 0.81 / 0.82 / 0.82, car 0.86 / 0.32 / 0.56 / 0.86 / 0.84 / 0.85, cow 0.61 / 0.39 / 0.51 / 0.84 / 0.61 / 0.61, horse 0.38 / 0.67 / 0.42 / 0.68 / 0.39 / 0.26, sheep 0.95 / 0.94 / 0.94 / 0.94 / 0.95 / 0.95

