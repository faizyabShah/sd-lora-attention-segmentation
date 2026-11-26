# Absent-class calibration: component study (`PREREG.md`)

`scripts/jc_absent.py`. One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; paired image-bootstrap 95% intervals (200). VOC-Sim = development. **COCO-Cap = screened** (run 28 already looked at this idea there); NX on COCO-Cap uses VOC-Sim profiles only.

## Data and checks

- voc: 10 images, 10 classes; recomputed vs runtime seed max |Δ| 1.8e-04; mean gate NC 1.000, NO 1.000, NL 1.000, NX 0.521; modules without a profile NC 110, NO 110, NL 110, NX 22 of 110.
- coco: 10 images, 10 classes; recomputed vs runtime seed max |Δ| 2.6e-04; mean gate NC 1.000, NO 1.000, NL 1.000, NX 0.479; modules without a profile NC 110, NO 110, NL 110, NX 22 of 110.

## VOC-Sim, DenseCRF

| Seed | Map | OOF | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.671 | 0.677 | 0.753 (0.425) |
| NC calibrated, cross-fitted | seed | 0.671 | 0.677 | 0.753 (0.425) |
| NO control (all-pixel mean) | seed | 0.671 | 0.677 | 0.753 (0.425) |
| NL calibrated, leave-one-out | seed | 0.671 | 0.677 | 0.753 (0.425) |
| NX calibrated, other benchmark's profiles | seed | 0.767 | 0.604 | 0.806 (0.350) |
| A BRACE seed | fine-only | 0.851 | 0.820 | 0.883 (0.350) |
| NC calibrated, cross-fitted | fine-only | 0.851 | 0.820 | 0.883 (0.350) |
| NO control (all-pixel mean) | fine-only | 0.851 | 0.820 | 0.883 (0.350) |
| NL calibrated, leave-one-out | fine-only | 0.851 | 0.820 | 0.883 (0.350) |
| NX calibrated, other benchmark's profiles | fine-only | 0.888 | 0.835 | 0.900 (0.375) |
| A BRACE seed | hierarchy | 0.863 | 0.880 | 0.893 (0.475) |
| NC calibrated, cross-fitted | hierarchy | 0.863 | 0.880 | 0.893 (0.475) |
| NO control (all-pixel mean) | hierarchy | 0.863 | 0.880 | 0.893 (0.475) |
| NL calibrated, leave-one-out | hierarchy | 0.863 | 0.880 | 0.893 (0.475) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.882 | 0.898 | 0.898 (0.550) |

## COCO-Cap (screened), DenseCRF

| Seed | Map | VOC-Sim-selected | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.632 (0.425) | 0.563 | 0.635 (0.375) |
| NC calibrated, cross-fitted | seed | 0.632 (0.425) | 0.563 | 0.635 (0.375) |
| NO control (all-pixel mean) | seed | 0.632 (0.425) | 0.563 | 0.635 (0.375) |
| NL calibrated, leave-one-out | seed | 0.632 (0.425) | 0.563 | 0.635 (0.375) |
| NX calibrated, other benchmark's profiles | seed | 0.639 (0.350) | 0.483 | 0.639 (0.350) |
| A BRACE seed | fine-only | 0.651 (0.350) | 0.697 | 0.717 (0.475) |
| NC calibrated, cross-fitted | fine-only | 0.651 (0.350) | 0.697 | 0.717 (0.475) |
| NO control (all-pixel mean) | fine-only | 0.651 (0.350) | 0.697 | 0.717 (0.475) |
| NL calibrated, leave-one-out | fine-only | 0.651 (0.350) | 0.697 | 0.717 (0.475) |
| NX calibrated, other benchmark's profiles | fine-only | 0.710 (0.375) | 0.671 | 0.727 (0.425) |
| A BRACE seed | hierarchy | 0.737 (0.475) | 0.753 | 0.753 (0.550) |
| NC calibrated, cross-fitted | hierarchy | 0.737 (0.475) | 0.753 | 0.753 (0.550) |
| NO control (all-pixel mean) | hierarchy | 0.737 (0.475) | 0.753 | 0.753 (0.550) |
| NL calibrated, leave-one-out | hierarchy | 0.737 (0.475) | 0.753 | 0.753 (0.550) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.768 (0.550) | 0.768 | 0.780 (0.475) |

## VOC-Sim, raw

| Seed | Map | OOF | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.634 | 0.628 | 0.718 (0.400) |
| NC calibrated, cross-fitted | seed | 0.634 | 0.628 | 0.718 (0.400) |
| NO control (all-pixel mean) | seed | 0.634 | 0.628 | 0.718 (0.400) |
| NL calibrated, leave-one-out | seed | 0.634 | 0.628 | 0.718 (0.400) |
| NX calibrated, other benchmark's profiles | seed | 0.733 | 0.572 | 0.754 (0.300) |
| A BRACE seed | fine-only | 0.811 | 0.808 | 0.854 (0.400) |
| NC calibrated, cross-fitted | fine-only | 0.811 | 0.808 | 0.854 (0.400) |
| NO control (all-pixel mean) | fine-only | 0.811 | 0.808 | 0.854 (0.400) |
| NL calibrated, leave-one-out | fine-only | 0.811 | 0.808 | 0.854 (0.400) |
| NX calibrated, other benchmark's profiles | fine-only | 0.854 | 0.823 | 0.876 (0.400) |
| A BRACE seed | hierarchy | 0.842 | 0.864 | 0.868 (0.500) |
| NC calibrated, cross-fitted | hierarchy | 0.843 | 0.864 | 0.868 (0.500) |
| NO control (all-pixel mean) | hierarchy | 0.843 | 0.864 | 0.868 (0.500) |
| NL calibrated, leave-one-out | hierarchy | 0.843 | 0.864 | 0.868 (0.500) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.859 | 0.878 | 0.878 (0.525) |

## COCO-Cap (screened), raw

| Seed | Map | VOC-Sim-selected | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.580 (0.400) | 0.533 | 0.580 (0.400) |
| NC calibrated, cross-fitted | seed | 0.580 (0.400) | 0.533 | 0.580 (0.400) |
| NO control (all-pixel mean) | seed | 0.580 (0.400) | 0.533 | 0.580 (0.400) |
| NL calibrated, leave-one-out | seed | 0.580 (0.400) | 0.533 | 0.580 (0.400) |
| NX calibrated, other benchmark's profiles | seed | 0.593 (0.300) | 0.468 | 0.599 (0.350) |
| A BRACE seed | fine-only | 0.652 (0.400) | 0.681 | 0.687 (0.500) |
| NC calibrated, cross-fitted | fine-only | 0.652 (0.400) | 0.681 | 0.687 (0.500) |
| NO control (all-pixel mean) | fine-only | 0.652 (0.400) | 0.681 | 0.687 (0.500) |
| NL calibrated, leave-one-out | fine-only | 0.652 (0.400) | 0.681 | 0.687 (0.500) |
| NX calibrated, other benchmark's profiles | fine-only | 0.690 (0.400) | 0.670 | 0.698 (0.450) |
| A BRACE seed | hierarchy | 0.703 (0.500) | 0.714 | 0.715 (0.625) |
| NC calibrated, cross-fitted | hierarchy | 0.703 (0.500) | 0.715 | 0.715 (0.625) |
| NO control (all-pixel mean) | hierarchy | 0.703 (0.500) | 0.715 | 0.715 (0.625) |
| NL calibrated, leave-one-out | hierarchy | 0.703 (0.500) | 0.715 | 0.715 (0.625) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.739 (0.525) | 0.738 | 0.739 (0.525) |

## Registered comparisons (DenseCRF)

| ID | Benchmark | Comparison | Threshold | Δ [95% CI] | Interval above zero? |
|---|---|---|---|---|---|
| P1 (primary) | VOC-Sim | NC − A, hierarchy | OOF | +0.000 [-0.000, +0.000] | no |
| P1b | VOC-Sim | NC − A, hierarchy | 0.55 | +0.000 [+0.000, +0.000] | no |
| P2 | COCO-Cap | NX − A, hierarchy | VOC-Sim-selected | +0.031 [-0.027, +0.089] | no |
| P3 | COCO-Cap | NC − A, hierarchy | VOC-Sim-selected | +0.000 [+0.000, +0.000] | no |
| P4 | VOC-Sim | NC − NO, hierarchy | OOF | +0.000 [+0.000, +0.000] | no |
| P4 | COCO-Cap | NC − NO, hierarchy | VOC-Sim-selected | +0.000 [+0.000, +0.000] | no |
| P5 | VOC-Sim | NC − A, seed | OOF | -0.000 [-0.000, +0.000] | no |
| P5 | VOC-Sim | NC − A, fine-only | OOF | -0.000 [-0.000, +0.000] | no |
| P5 | VOC-Sim | hierarchy − fine-only, A | OOF | +0.011 [-0.003, +0.023] | no |
| P5 | VOC-Sim | hierarchy − fine-only, NC | OOF | +0.011 [-0.003, +0.023] | no |
| P5 | COCO-Cap | NC − A, seed | VOC-Sim-selected | +0.000 [-0.000, +0.000] | no |
| P5 | COCO-Cap | NC − A, fine-only | VOC-Sim-selected | -0.000 [-0.000, +0.000] | no |
| P5 | COCO-Cap | hierarchy − fine-only, A | VOC-Sim-selected | +0.086 [+0.042, +0.141] | yes |
| P5 | COCO-Cap | hierarchy − fine-only, NC | VOC-Sim-selected | +0.087 [+0.042, +0.142] | yes |
| P6 | VOC-Sim | NL − NC, hierarchy | OOF | +0.000 [+0.000, +0.000] | no |
| P6 | COCO-Cap | NL − NC, hierarchy | VOC-Sim-selected | +0.000 [+0.000, +0.000] | no |
| — | COCO-Cap | NC − A, hierarchy (run 28 convention) | 0.55 | +0.000 [-0.000, +0.000] | no |
| — | COCO-Cap | NL − A, hierarchy (run 28 convention) | 0.55 | +0.000 [-0.000, +0.000] | no |
| — | COCO-Cap | NX − A, hierarchy | 0.55 | +0.015 [-0.006, +0.035] | no |

## Reading (rule fixed in `PREREG.md`)

- Development (P1): **fail**.
- Absent-class set matters on VOC-Sim (P4): **not supported**.
- P1 and P2 both above zero (bring to owner): **no**.

Per class, voc, hierarchy + DenseCRF, OOF (A / NC / NO / NL / NX): aeroplane 0.95 / 0.95 / 0.95 / 0.95 / 0.95, bird 0.74 / 0.74 / 0.74 / 0.74 / 0.75, boat 0.80 / 0.80 / 0.80 / 0.80 / 0.87, bus 0.95 / 0.95 / 0.95 / 0.95 / 0.95, cat 0.98 / 0.98 / 0.98 / 0.98 / 0.98, cow 0.90 / 0.90 / 0.90 / 0.90 / 0.94, dog 0.91 / 0.91 / 0.91 / 0.91 / 0.92, motorbike 0.86 / 0.86 / 0.86 / 0.86 / 0.88, potted plant 0.54 / 0.54 / 0.54 / 0.54 / 0.59, sofa 1.00 / 1.00 / 1.00 / 1.00 / 1.00

Per class, coco, hierarchy + DenseCRF, VOC-Sim-selected (A / NC / NO / NL / NX): aeroplane 0.87 / 0.87 / 0.87 / 0.87 / 0.89, bicycle 0.58 / 0.58 / 0.58 / 0.58 / 0.57, bird 0.95 / 0.95 / 0.95 / 0.95 / 0.93, bus 0.89 / 0.89 / 0.89 / 0.89 / 0.90, cat 0.79 / 0.79 / 0.79 / 0.79 / 0.86, cow 0.95 / 0.95 / 0.95 / 0.95 / 0.93, dog 0.93 / 0.93 / 0.93 / 0.93 / 0.84, motorbike 0.39 / 0.39 / 0.39 / 0.39 / 0.61, person 0.61 / 0.61 / 0.61 / 0.61 / 0.49, sofa 0.41 / 0.41 / 0.41 / 0.41 / 0.66

