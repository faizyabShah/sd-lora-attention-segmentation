# Absent-class calibration: component study (`PREREG.md`)

`scripts/jc_absent.py`. One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; paired image-bootstrap 95% intervals (2000). VOC-Sim = development. **COCO-Cap = screened** (run 28 already looked at this idea there); NX on COCO-Cap uses VOC-Sim profiles only.

## Data and checks

- voc: 490 images, 20 classes; recomputed vs runtime seed max |Δ| 5.3e-04; mean gate NC 0.607, NO 0.399, NL 0.609, NX 0.648; modules without a profile NC 0, NO 0, NL 0, NX 0 of 5390.
- coco: 707 images, 20 classes; recomputed vs runtime seed max |Δ| 6.2e-04; mean gate NC 0.589, NO 0.336, NL 0.594, NX 0.468; modules without a profile NC 0, NO 0, NL 0, NX 0 of 7777.

## VOC-Sim, DenseCRF

| Seed | Map | OOF | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.736 | 0.628 | 0.736 (0.400) |
| NC calibrated, cross-fitted | seed | 0.753 | 0.554 | 0.753 (0.300) |
| NO control (all-pixel mean) | seed | 0.757 | 0.530 | 0.757 (0.300) |
| NL calibrated, leave-one-out | seed | 0.753 | 0.554 | 0.753 (0.300) |
| NX calibrated, other benchmark's profiles | seed | 0.752 | 0.590 | 0.752 (0.325) |
| A BRACE seed | fine-only | 0.825 | 0.796 | 0.827 (0.425) |
| NC calibrated, cross-fitted | fine-only | 0.831 | 0.775 | 0.833 (0.400) |
| NO control (all-pixel mean) | fine-only | 0.831 | 0.772 | 0.831 (0.400) |
| NL calibrated, leave-one-out | fine-only | 0.831 | 0.775 | 0.833 (0.400) |
| NX calibrated, other benchmark's profiles | fine-only | 0.840 | 0.790 | 0.840 (0.425) |
| A BRACE seed | hierarchy | 0.832 | 0.833 | 0.833 (0.550) |
| NC calibrated, cross-fitted | hierarchy | 0.832 | 0.831 | 0.832 (0.525) |
| NO control (all-pixel mean) | hierarchy | 0.831 | 0.830 | 0.831 (0.525) |
| NL calibrated, leave-one-out | hierarchy | 0.830 | 0.830 | 0.831 (0.525) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.838 | 0.837 | 0.838 (0.525) |

## COCO-Cap (screened), DenseCRF

| Seed | Map | VOC-Sim-selected | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.629 (0.400) | 0.527 | 0.629 (0.400) |
| NC calibrated, cross-fitted | seed | 0.659 (0.300) | 0.510 | 0.670 (0.350) |
| NO control (all-pixel mean) | seed | 0.660 (0.300) | 0.464 | 0.663 (0.325) |
| NL calibrated, leave-one-out | seed | 0.658 (0.300) | 0.511 | 0.668 (0.350) |
| NX calibrated, other benchmark's profiles | seed | 0.657 (0.325) | 0.443 | 0.657 (0.325) |
| A BRACE seed | fine-only | 0.673 (0.425) | 0.698 | 0.702 (0.525) |
| NC calibrated, cross-fitted | fine-only | 0.708 (0.400) | 0.724 | 0.732 (0.500) |
| NO control (all-pixel mean) | fine-only | 0.697 (0.400) | 0.704 | 0.716 (0.500) |
| NL calibrated, leave-one-out | fine-only | 0.707 (0.400) | 0.720 | 0.730 (0.500) |
| NX calibrated, other benchmark's profiles | fine-only | 0.715 (0.425) | 0.710 | 0.725 (0.475) |
| A BRACE seed | hierarchy | 0.708 (0.550) | 0.708 | 0.719 (0.600) |
| NC calibrated, cross-fitted | hierarchy | 0.722 (0.525) | 0.730 | 0.740 (0.600) |
| NO control (all-pixel mean) | hierarchy | 0.719 (0.525) | 0.725 | 0.730 (0.600) |
| NL calibrated, leave-one-out | hierarchy | 0.723 (0.525) | 0.731 | 0.740 (0.625) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.721 (0.525) | 0.728 | 0.733 (0.575) |

## VOC-Sim, raw

| Seed | Map | OOF | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.706 | 0.613 | 0.706 (0.400) |
| NC calibrated, cross-fitted | seed | 0.724 | 0.550 | 0.724 (0.300) |
| NO control (all-pixel mean) | seed | 0.726 | 0.528 | 0.726 (0.300) |
| NL calibrated, leave-one-out | seed | 0.724 | 0.550 | 0.724 (0.300) |
| NX calibrated, other benchmark's profiles | seed | 0.722 | 0.583 | 0.722 (0.325) |
| A BRACE seed | fine-only | 0.811 | 0.792 | 0.813 (0.450) |
| NC calibrated, cross-fitted | fine-only | 0.817 | 0.774 | 0.818 (0.425) |
| NO control (all-pixel mean) | fine-only | 0.816 | 0.771 | 0.816 (0.400) |
| NL calibrated, leave-one-out | fine-only | 0.816 | 0.774 | 0.817 (0.425) |
| NX calibrated, other benchmark's profiles | fine-only | 0.826 | 0.787 | 0.826 (0.425) |
| A BRACE seed | hierarchy | 0.818 | 0.819 | 0.819 (0.550) |
| NC calibrated, cross-fitted | hierarchy | 0.816 | 0.818 | 0.818 (0.550) |
| NO control (all-pixel mean) | hierarchy | 0.817 | 0.817 | 0.817 (0.550) |
| NL calibrated, leave-one-out | hierarchy | 0.817 | 0.818 | 0.818 (0.550) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.824 | 0.824 | 0.824 (0.550) |

## COCO-Cap (screened), raw

| Seed | Map | VOC-Sim-selected | 0.55 | own best |
|---|---|---:|---:|---:|
| A BRACE seed | seed | 0.591 (0.400) | 0.510 | 0.591 (0.400) |
| NC calibrated, cross-fitted | seed | 0.620 (0.300) | 0.502 | 0.629 (0.350) |
| NO control (all-pixel mean) | seed | 0.620 (0.300) | 0.464 | 0.621 (0.325) |
| NL calibrated, leave-one-out | seed | 0.619 (0.300) | 0.503 | 0.628 (0.350) |
| NX calibrated, other benchmark's profiles | seed | 0.617 (0.325) | 0.446 | 0.617 (0.325) |
| A BRACE seed | fine-only | 0.670 (0.450) | 0.690 | 0.693 (0.525) |
| NC calibrated, cross-fitted | fine-only | 0.704 (0.425) | 0.717 | 0.723 (0.500) |
| NO control (all-pixel mean) | fine-only | 0.680 (0.400) | 0.700 | 0.707 (0.500) |
| NL calibrated, leave-one-out | fine-only | 0.703 (0.425) | 0.715 | 0.721 (0.500) |
| NX calibrated, other benchmark's profiles | fine-only | 0.701 (0.425) | 0.704 | 0.714 (0.500) |
| A BRACE seed | hierarchy | 0.692 (0.550) | 0.692 | 0.709 (0.625) |
| NC calibrated, cross-fitted | hierarchy | 0.717 (0.550) | 0.717 | 0.730 (0.625) |
| NO control (all-pixel mean) | hierarchy | 0.712 (0.550) | 0.712 | 0.721 (0.625) |
| NL calibrated, leave-one-out | hierarchy | 0.717 (0.550) | 0.717 | 0.731 (0.625) |
| NX calibrated, other benchmark's profiles | hierarchy | 0.715 (0.550) | 0.715 | 0.723 (0.600) |

## Registered comparisons (DenseCRF)

| ID | Benchmark | Comparison | Threshold | Δ [95% CI] | Interval above zero? |
|---|---|---|---|---|---|
| P1 (primary) | VOC-Sim | NC − A, hierarchy | OOF | +0.000 [-0.005, +0.006] | no |
| P1b | VOC-Sim | NC − A, hierarchy | 0.55 | -0.002 [-0.007, +0.003] | no |
| P2 | COCO-Cap | NX − A, hierarchy | VOC-Sim-selected | +0.013 [+0.001, +0.025] | yes |
| P3 | COCO-Cap | NC − A, hierarchy | VOC-Sim-selected | +0.014 [+0.003, +0.026] | yes |
| P4 | VOC-Sim | NC − NO, hierarchy | OOF | +0.001 [-0.001, +0.005] | no |
| P4 | COCO-Cap | NC − NO, hierarchy | VOC-Sim-selected | +0.004 [-0.004, +0.012] | no |
| P5 | VOC-Sim | NC − A, seed | OOF | +0.016 [+0.010, +0.023] | yes |
| P5 | VOC-Sim | NC − A, fine-only | OOF | +0.007 [+0.001, +0.012] | yes |
| P5 | VOC-Sim | hierarchy − fine-only, A | OOF | +0.007 [+0.001, +0.013] | yes |
| P5 | VOC-Sim | hierarchy − fine-only, NC | OOF | +0.001 [-0.006, +0.007] | no |
| P5 | COCO-Cap | NC − A, seed | VOC-Sim-selected | +0.030 [+0.018, +0.042] | yes |
| P5 | COCO-Cap | NC − A, fine-only | VOC-Sim-selected | +0.036 [+0.023, +0.047] | yes |
| P5 | COCO-Cap | hierarchy − fine-only, A | VOC-Sim-selected | +0.035 [+0.024, +0.046] | yes |
| P5 | COCO-Cap | hierarchy − fine-only, NC | VOC-Sim-selected | +0.014 [+0.004, +0.025] | yes |
| P6 | VOC-Sim | NL − NC, hierarchy | OOF | -0.002 [-0.004, -0.001] | no |
| P6 | COCO-Cap | NL − NC, hierarchy | VOC-Sim-selected | +0.001 [-0.002, +0.004] | no |
| — | COCO-Cap | NC − A, hierarchy (run 28 convention) | 0.55 | +0.022 [+0.010, +0.034] | yes |
| — | COCO-Cap | NL − A, hierarchy (run 28 convention) | 0.55 | +0.022 [+0.011, +0.035] | yes |
| — | COCO-Cap | NX − A, hierarchy | 0.55 | +0.020 [+0.007, +0.035] | yes |

## Reading (rule fixed in `PREREG.md`)

- Development (P1): **fail**.
- Absent-class set matters on VOC-Sim (P4): **not supported**.
- P1 and P2 both above zero (bring to owner): **no**.

Per class, voc, hierarchy + DenseCRF, OOF (A / NC / NO / NL / NX): aeroplane 0.88 / 0.89 / 0.89 / 0.89 / 0.89, bicycle 0.80 / 0.78 / 0.79 / 0.78 / 0.79, bird 0.92 / 0.92 / 0.92 / 0.92 / 0.93, boat 0.85 / 0.86 / 0.86 / 0.86 / 0.85, bottle 0.87 / 0.90 / 0.91 / 0.91 / 0.91, bus 0.92 / 0.92 / 0.92 / 0.92 / 0.92, car 0.85 / 0.86 / 0.86 / 0.85 / 0.86, cat 0.85 / 0.85 / 0.85 / 0.85 / 0.87, chair 0.79 / 0.78 / 0.78 / 0.78 / 0.78, cow 0.90 / 0.90 / 0.87 / 0.89 / 0.94, dining table 0.43 / 0.44 / 0.45 / 0.44 / 0.44, dog 0.93 / 0.93 / 0.93 / 0.93 / 0.92, horse 0.90 / 0.91 / 0.90 / 0.91 / 0.93, motorbike 0.87 / 0.87 / 0.87 / 0.87 / 0.87, person 0.81 / 0.82 / 0.83 / 0.82 / 0.84, potted plant 0.75 / 0.80 / 0.77 / 0.80 / 0.81, sheep 0.76 / 0.71 / 0.69 / 0.71 / 0.77, sofa 0.96 / 0.95 / 0.95 / 0.95 / 0.95, train 0.74 / 0.71 / 0.75 / 0.70 / 0.70, tv 0.85 / 0.82 / 0.81 / 0.82 / 0.80

Per class, coco, hierarchy + DenseCRF, VOC-Sim-selected (A / NC / NO / NL / NX): aeroplane 0.80 / 0.85 / 0.85 / 0.85 / 0.84, bicycle 0.63 / 0.65 / 0.66 / 0.65 / 0.64, bird 0.78 / 0.78 / 0.78 / 0.78 / 0.78, boat 0.59 / 0.61 / 0.61 / 0.60 / 0.64, bottle 0.57 / 0.60 / 0.58 / 0.59 / 0.63, bus 0.92 / 0.92 / 0.92 / 0.92 / 0.92, car 0.76 / 0.80 / 0.73 / 0.79 / 0.76, cat 0.92 / 0.92 / 0.92 / 0.92 / 0.91, chair 0.47 / 0.52 / 0.50 / 0.52 / 0.50, cow 0.87 / 0.90 / 0.87 / 0.90 / 0.86, dining table 0.47 / 0.54 / 0.58 / 0.55 / 0.55, dog 0.81 / 0.83 / 0.82 / 0.83 / 0.83, horse 0.72 / 0.77 / 0.75 / 0.77 / 0.77, motorbike 0.71 / 0.75 / 0.74 / 0.75 / 0.75, person 0.28 / 0.39 / 0.35 / 0.39 / 0.44, potted plant 0.74 / 0.66 / 0.67 / 0.68 / 0.69, sheep 0.87 / 0.87 / 0.86 / 0.87 / 0.89, sofa 0.70 / 0.74 / 0.74 / 0.73 / 0.71, train 0.75 / 0.67 / 0.71 / 0.67 / 0.65, tv 0.79 / 0.69 / 0.73 / 0.69 / 0.68

