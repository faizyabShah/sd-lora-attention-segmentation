# Seed × attention source × propagation rule (`PREREG.md`)

`scripts/jc_srcrule.py`. One live generation per image. Sources: **G** = the frozen runtime's generation self-attention (calls 10–20); **N** = NERVE's source, one pass of the generated image noised to t = 101 with an empty prompt. Rules: BRACE fine-only / capped hierarchy (frozen code, BRACE's modules) and NERVE's walk (released code, ported; NERVE's layers; on G the fused matrix is averaged over the 11 calls). Seeds: **CA** (live) and **CLIP** (NERVE's, from runs/17, stored image). Class-macro mIoU. Primary: threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap), with DenseCRF. Δ with image-bootstrap 95% interval (2000).

## VOC-Sim (490 images)

Regeneration bit-identical to the stored image: 238/490. CA × generation × hierarchy map vs stored capture: median max |Δ| 6.1e-04. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 1.7e-02 on all images, 1.4e-02 on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.705 | 0.727 | +0.022 [+0.012, +0.033] | 0.706 | 0.711 |
| generation SA, BRACE fine-only | 0.823 | 0.830 | +0.007 [-0.008, +0.022] | 0.846 | 0.818 |
| generation SA, BRACE hierarchy | 0.830 | 0.836 | +0.006 [-0.007, +0.018] | 0.847 | 0.825 |
| generation SA, NERVE walk | 0.824 | 0.835 | +0.011 [-0.004, +0.026] | 0.849 | 0.823 |
| NERVE pass SA, BRACE fine-only | 0.818 | 0.826 | +0.008 [-0.005, +0.021] | 0.839 | 0.816 |
| NERVE pass SA, BRACE hierarchy | 0.829 | 0.833 | +0.004 [-0.009, +0.015] | 0.845 | 0.823 |
| NERVE pass SA, NERVE walk | 0.826 | 0.839 | +0.014 [-0.000, +0.028] | 0.847 | 0.828 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.611 | 0.722 | +0.111 [+0.096, +0.124] | 0.704 | 0.619 |
| generation SA, BRACE fine-only | 0.796 | 0.828 | +0.033 [+0.017, +0.049] | 0.845 | 0.791 |
| generation SA, BRACE hierarchy | 0.830 | 0.834 | +0.004 [-0.009, +0.017] | 0.845 | 0.825 |
| generation SA, NERVE walk | 0.761 | 0.829 | +0.068 [+0.052, +0.084] | 0.842 | 0.776 |
| NERVE pass SA, BRACE fine-only | 0.783 | 0.827 | +0.044 [+0.028, +0.060] | 0.840 | 0.786 |
| NERVE pass SA, BRACE hierarchy | 0.829 | 0.829 | +0.001 [-0.013, +0.013] | 0.843 | 0.822 |
| NERVE pass SA, NERVE walk | 0.756 | 0.836 | +0.080 [+0.063, +0.096] | 0.846 | 0.777 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.705 | 0.727 | 0.706 | 0.711 |
| generation SA, BRACE fine-only | 0.810 | 0.814 | 0.833 | 0.804 |
| generation SA, BRACE hierarchy | 0.816 | 0.825 | 0.837 | 0.816 |
| generation SA, NERVE walk | 0.815 | 0.827 | 0.840 | 0.811 |
| NERVE pass SA, BRACE fine-only | 0.804 | 0.809 | 0.824 | 0.796 |
| NERVE pass SA, BRACE hierarchy | 0.814 | 0.821 | 0.833 | 0.808 |
| NERVE pass SA, NERVE walk | 0.816 | 0.828 | 0.838 | 0.812 |

## COCO-Cap (707 images)

Regeneration bit-identical to the stored image: 243/707. CA × generation × hierarchy map vs stored capture: median max |Δ| 2.5e-03. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 2.9e-02 on all images, 3.0e-02 on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.584 | 0.650 | +0.067 [+0.047, +0.087] | 0.591 | 0.611 |
| generation SA, BRACE fine-only | 0.662 | 0.704 | +0.042 [+0.020, +0.065] | 0.705 | 0.677 |
| generation SA, BRACE hierarchy | 0.696 | 0.704 | +0.008 [-0.012, +0.029] | 0.720 | 0.687 |
| generation SA, NERVE walk | 0.663 | 0.706 | +0.043 [+0.022, +0.066] | 0.703 | 0.668 |
| NERVE pass SA, BRACE fine-only | 0.658 | 0.715 | +0.057 [+0.034, +0.082] | 0.701 | 0.682 |
| NERVE pass SA, BRACE hierarchy | 0.680 | 0.707 | +0.027 [+0.006, +0.050] | 0.711 | 0.691 |
| NERVE pass SA, NERVE walk | 0.669 | 0.720 | +0.051 [+0.029, +0.075] | 0.706 | 0.680 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.501 | 0.642 | +0.142 [+0.121, +0.161] | 0.591 | 0.536 |
| generation SA, BRACE fine-only | 0.684 | 0.710 | +0.026 [+0.004, +0.049] | 0.707 | 0.695 |
| generation SA, BRACE hierarchy | 0.696 | 0.694 | -0.002 [-0.022, +0.020] | 0.712 | 0.681 |
| generation SA, NERVE walk | 0.670 | 0.718 | +0.048 [+0.026, +0.071] | 0.710 | 0.685 |
| NERVE pass SA, BRACE fine-only | 0.661 | 0.715 | +0.054 [+0.031, +0.078] | 0.701 | 0.688 |
| NERVE pass SA, BRACE hierarchy | 0.690 | 0.699 | +0.008 [-0.013, +0.030] | 0.705 | 0.685 |
| NERVE pass SA, NERVE walk | 0.658 | 0.729 | +0.071 [+0.049, +0.096] | 0.709 | 0.689 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.584 | 0.650 | 0.591 | 0.611 |
| generation SA, BRACE fine-only | 0.660 | 0.707 | 0.697 | 0.672 |
| generation SA, BRACE hierarchy | 0.682 | 0.699 | 0.712 | 0.681 |
| generation SA, NERVE walk | 0.654 | 0.710 | 0.699 | 0.669 |
| NERVE pass SA, BRACE fine-only | 0.653 | 0.709 | 0.689 | 0.665 |
| NERVE pass SA, BRACE hierarchy | 0.677 | 0.700 | 0.702 | 0.684 |
| NERVE pass SA, NERVE walk | 0.658 | 0.719 | 0.701 | 0.667 |

## Thresholds selected on full VOC-Sim (with DenseCRF)

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.400 | 0.625 | 0.600 | 0.400 |
| generation SA, BRACE fine-only | 0.425 | 0.525 | 0.525 | 0.450 |
| generation SA, BRACE hierarchy | 0.550 | 0.600 | 0.600 | 0.575 |
| generation SA, NERVE walk | 0.400 | 0.475 | 0.475 | 0.400 |
| NERVE pass SA, BRACE fine-only | 0.425 | 0.550 | 0.550 | 0.450 |
| NERVE pass SA, BRACE hierarchy | 0.525 | 0.600 | 0.575 | 0.575 |
| NERVE pass SA, NERVE walk | 0.400 | 0.500 | 0.500 | 0.400 |

## Registered contrasts (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | Seed | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images |
|---|---|---|---|---|---|
| R-G | rule: NERVE walk − BRACE hierarchy, generation SA | CA | -0.006 [-0.014, +0.003] | -0.033 [-0.045, -0.021] | -0.027 [-0.049, -0.007] |
| R-G | rule: NERVE walk − BRACE hierarchy, generation SA | CLIP | -0.001 [-0.009, +0.008] | +0.002 [-0.009, +0.013] | +0.009 [-0.011, +0.028] |
| R-N | rule: NERVE walk − BRACE hierarchy, NERVE pass SA | CA | -0.004 [-0.013, +0.006] | -0.011 [-0.021, -0.001] | -0.011 [-0.027, +0.006] |
| R-N | rule: NERVE walk − BRACE hierarchy, NERVE pass SA | CLIP | +0.006 [-0.000, +0.013] | +0.013 [+0.004, +0.023] | +0.006 [-0.007, +0.021] |
| S-H | source: NERVE pass − generation, BRACE hierarchy | CA | -0.001 [-0.005, +0.004] | -0.016 [-0.024, -0.007] | -0.023 [-0.039, -0.004] |
| S-H | source: NERVE pass − generation, BRACE hierarchy | CLIP | -0.003 [-0.008, +0.002] | +0.003 [-0.008, +0.016] | -0.003 [-0.026, +0.026] |
| S-W | source: NERVE pass − generation, NERVE walk | CA | +0.001 [-0.005, +0.008] | +0.006 [-0.002, +0.014] | -0.007 [-0.023, +0.013] |
| S-W | source: NERVE pass − generation, NERVE walk | CLIP | +0.004 [-0.000, +0.008] | +0.014 [+0.002, +0.027] | -0.006 [-0.026, +0.021] |
| I-G | interaction: (walk − hierarchy) on CLIP minus on CA, source G | — | — | +0.036 [+0.022, +0.050] | — |
| I-N | interaction: (walk − hierarchy) on CLIP minus on CA, source N | — | — | +0.024 [+0.012, +0.037] | — |

**Reading rule (COCO-Cap, CLIP seed).** The walk's advantage is carried by the rule if R-G and R-N are both above zero: **no**. By the attention source if S-H and S-W are both above zero: **no**. Result: **neither alone**.

Reference (runs/17, CLIP seed): NERVE walk on NERVE pass − BRACE hierarchy on generation = +0.016 [+0.001, +0.033] here (runs/17: +0.033).

## Amendment 1: rank–distribution swap (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | Cell | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images |
|---|---|---|---|---|---|
| V | values: CA rank + CLIP values − CA | seed alone | +0.002 [-0.007, +0.010] | +0.007 [-0.009, +0.023] | -0.001 [-0.027, +0.025] |
| K | ranking: CLIP − CA rank + CLIP values | seed alone | +0.020 [+0.015, +0.027] | +0.060 [+0.050, +0.071] | +0.050 [+0.037, +0.068] |
| V' | values: CLIP rank + CA values − CLIP | seed alone | -0.016 [-0.026, -0.006] | -0.039 [-0.058, -0.022] | -0.032 [-0.062, -0.005] |
| V | values: CA rank + CLIP values − CA | generation SA, BRACE fine-only | +0.024 [+0.013, +0.034] | +0.043 [+0.027, +0.059] | +0.053 [+0.031, +0.072] |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, BRACE fine-only | -0.017 [-0.027, -0.007] | -0.001 [-0.021, +0.019] | -0.020 [-0.048, +0.014] |
| V' | values: CLIP rank + CA values − CLIP | generation SA, BRACE fine-only | -0.011 [-0.020, -0.003] | -0.027 [-0.043, -0.012] | -0.025 [-0.053, -0.001] |
| V | values: CA rank + CLIP values − CA | generation SA, BRACE hierarchy | +0.017 [+0.009, +0.025] | +0.024 [+0.013, +0.036] | +0.026 [+0.011, +0.040] |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, BRACE hierarchy | -0.011 [-0.021, -0.001] | -0.016 [-0.035, +0.004] | -0.021 [-0.048, +0.012] |
| V' | values: CLIP rank + CA values − CLIP | generation SA, BRACE hierarchy | -0.011 [-0.019, -0.003] | -0.017 [-0.031, -0.004] | -0.017 [-0.037, +0.002] |
| V | values: CA rank + CLIP values − CA | generation SA, NERVE walk | +0.024 [+0.012, +0.036] | +0.040 [+0.022, +0.056] | +0.054 [+0.033, +0.073] |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, NERVE walk | -0.014 [-0.024, -0.002] | +0.004 [-0.015, +0.023] | -0.013 [-0.040, +0.020] |
| V' | values: CLIP rank + CA values − CLIP | generation SA, NERVE walk | -0.012 [-0.022, -0.002] | -0.038 [-0.055, -0.022] | -0.025 [-0.049, -0.001] |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, BRACE fine-only | +0.021 [+0.011, +0.030] | +0.043 [+0.025, +0.062] | +0.045 [+0.026, +0.063] |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, BRACE fine-only | -0.013 [-0.022, -0.003] | +0.014 [-0.004, +0.033] | -0.007 [-0.034, +0.026] |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, BRACE fine-only | -0.010 [-0.018, -0.004] | -0.033 [-0.053, -0.015] | -0.027 [-0.055, -0.005] |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, BRACE hierarchy | +0.016 [+0.008, +0.023] | +0.030 [+0.017, +0.045] | +0.035 [+0.017, +0.054] |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, BRACE hierarchy | -0.012 [-0.022, -0.002] | -0.003 [-0.021, +0.017] | -0.011 [-0.035, +0.020] |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, BRACE hierarchy | -0.010 [-0.019, -0.002] | -0.017 [-0.032, -0.002] | -0.024 [-0.049, -0.004] |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, NERVE walk | +0.021 [+0.011, +0.031] | +0.037 [+0.019, +0.055] | +0.048 [+0.028, +0.066] |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, NERVE walk | -0.008 [-0.017, +0.002] | +0.014 [-0.003, +0.034] | -0.006 [-0.030, +0.025] |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, NERVE walk | -0.012 [-0.020, -0.003] | -0.040 [-0.059, -0.023] | -0.027 [-0.054, -0.004] |

**Registered reading (COCO-Cap; cells: generation × hierarchy and NERVE pass × walk).** Calibration has headroom (V above zero in either cell): **yes**. CLIP's spatial ranking is essential (K above zero in both cells): **no**.

**COCO-Cap failure modes, VOC-sel. threshold with DenseCRF (descriptive)**

| Seed | Cell | mIoU | small third | middle | large | images below 0.3 IoU | false-positive area / image | macro precision | macro recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ca | seed alone | 0.584 | 0.408 | 0.618 | 0.682 | 0.151 | 0.0797 | 0.685 | 0.807 |
| clip | seed alone | 0.650 | 0.532 | 0.697 | 0.734 | 0.076 | 0.0567 | 0.760 | 0.837 |
| hA | seed alone | 0.591 | 0.428 | 0.610 | 0.682 | 0.132 | 0.0726 | 0.709 | 0.798 |
| hB | seed alone | 0.611 | 0.454 | 0.680 | 0.711 | 0.123 | 0.0735 | 0.707 | 0.835 |
| ca | generation SA, BRACE hierarchy | 0.696 | 0.584 | 0.764 | 0.814 | 0.112 | 0.0695 | 0.756 | 0.905 |
| clip | generation SA, BRACE hierarchy | 0.704 | 0.650 | 0.761 | 0.818 | 0.074 | 0.0563 | 0.768 | 0.889 |
| hA | generation SA, BRACE hierarchy | 0.720 | 0.629 | 0.768 | 0.813 | 0.091 | 0.0535 | 0.794 | 0.886 |
| hB | generation SA, BRACE hierarchy | 0.687 | 0.613 | 0.757 | 0.813 | 0.089 | 0.0657 | 0.746 | 0.898 |
| ca | generation SA, NERVE walk | 0.663 | 0.551 | 0.734 | 0.792 | 0.124 | 0.0850 | 0.723 | 0.900 |
| clip | generation SA, NERVE walk | 0.706 | 0.653 | 0.762 | 0.811 | 0.079 | 0.0550 | 0.773 | 0.887 |
| hA | generation SA, NERVE walk | 0.703 | 0.616 | 0.750 | 0.798 | 0.083 | 0.0561 | 0.777 | 0.884 |
| hB | generation SA, NERVE walk | 0.668 | 0.592 | 0.744 | 0.797 | 0.102 | 0.0756 | 0.723 | 0.902 |
| ca | NERVE pass SA, BRACE hierarchy | 0.680 | 0.569 | 0.750 | 0.809 | 0.122 | 0.0774 | 0.734 | 0.911 |
| clip | NERVE pass SA, BRACE hierarchy | 0.707 | 0.651 | 0.768 | 0.815 | 0.075 | 0.0530 | 0.780 | 0.881 |
| hA | NERVE pass SA, BRACE hierarchy | 0.711 | 0.622 | 0.760 | 0.811 | 0.085 | 0.0570 | 0.781 | 0.890 |
| hB | NERVE pass SA, BRACE hierarchy | 0.691 | 0.626 | 0.762 | 0.812 | 0.083 | 0.0618 | 0.755 | 0.891 |
| ca | NERVE pass SA, NERVE walk | 0.669 | 0.559 | 0.738 | 0.791 | 0.122 | 0.0824 | 0.728 | 0.900 |
| clip | NERVE pass SA, NERVE walk | 0.720 | 0.684 | 0.778 | 0.810 | 0.066 | 0.0477 | 0.795 | 0.881 |
| hA | NERVE pass SA, NERVE walk | 0.706 | 0.629 | 0.757 | 0.792 | 0.082 | 0.0515 | 0.788 | 0.874 |
| hB | NERVE pass SA, NERVE walk | 0.680 | 0.604 | 0.762 | 0.794 | 0.095 | 0.0699 | 0.737 | 0.900 |

## N's sanity convention for the NERVE cell (CLIP × NERVE pass × walk, raw, threshold 0.5)

- VOC-Sim: 0.8267 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)
- COCO-Cap: 0.7115 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)

Per class, VOC-Sim, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.88 / 0.87 / 0.87 / 0.87 / 0.90 / 0.85 / 0.89 / 0.86, bicycle 0.79 / 0.74 / 0.76 / 0.75 / 0.80 / 0.75 / 0.79 / 0.77, bird 0.92 / 0.91 / 0.92 / 0.93 / 0.92 / 0.90 / 0.91 / 0.92, boat 0.86 / 0.82 / 0.86 / 0.83 / 0.85 / 0.84 / 0.87 / 0.86, bottle 0.88 / 0.90 / 0.90 / 0.87 / 0.83 / 0.82 / 0.81 / 0.79, bus 0.92 / 0.93 / 0.90 / 0.92 / 0.89 / 0.89 / 0.89 / 0.89, car 0.85 / 0.86 / 0.86 / 0.84 / 0.83 / 0.82 / 0.82 / 0.82, cat 0.85 / 0.79 / 0.86 / 0.79 / 0.91 / 0.92 / 0.90 / 0.92, chair 0.79 / 0.79 / 0.76 / 0.77 / 0.77 / 0.84 / 0.77 / 0.84, cow 0.90 / 0.92 / 0.93 / 0.93 / 0.96 / 0.95 / 0.96 / 0.96, dining-table 0.42 / 0.47 / 0.43 / 0.45 / 0.40 / 0.41 / 0.41 / 0.42, dog 0.93 / 0.94 / 0.92 / 0.93 / 0.95 / 0.95 / 0.94 / 0.95, horse 0.90 / 0.90 / 0.90 / 0.91 / 0.92 / 0.94 / 0.92 / 0.93, motorbike 0.87 / 0.85 / 0.86 / 0.86 / 0.86 / 0.85 / 0.86 / 0.86, person 0.81 / 0.78 / 0.80 / 0.75 / 0.78 / 0.78 / 0.77 / 0.76, potted-plant 0.72 / 0.73 / 0.72 / 0.73 / 0.79 / 0.80 / 0.77 / 0.82, sheep 0.75 / 0.81 / 0.82 / 0.90 / 0.90 / 0.92 / 0.93 / 0.94, sofa 0.96 / 0.96 / 0.95 / 0.96 / 0.91 / 0.90 / 0.92 / 0.91, train 0.74 / 0.69 / 0.71 / 0.69 / 0.84 / 0.87 / 0.85 / 0.88, tv 0.85 / 0.84 / 0.85 / 0.83 / 0.71 / 0.67 / 0.68 / 0.67

Per class, COCO-Cap, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.80 / 0.66 / 0.75 / 0.68 / 0.67 / 0.62 / 0.68 / 0.64, bicycle 0.63 / 0.52 / 0.59 / 0.54 / 0.70 / 0.68 / 0.71 / 0.72, bird 0.78 / 0.78 / 0.78 / 0.80 / 0.78 / 0.80 / 0.79 / 0.83, boat 0.59 / 0.55 / 0.60 / 0.59 / 0.64 / 0.66 / 0.66 / 0.69, bottle 0.50 / 0.50 / 0.49 / 0.51 / 0.45 / 0.52 / 0.55 / 0.57, bus 0.92 / 0.94 / 0.92 / 0.94 / 0.87 / 0.86 / 0.85 / 0.84, car 0.76 / 0.66 / 0.73 / 0.68 / 0.78 / 0.75 / 0.75 / 0.76, cat 0.87 / 0.85 / 0.87 / 0.86 / 0.88 / 0.88 / 0.88 / 0.89, chair 0.47 / 0.45 / 0.43 / 0.42 / 0.59 / 0.63 / 0.53 / 0.57, cow 0.87 / 0.84 / 0.88 / 0.88 / 0.89 / 0.89 / 0.91 / 0.91, dining table 0.45 / 0.53 / 0.49 / 0.50 / 0.56 / 0.62 / 0.56 / 0.59, dog 0.75 / 0.76 / 0.70 / 0.71 / 0.79 / 0.81 / 0.74 / 0.78, horse 0.72 / 0.66 / 0.70 / 0.70 / 0.74 / 0.68 / 0.75 / 0.73, motorbike 0.67 / 0.63 / 0.67 / 0.66 / 0.71 / 0.70 / 0.72 / 0.71, person 0.28 / 0.28 / 0.28 / 0.26 / 0.64 / 0.67 / 0.67 / 0.68, potted plant 0.74 / 0.65 / 0.71 / 0.68 / 0.79 / 0.79 / 0.80 / 0.85, sheep 0.87 / 0.89 / 0.88 / 0.90 / 0.89 / 0.90 / 0.90 / 0.92, sofa 0.70 / 0.67 / 0.68 / 0.67 / 0.72 / 0.71 / 0.73 / 0.74, train 0.75 / 0.68 / 0.71 / 0.67 / 0.73 / 0.71 / 0.72 / 0.72, tv 0.79 / 0.75 / 0.75 / 0.74 / 0.26 / 0.27 / 0.24 / 0.25

