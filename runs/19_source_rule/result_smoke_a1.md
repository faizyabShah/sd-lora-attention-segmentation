# Seed × attention source × propagation rule (`PREREG.md`)

`scripts/jc_srcrule.py`. One live generation per image. Sources: **G** = the frozen runtime's generation self-attention (calls 10–20); **N** = NERVE's source, one pass of the generated image noised to t = 101 with an empty prompt. Rules: BRACE fine-only / capped hierarchy (frozen code, BRACE's modules) and NERVE's walk (released code, ported; NERVE's layers; on G the fused matrix is averaged over the 11 calls). Seeds: **CA** (live) and **CLIP** (NERVE's, from runs/17, stored image). Class-macro mIoU. Primary: threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap), with DenseCRF. Δ with image-bootstrap 95% interval (2000).

## VOC-Sim (4 images)

Regeneration bit-identical to the stored image: 2/4. CA × generation × hierarchy map vs stored capture: median max |Δ| 3.4e-04. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 1.1e-02 on all images, 1.1e-01 on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.751 | 0.604 | -0.147 | 0.646 | 0.708 |
| generation SA, BRACE fine-only | 0.887 | 0.799 | -0.088 | 0.799 | 0.872 |
| generation SA, BRACE hierarchy | 0.849 | 0.832 | -0.018 | 0.844 | 0.812 |
| generation SA, NERVE walk | 0.880 | 0.804 | -0.076 | 0.837 | 0.870 |
| NERVE pass SA, BRACE fine-only | 0.883 | 0.807 | -0.076 | 0.812 | 0.885 |
| NERVE pass SA, BRACE hierarchy | 0.874 | 0.877 | +0.003 | 0.863 | 0.837 |
| NERVE pass SA, NERVE walk | 0.865 | 0.841 | -0.024 | 0.819 | 0.888 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.639 | 0.687 | +0.048 | 0.686 | 0.604 |
| generation SA, BRACE fine-only | 0.846 | 0.839 | -0.008 | 0.876 | 0.814 |
| generation SA, BRACE hierarchy | 0.917 | 0.900 | -0.017 | 0.935 | 0.883 |
| generation SA, NERVE walk | 0.844 | 0.845 | +0.001 | 0.893 | 0.809 |
| NERVE pass SA, BRACE fine-only | 0.868 | 0.900 | +0.032 | 0.888 | 0.891 |
| NERVE pass SA, BRACE hierarchy | 0.925 | 0.915 | -0.009 | 0.941 | 0.907 |
| NERVE pass SA, NERVE walk | 0.829 | 0.877 | +0.048 | 0.879 | 0.843 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.751 | 0.604 | 0.646 | 0.708 |
| generation SA, BRACE fine-only | 0.856 | 0.749 | 0.761 | 0.770 |
| generation SA, BRACE hierarchy | 0.823 | 0.792 | 0.817 | 0.768 |
| generation SA, NERVE walk | 0.863 | 0.752 | 0.795 | 0.775 |
| NERVE pass SA, BRACE fine-only | 0.850 | 0.775 | 0.763 | 0.849 |
| NERVE pass SA, BRACE hierarchy | 0.847 | 0.841 | 0.837 | 0.799 |
| NERVE pass SA, NERVE walk | 0.850 | 0.794 | 0.787 | 0.830 |

## COCO-Cap (4 images)

Regeneration bit-identical to the stored image: 0/4. CA × generation × hierarchy map vs stored capture: median max |Δ| 1.1e-02. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 3.9e-02 on all images, nan on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.451 | 0.446 | -0.005 | 0.505 | 0.352 |
| generation SA, BRACE fine-only | 0.496 | 0.770 | +0.273 | 0.800 | 0.631 |
| generation SA, BRACE hierarchy | 0.733 | 0.768 | +0.035 | 0.803 | 0.697 |
| generation SA, NERVE walk | 0.648 | 0.783 | +0.136 | 0.787 | 0.566 |
| NERVE pass SA, BRACE fine-only | 0.608 | 0.768 | +0.160 | 0.780 | 0.634 |
| NERVE pass SA, BRACE hierarchy | 0.731 | 0.810 | +0.079 | 0.806 | 0.712 |
| NERVE pass SA, NERVE walk | 0.603 | 0.796 | +0.193 | 0.788 | 0.514 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA | CA rank + CLIP values (hA) | CLIP rank + CA values (hB) |
|---|---:|---:|---|---:|---:|
| seed alone | 0.513 | 0.451 | -0.062 | 0.432 | 0.445 |
| generation SA, BRACE fine-only | 0.741 | 0.772 | +0.031 | 0.782 | 0.695 |
| generation SA, BRACE hierarchy | 0.737 | 0.810 | +0.072 | 0.809 | 0.748 |
| generation SA, NERVE walk | 0.742 | 0.756 | +0.013 | 0.707 | 0.667 |
| NERVE pass SA, BRACE fine-only | 0.790 | 0.717 | -0.073 | 0.710 | 0.653 |
| NERVE pass SA, BRACE hierarchy | 0.731 | 0.814 | +0.083 | 0.798 | 0.689 |
| NERVE pass SA, NERVE walk | 0.769 | 0.721 | -0.048 | 0.697 | 0.649 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.451 | 0.446 | 0.505 | 0.352 |
| generation SA, BRACE fine-only | 0.568 | 0.664 | 0.712 | 0.564 |
| generation SA, BRACE hierarchy | 0.650 | 0.704 | 0.786 | 0.598 |
| generation SA, NERVE walk | 0.580 | 0.689 | 0.760 | 0.484 |
| NERVE pass SA, BRACE fine-only | 0.503 | 0.707 | 0.730 | 0.570 |
| NERVE pass SA, BRACE hierarchy | 0.655 | 0.777 | 0.778 | 0.687 |
| NERVE pass SA, NERVE walk | 0.546 | 0.792 | 0.751 | 0.487 |

## Thresholds selected on full VOC-Sim (with DenseCRF)

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.400 | 0.525 | 0.700 | 0.375 |
| generation SA, BRACE fine-only | 0.300 | 0.350 | 0.375 | 0.300 |
| generation SA, BRACE hierarchy | 0.500 | 0.450 | 0.500 | 0.425 |
| generation SA, NERVE walk | 0.350 | 0.300 | 0.325 | 0.300 |
| NERVE pass SA, BRACE fine-only | 0.350 | 0.450 | 0.425 | 0.400 |
| NERVE pass SA, BRACE hierarchy | 0.500 | 0.500 | 0.525 | 0.450 |
| NERVE pass SA, NERVE walk | 0.300 | 0.325 | 0.325 | 0.300 |

## Registered contrasts (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | Seed | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images |
|---|---|---|---|---|---|
| R-G | rule: NERVE walk − BRACE hierarchy, generation SA | CA | +0.031 | -0.085 | +nan |
| R-G | rule: NERVE walk − BRACE hierarchy, generation SA | CLIP | -0.028 | +0.015 | +nan |
| R-N | rule: NERVE walk − BRACE hierarchy, NERVE pass SA | CA | -0.009 | -0.128 | +nan |
| R-N | rule: NERVE walk − BRACE hierarchy, NERVE pass SA | CLIP | -0.036 | -0.014 | +nan |
| S-H | source: NERVE pass − generation, BRACE hierarchy | CA | +0.025 | -0.002 | +nan |
| S-H | source: NERVE pass − generation, BRACE hierarchy | CLIP | +0.045 | +0.042 | +nan |
| S-W | source: NERVE pass − generation, NERVE walk | CA | -0.016 | -0.045 | +nan |
| S-W | source: NERVE pass − generation, NERVE walk | CLIP | +0.037 | +0.012 | +nan |
| I-G | interaction: (walk − hierarchy) on CLIP minus on CA, source G | — | — | +0.100 [-0.036, +0.237] | — |
| I-N | interaction: (walk − hierarchy) on CLIP minus on CA, source N | — | — | +0.114 [-0.014, +0.241] | — |

**Reading rule (COCO-Cap, CLIP seed).** The walk's advantage is carried by the rule if R-G and R-N are both above zero: **no**. By the attention source if S-H and S-W are both above zero: **no**. Result: **neither alone**.

Reference (runs/17, CLIP seed): NERVE walk on NERVE pass − BRACE hierarchy on generation = +0.028 here (runs/17: +0.033).

## Amendment 1: rank–distribution swap (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | Cell | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images |
|---|---|---|---|---|---|
| V | values: CA rank + CLIP values − CA | seed alone | -0.105 | +0.054 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | seed alone | -0.042 | -0.059 | +nan |
| V' | values: CLIP rank + CA values − CLIP | seed alone | +0.104 | -0.094 | +nan |
| V | values: CA rank + CLIP values − CA | generation SA, BRACE fine-only | -0.087 | +0.303 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, BRACE fine-only | -0.000 | -0.030 | +nan |
| V' | values: CLIP rank + CA values − CLIP | generation SA, BRACE fine-only | +0.073 | -0.139 | +nan |
| V | values: CA rank + CLIP values − CA | generation SA, BRACE hierarchy | -0.005 | +0.070 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, BRACE hierarchy | -0.012 | -0.035 | +nan |
| V' | values: CLIP rank + CA values − CLIP | generation SA, BRACE hierarchy | -0.020 | -0.071 | +nan |
| V | values: CA rank + CLIP values − CA | generation SA, NERVE walk | -0.044 | +0.140 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | generation SA, NERVE walk | -0.033 | -0.004 | +nan |
| V' | values: CLIP rank + CA values − CLIP | generation SA, NERVE walk | +0.066 | -0.217 | +nan |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, BRACE fine-only | -0.071 | +0.172 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, BRACE fine-only | -0.005 | -0.012 | +nan |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, BRACE fine-only | +0.078 | -0.134 | +nan |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, BRACE hierarchy | -0.011 | +0.075 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, BRACE hierarchy | +0.014 | +0.005 | +nan |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, BRACE hierarchy | -0.040 | -0.098 | +nan |
| V | values: CA rank + CLIP values − CA | NERVE pass SA, NERVE walk | -0.045 | +0.185 | +nan |
| K | ranking: CLIP − CA rank + CLIP values | NERVE pass SA, NERVE walk | +0.021 | +0.007 | +nan |
| V' | values: CLIP rank + CA values − CLIP | NERVE pass SA, NERVE walk | +0.047 | -0.282 | +nan |

**Registered reading (COCO-Cap; cells: generation × hierarchy and NERVE pass × walk).** Calibration has headroom (V above zero in either cell): **no**. CLIP's spatial ranking is essential (K above zero in both cells): **no**.

**COCO-Cap failure modes, VOC-sel. threshold with DenseCRF (descriptive)**

| Seed | Cell | mIoU | small third | middle | large | images below 0.3 IoU | false-positive area / image | macro precision | macro recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ca | seed alone | 0.451 | 0.330 | 0.642 | 0.417 | 0.250 | 0.0791 | 0.480 | 0.880 |
| clip | seed alone | 0.446 | 0.433 | 0.545 | 0.403 | 0.250 | 0.0991 | 0.457 | 0.908 |
| hA | seed alone | 0.505 | 0.519 | 0.617 | 0.443 | 0.000 | 0.0475 | 0.563 | 0.814 |
| hB | seed alone | 0.352 | 0.316 | 0.589 | 0.251 | 0.250 | 0.1147 | 0.376 | 0.841 |
| ca | generation SA, BRACE hierarchy | 0.733 | 0.785 | 0.644 | 0.751 | 0.000 | 0.0205 | 0.820 | 0.889 |
| clip | generation SA, BRACE hierarchy | 0.768 | 0.773 | 0.573 | 0.863 | 0.000 | 0.0154 | 0.848 | 0.892 |
| hA | generation SA, BRACE hierarchy | 0.803 | 0.758 | 0.620 | 0.916 | 0.000 | 0.0098 | 0.901 | 0.878 |
| hB | generation SA, BRACE hierarchy | 0.697 | 0.786 | 0.501 | 0.751 | 0.000 | 0.0302 | 0.756 | 0.917 |
| ca | generation SA, NERVE walk | 0.648 | 0.800 | 0.665 | 0.563 | 0.250 | 0.0693 | 0.742 | 0.880 |
| clip | generation SA, NERVE walk | 0.783 | 0.754 | 0.556 | 0.912 | 0.000 | 0.0149 | 0.863 | 0.895 |
| hA | generation SA, NERVE walk | 0.787 | 0.739 | 0.609 | 0.901 | 0.000 | 0.0101 | 0.895 | 0.867 |
| hB | generation SA, NERVE walk | 0.566 | 0.768 | 0.470 | 0.514 | 0.250 | 0.1170 | 0.637 | 0.924 |
| ca | NERVE pass SA, BRACE hierarchy | 0.731 | 0.788 | 0.653 | 0.741 | 0.000 | 0.0210 | 0.823 | 0.887 |
| clip | NERVE pass SA, BRACE hierarchy | 0.810 | 0.755 | 0.657 | 0.915 | 0.000 | 0.0090 | 0.912 | 0.879 |
| hA | NERVE pass SA, BRACE hierarchy | 0.806 | 0.753 | 0.665 | 0.903 | 0.000 | 0.0088 | 0.919 | 0.870 |
| hB | NERVE pass SA, BRACE hierarchy | 0.712 | 0.774 | 0.592 | 0.742 | 0.000 | 0.0169 | 0.831 | 0.849 |
| ca | NERVE pass SA, NERVE walk | 0.603 | 0.796 | 0.593 | 0.511 | 0.250 | 0.1299 | 0.682 | 0.893 |
| clip | NERVE pass SA, NERVE walk | 0.796 | 0.769 | 0.603 | 0.906 | 0.000 | 0.0123 | 0.882 | 0.889 |
| hA | NERVE pass SA, NERVE walk | 0.788 | 0.768 | 0.586 | 0.900 | 0.000 | 0.0120 | 0.884 | 0.877 |
| hB | NERVE pass SA, NERVE walk | 0.514 | 0.779 | 0.558 | 0.359 | 0.250 | 0.1223 | 0.662 | 0.803 |

## N's sanity convention for the NERVE cell (CLIP × NERVE pass × walk, raw, threshold 0.5)

- VOC-Sim: 0.8728 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)
- COCO-Cap: 0.7923 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)

Per class, VOC-Sim, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.95 / 0.94 / 0.96 / 0.97 / 0.95 / 0.94 / 0.96 / 0.96, bottle 0.90 / 0.92 / 0.91 / 0.91 / 0.88 / 0.88 / 0.89 / 0.88, cow 0.98 / 0.97 / 0.98 / 0.97 / 0.98 / 0.98 / 0.98 / 0.98, person 0.57 / 0.69 / 0.64 / 0.61 / 0.52 / 0.42 / 0.67 / 0.55

Per class, COCO-Cap, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.88 / 0.88 / 0.86 / 0.87 / 0.87 / 0.86 / 0.87 / 0.85, bottle 0.64 / 0.66 / 0.65 / 0.59 / 0.57 / 0.56 / 0.66 / 0.60, cow 0.79 / 0.80 / 0.79 / 0.80 / 0.77 / 0.75 / 0.76 / 0.77, person 0.62 / 0.24 / 0.63 / 0.15 / 0.85 / 0.96 / 0.96 / 0.96

