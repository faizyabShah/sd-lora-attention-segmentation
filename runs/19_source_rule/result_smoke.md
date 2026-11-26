# Seed × attention source × propagation rule (`PREREG.md`)

`scripts/jc_srcrule.py`. One live generation per image. Sources: **G** = the frozen runtime's generation self-attention (calls 10–20); **N** = NERVE's source, one pass of the generated image noised to t = 101 with an empty prompt. Rules: BRACE fine-only / capped hierarchy (frozen code, BRACE's modules) and NERVE's walk (released code, ported; NERVE's layers; on G the fused matrix is averaged over the 11 calls). Seeds: **CA** (live) and **CLIP** (NERVE's, from runs/17, stored image). Class-macro mIoU. Primary: threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap), with DenseCRF. Δ with image-bootstrap 95% interval (2000).

## VOC-Sim (4 images)

Regeneration bit-identical to the stored image: 2/4. CA × generation × hierarchy map vs stored capture: median max |Δ| 3.4e-04. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 1.1e-02 on all images, 1.1e-01 on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA |
|---|---:|---:|---|
| seed alone | 0.751 | 0.604 | -0.147 |
| generation SA, BRACE fine-only | 0.887 | 0.799 | -0.088 |
| generation SA, BRACE hierarchy | 0.849 | 0.832 | -0.018 |
| generation SA, NERVE walk | 0.880 | 0.804 | -0.076 |
| NERVE pass SA, BRACE fine-only | 0.883 | 0.807 | -0.076 |
| NERVE pass SA, BRACE hierarchy | 0.874 | 0.877 | +0.003 |
| NERVE pass SA, NERVE walk | 0.865 | 0.841 | -0.024 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA |
|---|---:|---:|---|
| seed alone | 0.639 | 0.687 | +0.048 |
| generation SA, BRACE fine-only | 0.846 | 0.839 | -0.008 |
| generation SA, BRACE hierarchy | 0.917 | 0.900 | -0.017 |
| generation SA, NERVE walk | 0.844 | 0.845 | +0.001 |
| NERVE pass SA, BRACE fine-only | 0.868 | 0.900 | +0.032 |
| NERVE pass SA, BRACE hierarchy | 0.925 | 0.915 | -0.009 |
| NERVE pass SA, NERVE walk | 0.829 | 0.877 | +0.048 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA seed | CLIP seed |
|---|---:|---:|
| seed alone | 0.751 | 0.604 |
| generation SA, BRACE fine-only | 0.856 | 0.749 |
| generation SA, BRACE hierarchy | 0.823 | 0.792 |
| generation SA, NERVE walk | 0.863 | 0.752 |
| NERVE pass SA, BRACE fine-only | 0.850 | 0.775 |
| NERVE pass SA, BRACE hierarchy | 0.847 | 0.841 |
| NERVE pass SA, NERVE walk | 0.850 | 0.794 |

## COCO-Cap (4 images)

Regeneration bit-identical to the stored image: 0/4. CA × generation × hierarchy map vs stored capture: median max |Δ| 1.1e-02. CLIP × NERVE pass × walk vs runs/17's NERVE walk (min–max maps): median max |Δ| 3.9e-02 on all images, nan on bit-identical images.

**VOC-sel. threshold, with DenseCRF (primary)**

| Cell | CA seed | CLIP seed | CLIP − CA |
|---|---:|---:|---|
| seed alone | 0.451 | 0.446 | -0.005 |
| generation SA, BRACE fine-only | 0.496 | 0.770 | +0.273 |
| generation SA, BRACE hierarchy | 0.733 | 0.768 | +0.035 |
| generation SA, NERVE walk | 0.648 | 0.783 | +0.136 |
| NERVE pass SA, BRACE fine-only | 0.608 | 0.768 | +0.160 |
| NERVE pass SA, BRACE hierarchy | 0.731 | 0.810 | +0.079 |
| NERVE pass SA, NERVE walk | 0.603 | 0.796 | +0.193 |

**fixed threshold 0.55, with DenseCRF**

| Cell | CA seed | CLIP seed | CLIP − CA |
|---|---:|---:|---|
| seed alone | 0.513 | 0.451 | -0.062 |
| generation SA, BRACE fine-only | 0.741 | 0.772 | +0.031 |
| generation SA, BRACE hierarchy | 0.737 | 0.810 | +0.072 |
| generation SA, NERVE walk | 0.742 | 0.756 | +0.013 |
| NERVE pass SA, BRACE fine-only | 0.790 | 0.717 | -0.073 |
| NERVE pass SA, BRACE hierarchy | 0.731 | 0.814 | +0.083 |
| NERVE pass SA, NERVE walk | 0.769 | 0.721 | -0.048 |

**Raw (no DenseCRF), VOC-sel. threshold**

| Cell | CA seed | CLIP seed |
|---|---:|---:|
| seed alone | 0.451 | 0.446 |
| generation SA, BRACE fine-only | 0.568 | 0.664 |
| generation SA, BRACE hierarchy | 0.650 | 0.704 |
| generation SA, NERVE walk | 0.580 | 0.689 |
| NERVE pass SA, BRACE fine-only | 0.503 | 0.707 |
| NERVE pass SA, BRACE hierarchy | 0.655 | 0.777 |
| NERVE pass SA, NERVE walk | 0.546 | 0.792 |

## Thresholds selected on full VOC-Sim (with DenseCRF)

| Cell | CA | CLIP |
|---|---:|---:|
| seed alone | 0.400 | 0.525 |
| generation SA, BRACE fine-only | 0.300 | 0.350 |
| generation SA, BRACE hierarchy | 0.500 | 0.450 |
| generation SA, NERVE walk | 0.350 | 0.300 |
| NERVE pass SA, BRACE fine-only | 0.350 | 0.450 |
| NERVE pass SA, BRACE hierarchy | 0.500 | 0.500 |
| NERVE pass SA, NERVE walk | 0.300 | 0.325 |

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

## N's sanity convention for the NERVE cell (CLIP × NERVE pass × walk, raw, threshold 0.5)

- VOC-Sim: 0.8728 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)
- COCO-Cap: 0.7923 (runs/17 used bilinear-then-min–max; here min–max-then-bilinear, the project's scoring)

Per class, VOC-Sim, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.95 / 0.94 / 0.96 / 0.97 / 0.95 / 0.94 / 0.96 / 0.96, bottle 0.90 / 0.92 / 0.91 / 0.91 / 0.88 / 0.88 / 0.89 / 0.88, cow 0.98 / 0.97 / 0.98 / 0.97 / 0.98 / 0.98 / 0.98 / 0.98, person 0.57 / 0.69 / 0.64 / 0.61 / 0.52 / 0.42 / 0.67 / 0.55

Per class, COCO-Cap, VOC-sel. + DenseCRF (CA G|V3 / CA G|W / CA N|V3 / CA N|W / CLIP G|V3 / CLIP G|W / CLIP N|V3 / CLIP N|W): aeroplane 0.88 / 0.88 / 0.86 / 0.87 / 0.87 / 0.86 / 0.87 / 0.85, bottle 0.64 / 0.66 / 0.65 / 0.59 / 0.57 / 0.56 / 0.66 / 0.60, cow 0.79 / 0.80 / 0.79 / 0.80 / 0.77 / 0.75 / 0.76 / 0.77, person 0.62 / 0.24 / 0.63 / 0.15 / 0.85 / 0.96 / 0.96 / 0.96

