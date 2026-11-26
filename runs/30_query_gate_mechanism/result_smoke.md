# Query-profile gate: mechanism and class-adaptive follow-up (`PREREG.md`)

`scripts/jc_qgate.py`. **Exploratory.** One live SD1.5 generation per image on renyi (A100); all arms paired. Profiles from run 29's statistics, cross-fitted. VOC-Sim primary = OOF thresholds; COCO-Cap (**screened**) primary = VOC-Sim-selected thresholds. Paired image bootstrap (200).

## Checks

- voc: 10 images; recomputed vs runtime seed max |Δ| 2.0e-04.
- coco: 10 images; recomputed vs runtime seed max |Δ| 2.5e-04.

## VOC-Sim: mIoU, DenseCRF

| Seed | Seed (OOF) | Fine-only (OOF) | Hierarchy (OOF) | Hierarchy (0.55) | Hierarchy raw (OOF) |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.690 | 0.853 | 0.867 | 0.879 | 0.843 |
| NC absent-class gate | 0.731 | 0.869 | 0.890 | 0.883 | 0.848 |
| NH NC half strength | 0.710 | 0.872 | 0.893 | 0.885 | 0.854 |
| NO all-pixel contrast | 0.752 | 0.852 | 0.876 | 0.863 | 0.817 |
| PD object direction, all-pixel anchor | 0.704 | 0.842 | 0.883 | 0.875 | 0.847 |
| PI object direction, image's own anchor | 0.632 | 0.821 | 0.804 | 0.824 | 0.780 |
| CAg class-adaptive strength | — | — | 0.867 | — | — |
| CAa class-adaptive any arm | — | — | 0.867 | — | — |

## COCO-Cap (screened): mIoU, DenseCRF

| Seed | Seed (VOC-Sim-selected) | Fine-only (VOC-Sim-selected) | Hierarchy (VOC-Sim-selected) | Hierarchy (0.55) | Hierarchy raw (VOC-Sim-selected) |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.632 | 0.651 | 0.739 | 0.753 | 0.704 |
| NC absent-class gate | 0.716 | 0.746 | 0.745 | 0.771 | 0.727 |
| NH NC half strength | 0.688 | 0.710 | 0.757 | 0.765 | 0.729 |
| NO all-pixel contrast | 0.670 | 0.695 | 0.767 | 0.779 | 0.734 |
| PD object direction, all-pixel anchor | 0.663 | 0.703 | 0.759 | 0.759 | 0.710 |
| PI object direction, image's own anchor | 0.673 | 0.695 | 0.754 | 0.754 | 0.703 |
| CAg class-adaptive strength | — | — | 0.758 | — | — |
| CAa class-adaptive any arm | — | — | 0.763 | — | — |

## Comparisons (DenseCRF, primary thresholds; exploratory)

| Comparison | Map | VOC-Sim | COCO-Cap (screened) |
|---|---|---|---|
| NC − A | hierarchy | +0.023 [-0.013, +0.055] | +0.006 [-0.045, +0.062] |
| NH − A | hierarchy | +0.026 [-0.008, +0.055] | +0.018 [-0.015, +0.052] |
| NO − A | hierarchy | +0.009 [-0.010, +0.028] | +0.029 [-0.006, +0.056] |
| PD − A | hierarchy | +0.016 [-0.009, +0.040] | +0.020 [-0.046, +0.094] |
| PI − A | hierarchy | -0.063 [-0.117, -0.008] | +0.015 [-0.051, +0.087] |
| NO − NC | hierarchy | -0.015 [-0.030, +0.003] | +0.023 [-0.026, +0.064] |
| PD − NC | hierarchy | -0.007 [-0.025, +0.009] | +0.014 [-0.001, +0.031] |
| PI − NC | hierarchy | -0.087 [-0.150, -0.017] | +0.009 [-0.006, +0.025] |
| NH − NC | hierarchy | +0.002 [-0.002, +0.006] | +0.012 [-0.011, +0.033] |
| CAg − A | hierarchy | +0.000 [+0.000, +0.000] | +0.019 [-0.000, +0.039] |
| CAa − A | hierarchy | +0.000 [+0.000, +0.000] | +0.024 [+0.003, +0.045] |
| NC − A | seed | +0.040 [-0.027, +0.099] | +0.084 [+0.032, +0.146] |
| NH − A | seed | +0.020 [-0.033, +0.076] | +0.056 [+0.019, +0.095] |
| NO − A | seed | +0.062 [+0.002, +0.114] | +0.038 [+0.014, +0.065] |
| PD − A | seed | +0.014 [-0.058, +0.094] | +0.031 [-0.009, +0.073] |
| PI − A | seed | -0.058 [-0.131, +0.018] | +0.041 [-0.009, +0.107] |
| NC − A | fine-only | +0.017 [-0.003, +0.034] | +0.095 [+0.038, +0.156] |
| NH − A | fine-only | +0.020 [+0.003, +0.032] | +0.060 [+0.019, +0.098] |
| NO − A | fine-only | -0.001 [-0.007, +0.006] | +0.045 [-0.013, +0.098] |
| PD − A | fine-only | -0.010 [-0.025, +0.006] | +0.052 [+0.008, +0.096] |
| PI − A | fine-only | -0.032 [-0.060, -0.007] | +0.044 [-0.006, +0.091] |

Class-adaptive choices (from VOC-Sim, applied to COCO-Cap): **CAg**: bird NH, boat NH, bus NC, cat NC, motorbike NH, potted plant NC (rest A); **CAa**: bird PD, boat NH, bus PD, cat NC, motorbike NH, potted plant NC, sofa NO (rest A)


## Diagnostics, VOC-Sim

D1 mean gate by region (image mean, then mean over images):

| Gate | fg | near bg | far bg | fg − far |
|---|---:|---:|---:|---:|
| NC | 0.779 | 0.588 | 0.446 | +0.332 |
| NO | 0.648 | 0.362 | 0.203 | +0.446 |
| PD | 0.688 | 0.568 | 0.421 | +0.267 |
| PI | 0.556 | 0.395 | 0.292 | +0.264 |

D2 share of seed mass by region:

| Seed | fg | near bg | far bg |
|---|---:|---:|---:|
| A | 0.668 | 0.172 | 0.160 |
| NC | 0.734 | 0.164 | 0.101 |
| NH | 0.712 | 0.169 | 0.119 |
| NO | 0.747 | 0.161 | 0.092 |
| PD | 0.699 | 0.176 | 0.126 |
| PI | 0.724 | 0.167 | 0.109 |

D1 per class, NC gate fg / near / far (and NO fg / far):

| Class | NC fg | NC near | NC far | NO fg | NO far | PI fg | PI far |
|---|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.82 | 0.67 | 0.50 | 0.76 | 0.13 | 0.51 | 0.36 |
| bird | 0.76 | 0.66 | 0.53 | 0.67 | 0.24 | 0.67 | 0.41 |
| boat | 0.67 | 0.56 | 0.53 | 0.64 | 0.24 | 0.55 | 0.38 |
| bus | 0.81 | 0.44 | 0.27 | 0.73 | 0.16 | 0.56 | 0.27 |
| cat | 0.78 | 0.59 | 0.48 | 0.60 | 0.24 | 0.55 | 0.20 |
| cow | 0.74 | 0.56 | 0.45 | 0.62 | 0.21 | 0.47 | 0.25 |
| dog | 0.77 | 0.48 | 0.35 | 0.67 | 0.13 | 0.66 | 0.28 |
| motorbike | 0.80 | 0.52 | 0.44 | 0.72 | 0.27 | 0.58 | 0.19 |
| potted plant | 0.76 | 0.65 | nan | 0.56 | nan | 0.48 | nan |
| sofa | 0.86 | 0.75 | nan | 0.51 | nan | 0.54 | nan |

D3 per-image IoU gain NC − A (hierarchy + CRF, OOF); mean +0.0234:

| Feature | Spearman ρ | gain, low tertile | mid | high |
|---|---:|---:|---:|---:|
| object area | +0.370 | +0.0127 (n=4) | -0.0240 (n=3) | +0.0849 (n=3) |
| far-bg seed mass of A | -0.345 | +0.0652 (n=4) | -0.0264 (n=3) | +0.0173 (n=3) |

## Diagnostics, COCO-Cap

D1 mean gate by region (image mean, then mean over images):

| Gate | fg | near bg | far bg | fg − far |
|---|---:|---:|---:|---:|
| NC | 0.837 | 0.611 | 0.495 | +0.342 |
| NO | 0.660 | 0.357 | 0.183 | +0.477 |
| PD | 0.682 | 0.539 | 0.399 | +0.283 |
| PI | 0.647 | 0.467 | 0.316 | +0.331 |

D2 share of seed mass by region:

| Seed | fg | near bg | far bg |
|---|---:|---:|---:|
| A | 0.509 | 0.285 | 0.206 |
| NC | 0.578 | 0.277 | 0.145 |
| NH | 0.557 | 0.283 | 0.159 |
| NO | 0.605 | 0.282 | 0.113 |
| PD | 0.519 | 0.290 | 0.191 |
| PI | 0.542 | 0.287 | 0.171 |

D1 per class, NC gate fg / near / far (and NO fg / far):

| Class | NC fg | NC near | NC far | NO fg | NO far | PI fg | PI far |
|---|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.76 | 0.73 | 0.58 | 0.69 | 0.15 | 0.57 | 0.33 |
| bicycle | 0.80 | 0.48 | 0.34 | 0.69 | 0.14 | 0.66 | 0.33 |
| bird | 0.83 | 0.61 | 0.47 | 0.64 | 0.15 | 0.66 | 0.21 |
| bus | 0.80 | 0.48 | 0.32 | 0.75 | 0.15 | 0.63 | 0.29 |
| cat | 0.85 | 0.59 | 0.52 | 0.71 | 0.21 | 0.66 | 0.36 |
| cow | 0.84 | 0.43 | 0.31 | 0.66 | 0.10 | 0.80 | 0.15 |
| dog | 0.91 | 0.67 | 0.59 | 0.67 | 0.23 | 0.65 | 0.31 |
| motorbike | 0.83 | 0.51 | 0.33 | 0.73 | 0.12 | 0.67 | 0.29 |
| person | 0.93 | 0.90 | 0.84 | 0.44 | 0.17 | 0.44 | 0.62 |
| sofa | 0.82 | 0.73 | 0.65 | 0.62 | 0.40 | 0.72 | 0.26 |

D3 per-image IoU gain NC − A (hierarchy + CRF, VOC-Sim-selected); mean +0.0060:

| Feature | Spearman ρ | gain, low tertile | mid | high |
|---|---:|---:|---:|---:|
| object area | -0.345 | +0.0476 (n=4) | +0.0004 (n=3) | -0.0437 (n=3) |
| far-bg seed mass of A | +0.612 | +0.0009 (n=4) | -0.0914 (n=3) | +0.1103 (n=3) |
| other classes in caption (0 / 1 / ≥2) | +0.527 | -0.0368 (n=7) | +0.1060 (n=3) | — |

Per class, voc, hierarchy + DenseCRF, OOF (A / NC / NH / NO / PD / PI / CAg / CAa): aeroplane 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95, bird 0.74 / 0.74 / 0.74 / 0.73 / 0.74 / 0.74 / 0.74 / 0.74, boat 0.81 / 0.88 / 0.89 / 0.83 / 0.85 / 0.86 / 0.81 / 0.81, bus 0.94 / 0.95 / 0.95 / 0.94 / 0.95 / 0.93 / 0.94 / 0.94, cat 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98, cow 0.94 / 0.86 / 0.88 / 0.89 / 0.94 / 0.48 / 0.94 / 0.94, dog 0.91 / 0.90 / 0.90 / 0.90 / 0.90 / 0.90 / 0.91 / 0.91, motorbike 0.86 / 0.87 / 0.88 / 0.86 / 0.82 / 0.79 / 0.86 / 0.86, potted plant 0.53 / 0.79 / 0.77 / 0.68 / 0.72 / 0.43 / 0.53 / 0.53, sofa 1.00 / 1.00 / 0.99 / 1.00 / 0.99 / 0.98 / 1.00 / 1.00


Per class, coco, hierarchy + DenseCRF, VOC-Sim-selected (A / NC / NH / NO / PD / PI / CAg / CAa): aeroplane 0.87 / 0.87 / 0.87 / 0.86 / 0.87 / 0.87 / 0.87 / 0.87, bicycle 0.58 / 0.59 / 0.59 / 0.58 / 0.62 / 0.59 / 0.58 / 0.58, bird 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95, bus 0.89 / 0.89 / 0.89 / 0.89 / 0.89 / 0.89 / 0.89 / 0.89, cat 0.79 / 0.81 / 0.82 / 0.76 / 0.83 / 0.82 / 0.81 / 0.81, cow 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95, dog 0.93 / 0.94 / 0.93 / 0.94 / 0.93 / 0.94 / 0.93 / 0.93, motorbike 0.39 / 0.57 / 0.56 / 0.57 / 0.57 / 0.57 / 0.56 / 0.56, person 0.62 / 0.34 / 0.51 / 0.72 / 0.30 / 0.30 / 0.62 / 0.62, sofa 0.41 / 0.55 / 0.49 / 0.46 / 0.66 / 0.66 / 0.41 / 0.46

