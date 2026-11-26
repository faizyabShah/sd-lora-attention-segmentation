# Query-profile gate: mechanism and class-adaptive follow-up (`PREREG.md`)

`scripts/jc_qgate.py`. **Exploratory.** One live SD1.5 generation per image on renyi (A100); all arms paired. Profiles from run 29's statistics, cross-fitted. VOC-Sim primary = OOF thresholds; COCO-Cap (**screened**) primary = VOC-Sim-selected thresholds. Paired image bootstrap (2000).

## Checks

- voc: 490 images; recomputed vs runtime seed max |Δ| 5.3e-04.
- coco: 707 images; recomputed vs runtime seed max |Δ| 6.2e-04.

## VOC-Sim: mIoU, DenseCRF

| Seed | Seed (OOF) | Fine-only (OOF) | Hierarchy (OOF) | Hierarchy (0.55) | Hierarchy raw (OOF) |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.736 | 0.825 | 0.832 | 0.833 | 0.818 |
| NC absent-class gate | 0.753 | 0.831 | 0.832 | 0.831 | 0.816 |
| NH NC half strength | 0.755 | 0.831 | 0.832 | 0.834 | 0.820 |
| NO all-pixel contrast | 0.757 | 0.831 | 0.831 | 0.830 | 0.817 |
| PD object direction, all-pixel anchor | 0.739 | 0.826 | 0.831 | 0.832 | 0.818 |
| PI object direction, image's own anchor | 0.679 | 0.802 | 0.807 | 0.803 | 0.792 |
| CAg class-adaptive strength | — | — | 0.836 | — | — |
| CAa class-adaptive any arm | — | — | 0.839 | — | — |

## COCO-Cap (screened): mIoU, DenseCRF

| Seed | Seed (VOC-Sim-selected) | Fine-only (VOC-Sim-selected) | Hierarchy (VOC-Sim-selected) | Hierarchy (0.55) | Hierarchy raw (VOC-Sim-selected) |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.629 | 0.673 | 0.709 | 0.709 | 0.693 |
| NC absent-class gate | 0.659 | 0.708 | 0.722 | 0.730 | 0.717 |
| NH NC half strength | 0.661 | 0.706 | 0.716 | 0.725 | 0.710 |
| NO all-pixel contrast | 0.660 | 0.697 | 0.719 | 0.725 | 0.712 |
| PD object direction, all-pixel anchor | 0.634 | 0.697 | 0.703 | 0.712 | 0.698 |
| PI object direction, image's own anchor | 0.625 | 0.660 | 0.697 | 0.712 | 0.692 |
| CAg class-adaptive strength | — | — | 0.726 | — | — |
| CAa class-adaptive any arm | — | — | 0.721 | — | — |

## Comparisons (DenseCRF, primary thresholds; exploratory)

| Comparison | Map | VOC-Sim | COCO-Cap (screened) |
|---|---|---|---|
| NC − A | hierarchy | +0.000 [-0.005, +0.006] | +0.014 [+0.002, +0.026] |
| NH − A | hierarchy | +0.001 [-0.004, +0.005] | +0.007 [-0.002, +0.016] |
| NO − A | hierarchy | -0.001 [-0.007, +0.005] | +0.011 [+0.001, +0.021] |
| PD − A | hierarchy | -0.001 [-0.006, +0.004] | -0.006 [-0.019, +0.008] |
| PI − A | hierarchy | -0.025 [-0.034, -0.016] | -0.011 [-0.026, +0.005] |
| NO − NC | hierarchy | -0.001 [-0.005, +0.001] | -0.003 [-0.011, +0.005] |
| PD − NC | hierarchy | -0.001 [-0.006, +0.004] | -0.019 [-0.032, -0.005] |
| PI − NC | hierarchy | -0.025 [-0.032, -0.017] | -0.025 [-0.039, -0.010] |
| NH − NC | hierarchy | +0.000 [-0.002, +0.003] | -0.006 [-0.012, -0.001] |
| CAg − A | hierarchy | +0.004 [+0.001, +0.008] | +0.018 [+0.008, +0.027] |
| CAa − A | hierarchy | +0.008 [+0.003, +0.012] | +0.012 [+0.003, +0.022] |
| NC − A | seed | +0.016 [+0.010, +0.023] | +0.030 [+0.018, +0.042] |
| NH − A | seed | +0.018 [+0.014, +0.023] | +0.032 [+0.024, +0.041] |
| NO − A | seed | +0.020 [+0.015, +0.026] | +0.031 [+0.023, +0.039] |
| PD − A | seed | +0.003 [-0.003, +0.009] | +0.005 [-0.007, +0.016] |
| PI − A | seed | -0.058 [-0.066, -0.047] | -0.004 [-0.017, +0.009] |
| NC − A | fine-only | +0.007 [+0.001, +0.012] | +0.036 [+0.023, +0.047] |
| NH − A | fine-only | +0.007 [+0.003, +0.011] | +0.034 [+0.025, +0.043] |
| NO − A | fine-only | +0.007 [+0.000, +0.013] | +0.024 [+0.013, +0.035] |
| PD − A | fine-only | +0.001 [-0.004, +0.007] | +0.025 [+0.008, +0.040] |
| PI − A | fine-only | -0.023 [-0.031, -0.013] | -0.013 [-0.031, +0.006] |

Class-adaptive choices (from VOC-Sim, applied to COCO-Cap): **CAg**: aeroplane NC, bird NC, boat NC, bottle NC, bus NC, car NC, cat NC, cow NH, dining table NC, dog NH, horse NC, motorbike NC, person NC, potted plant NC (rest A); **CAa**: aeroplane NC, bird PD, boat NO, bottle NO, bus NO, car NO, cat PD, cow PD, dining table NO, dog PD, horse PD, motorbike NO, person NO, potted plant NC, sheep PD, train NO (rest A)


## Diagnostics, VOC-Sim

D1 mean gate by region (image mean, then mean over images):

| Gate | fg | near bg | far bg | fg − far |
|---|---:|---:|---:|---:|
| NC | 0.764 | 0.584 | 0.462 | +0.302 |
| NO | 0.638 | 0.363 | 0.210 | +0.428 |
| PD | 0.677 | 0.577 | 0.434 | +0.243 |
| PI | 0.569 | 0.421 | 0.289 | +0.281 |

D2 share of seed mass by region:

| Seed | fg | near bg | far bg |
|---|---:|---:|---:|
| A | 0.603 | 0.242 | 0.155 |
| NC | 0.677 | 0.226 | 0.097 |
| NH | 0.649 | 0.234 | 0.116 |
| NO | 0.704 | 0.220 | 0.077 |
| PD | 0.643 | 0.239 | 0.117 |
| PI | 0.686 | 0.223 | 0.091 |

D1 per class, NC gate fg / near / far (and NO fg / far):

| Class | NC fg | NC near | NC far | NO fg | NO far | PI fg | PI far |
|---|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.80 | 0.70 | 0.54 | 0.72 | 0.16 | 0.61 | 0.33 |
| bicycle | 0.74 | 0.51 | 0.33 | 0.64 | 0.14 | 0.53 | 0.28 |
| bird | 0.76 | 0.60 | 0.50 | 0.65 | 0.19 | 0.63 | 0.27 |
| boat | 0.79 | 0.65 | 0.52 | 0.72 | 0.23 | 0.60 | 0.33 |
| bottle | 0.72 | 0.59 | 0.49 | 0.60 | 0.18 | 0.53 | 0.34 |
| bus | 0.80 | 0.50 | 0.32 | 0.73 | 0.19 | 0.62 | 0.28 |
| car | 0.74 | 0.47 | 0.34 | 0.66 | 0.21 | 0.56 | 0.29 |
| cat | 0.76 | 0.59 | 0.48 | 0.59 | 0.21 | 0.51 | 0.25 |
| chair | 0.73 | 0.64 | 0.47 | 0.63 | 0.15 | 0.54 | 0.35 |
| cow | 0.75 | 0.51 | 0.41 | 0.60 | 0.14 | 0.57 | 0.25 |
| dining table | 0.78 | 0.72 | 0.64 | 0.64 | 0.45 | 0.59 | 0.33 |
| dog | 0.76 | 0.51 | 0.38 | 0.64 | 0.14 | 0.58 | 0.24 |
| horse | 0.79 | 0.53 | 0.42 | 0.62 | 0.13 | 0.63 | 0.23 |
| motorbike | 0.78 | 0.52 | 0.39 | 0.68 | 0.17 | 0.57 | 0.26 |
| person | 0.80 | 0.72 | 0.64 | 0.57 | 0.34 | 0.50 | 0.36 |
| potted plant | 0.74 | 0.60 | 0.47 | 0.59 | 0.21 | 0.54 | 0.21 |
| sheep | 0.75 | 0.50 | 0.40 | 0.59 | 0.16 | 0.56 | 0.22 |
| sofa | 0.82 | 0.66 | 0.60 | 0.59 | 0.27 | 0.62 | 0.28 |
| train | 0.77 | 0.59 | 0.46 | 0.72 | 0.24 | 0.57 | 0.36 |
| tv | 0.71 | 0.56 | 0.46 | 0.59 | 0.29 | 0.51 | 0.31 |

D3 per-image IoU gain NC − A (hierarchy + CRF, OOF); mean +0.0002:

| Feature | Spearman ρ | gain, low tertile | mid | high |
|---|---:|---:|---:|---:|
| object area | -0.034 | +0.0049 (n=164) | -0.0016 (n=163) | -0.0027 (n=163) |
| far-bg seed mass of A | +0.059 | -0.0050 (n=164) | +0.0024 (n=163) | +0.0033 (n=163) |

## Diagnostics, COCO-Cap

D1 mean gate by region (image mean, then mean over images):

| Gate | fg | near bg | far bg | fg − far |
|---|---:|---:|---:|---:|
| NC | 0.784 | 0.606 | 0.499 | +0.285 |
| NO | 0.638 | 0.362 | 0.217 | +0.421 |
| PD | 0.674 | 0.570 | 0.428 | +0.246 |
| PI | 0.620 | 0.481 | 0.332 | +0.288 |

D2 share of seed mass by region:

| Seed | fg | near bg | far bg |
|---|---:|---:|---:|
| A | 0.452 | 0.259 | 0.289 |
| NC | 0.505 | 0.251 | 0.244 |
| NH | 0.492 | 0.257 | 0.251 |
| NO | 0.536 | 0.258 | 0.206 |
| PD | 0.467 | 0.260 | 0.273 |
| PI | 0.489 | 0.257 | 0.254 |

D1 per class, NC gate fg / near / far (and NO fg / far):

| Class | NC fg | NC near | NC far | NO fg | NO far | PI fg | PI far |
|---|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.78 | 0.68 | 0.52 | 0.74 | 0.18 | 0.62 | 0.32 |
| bicycle | 0.83 | 0.60 | 0.45 | 0.70 | 0.19 | 0.64 | 0.32 |
| bird | 0.75 | 0.63 | 0.54 | 0.64 | 0.21 | 0.67 | 0.33 |
| boat | 0.79 | 0.70 | 0.58 | 0.67 | 0.24 | 0.60 | 0.35 |
| bottle | 0.78 | 0.63 | 0.54 | 0.63 | 0.27 | 0.61 | 0.38 |
| bus | 0.79 | 0.53 | 0.39 | 0.71 | 0.21 | 0.62 | 0.29 |
| car | 0.74 | 0.48 | 0.36 | 0.63 | 0.17 | 0.58 | 0.34 |
| cat | 0.78 | 0.54 | 0.46 | 0.61 | 0.22 | 0.66 | 0.30 |
| chair | 0.72 | 0.62 | 0.55 | 0.60 | 0.22 | 0.56 | 0.40 |
| cow | 0.79 | 0.53 | 0.44 | 0.62 | 0.16 | 0.66 | 0.28 |
| dining table | 0.70 | 0.58 | 0.54 | 0.52 | 0.30 | 0.53 | 0.37 |
| dog | 0.79 | 0.53 | 0.43 | 0.63 | 0.16 | 0.68 | 0.29 |
| horse | 0.81 | 0.57 | 0.46 | 0.65 | 0.17 | 0.71 | 0.29 |
| motorbike | 0.80 | 0.53 | 0.37 | 0.70 | 0.15 | 0.65 | 0.28 |
| person | 0.87 | 0.81 | 0.75 | 0.55 | 0.21 | 0.47 | 0.46 |
| potted plant | 0.65 | 0.58 | 0.52 | 0.52 | 0.24 | 0.52 | 0.31 |
| sheep | 0.78 | 0.53 | 0.44 | 0.62 | 0.21 | 0.65 | 0.31 |
| sofa | 0.83 | 0.71 | 0.67 | 0.56 | 0.41 | 0.71 | 0.35 |
| train | 0.74 | 0.58 | 0.46 | 0.69 | 0.31 | 0.53 | 0.39 |
| tv | 0.71 | 0.69 | 0.54 | 0.58 | 0.33 | 0.54 | 0.38 |

D3 per-image IoU gain NC − A (hierarchy + CRF, VOC-Sim-selected); mean +0.0167:

| Feature | Spearman ρ | gain, low tertile | mid | high |
|---|---:|---:|---:|---:|
| object area | -0.125 | +0.0307 (n=236) | +0.0052 (n=235) | +0.0141 (n=236) |
| far-bg seed mass of A | +0.153 | +0.0114 (n=236) | +0.0021 (n=235) | +0.0364 (n=236) |
| other classes in caption (0 / 1 / ≥2) | +0.055 | +0.0129 (n=499) | +0.0286 (n=193) | -0.0109 (n=15) |

Per class, voc, hierarchy + DenseCRF, OOF (A / NC / NH / NO / PD / PI / CAg / CAa): aeroplane 0.88 / 0.89 / 0.89 / 0.89 / 0.88 / 0.88 / 0.89 / 0.89, bicycle 0.80 / 0.78 / 0.79 / 0.79 / 0.78 / 0.76 / 0.80 / 0.80, bird 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92, boat 0.85 / 0.86 / 0.86 / 0.86 / 0.86 / 0.82 / 0.86 / 0.86, bottle 0.87 / 0.90 / 0.90 / 0.91 / 0.87 / 0.85 / 0.90 / 0.90, bus 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.92, car 0.85 / 0.86 / 0.85 / 0.86 / 0.86 / 0.85 / 0.86 / 0.85, cat 0.85 / 0.85 / 0.85 / 0.85 / 0.87 / 0.86 / 0.84 / 0.87, chair 0.79 / 0.78 / 0.79 / 0.78 / 0.79 / 0.75 / 0.79 / 0.79, cow 0.90 / 0.90 / 0.90 / 0.87 / 0.91 / 0.89 / 0.90 / 0.90, dining table 0.43 / 0.44 / 0.43 / 0.45 / 0.43 / 0.42 / 0.43 / 0.45, dog 0.93 / 0.93 / 0.94 / 0.93 / 0.94 / 0.93 / 0.94 / 0.93, horse 0.90 / 0.91 / 0.91 / 0.90 / 0.92 / 0.90 / 0.91 / 0.91, motorbike 0.87 / 0.87 / 0.87 / 0.87 / 0.87 / 0.86 / 0.87 / 0.86, person 0.81 / 0.82 / 0.82 / 0.83 / 0.81 / 0.76 / 0.81 / 0.82, potted plant 0.75 / 0.80 / 0.78 / 0.77 / 0.77 / 0.75 / 0.80 / 0.80, sheep 0.76 / 0.71 / 0.73 / 0.69 / 0.76 / 0.64 / 0.76 / 0.76, sofa 0.96 / 0.95 / 0.95 / 0.95 / 0.95 / 0.94 / 0.96 / 0.96, train 0.74 / 0.71 / 0.73 / 0.75 / 0.71 / 0.66 / 0.72 / 0.75, tv 0.85 / 0.82 / 0.82 / 0.81 / 0.81 / 0.77 / 0.85 / 0.85


Per class, coco, hierarchy + DenseCRF, VOC-Sim-selected (A / NC / NH / NO / PD / PI / CAg / CAa): aeroplane 0.80 / 0.85 / 0.84 / 0.85 / 0.77 / 0.77 / 0.85 / 0.85, bicycle 0.63 / 0.65 / 0.63 / 0.66 / 0.67 / 0.67 / 0.63 / 0.63, bird 0.78 / 0.78 / 0.78 / 0.78 / 0.77 / 0.78 / 0.78 / 0.77, boat 0.59 / 0.61 / 0.59 / 0.61 / 0.60 / 0.60 / 0.61 / 0.61, bottle 0.57 / 0.60 / 0.57 / 0.58 / 0.60 / 0.56 / 0.60 / 0.58, bus 0.92 / 0.92 / 0.92 / 0.92 / 0.92 / 0.91 / 0.92 / 0.92, car 0.76 / 0.80 / 0.78 / 0.73 / 0.78 / 0.78 / 0.80 / 0.73, cat 0.92 / 0.92 / 0.92 / 0.92 / 0.91 / 0.91 / 0.92 / 0.91, chair 0.48 / 0.52 / 0.49 / 0.50 / 0.47 / 0.46 / 0.48 / 0.48, cow 0.87 / 0.90 / 0.90 / 0.87 / 0.89 / 0.88 / 0.90 / 0.89, dining table 0.47 / 0.54 / 0.52 / 0.58 / 0.53 / 0.65 / 0.54 / 0.58, dog 0.81 / 0.83 / 0.83 / 0.82 / 0.83 / 0.83 / 0.83 / 0.83, horse 0.72 / 0.77 / 0.76 / 0.75 / 0.78 / 0.78 / 0.77 / 0.78, motorbike 0.71 / 0.75 / 0.74 / 0.74 / 0.74 / 0.74 / 0.75 / 0.74, person 0.28 / 0.39 / 0.35 / 0.35 / 0.14 / 0.15 / 0.39 / 0.35, potted plant 0.75 / 0.66 / 0.68 / 0.67 / 0.69 / 0.64 / 0.66 / 0.66, sheep 0.87 / 0.87 / 0.89 / 0.86 / 0.90 / 0.88 / 0.87 / 0.90, sofa 0.70 / 0.74 / 0.72 / 0.74 / 0.71 / 0.70 / 0.70 / 0.70, train 0.75 / 0.67 / 0.71 / 0.71 / 0.68 / 0.59 / 0.75 / 0.71, tv 0.79 / 0.69 / 0.70 / 0.73 / 0.68 / 0.67 / 0.79 / 0.79

