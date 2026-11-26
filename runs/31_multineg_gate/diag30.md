# Run 31, Part A: gate diagnostics on run 30 captures (`PREREG.md`)

Module-mean gate maps saved by run 30 (NC = absent-class gate, NO = all-pixel gate); regions from GT at 64 px (object; near background within 6 px; far background). Ground truth only defines regions and scores.


## VOC-Sim (490 images)

### A1 gate distributions

| Gate | Region | q05 | q25 | q50 | q75 | q95 | share g>0.3 | g>0.5 | g>0.7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC | fg | 0.55 | 0.70 | 0.79 | 0.85 | 0.91 | 1.00 | 0.97 | 0.74 |
| NC | near | 0.33 | 0.48 | 0.59 | 0.69 | 0.82 | 0.97 | 0.71 | 0.23 |
| NC | far | 0.24 | 0.37 | 0.46 | 0.55 | 0.70 | 0.87 | 0.43 | 0.05 |
| NO | fg | 0.42 | 0.56 | 0.65 | 0.73 | 0.82 | 0.99 | 0.82 | 0.27 |
| NO | near | 0.13 | 0.23 | 0.34 | 0.48 | 0.66 | 0.58 | 0.22 | 0.03 |
| NO | far | 0.07 | 0.12 | 0.18 | 0.26 | 0.45 | 0.19 | 0.04 | 0.00 |

Object retention = share of object pixels with g > t; far rejection = 1 − share of far pixels with g > t.

| Gate | AUROC object vs far (per-image mean) | pooled | AUROC object vs near (per-image mean) | pooled |
|---|---:|---:|---:|---:|
| NC | 0.968 | 0.949 | 0.866 | 0.836 |
| NO | 0.988 | 0.985 | 0.912 | 0.898 |

Per class (NC): median per-image AUROC object vs far; object retention and far rejection at 0.5.

| Class | AUROC | retention | rejection |
|---|---:|---:|---:|
| aeroplane | 0.995 | 0.97 | 0.17 |
| bicycle | 1.000 | 0.96 | 0.98 |
| bird | 0.997 | 0.99 | 0.50 |
| boat | 0.990 | 0.94 | 0.35 |
| bottle | 0.970 | 0.93 | 0.37 |
| bus | 0.999 | 0.99 | 0.84 |
| car | 1.000 | 0.96 | 0.92 |
| cat | 0.999 | 0.99 | 0.76 |
| chair | 0.994 | 0.98 | 0.55 |
| cow | 1.000 | 0.99 | 0.98 |
| dining table | 0.928 | 0.98 | 0.12 |
| dog | 1.000 | 0.98 | 0.99 |
| horse | 0.998 | 0.99 | 0.90 |
| motorbike | 0.999 | 0.99 | 0.85 |
| person | 0.959 | 0.99 | 0.11 |
| potted plant | 0.965 | 0.95 | 0.72 |
| sheep | 1.000 | 0.99 | 0.72 |
| sofa | 0.957 | 0.98 | 0.19 |
| train | 0.987 | 0.99 | 0.62 |
| tv | 0.991 | 0.88 | 0.69 |

### A2 seed-level sharpening (approximation; no self-attention rerun)

| Seed | far-background share of seed mass | raw seed mIoU (OOF) |
|---|---:|---:|
| BRACE seed | 0.155 | 0.706 |
| seed × NC gate^0.5 | 0.133 | 0.718 |
| seed × NC gate^1 | 0.115 | 0.722 |
| seed × NC gate^2 | 0.086 | 0.698 |
| seed × NC gate^4 | 0.051 | 0.581 |

## COCO-Cap (707 images)

### A1 gate distributions

| Gate | Region | q05 | q25 | q50 | q75 | q95 | share g>0.3 | g>0.5 | g>0.7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC | fg | 0.53 | 0.72 | 0.82 | 0.89 | 0.94 | 1.00 | 0.95 | 0.75 |
| NC | near | 0.31 | 0.50 | 0.61 | 0.73 | 0.87 | 0.96 | 0.75 | 0.30 |
| NC | far | 0.23 | 0.41 | 0.50 | 0.58 | 0.79 | 0.89 | 0.52 | 0.11 |
| NO | fg | 0.38 | 0.55 | 0.65 | 0.74 | 0.83 | 0.95 | 0.75 | 0.28 |
| NO | near | 0.11 | 0.22 | 0.34 | 0.49 | 0.68 | 0.58 | 0.22 | 0.04 |
| NO | far | 0.06 | 0.12 | 0.18 | 0.29 | 0.47 | 0.22 | 0.04 | 0.00 |

Object retention = share of object pixels with g > t; far rejection = 1 − share of far pixels with g > t.

| Gate | AUROC object vs far (per-image mean) | pooled | AUROC object vs near (per-image mean) | pooled |
|---|---:|---:|---:|---:|
| NC | 0.941 | 0.915 | 0.846 | 0.809 |
| NO | 0.970 | 0.973 | 0.898 | 0.879 |

Per class (NC): median per-image AUROC object vs far; object retention and far rejection at 0.5.

| Class | AUROC | retention | rejection |
|---|---:|---:|---:|
| aeroplane | 0.994 | 0.98 | 0.30 |
| bicycle | 0.999 | 0.99 | 0.69 |
| bird | 0.981 | 0.98 | 0.22 |
| boat | 0.980 | 0.85 | 0.18 |
| bottle | 0.977 | 0.99 | 0.39 |
| bus | 0.998 | 0.99 | 0.72 |
| car | 0.996 | 0.87 | 0.80 |
| cat | 0.993 | 0.98 | 0.65 |
| chair | 0.910 | 0.96 | 0.34 |
| cow | 0.996 | 0.96 | 0.77 |
| dining table | 0.940 | 0.93 | 0.15 |
| dog | 0.998 | 0.96 | 0.81 |
| horse | 0.996 | 0.98 | 0.50 |
| motorbike | 0.998 | 0.98 | 0.89 |
| person | 0.874 | 1.00 | 0.01 |
| potted plant | 0.937 | 0.77 | 0.32 |
| sheep | 0.995 | 0.96 | 0.67 |
| sofa | 0.909 | 0.98 | 0.10 |
| train | 0.929 | 0.95 | 0.57 |
| tv | 0.793 | 0.84 | 0.35 |

### A2 seed-level sharpening (approximation; no self-attention rerun)

| Seed | far-background share of seed mass | raw seed mIoU (VOC-Sim-selected) |
|---|---:|---:|
| BRACE seed | 0.289 | 0.592 (thr 0.400) |
| seed × NC gate^0.5 | 0.261 | 0.609 (thr 0.350) |
| seed × NC gate^1 | 0.235 | 0.614 (thr 0.300) |
| seed × NC gate^2 | 0.194 | 0.615 (thr 0.300) |
| seed × NC gate^4 | 0.138 | 0.539 (thr 0.300) |

A2 is at the seed only. After propagation the ceiling for a perfectly clean background is run 24's oracle (about +0.05 / +0.06 BRACE mIoU).
