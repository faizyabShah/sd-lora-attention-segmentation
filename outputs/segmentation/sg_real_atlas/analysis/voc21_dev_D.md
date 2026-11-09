# D development, standard multi-class protocol (200 D photos, VOC train; val untouched)

| variant | VOC21 | VOC20 | VOC21 +CRF | VOC20 +CRF |
|---|---:|---:|---:|---:|
| sclip | 62.08 | 82.42 | 64.77 | 87.24 |
| pool | 62.28 | 83.18 | 64.54 | 87.24 |
| c16 | 43.58 | 81.29 | 39.14 | 78.50 |
| resid | 44.22 | 81.49 | 39.38 | 78.73 |
| geo | 60.97 | 87.19 | 56.66 | 85.41 |

Δ vs SCLIP + CRF and vs SCLIP standalone (photo-bootstrap 95% CI):

| variant | Δ VOC21 vs sclip+crf | CI | Δ VOC21 vs sclip | CI | Δ VOC20 vs sclip | CI |
|---|---:|---|---:|---|---:|---|
| pool | -2.49 | -4.63…-0.41 | +0.20 | +0.03…+0.48 | +0.77 | +0.65…+0.99 |
| pool+crf | -0.23 | -0.37…-0.10 | +2.47 | +0.32…+4.66 | +4.82 | +2.02…+7.47 |
| c16 | -21.18 | -24.62…-16.79 | -18.49 | -22.32…-14.26 | -1.13 | -5.87…+3.53 |
| c16+crf | -25.63 | -28.97…-22.14 | -22.94 | -26.67…-19.35 | -3.92 | -8.60…+0.46 |
| resid | -20.55 | -23.90…-16.66 | -17.86 | -21.49…-13.88 | -0.92 | -5.49…+3.64 |
| resid+crf | -25.39 | -28.74…-21.91 | -22.70 | -26.43…-19.12 | -3.69 | -8.29…+0.70 |
| geo | -3.80 | -6.55…-1.64 | -1.11 | -3.54…+0.67 | +4.77 | +2.64…+7.25 |
| geo+crf | -8.11 | -10.29…-5.77 | -5.42 | -8.16…-2.13 | +2.99 | -0.63…+6.48 |

VOC21 error types (dataset level):

- sclip: background→object 8.3% of bg pixels; object→background 15.2% of object pixels; wrong object 5.2%
- sclip+crf: background→object 6.7% of bg pixels; object→background 15.1% of object pixels; wrong object 4.9%
- pool: background→object 8.5% of bg pixels; object→background 14.6% of object pixels; wrong object 5.3%
- c16: background→object 18.9% of bg pixels; object→background 23.8% of object pixels; wrong object 8.3%
- resid: background→object 18.9% of bg pixels; object→background 23.3% of object pixels; wrong object 8.3%

Per-class VOC21 IoU:

| class | sclip | sclip+crf | c16 | resid |
|---|---:|---:|---:|---:|
| background | 87.1 | 88.7 | 75.0 | 75.1 |
| aeroplane | 51.2 | 61.3 | 40.2 | 39.9 |
| bicycle | 33.6 | 35.5 | 16.5 | 17.3 |
| bird | 74.8 | 70.1 | 57.7 | 57.8 |
| ship | 39.3 | 38.6 | 11.3 | 11.6 |
| bottle | 50.0 | 62.5 | 27.1 | 35.0 |
| bus | 76.7 | 79.1 | 56.9 | 57.5 |
| car | 65.2 | 62.4 | 36.4 | 36.2 |
| cat | 78.9 | 78.1 | 34.2 | 35.0 |
| chair | 38.2 | 39.7 | 22.2 | 22.9 |
| cow | 76.3 | 84.6 | 72.8 | 72.8 |
| table | 61.9 | 67.0 | 59.2 | 57.1 |
| dog | 80.3 | 83.6 | 45.3 | 47.9 |
| horse | 53.7 | 53.1 | 29.9 | 30.1 |
| motorbike | 76.9 | 85.1 | 60.4 | 60.7 |
| person | 65.4 | 68.1 | 38.2 | 38.2 |
| pottedplant | 55.2 | 48.0 | 31.0 | 31.6 |
| sheep | 76.8 | 82.5 | 57.4 | 58.4 |
| sofa | 42.5 | 43.7 | 31.4 | 31.3 |
| train | 63.1 | 63.7 | 59.3 | 58.8 |
| television monitor | 56.5 | 64.8 | 52.7 | 53.2 |

Runtime: 31.9 s/photo mean (median 31.9): SCLIP + one DDIM inversion + one SD run + propagation + all CRFs, on one GPU node.

**Chosen for val by the registered rule:** `geo`
