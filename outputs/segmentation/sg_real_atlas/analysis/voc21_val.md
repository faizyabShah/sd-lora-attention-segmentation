# VOC2012 val (1449 photos), standard protocol: SCLIP ViT-B/16 + SD1.5 self-attention

Pre-registered in `voc21_prereg.md`; the variant was chosen on D by the registered rule (`voc21_choice.json`: **geo**) before val was scored. Nothing tuned on val. No class-presence information used. Dataset-level confusion (mmseg IoUMetric style). Our SCLIP port reproduces the repo's B16 numbers (59.63 / 81.54).

| variant | VOC21 | VOC20 | VOC21 +CRF | VOC20 +CRF |
|---|---:|---:|---:|---:|
| sclip | 59.64 | 81.49 | 60.56 | 83.39 |
| pool | 59.81 | 82.27 | 60.31 | 83.41 |
| c16 | 41.09 | 78.85 | 37.82 | 76.95 |
| resid | 41.45 | 79.08 | 37.95 | 77.09 |
| geo | 58.56 | 84.32 | 54.86 | 81.85 |

**Primary (registered):** geo − sclip+crf VOC21 = -2.00 (photo-bootstrap 95% CI -2.63…-1.37) → **FAIL**

Δ vs SCLIP + CRF and vs SCLIP standalone (photo-bootstrap 95% CI) — secondary:

| variant | Δ VOC21 vs sclip+crf | CI | Δ VOC21 vs sclip | CI | Δ VOC20 vs sclip | CI |
|---|---:|---|---:|---|---:|---|
| pool | -0.75 | -1.41…-0.11 | +0.17 | +0.11…+0.25 | +0.78 | +0.72…+0.85 |
| pool+crf | -0.25 | -0.35…-0.16 | +0.67 | +0.01…+1.34 | +1.91 | +1.16…+2.68 |
| c16 | -19.46 | -20.92…-18.11 | -18.54 | -19.83…-17.14 | -2.64 | -4.24…-1.11 |
| c16+crf | -22.74 | -24.35…-21.26 | -21.82 | -23.27…-20.36 | -4.54 | -6.30…-2.80 |
| resid | -19.11 | -20.55…-17.76 | -18.19 | -19.45…-16.81 | -2.41 | -3.99…-0.93 |
| resid+crf | -22.61 | -24.23…-21.13 | -21.69 | -23.15…-20.24 | -4.40 | -6.16…-2.68 |
| geo | -2.00 | -2.63…-1.37 | -1.08 | -1.64…-0.45 | +2.83 | +2.11…+3.63 |
| geo+crf | -5.70 | -6.46…-4.94 | -4.78 | -5.72…-3.78 | +0.36 | -0.80…+1.47 |

VOC21 error types (dataset level):

- sclip: background→object 10.7% of bg pixels; object→background 12.1% of object pixels; wrong object 5.0%
- sclip+crf: background→object 9.6% of bg pixels; object→background 11.4% of object pixels; wrong object 5.6%
- pool: background→object 11.0% of bg pixels; object→background 11.4% of object pixels; wrong object 5.0%
- c16: background→object 25.4% of bg pixels; object→background 16.2% of object pixels; wrong object 8.7%
- resid: background→object 25.1% of bg pixels; object→background 16.0% of object pixels; wrong object 8.7%
- geo: background→object 11.4% of bg pixels; object→background 11.8% of object pixels; wrong object 5.2%

Per-class VOC21 IoU:

| class | sclip | sclip+crf | geo |
|---|---:|---:|---:|
| background | 85.5 | 86.8 | 84.9 |
| aeroplane | 50.6 | 54.7 | 47.4 |
| bicycle | 30.3 | 31.2 | 27.6 |
| bird | 71.5 | 65.4 | 63.2 |
| ship | 50.6 | 44.6 | 46.3 |
| bottle | 57.1 | 63.3 | 60.0 |
| bus | 81.4 | 82.7 | 83.6 |
| car | 65.2 | 62.0 | 61.5 |
| cat | 82.9 | 85.3 | 78.0 |
| chair | 28.8 | 29.2 | 28.0 |
| cow | 76.4 | 77.3 | 73.8 |
| table | 47.0 | 49.3 | 52.0 |
| dog | 80.3 | 80.7 | 76.8 |
| horse | 58.1 | 60.4 | 54.6 |
| motorbike | 64.2 | 64.6 | 64.4 |
| person | 67.7 | 69.1 | 67.4 |
| pottedplant | 39.4 | 41.8 | 42.8 |
| sheep | 72.1 | 75.4 | 71.7 |
| sofa | 45.3 | 47.4 | 46.3 |
| train | 58.4 | 57.1 | 58.0 |
| television monitor | 39.5 | 43.5 | 41.4 |

Runtime: 22.5 s/photo mean (median 20.3): SCLIP + one DDIM inversion + one SD run + propagation + all CRFs, on one GPU node.
