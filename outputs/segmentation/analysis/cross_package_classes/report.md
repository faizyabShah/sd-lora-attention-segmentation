# Cross-package per-class analysis

Comparisons within one package, from the committed per-class tables; nothing is selected.

| Comparison | Group | VOC mean Δ | VOC classes improved | COCO mean Δ | COCO classes improved | VOC–COCO class correlation |
|---|---|---:|---:|---:|---:|---:|
| P1 two passes | raw | +0.0120 | 14/20 | +0.0073 | 14/20 | +0.42 |
| SG2 consensus graph | raw | +0.0098 | 16/20 | +0.0063 | 13/20 | +0.37 |
| SG8 sign consensus | raw | +0.0081 | 12/20 | +0.0033 | 13/20 | +0.19 |
| SG11 robust consensus | raw | +0.0098 | 14/20 | +0.0048 | 13/20 | +0.22 |
| SG12 top-50% observations | raw | +0.0099 | 16/20 | +0.0100 | 17/20 | +0.04 |
| SG13 region labels (R1) | raw | -0.0203 | 8/20 | -0.0166 | 7/20 | +0.46 |
| SG18 middle calls (T1) vs P1 | raw | -0.0048 | 12/20 | +0.0037 | 12/20 | +0.20 |
| SG2 + CRF vs FTF | final | -0.0021 | 9/20 | -0.0110 | 4/20 | +0.41 |
| SG12 + CRF vs FTF | final | -0.0001 | 11/20 | -0.0056 | 5/20 | +0.44 |
| T1 + CRF vs FTF | final | -0.0013 | 13/20 | +0.0089 | 14/20 | +0.23 |
| SG22 MID-P2 + CRF vs FTF | final | +0.0053 | 16/20 | +0.0067 | 17/20 | +0.17 |
| SG24 A3 vs A0 (both MID-P2 + CRF) | final | +0.0058 | 11/20 | +0.0183 | 11/20 | -0.01 |
| SG25 16->64 + CRF vs fine | final | -0.0060 | 9/20 | +0.0354 | 19/20 | +0.23 |
| SG26 S5 + CRF vs S0 | final | -0.0042 | 7/20 | +0.0200 | 17/20 | +0.09 |

Classes on VOC, by mean Δ over all 14 comparisons (gains / losses = comparisons with Δ > +0.005 / < −0.005):

| Class | Mean Δ | Gains | Losses |
|---|---:|---:|---:|
| train | +0.0202 | 11 | 2 |
| bird | +0.0099 | 10 | 1 |
| boat | +0.0090 | 8 | 4 |
| bottle | +0.0086 | 11 | 1 |
| cow | +0.0082 | 9 | 3 |
| sheep | +0.0073 | 6 | 4 |
| bus | +0.0060 | 7 | 1 |
| tv | +0.0056 | 7 | 4 |
| car | +0.0054 | 9 | 2 |
| dining-table | +0.0032 | 7 | 4 |
| dog | +0.0023 | 5 | 1 |
| horse | +0.0013 | 3 | 2 |
| motorbike | +0.0007 | 6 | 3 |
| sofa | -0.0019 | 3 | 7 |
| aeroplane | -0.0020 | 5 | 7 |
| cat | -0.0036 | 6 | 7 |
| chair | -0.0037 | 6 | 8 |
| bicycle | -0.0070 | 3 | 6 |
| potted-plant | -0.0105 | 1 | 8 |
| person | -0.0271 | 7 | 7 |

Classes on COCO, by mean Δ over all 14 comparisons (gains / losses = comparisons with Δ > +0.005 / < −0.005):

| Class | Mean Δ | Gains | Losses |
|---|---:|---:|---:|
| aeroplane | +0.0299 | 12 | 2 |
| boat | +0.0204 | 11 | 2 |
| bottle | +0.0177 | 11 | 2 |
| motorbike | +0.0140 | 10 | 3 |
| bicycle | +0.0138 | 8 | 5 |
| horse | +0.0138 | 11 | 3 |
| chair | +0.0135 | 9 | 4 |
| car | +0.0121 | 6 | 8 |
| sheep | +0.0080 | 10 | 2 |
| train | +0.0077 | 6 | 3 |
| cow | +0.0049 | 9 | 1 |
| bus | +0.0037 | 4 | 3 |
| potted plant | +0.0017 | 6 | 5 |
| sofa | +0.0016 | 6 | 5 |
| dog | -0.0004 | 2 | 1 |
| tv | -0.0005 | 3 | 5 |
| bird | -0.0031 | 4 | 6 |
| person | -0.0067 | 8 | 4 |
| cat | -0.0095 | 6 | 5 |
| dining table | -0.0118 | 4 | 9 |
