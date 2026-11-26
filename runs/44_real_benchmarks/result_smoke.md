# Run 44: real-photo comparison on PASCAL VOC, PASCAL Context and COCO-Object (`PREREG.md`)

Class word given; one binary mask per (photo, class) primary item (class area >= 1%). Every method as released on the same photos (PREREG section 4); ours: MK4-LR, gamma 1, fine-only, fixed 0.55. Class-macro mIoU; 95% paired photo bootstrap (50 samples). Items: PASCAL VOC (20) 3 (3 photos), PASCAL Context (59) 3 (1 photos), COCO-Object (80) 3 (1 photos).


## 1. Primary contrasts (ours = MK4-LR, V0, raw 0.55)

| # | Contrast | Column | Delta mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | ours - no gate | PASCAL VOC (20) | — | — |
| P2 | ours - no gate | PASCAL Context (59) | — | — |
| P3 | ours - no gate | COCO-Object (80) | — | — |
| P4 | ours - NERVE | PASCAL VOC (20) | — | — |
| P5 | ours - NERVE | PASCAL Context (59) | — | — |
| P6 | ours - NERVE | COCO-Object (80) | — | — |

## 2. Main table (mIoU)

| Method | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---:|---:|---:|
| DAAM | — | — | — |
| OVAM (no token opt.) | — | — | — |
| DiffSegmenter | — | — | — |
| iSeg | — | 0.089 | 0.150 |
| SCLIP | 0.620 | 0.048 | — |
| ClearCLIP | 0.381 | 0.076 | — |
| NACLIP | 0.655 | 0.008 | — |
| ProxyCLIP | 0.753 | 0.051 | — |
| NERVE | — | 0.091 | — |
| OVDiff | — | — | — |
| No gate (ours, ablation) | — | — | — |
| MK4 (ours, ablation) | — | — | — |
| **Ours (MK4-LR), raw** | — | — | — |
| **Ours + DenseCRF** | — | — | — |
| OVAM + DenseCRF | — | — | — |
| No gate + DenseCRF | — | — | — |
| iSeg, class given | — | — | — |

## 3. Ours - each baseline (raw 0.55)

| Baseline | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---|---|---|
| DAAM | — | — | — |
| OVAM (no token opt.) | — | — | — |
| DiffSegmenter | — | — | — |
| iSeg | — | — | — |
| SCLIP | — | — | — |
| ClearCLIP | — | — | — |
| NACLIP | — | — | — |
| ProxyCLIP | — | — | — |
| NERVE | — | — | — |
| OVDiff | — | — | — |
| No gate (ours, ablation) | — | — | — |
| MK4 (ours, ablation) | — | — | — |
| OVAM + DenseCRF | — | — | — |
| No gate + DenseCRF | — | — | — |
| iSeg, class given | — | — | — |

## 4. Subsets (raw 0.55)


## 5. Every threshold (ours, raw; descriptive)

| Column | Map | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 6. Per class (raw 0.55)


PASCAL VOC (20):

| Class | items | SCLIP | ClearCLIP | NACLIP | ProxyCLIP |
|---|---:|---:|---:|---:|---:|
| aeroplane | 1 | 0.457 | 0.193 | 0.636 | 0.712 |
| boat | 1 | 0.702 | 0.357 | 0.593 | 0.846 |
| train | 1 | 0.700 | 0.595 | 0.735 | 0.700 |

PASCAL Context (59):

| Class | items | iSeg | SCLIP | ClearCLIP | NACLIP | ProxyCLIP | NERVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| dining table | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| tv | 1 | 0.267 | 0.144 | 0.228 | 0.024 | 0.152 | 0.272 |
| window | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

COCO-Object (80):

| Class | items | iSeg |
|---|---:|---:|
| chair | 1 | 0.000 |
| person | 1 | 0.000 |
| tv | 1 | 0.451 |

## 7. Coverage

- PASCAL VOC (20): methods with missing items: {'daam|0.4': 1, 'diffseg|0.5': 1, 'iseg|own': 1, 'nerve|own': 1, 'ours:u-|V0|crf': 1, 'ours:u-|V0|raw': 1, 'ours:u-|seed|raw': 1, 'ours:uL|V0|crf': 1, 'ours:uL|V0|raw': 1, 'ours:uL|seed|raw': 1, 'ours:uM|V0|raw': 1, 'ours:uM|seed|raw': 1, 'ovam|0.4': 1, 'ovam|0.4|crf': 1}
- PASCAL Context (59): methods with missing items: {'daam|0.4': 1, 'diffseg|0.5': 2, 'ours:u-|V0|crf': 1, 'ours:u-|V0|raw': 1, 'ours:u-|seed|raw': 1, 'ours:uL|V0|crf': 1, 'ours:uL|V0|raw': 1, 'ours:uL|seed|raw': 1, 'ours:uM|V0|raw': 1, 'ours:uM|seed|raw': 1, 'ovam|0.4': 1, 'ovam|0.4|crf': 1}
- COCO-Object (80): methods with missing items: {'daam|0.4': 1, 'diffseg|0.5': 1, 'ours:u-|V0|crf': 1, 'ours:u-|V0|raw': 1, 'ours:u-|seed|raw': 1, 'ours:uL|V0|crf': 1, 'ours:uL|V0|raw': 1, 'ours:uL|seed|raw': 1, 'ours:uM|V0|raw': 1, 'ours:uM|seed|raw': 1, 'ovam|0.4': 1, 'ovam|0.4|crf': 1}

BRACE stays frozen; nothing here is a promotion.
