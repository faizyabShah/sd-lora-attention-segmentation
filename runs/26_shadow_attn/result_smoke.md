# Supervised shadow attention for BRACE: CA-only vs CA + SA (`PREREG.md`)

`scripts/jc_shadow.py`. **Exploratory, supervised** (human masks of the COCO-Cap training images). The generator is untouched: trained low-rank *copies* of the attention only produce the maps BRACE reads. All arms from one live generation per test image (paired). *VOC-sel.* = 5-fold OOF on VOC-Sim; the full-VOC-Sim best threshold applied unchanged on COCO-Cap test. Class-macro mIoU; Δ with paired image-bootstrap 95% interval (200).

## Data and checks

- Split: 160 train / 40 validation / 507 COCO-Cap test; VOC-Sim (all) never trained on or used for selection.
- VOC-Sim: 1 images. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 1.3e-04, BRACE map 1.1e-05.
- COCO-Cap test: 1 images. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 2.1e-04, BRACE map 2.0e-05.
- Validation (40 images, raw shadow BRACE map @0.55): A0@0 0.725, B@2 0.725, B@4 0.725, C@2 0.725, C@4 0.725; selected B step 2, C step 2.
- Training B: loss first 1 steps 1.910 → last 1 0.833; 14.8 s/step.
- Training C: loss first 1 steps 1.910 → last 1 0.831; 16.2 s/step.

## With DenseCRF (primary)

| Arm | Map | VOC-Sim, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap test, VOC-sel. | COCO-Cap test, 0.55 |
|---|---|---:|---:|---:|---:|
| A0 frozen BRACE | seed | 0.744 | 0.722 | 0.754 | 0.726 |
| A0 frozen BRACE | BRACE hierarchy | 0.936 | 0.945 | 0.877 | 0.886 |
| B CA copies (step 2) | seed | 0.745 | 0.721 | 0.754 | 0.726 |
| B CA copies (step 2) | BRACE hierarchy | 0.936 | 0.945 | 0.877 | 0.886 |
| C CA + SA copies (step 2) | seed | 0.745 | 0.721 | 0.754 | 0.726 |
| C CA + SA copies (step 2) | BRACE hierarchy | 0.936 | 0.945 | 0.877 | 0.886 |

## Raw (secondary)

| Arm | Map | VOC-Sim, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap test, VOC-sel. | COCO-Cap test, 0.55 |
|---|---|---:|---:|---:|---:|
| A0 frozen BRACE | seed | 0.665 | 0.669 | 0.571 | 0.602 |
| A0 frozen BRACE | BRACE hierarchy | 0.818 | 0.917 | 0.800 | 0.783 |
| B CA copies (step 2) | seed | 0.666 | 0.668 | 0.571 | 0.603 |
| B CA copies (step 2) | BRACE hierarchy | 0.818 | 0.917 | 0.800 | 0.783 |
| C CA + SA copies (step 2) | seed | 0.666 | 0.668 | 0.571 | 0.603 |
| C CA + SA copies (step 2) | BRACE hierarchy | 0.818 | 0.917 | 0.800 | 0.783 |

## Registered comparisons (BRACE hierarchy + DenseCRF)

| ID | Comparison | VOC-Sim, VOC-sel. | COCO-Cap test, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap test, 0.55 | Verdict (0.55) |
|---|---|---|---|---|---|---|
| S1 | C − B | +0.001 [+0.001, +0.001] | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | does not hold |
| S2 | B − A0 | -0.000 [-0.000, -0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | holds |
| S3 | C − A0 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | does not hold |

Rule: holds if the COCO-Cap test Δ at 0.55 is positive with interval above zero and the VOC-Sim Δ at 0.55 is not below −0.005.

Descriptive, seed + DenseCRF @0.55, C − B / B − A0: VOC-Sim +0.000 [+0.000, +0.000] / -0.001 [-0.001, -0.001]; COCO-Cap test +0.000 [+0.000, +0.000] / -0.000 [-0.000, -0.000]

Per class, VOC-Sim, BRACE hierarchy + DenseCRF @0.55 (A0 / B / C): aeroplane 0.95 / 0.95 / 0.95

Per class, COCO-Cap test, BRACE hierarchy + DenseCRF @0.55 (A0 / B / C): aeroplane 0.89 / 0.89 / 0.89

