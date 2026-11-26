# Few-shot supervised shadow attention (`PREREG.md`, amendment 1)

`scripts/jc_fewshot.py`. **Exploratory, supervised, few-shot** (masks of 3 VOC-Sim images per class). Shadow low-rank copies (rank 4, 60 steps, lr 0.001); the generator is untouched. All arms from one live generation per image (paired). *train-sel.* = each arm's best threshold on the 60 training images, applied unchanged to COCO-Cap. Primary test set: all COCO-Cap images (amendment 1); the registered 60-image subset is also reported. Class-macro mIoU; Δ with paired image-bootstrap 95% interval (2000).

## Data and checks

- VOC-Sim training images (in-sample): 60 images, 20 classes. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 3.3e-04, BRACE map 3.0e-04.
- COCO-Cap all: 707 images, 20 classes. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 6.2e-04, BRACE map 4.3e-03.
- Training B: loss first 15 steps 0.771 → last 15 0.601 (seed term 0.771 → 0.601); 15.6 s/step.
- Training C: loss first 15 steps 0.473 → last 15 0.549 (seed term 0.786 → 0.709); 18.9 s/step.
- Training D: loss first 15 steps 0.468 → last 15 0.461 (seed term 0.787 → 0.694); 7.6 s/step.

## With DenseCRF (primary)

| Arm | Map | Train (in-sample), own best | train-sel. thr. | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | COCO-Cap 60, train-sel. | COCO-Cap 60, 0.55 |
|---|---|---:|---:|---:|---:|---:|---:|
| A frozen BRACE | seed | 0.756 | 0.400 | 0.629 | 0.527 | 0.623 | 0.529 |
| A frozen BRACE | BRACE hierarchy | 0.832 | 0.500 | 0.682 | 0.708 | 0.708 | 0.743 |
| B CA, seed supervision | seed | 0.841 | 0.450 | 0.678 | 0.622 | 0.684 | 0.642 |
| B CA, seed supervision | BRACE hierarchy | 0.867 | 0.600 | 0.713 | 0.707 | 0.754 | 0.738 |
| C CA, hierarchy supervision | seed | 0.803 | 0.400 | 0.666 | 0.561 | 0.681 | 0.561 |
| C CA, hierarchy supervision | BRACE hierarchy | 0.866 | 0.525 | 0.708 | 0.718 | 0.744 | 0.747 |
| D CA + SA, hierarchy supervision | seed | 0.810 | 0.400 | 0.669 | 0.576 | 0.678 | 0.576 |
| D CA + SA, hierarchy supervision | BRACE hierarchy | 0.885 | 0.425 | 0.693 | 0.719 | 0.704 | 0.726 |

## Raw (secondary)

| Arm | Map | Train (in-sample), own best | train-sel. thr. | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | COCO-Cap 60, train-sel. | COCO-Cap 60, 0.55 |
|---|---|---:|---:|---:|---:|---:|---:|
| A frozen BRACE | seed | 0.725 | 0.400 | 0.591 | 0.510 | 0.580 | 0.515 |
| A frozen BRACE | BRACE hierarchy | 0.818 | 0.525 | 0.679 | 0.692 | 0.707 | 0.727 |
| B CA, seed supervision | seed | 0.811 | 0.450 | 0.640 | 0.597 | 0.639 | 0.611 |
| B CA, seed supervision | BRACE hierarchy | 0.861 | 0.600 | 0.703 | 0.694 | 0.742 | 0.721 |
| C CA, hierarchy supervision | seed | 0.771 | 0.400 | 0.624 | 0.544 | 0.626 | 0.547 |
| C CA, hierarchy supervision | BRACE hierarchy | 0.853 | 0.550 | 0.705 | 0.705 | 0.734 | 0.734 |
| D CA + SA, hierarchy supervision | seed | 0.776 | 0.400 | 0.628 | 0.555 | 0.627 | 0.558 |
| D CA + SA, hierarchy supervision | BRACE hierarchy | 0.879 | 0.475 | 0.699 | 0.716 | 0.711 | 0.731 |

## Comparisons (BRACE hierarchy + DenseCRF)

| ID | Comparison | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | Positive (all, 0.55)? | COCO-Cap 60, 0.55 |
|---|---|---|---|---|---|
| F1 | D − C (SA copies add to hierarchy-supervised CA?) | -0.015 [-0.025, -0.006] | +0.001 [-0.007, +0.009] | no | -0.020 [-0.035, -0.006] |
| F2 | C − B (hierarchy vs direct supervision of CA) | -0.004 [-0.023, +0.013] | +0.011 [-0.006, +0.026] | no | +0.009 [-0.033, +0.053] |
| F3 | B − A | +0.031 [+0.012, +0.050] | -0.001 [-0.018, +0.016] | no | -0.005 [-0.051, +0.041] |
| F4 | C − A | +0.026 [+0.017, +0.036] | +0.010 [+0.000, +0.020] | no | +0.004 [-0.014, +0.022] |
| F5 | D − A | +0.011 [-0.000, +0.022] | +0.011 [-0.000, +0.023] | no | -0.017 [-0.040, +0.009] |

"Positive" = 95% interval above zero on all COCO-Cap at 0.55. Exploratory.

Seed + DenseCRF @0.55, COCO-Cap all, vs A: B +0.095 [+0.078, +0.112]; C +0.034 [+0.024, +0.044]; D +0.049 [+0.039, +0.058]

Per class, COCO-Cap all, BRACE hierarchy + DenseCRF @0.55 (A / B / C / D): aeroplane 0.80 / 0.85 / 0.86 / 0.89, bicycle 0.63 / 0.64 / 0.66 / 0.56, bird 0.78 / 0.78 / 0.78 / 0.79, boat 0.59 / 0.61 / 0.58 / 0.63, bottle 0.57 / 0.48 / 0.58 / 0.53, bus 0.92 / 0.91 / 0.92 / 0.93, car 0.76 / 0.68 / 0.74 / 0.73, cat 0.92 / 0.91 / 0.92 / 0.92, chair 0.47 / 0.43 / 0.48 / 0.47, cow 0.87 / 0.91 / 0.90 / 0.90, dining table 0.47 / 0.53 / 0.50 / 0.51, dog 0.81 / 0.79 / 0.82 / 0.82, horse 0.72 / 0.73 / 0.77 / 0.75, motorbike 0.71 / 0.70 / 0.73 / 0.71, person 0.28 / 0.67 / 0.28 / 0.34, potted plant 0.74 / 0.73 / 0.67 / 0.71, sheep 0.87 / 0.85 / 0.89 / 0.90, sofa 0.70 / 0.35 / 0.66 / 0.64, train 0.75 / 0.87 / 0.82 / 0.86, tv 0.79 / 0.72 / 0.80 / 0.81

