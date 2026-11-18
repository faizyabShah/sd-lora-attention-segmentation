# Few-shot supervised shadow attention (`PREREG.md`, amendment 1)

`scripts/jc_fewshot.py`. **Exploratory, supervised, few-shot** (masks of 3 VOC-Sim images per class). Shadow low-rank copies (rank 4, 60 steps, lr 0.001); the generator is untouched. All arms from one live generation per image (paired). *train-sel.* = each arm's best threshold on the 60 training images, applied unchanged to COCO-Cap. Primary test set: all COCO-Cap images (amendment 1); the registered 60-image subset is also reported. Class-macro mIoU; Δ with paired image-bootstrap 95% interval (20).

## Data and checks

- VOC-Sim training images (in-sample): 1 images, 1 classes. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 1.5e-04, BRACE map 2.1e-05.
- COCO-Cap all: 1 images, 1 classes. Zero-adapter shadow vs frozen BRACE: max |Δ| seed 1.8e-04, BRACE map 1.2e-05.
- Training B: loss first 1 steps 0.742 → last 1 0.653 (seed term 0.742 → 0.653); 9.9 s/step.
- Training C: loss first 1 steps 0.340 → last 1 0.355 (seed term 0.742 → 0.653); 10.0 s/step.
- Training D: loss first 1 steps 0.340 → last 1 0.355 (seed term 0.742 → 0.653); 10.2 s/step.

## With DenseCRF (primary)

| Arm | Map | Train (in-sample), own best | train-sel. thr. | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | COCO-Cap 60, train-sel. | COCO-Cap 60, 0.55 |
|---|---|---:|---:|---:|---:|---:|---:|
| A frozen BRACE | seed | 0.775 | 0.500 | 0.871 | 0.820 | 0.871 | 0.820 |
| A frozen BRACE | BRACE hierarchy | 0.958 | 0.600 | 0.931 | 0.930 | 0.931 | 0.930 |
| B CA, seed supervision | seed | 0.793 | 0.475 | 0.881 | 0.812 | 0.881 | 0.812 |
| B CA, seed supervision | BRACE hierarchy | 0.958 | 0.600 | 0.931 | 0.930 | 0.931 | 0.930 |
| C CA, hierarchy supervision | seed | 0.783 | 0.500 | 0.864 | 0.810 | 0.864 | 0.810 |
| C CA, hierarchy supervision | BRACE hierarchy | 0.956 | 0.600 | 0.931 | 0.930 | 0.931 | 0.930 |
| D CA + SA, hierarchy supervision | seed | 0.783 | 0.500 | 0.864 | 0.810 | 0.864 | 0.810 |
| D CA + SA, hierarchy supervision | BRACE hierarchy | 0.956 | 0.600 | 0.931 | 0.930 | 0.931 | 0.930 |

## Raw (secondary)

| Arm | Map | Train (in-sample), own best | train-sel. thr. | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | COCO-Cap 60, train-sel. | COCO-Cap 60, 0.55 |
|---|---|---:|---:|---:|---:|---:|---:|
| A frozen BRACE | seed | 0.693 | 0.450 | 0.783 | 0.774 | 0.783 | 0.774 |
| A frozen BRACE | BRACE hierarchy | 0.947 | 0.600 | 0.885 | 0.864 | 0.885 | 0.864 |
| B CA, seed supervision | seed | 0.711 | 0.450 | 0.783 | 0.771 | 0.783 | 0.771 |
| B CA, seed supervision | BRACE hierarchy | 0.948 | 0.600 | 0.885 | 0.865 | 0.885 | 0.865 |
| C CA, hierarchy supervision | seed | 0.703 | 0.450 | 0.780 | 0.769 | 0.780 | 0.769 |
| C CA, hierarchy supervision | BRACE hierarchy | 0.947 | 0.625 | 0.893 | 0.865 | 0.893 | 0.865 |
| D CA + SA, hierarchy supervision | seed | 0.703 | 0.450 | 0.780 | 0.769 | 0.780 | 0.769 |
| D CA + SA, hierarchy supervision | BRACE hierarchy | 0.948 | 0.625 | 0.893 | 0.866 | 0.893 | 0.866 |

## Comparisons (BRACE hierarchy + DenseCRF)

| ID | Comparison | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | Positive (all, 0.55)? | COCO-Cap 60, 0.55 |
|---|---|---|---|---|---|
| F1 | D − C (SA copies add to hierarchy-supervised CA?) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes | +0.000 [+0.000, +0.000] |
| F2 | C − B (hierarchy vs direct supervision of CA) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes | +0.000 [+0.000, +0.000] |
| F3 | B − A | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes | +0.000 [+0.000, +0.000] |
| F4 | C − A | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes | +0.000 [+0.000, +0.000] |
| F5 | D − A | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes | +0.000 [+0.000, +0.000] |

"Positive" = 95% interval above zero on all COCO-Cap at 0.55. Exploratory.

Seed + DenseCRF @0.55, COCO-Cap all, vs A: B -0.008 [-0.008, -0.008]; C -0.010 [-0.010, -0.010]; D -0.010 [-0.010, -0.010]

Per class, COCO-Cap all, BRACE hierarchy + DenseCRF @0.55 (A / B / C / D): aeroplane 0.93 / 0.93 / 0.93 / 0.93

