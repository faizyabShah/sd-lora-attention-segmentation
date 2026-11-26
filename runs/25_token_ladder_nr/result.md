# Stage ladder: class word vs OVAM's optimized token, corrected data (`PREREG.md`)

`scripts/jc_ladder_nr.py`. One live generation per image (frozen runtime, runs/20 corrected prompts, safety checker off); both query columns and all stages from that generation. **Class word** = frozen attribution (BRACE). **Optimized token** = OVAM's released VOC-Sim token (few-shot: needs annotated images; the same VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold applied unchanged on COCO-Cap. Δ with paired image-bootstrap 95% interval (2000). Class-word results are in-sample for the method design. Supersedes runs/15 (old prompts).

## Data checks

- VOC-Sim: 490 images; image identical to the runs/20 capture on 164/490; class-word BRACE map vs runs/20: median max |Δ| 2.4e-03.
- COCO-Cap: 707 images; image identical to the runs/20 capture on 177/707; class-word BRACE map vs runs/20: median max |Δ| 3.7e-03.

## Stage table, with DenseCRF (primary), SA-guided layer weights

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.736 | 0.810 | 0.628 | 0.781 | 0.629 | 0.667 | 0.527 | 0.661 |
| CA + fine SA | 0.824 | 0.874 | 0.796 | 0.854 | 0.673 | 0.700 | 0.698 | 0.712 |
| CA + hierarchy (BRACE) | 0.831 | 0.864 | 0.832 | 0.864 | 0.708 | 0.693 | 0.708 | 0.693 |

## Stage table, raw (secondary), SA-guided layer weights

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.706 | 0.777 | 0.613 | 0.756 | 0.591 | 0.633 | 0.510 | 0.627 |
| CA + fine SA | 0.811 | 0.861 | 0.792 | 0.848 | 0.670 | 0.684 | 0.690 | 0.703 |
| CA + hierarchy (BRACE) | 0.818 | 0.850 | 0.819 | 0.849 | 0.693 | 0.685 | 0.693 | 0.678 |

## Uniform layer weights, with DenseCRF (secondary)

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.748 | 0.833 | 0.603 | 0.792 | 0.616 | 0.684 | 0.458 | 0.665 |
| CA + fine SA | 0.820 | 0.873 | 0.786 | 0.852 | 0.657 | 0.682 | 0.668 | 0.708 |
| CA + hierarchy (BRACE) | 0.828 | 0.860 | 0.829 | 0.860 | 0.674 | 0.688 | 0.683 | 0.688 |

## Registered comparisons (with DenseCRF)

| ID | Comparison | Query | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 | Verdict (0.55) |
|---|---|---|---|---|---|---|---|
| F | CA + fine − CA | class word | +0.088 [+0.077, +0.101] | +0.044 [+0.028, +0.061] | +0.169 [+0.156, +0.183] | +0.171 [+0.152, +0.190] | holds |
| F | CA + fine − CA | optimized token | +0.064 [+0.057, +0.073] | +0.032 [+0.022, +0.044] | +0.074 [+0.064, +0.083] | +0.051 [+0.038, +0.065] | holds |
| H | CA + hierarchy − CA + fine | class word | +0.007 [+0.001, +0.013] | +0.036 [+0.024, +0.046] | +0.036 [+0.025, +0.046] | +0.010 [-0.005, +0.026] | does not hold |
| H | CA + hierarchy − CA + fine | optimized token | -0.011 [-0.017, -0.005] | -0.007 [-0.014, -0.000] | +0.009 [+0.001, +0.017] | -0.019 [-0.031, -0.009] | does not hold |
| — | CA + hierarchy − CA | class word | +0.095 [+0.083, +0.108] | +0.079 [+0.063, +0.096] | +0.205 [+0.191, +0.218] | +0.182 [+0.158, +0.205] | — |
| — | CA + hierarchy − CA | optimized token | +0.054 [+0.044, +0.064] | +0.026 [+0.013, +0.040] | +0.083 [+0.072, +0.093] | +0.032 [+0.014, +0.050] | — |
| Q | token − word: CA | — | +0.073 [+0.064, +0.082] | +0.038 [+0.025, +0.052] | +0.153 [+0.141, +0.164] | +0.135 [+0.116, +0.153] | descriptive |
| Q | token − word: CA + fine SA | — | +0.050 [+0.038, +0.061] | +0.027 [+0.012, +0.042] | +0.058 [+0.045, +0.070] | +0.014 [-0.002, +0.032] | descriptive |
| Q | token − word: CA + hierarchy (BRACE) | — | +0.032 [+0.023, +0.041] | -0.015 [-0.030, -0.000] | +0.031 [+0.022, +0.040] | -0.015 [-0.030, -0.000] | descriptive |

Rule (F, H, per query): holds if the COCO-Cap Δ at 0.55 is positive with interval above zero and the VOC-Sim Δ at 0.55 is not below −0.005.

Per class, VOC-Sim, CA + hierarchy + DenseCRF @0.55 (word / token): aeroplane 0.88 / 0.90, bicycle 0.79 / 0.80, bird 0.92 / 0.93, boat 0.86 / 0.85, bottle 0.88 / 0.90, bus 0.92 / 0.92, car 0.85 / 0.86, cat 0.85 / 0.88, chair 0.79 / 0.80, cow 0.90 / 0.95, dining-table 0.44 / 0.45, dog 0.93 / 0.93, horse 0.90 / 0.92, motorbike 0.87 / 0.88, person 0.81 / 0.86, potted-plant 0.75 / 0.82, sheep 0.75 / 0.87, sofa 0.96 / 0.96, train 0.74 / 0.88, tv 0.85 / 0.92

Per class, COCO-Cap, CA + hierarchy + DenseCRF @0.55 (word / token): aeroplane 0.80 / 0.87, bicycle 0.63 / 0.60, bird 0.78 / 0.77, boat 0.59 / 0.57, bottle 0.57 / 0.52, bus 0.92 / 0.92, car 0.76 / 0.66, cat 0.92 / 0.89, chair 0.47 / 0.46, cow 0.87 / 0.87, dining table 0.47 / 0.53, dog 0.81 / 0.75, horse 0.72 / 0.66, motorbike 0.71 / 0.68, person 0.28 / 0.63, potted plant 0.74 / 0.70, sheep 0.87 / 0.82, sofa 0.70 / 0.34, train 0.75 / 0.85, tv 0.79 / 0.73

