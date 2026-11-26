# Run 37: run 36 gates on real PASCAL VOC photos, strengths and RW weights fitted on VOC train (`PREREG.md`)

**Exploratory; third look at VOC val.** VOC 2012 val: 12 photos, 12 items. Fitted on VOC train (12 items): thresholds per map, gate strength γ per gate family and map (grid [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), RW module weights (40 items). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals (50 samples, paired).


## 1. Primary contrasts (val, + DenseCRF, train-selected threshold and γ; source C, SA weights on)

| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − MK4 | capped hierarchy (V3) | -0.047 [-0.120, +0.033] | not resolved |
| P2 | MK4-LR − MK4 | fine-only (V0) | +0.017 [-0.010, +0.044] | not resolved |
| P3 | MK4-RW − MK4 | capped hierarchy (V3) | -0.073 [-0.146, -0.002] | **hurts** |
| P4 | MK4-RW − MK4 | fine-only (V0) | -0.003 [-0.033, +0.023] | not resolved |
| P5 | SN − MK4 | capped hierarchy (V3) | -0.021 [-0.048, +0.004] | not resolved |
| P6 | SN − MK4 | fine-only (V0) | +0.013 [-0.029, +0.056] | not resolved |

- MK4-LR: does not hold on real photos (helps on V3 or V0, hurts on neither).
- MK4-RW: does not hold on real photos (helps on V3 or V0, hurts on neither).
- SN: does not hold on real photos (helps on V3 or V0, hurts on neither).

## 2. Train-selected γ

| Source | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|
| C |  | seed only | 0.5 | 1.5 | 2.0 | 2.0 |
| C |  | fine-only (V0) | 0.5 | 2.0 | 2.0 | 2.0 |
| C |  | capped hierarchy (V3) | 0.5 | 2.0 | 2.0 | 2.0 |
| C | ✓ | seed only | 1.5 | 1.5 | 2.0 | 2.0 |
| C | ✓ | fine-only (V0) | 2.0 | 2.0 | 2.0 | 2.0 |
| C | ✓ | capped hierarchy (V3) | 0.75 | 2.0 | 2.0 | 2.0 |
| A |  | seed only | 0.5 | 2.0 | 2.0 | 2.0 |
| A |  | fine-only (V0) | 0.5 | 2.0 | 2.0 | 2.0 |
| A |  | capped hierarchy (V3) | 0.5 | 2.0 | 2.0 | 2.0 |
| A | ✓ | seed only | 2.0 | 2.0 | 2.0 | 2.0 |
| A | ✓ | fine-only (V0) | 2.0 | 2.0 | 2.0 | 2.0 |
| A | ✓ | capped hierarchy (V3) | 1.0 | 2.0 | 2.0 | 2.0 |

## 3. Source C (real COCO photos, captions; label-free profiles)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.647 | 0.733 | 0.686 | 0.713 | 0.681 |
|  | capped hierarchy (V3) | 0.733 | 0.739 | 0.717 | 0.669 | 0.670 |
|  | seed only | 0.607 | 0.645 | 0.578 | 0.647 | 0.631 |
| ✓ | fine-only (V0) | 0.695 | 0.667 | 0.680 | 0.684 | 0.664 |
| ✓ | capped hierarchy (V3) | 0.756 | 0.733 | 0.712 | 0.686 | 0.660 |
| ✓ | seed only | 0.608 | 0.586 | 0.571 | 0.627 | 0.632 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.085 [+0.025, +0.161] | -0.047 [-0.080, -0.015] | -0.020 [-0.109, +0.068] | -0.052 [-0.135, +0.028] |
|  | capped hierarchy (V3) | +0.006 [-0.022, +0.039] | -0.022 [-0.046, +0.002] | -0.070 [-0.146, +0.009] | -0.069 [-0.143, +0.009] |
|  | seed only | +0.037 [-0.030, +0.088] | -0.067 [-0.116, -0.045] | +0.002 [-0.044, +0.052] | -0.013 [-0.059, +0.032] |
| ✓ | fine-only (V0) | -0.028 [-0.104, +0.069] | +0.013 [-0.029, +0.056] | +0.017 [-0.010, +0.044] | -0.003 [-0.033, +0.023] |
| ✓ | capped hierarchy (V3) | -0.023 [-0.055, +0.003] | -0.021 [-0.048, +0.004] | -0.047 [-0.120, +0.033] | -0.073 [-0.146, -0.002] |
| ✓ | seed only | -0.021 [-0.124, +0.047] | -0.016 [-0.047, +0.009] | +0.040 [+0.010, +0.078] | +0.046 [+0.016, +0.082] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.048 [+0.011, +0.088] | +0.023 [+0.002, +0.048] | +0.085 [+0.026, +0.156] | +0.061 [+0.004, +0.121] |
| MK4 | -0.066 [-0.130, +0.006] | -0.006 [-0.014, +0.005] | +0.006 [-0.022, +0.033] | +0.066 [-0.021, +0.147] |
| SN | -0.006 [-0.058, +0.052] | -0.005 [-0.024, +0.014] | +0.032 [-0.020, +0.071] | +0.032 [-0.024, +0.099] |
| MK4-LR | -0.028 [-0.045, -0.003] | +0.017 [+0.009, +0.032] | -0.044 [-0.074, -0.004] | +0.001 [-0.030, +0.032] |
| MK4-RW | -0.016 [-0.028, -0.001] | -0.010 [-0.016, -0.001] | -0.011 [-0.049, +0.027] | -0.005 [-0.049, +0.035] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.710 | 0.665 | 0.689 | 0.690 | -0.021 [-0.069, +0.038] |
| capped hierarchy (V3) | 0.733 | 0.735 | 0.696 | 0.696 | -0.037 [-0.097, +0.019] |

## 4. Source A (VOC train labels; reference)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.647 | 0.637 | 0.550 | 0.644 | 0.614 |
|  | capped hierarchy (V3) | 0.733 | 0.669 | 0.621 | 0.646 | 0.678 |
|  | seed only | 0.607 | 0.651 | 0.566 | 0.648 | 0.657 |
| ✓ | fine-only (V0) | 0.695 | 0.627 | 0.598 | 0.617 | 0.595 |
| ✓ | capped hierarchy (V3) | 0.756 | 0.676 | 0.607 | 0.670 | 0.691 |
| ✓ | seed only | 0.608 | 0.558 | 0.576 | 0.631 | 0.634 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | -0.010 [-0.117, +0.090] | -0.087 [-0.144, -0.036] | +0.007 [-0.072, +0.084] | -0.023 [-0.087, +0.046] |
|  | capped hierarchy (V3) | -0.064 [-0.128, +0.009] | -0.047 [-0.088, -0.015] | -0.023 [-0.085, +0.044] | +0.010 [-0.022, +0.039] |
|  | seed only | +0.044 [-0.030, +0.116] | -0.086 [-0.132, -0.053] | -0.004 [-0.067, +0.044] | +0.006 [-0.047, +0.043] |
| ✓ | fine-only (V0) | -0.068 [-0.173, +0.020] | -0.029 [-0.099, +0.030] | -0.010 [-0.072, +0.060] | -0.032 [-0.091, +0.031] |
| ✓ | capped hierarchy (V3) | -0.080 [-0.151, -0.009] | -0.069 [-0.114, -0.041] | -0.005 [-0.072, +0.054] | +0.015 [-0.020, +0.042] |
| ✓ | seed only | -0.049 [-0.144, +0.029] | +0.018 [-0.008, +0.038] | +0.073 [+0.024, +0.113] | +0.075 [+0.034, +0.114] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.048 [+0.011, +0.088] | +0.023 [+0.002, +0.048] | +0.085 [+0.026, +0.156] | +0.061 [+0.004, +0.121] |
| MK4 | -0.010 [-0.035, +0.018] | +0.007 [-0.008, +0.025] | +0.031 [+0.016, +0.049] | +0.049 [+0.028, +0.071] |
| SN | +0.048 [+0.030, +0.078] | -0.015 [-0.027, -0.002] | +0.071 [+0.009, +0.133] | +0.008 [-0.041, +0.045] |
| MK4-LR | -0.027 [-0.042, -0.005] | +0.025 [+0.011, +0.047] | +0.002 [-0.026, +0.033] | +0.053 [+0.029, +0.073] |
| MK4-RW | -0.019 [-0.031, -0.007] | +0.013 [+0.007, +0.028] | +0.064 [+0.008, +0.105] | +0.096 [+0.049, +0.135] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.667 | 0.616 | 0.648 | 0.675 | -0.019 [-0.063, +0.033] |
| capped hierarchy (V3) | 0.676 | 0.676 | 0.683 | 0.686 | +0.007 [-0.038, +0.042] |

## 5. Other rules, source C, SA weights on (train-selected γ)

| Map | Gate | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.695 | 0.704 | 0.713 | 0.717 |
| fine-only (V0) | MK4 | 0.667 | 0.705 | 0.702 | 0.727 |
| fine-only (V0) | SN | 0.680 | 0.680 | 0.702 | 0.702 |
| fine-only (V0) | MK4-LR | 0.684 | 0.657 | 0.714 | 0.692 |
| fine-only (V0) | MK4-RW | 0.664 | 0.664 | 0.700 | 0.700 |
| capped hierarchy (V3) | none | 0.756 | 0.758 | 0.753 | 0.750 |
| capped hierarchy (V3) | MK4 | 0.733 | 0.775 | 0.755 | 0.776 |
| capped hierarchy (V3) | SN | 0.712 | 0.767 | 0.738 | 0.774 |
| capped hierarchy (V3) | MK4-LR | 0.686 | 0.728 | 0.708 | 0.740 |
| capped hierarchy (V3) | MK4-RW | 0.660 | 0.729 | 0.695 | 0.741 |

## 6. Per class IoU (primary rule, source C, SA weights on)

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 1 | 0.789 | 0.603 | 0.641 | 0.566 | 0.465 | 0.844 | 0.768 | 0.825 | 0.775 | 0.739 |
| bicycle | 1 | 0.292 | 0.333 | 0.438 | 0.418 | 0.428 | 0.410 | 0.436 | 0.450 | 0.441 | 0.417 |
| bus | 1 | 0.836 | 0.830 | 0.806 | 0.840 | 0.814 | 0.858 | 0.850 | 0.871 | 0.870 | 0.859 |
| cat | 1 | 0.979 | 0.961 | 0.945 | 0.961 | 0.952 | 0.974 | 0.945 | 0.960 | 0.961 | 0.953 |
| chair | 1 | 0.286 | 0.210 | 0.228 | 0.381 | 0.326 | 0.181 | 0.217 | 0.204 | 0.325 | 0.311 |
| dog | 2 | 0.944 | 0.860 | 0.818 | 0.920 | 0.877 | 0.864 | 0.957 | 0.943 | 0.949 | 0.951 |
| motorbike | 1 | 0.873 | 0.897 | 0.844 | 0.896 | 0.883 | 0.842 | 0.714 | 0.783 | 0.832 | 0.806 |
| person | 1 | 0.834 | 0.955 | 0.908 | 0.934 | 0.913 | 0.492 | 0.717 | 0.646 | 0.791 | 0.732 |
| potted plant | 1 | 0.933 | 0.905 | 0.897 | 0.256 | 0.251 | 0.903 | 0.255 | 0.495 | 0.258 | 0.254 |
| train | 1 | 0.825 | 0.844 | 0.846 | 0.848 | 0.865 | 0.834 | 0.823 | 0.841 | 0.831 | 0.818 |
| tv | 1 | 0.724 | 0.664 | 0.464 | 0.522 | 0.483 | 0.443 | 0.654 | 0.462 | 0.495 | 0.468 |

## 7. RW module weights (VOC train, all items; w = clip(2·(AUROC − 0.5), 0, 1) of the MK4 gate, object vs far)

| Source | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C AUROC | 0.574 | 0.488 | 0.663 | 0.819 | 0.972 | 0.952 | 0.918 | 0.889 | 0.506 | 0.479 | 0.435 |
| A AUROC | 0.671 | 0.483 | 0.723 | 0.861 | 0.992 | 0.993 | 0.962 | 0.884 | 0.482 | 0.458 | 0.424 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Mean gate value at γ = 1 over modules and pixels (val): C/MK4 0.472, C/SN 0.317, C/MK4-LR 0.411, C/MK4-RW 0.412, A/MK4 0.469, A/SN 0.319, A/MK4-LR 0.401, A/MK4-RW 0.405.

Run 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, full stack C 0.645 (uncapped hierarchy + MK4).

