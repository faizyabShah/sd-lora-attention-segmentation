# Run 37: run 36 gates on real PASCAL VOC photos, strengths and RW weights fitted on VOC train (`PREREG.md`)

**Exploratory; third look at VOC val.** VOC 2012 val: 1449 photos, 2148 items. Fitted on VOC train (2170 items): thresholds per map, gate strength γ per gate family and map (grid [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), RW module weights (2170 items). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals (2000 samples, paired).


## 1. Primary contrasts (val, + DenseCRF, train-selected threshold and γ; source C, SA weights on)

| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − MK4 | capped hierarchy (V3) | +0.021 [+0.014, +0.028] | **helps** |
| P2 | MK4-LR − MK4 | fine-only (V0) | +0.038 [+0.028, +0.048] | **helps** |
| P3 | MK4-RW − MK4 | capped hierarchy (V3) | +0.025 [+0.019, +0.032] | **helps** |
| P4 | MK4-RW − MK4 | fine-only (V0) | +0.042 [+0.032, +0.051] | **helps** |
| P5 | SN − MK4 | capped hierarchy (V3) | -0.018 [-0.025, -0.012] | **hurts** |
| P6 | SN − MK4 | fine-only (V0) | -0.021 [-0.029, -0.015] | **hurts** |

- MK4-LR: **holds on real photos** (helps on V3 or V0, hurts on neither).
- MK4-RW: **holds on real photos** (helps on V3 or V0, hurts on neither).
- SN: does not hold on real photos (helps on V3 or V0, hurts on neither).

## 2. Train-selected γ

| Source | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|
| C |  | seed only | 0.25 | 0.25 | 1.5 | 1.5 |
| C |  | fine-only (V0) | 0.5 | 0.25 | 1.5 | 1.5 |
| C |  | capped hierarchy (V3) | 0.5 | 0.5 | 2.0 | 2.0 |
| C | ✓ | seed only | 0.25 | 1.0 | 1.5 | 1.5 |
| C | ✓ | fine-only (V0) | 0.5 | 0.25 | 1.5 | 1.5 |
| C | ✓ | capped hierarchy (V3) | 0.75 | 1.0 | 1.5 | 1.5 |
| A |  | seed only | 0.25 | 0.75 | 1.5 | 1.5 |
| A |  | fine-only (V0) | 0.5 | 0.5 | 2.0 | 2.0 |
| A |  | capped hierarchy (V3) | 0.5 | 0.5 | 2.0 | 2.0 |
| A | ✓ | seed only | 0.5 | 1.0 | 1.5 | 1.5 |
| A | ✓ | fine-only (V0) | 0.75 | 1.0 | 2.0 | 2.0 |
| A | ✓ | capped hierarchy (V3) | 1.0 | 1.5 | 2.0 | 2.0 |

## 3. Source C (real COCO photos, captions; label-free profiles)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.596 | 0.623 | 0.597 | 0.677 | 0.682 |
|  | capped hierarchy (V3) | 0.591 | 0.626 | 0.597 | 0.662 | 0.666 |
|  | seed only | 0.539 | 0.556 | 0.549 | 0.619 | 0.624 |
| ✓ | fine-only (V0) | 0.619 | 0.646 | 0.624 | 0.684 | 0.687 |
| ✓ | capped hierarchy (V3) | 0.608 | 0.646 | 0.628 | 0.668 | 0.672 |
| ✓ | seed only | 0.558 | 0.574 | 0.574 | 0.622 | 0.626 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.027 [+0.017, +0.038] | -0.026 [-0.034, -0.017] | +0.054 [+0.043, +0.064] | +0.059 [+0.048, +0.069] |
|  | capped hierarchy (V3) | +0.035 [+0.026, +0.044] | -0.029 [-0.036, -0.022] | +0.036 [+0.026, +0.046] | +0.040 [+0.030, +0.050] |
|  | seed only | +0.017 [+0.011, +0.022] | -0.007 [-0.010, -0.003] | +0.064 [+0.056, +0.072] | +0.069 [+0.061, +0.076] |
| ✓ | fine-only (V0) | +0.027 [+0.019, +0.036] | -0.021 [-0.029, -0.015] | +0.038 [+0.028, +0.048] | +0.042 [+0.032, +0.051] |
| ✓ | capped hierarchy (V3) | +0.038 [+0.029, +0.047] | -0.018 [-0.025, -0.012] | +0.021 [+0.014, +0.028] | +0.025 [+0.019, +0.032] |
| ✓ | seed only | +0.016 [+0.012, +0.020] | +0.000 [-0.005, +0.005] | +0.048 [+0.042, +0.055] | +0.052 [+0.046, +0.059] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.023 [+0.016, +0.029] | +0.018 [+0.011, +0.024] | -0.005 [-0.015, +0.004] | -0.010 [-0.019, -0.002] |
| MK4 | +0.023 [+0.017, +0.028] | +0.020 [+0.015, +0.026] | +0.003 [-0.007, +0.012] | +0.001 [-0.007, +0.008] |
| SN | +0.027 [+0.021, +0.033] | +0.031 [+0.024, +0.038] | -0.000 [-0.010, +0.009] | +0.004 [-0.005, +0.013] |
| MK4-LR | +0.007 [+0.003, +0.011] | +0.006 [+0.002, +0.010] | -0.015 [-0.023, -0.007] | -0.016 [-0.024, -0.008] |
| MK4-RW | +0.006 [+0.002, +0.010] | +0.006 [+0.001, +0.010] | -0.016 [-0.024, -0.008] | -0.016 [-0.023, -0.008] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.636 | 0.629 | 0.682 | 0.686 | +0.046 [+0.038, +0.054] |
| capped hierarchy (V3) | 0.644 | 0.628 | 0.664 | 0.666 | +0.020 [+0.013, +0.026] |

## 4. Source A (VOC train labels; reference)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.596 | 0.624 | 0.597 | 0.674 | 0.678 |
|  | capped hierarchy (V3) | 0.591 | 0.624 | 0.597 | 0.656 | 0.658 |
|  | seed only | 0.539 | 0.557 | 0.558 | 0.633 | 0.636 |
| ✓ | fine-only (V0) | 0.619 | 0.647 | 0.636 | 0.681 | 0.683 |
| ✓ | capped hierarchy (V3) | 0.608 | 0.644 | 0.632 | 0.663 | 0.666 |
| ✓ | seed only | 0.558 | 0.577 | 0.585 | 0.631 | 0.633 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.029 [+0.019, +0.038] | -0.027 [-0.036, -0.018] | +0.049 [+0.037, +0.061] | +0.054 [+0.042, +0.065] |
|  | capped hierarchy (V3) | +0.033 [+0.025, +0.042] | -0.026 [-0.033, -0.019] | +0.032 [+0.023, +0.042] | +0.034 [+0.025, +0.044] |
|  | seed only | +0.018 [+0.013, +0.024] | +0.000 [-0.005, +0.005] | +0.076 [+0.068, +0.083] | +0.079 [+0.071, +0.086] |
| ✓ | fine-only (V0) | +0.028 [+0.019, +0.038] | -0.011 [-0.019, -0.004] | +0.034 [+0.025, +0.043] | +0.037 [+0.027, +0.045] |
| ✓ | capped hierarchy (V3) | +0.036 [+0.026, +0.045] | -0.013 [-0.019, -0.005] | +0.019 [+0.011, +0.027] | +0.022 [+0.014, +0.030] |
| ✓ | seed only | +0.019 [+0.014, +0.025] | +0.008 [+0.003, +0.012] | +0.054 [+0.047, +0.060] | +0.056 [+0.049, +0.062] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.023 [+0.016, +0.029] | +0.018 [+0.011, +0.024] | -0.005 [-0.015, +0.004] | -0.010 [-0.019, -0.002] |
| MK4 | +0.023 [+0.017, +0.029] | +0.020 [+0.013, +0.028] | -0.000 [-0.010, +0.009] | -0.003 [-0.011, +0.005] |
| SN | +0.039 [+0.031, +0.046] | +0.034 [+0.026, +0.042] | +0.000 [-0.009, +0.010] | -0.004 [-0.012, +0.004] |
| MK4-LR | +0.007 [+0.003, +0.012] | +0.007 [+0.004, +0.011] | -0.018 [-0.025, -0.010] | -0.018 [-0.026, -0.010] |
| MK4-RW | +0.006 [+0.001, +0.010] | +0.008 [+0.004, +0.012] | -0.020 [-0.028, -0.012] | -0.017 [-0.025, -0.010] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.642 | 0.636 | 0.677 | 0.679 | +0.035 [+0.027, +0.043] |
| capped hierarchy (V3) | 0.644 | 0.629 | 0.655 | 0.657 | +0.011 [+0.004, +0.018] |

## 5. Other rules, source C, SA weights on (train-selected γ)

| Map | Gate | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.619 | 0.615 | 0.624 | 0.623 |
| fine-only (V0) | MK4 | 0.646 | 0.641 | 0.653 | 0.651 |
| fine-only (V0) | SN | 0.624 | 0.621 | 0.630 | 0.629 |
| fine-only (V0) | MK4-LR | 0.684 | 0.666 | 0.688 | 0.679 |
| fine-only (V0) | MK4-RW | 0.687 | 0.671 | 0.691 | 0.684 |
| capped hierarchy (V3) | none | 0.608 | 0.596 | 0.616 | 0.595 |
| capped hierarchy (V3) | MK4 | 0.646 | 0.632 | 0.653 | 0.632 |
| capped hierarchy (V3) | SN | 0.628 | 0.616 | 0.635 | 0.616 |
| capped hierarchy (V3) | MK4-LR | 0.668 | 0.661 | 0.674 | 0.663 |
| capped hierarchy (V3) | MK4-RW | 0.672 | 0.665 | 0.677 | 0.666 |

## 6. Per class IoU (primary rule, source C, SA weights on)

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 90 | 0.813 | 0.857 | 0.844 | 0.860 | 0.861 | 0.806 | 0.830 | 0.813 | 0.845 | 0.844 |
| bicycle | 79 | 0.311 | 0.345 | 0.345 | 0.362 | 0.376 | 0.295 | 0.336 | 0.314 | 0.377 | 0.378 |
| bird | 103 | 0.710 | 0.728 | 0.746 | 0.772 | 0.764 | 0.757 | 0.761 | 0.763 | 0.804 | 0.804 |
| boat | 72 | 0.646 | 0.726 | 0.718 | 0.742 | 0.740 | 0.660 | 0.732 | 0.675 | 0.746 | 0.749 |
| bottle | 96 | 0.465 | 0.431 | 0.402 | 0.448 | 0.459 | 0.482 | 0.428 | 0.445 | 0.502 | 0.503 |
| bus | 74 | 0.828 | 0.830 | 0.811 | 0.835 | 0.835 | 0.838 | 0.816 | 0.830 | 0.811 | 0.814 |
| car | 127 | 0.471 | 0.523 | 0.496 | 0.541 | 0.544 | 0.461 | 0.495 | 0.473 | 0.542 | 0.542 |
| cat | 119 | 0.829 | 0.856 | 0.833 | 0.874 | 0.875 | 0.846 | 0.864 | 0.845 | 0.898 | 0.900 |
| chair | 123 | 0.305 | 0.313 | 0.303 | 0.372 | 0.377 | 0.316 | 0.334 | 0.297 | 0.430 | 0.438 |
| cow | 71 | 0.827 | 0.870 | 0.868 | 0.880 | 0.879 | 0.854 | 0.874 | 0.876 | 0.891 | 0.892 |
| dining table | 75 | 0.463 | 0.451 | 0.437 | 0.472 | 0.477 | 0.476 | 0.454 | 0.462 | 0.471 | 0.475 |
| dog | 128 | 0.807 | 0.843 | 0.830 | 0.846 | 0.853 | 0.812 | 0.821 | 0.814 | 0.853 | 0.863 |
| horse | 79 | 0.765 | 0.831 | 0.819 | 0.827 | 0.840 | 0.785 | 0.856 | 0.824 | 0.881 | 0.883 |
| motorbike | 76 | 0.742 | 0.759 | 0.756 | 0.766 | 0.766 | 0.729 | 0.744 | 0.734 | 0.754 | 0.755 |
| person | 446 | 0.410 | 0.517 | 0.439 | 0.640 | 0.650 | 0.364 | 0.466 | 0.382 | 0.669 | 0.684 |
| potted plant | 85 | 0.253 | 0.288 | 0.254 | 0.294 | 0.295 | 0.294 | 0.333 | 0.282 | 0.365 | 0.370 |
| sheep | 57 | 0.745 | 0.861 | 0.857 | 0.871 | 0.872 | 0.846 | 0.885 | 0.884 | 0.879 | 0.880 |
| sofa | 90 | 0.671 | 0.709 | 0.649 | 0.735 | 0.743 | 0.640 | 0.714 | 0.640 | 0.736 | 0.749 |
| train | 84 | 0.733 | 0.756 | 0.738 | 0.776 | 0.784 | 0.720 | 0.741 | 0.725 | 0.767 | 0.772 |
| tv | 74 | 0.371 | 0.434 | 0.420 | 0.443 | 0.446 | 0.389 | 0.427 | 0.405 | 0.457 | 0.456 |

## 7. RW module weights (VOC train, all items; w = clip(2·(AUROC − 0.5), 0, 1) of the MK4 gate, object vs far)

| Source | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C AUROC | 0.587 | 0.480 | 0.783 | 0.851 | 0.950 | 0.951 | 0.928 | 0.918 | 0.488 | 0.494 | 0.434 |
| A AUROC | 0.633 | 0.487 | 0.820 | 0.898 | 0.965 | 0.971 | 0.957 | 0.915 | 0.466 | 0.469 | 0.430 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Mean gate value at γ = 1 over modules and pixels (val): C/MK4 0.479, C/SN 0.287, C/MK4-LR 0.400, C/MK4-RW 0.400, A/MK4 0.492, A/SN 0.296, A/MK4-LR 0.420, A/MK4-RW 0.421.

Run 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, full stack C 0.645 (uncapped hierarchy + MK4).

