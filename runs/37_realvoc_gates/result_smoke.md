# Run 37: run 36 gates on real PASCAL VOC photos (`PREREG.md`)

**Exploratory; third look at VOC val.** VOC 2012 val: 12 photos, 12 items; thresholds chosen on VOC train (12 items, raw maps). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals (50 samples, paired). Gate strengths fixed from run 36 regime X: MK4 γ = 1.000, SN γ = 0.532, MK4-LR γ = 0.716, MK4-RW γ = 0.765. MK4-RW weights from run 36 regime X.


## 1. Primary contrasts (val, + DenseCRF, train-selected threshold; source C, SA weights on)

| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − MK4 | capped hierarchy (V3) | -0.004 [-0.041, +0.025] | not resolved |
| P2 | MK4-LR − MK4 | fine-only (V0) | +0.012 [-0.015, +0.038] | not resolved |
| P3 | MK4-RW − MK4 | capped hierarchy (V3) | -0.003 [-0.036, +0.022] | not resolved |
| P4 | MK4-RW − MK4 | fine-only (V0) | +0.021 [-0.006, +0.048] | not resolved |
| P5 | SN − MK4 | capped hierarchy (V3) | -0.011 [-0.059, +0.021] | not resolved |
| P6 | SN − MK4 | fine-only (V0) | -0.020 [-0.063, +0.024] | not resolved |

- MK4-LR: does not hold on real photos (helps on V3 or V0, hurts on neither).
- MK4-RW: does not hold on real photos (helps on V3 or V0, hurts on neither).
- SN: does not hold on real photos (helps on V3 or V0, hurts on neither).

## 2. Source C (real COCO photos, captions; label-free)

mIoU, primary rule (seed only: raw, train threshold):

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.647 | 0.754 | 0.645 | 0.694 | 0.750 |
|  | capped hierarchy (V3) | 0.733 | 0.716 | 0.675 | 0.733 | 0.737 |
|  | uncapped hierarchy (V1) | 0.655 | 0.685 | 0.682 | 0.698 | 0.704 |
|  | seed only | 0.607 | 0.583 | 0.618 | 0.698 | 0.704 |
| ✓ | fine-only (V0) | 0.695 | 0.710 | 0.690 | 0.722 | 0.731 |
| ✓ | capped hierarchy (V3) | 0.756 | 0.733 | 0.722 | 0.729 | 0.730 |
| ✓ | uncapped hierarchy (V1) | 0.638 | 0.739 | 0.725 | 0.705 | 0.710 |
| ✓ | seed only | 0.608 | 0.592 | 0.618 | 0.675 | 0.678 |

Paired differences (primary rule; seed only: raw):

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.106 [+0.031, +0.191] | -0.108 [-0.181, -0.026] | -0.060 [-0.128, +0.008] | -0.004 [-0.040, +0.033] |
|  | capped hierarchy (V3) | -0.016 [-0.056, +0.039] | -0.042 [-0.130, +0.031] | +0.017 [-0.035, +0.060] | +0.020 [-0.028, +0.059] |
|  | uncapped hierarchy (V1) | +0.029 [-0.081, +0.164] | -0.003 [-0.114, +0.104] | +0.014 [-0.070, +0.080] | +0.019 [-0.058, +0.080] |
|  | seed only | -0.024 [-0.112, +0.040] | +0.035 [+0.010, +0.069] | +0.114 [+0.070, +0.184] | +0.121 [+0.073, +0.194] |
| ✓ | fine-only (V0) | +0.015 [-0.031, +0.067] | -0.020 [-0.063, +0.024] | +0.012 [-0.015, +0.038] | +0.021 [-0.006, +0.048] |
| ✓ | capped hierarchy (V3) | -0.023 [-0.054, +0.009] | -0.011 [-0.059, +0.021] | -0.004 [-0.041, +0.025] | -0.003 [-0.036, +0.022] |
| ✓ | uncapped hierarchy (V1) | +0.100 [+0.001, +0.208] | -0.014 [-0.061, +0.021] | -0.033 [-0.068, -0.002] | -0.029 [-0.059, -0.002] |
| ✓ | seed only | -0.015 [-0.112, +0.055] | +0.026 [+0.004, +0.071] | +0.082 [+0.047, +0.148] | +0.086 [+0.050, +0.153] |

Component effects for MK4-LR (primary rule): SA weights and hierarchy, with the gate fixed:

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.048 [+0.011, +0.088] | +0.023 [+0.002, +0.048] | +0.085 [+0.026, +0.156] | +0.061 [+0.004, +0.121] |
| MK4 | -0.044 [-0.070, -0.020] | +0.017 [+0.009, +0.024] | -0.037 [-0.086, +0.010] | +0.023 [-0.023, +0.085] |
| MK4-LR | +0.028 [-0.002, +0.057] | -0.004 [-0.014, +0.007] | +0.040 [-0.022, +0.103] | +0.007 [-0.042, +0.056] |

## 3. Source A (VOC train labels; reference)

mIoU, primary rule (seed only: raw, train threshold):

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.647 | 0.657 | 0.577 | 0.666 | 0.694 |
|  | capped hierarchy (V3) | 0.733 | 0.665 | 0.659 | 0.659 | 0.674 |
|  | uncapped hierarchy (V1) | 0.655 | 0.668 | 0.666 | 0.631 | 0.643 |
|  | seed only | 0.607 | 0.577 | 0.627 | 0.676 | 0.681 |
| ✓ | fine-only (V0) | 0.695 | 0.667 | 0.627 | 0.679 | 0.681 |
| ✓ | capped hierarchy (V3) | 0.756 | 0.676 | 0.718 | 0.707 | 0.704 |
| ✓ | uncapped hierarchy (V1) | 0.638 | 0.677 | 0.724 | 0.671 | 0.671 |
| ✓ | seed only | 0.608 | 0.610 | 0.637 | 0.661 | 0.663 |

Paired differences (primary rule; seed only: raw):

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.010 [-0.107, +0.127] | -0.080 [-0.157, -0.021] | +0.008 [-0.045, +0.077] | +0.037 [+0.012, +0.078] |
|  | capped hierarchy (V3) | -0.068 [-0.138, +0.020] | -0.005 [-0.101, +0.100] | -0.005 [-0.064, +0.056] | +0.009 [-0.034, +0.052] |
|  | uncapped hierarchy (V1) | +0.012 [-0.110, +0.133] | -0.002 [-0.096, +0.110] | -0.037 [-0.103, +0.040] | -0.025 [-0.077, +0.035] |
|  | seed only | -0.031 [-0.125, +0.050] | +0.050 [+0.031, +0.076] | +0.100 [+0.059, +0.151] | +0.105 [+0.061, +0.161] |
| ✓ | fine-only (V0) | -0.028 [-0.111, +0.063] | -0.040 [-0.082, -0.003] | +0.012 [-0.011, +0.038] | +0.014 [-0.005, +0.036] |
| ✓ | capped hierarchy (V3) | -0.080 [-0.151, -0.009] | +0.042 [-0.043, +0.116] | +0.031 [-0.018, +0.071] | +0.028 [-0.014, +0.062] |
| ✓ | uncapped hierarchy (V1) | +0.038 [-0.094, +0.156] | +0.047 [-0.035, +0.123] | -0.006 [-0.041, +0.026] | -0.006 [-0.037, +0.023] |
| ✓ | seed only | +0.002 [-0.092, +0.082] | +0.027 [+0.006, +0.056] | +0.051 [+0.023, +0.094] | +0.053 [+0.023, +0.098] |

Component effects for MK4-LR (primary rule): SA weights and hierarchy, with the gate fixed:

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.048 [+0.011, +0.088] | +0.023 [+0.002, +0.048] | +0.085 [+0.026, +0.156] | +0.061 [+0.004, +0.121] |
| MK4 | +0.010 [+0.001, +0.024] | +0.011 [+0.001, +0.022] | +0.008 [-0.017, +0.034] | +0.009 [-0.016, +0.037] |
| MK4-LR | +0.013 [-0.027, +0.047] | +0.047 [+0.010, +0.078] | -0.006 [-0.039, +0.026] | +0.028 [-0.010, +0.067] |

## 4. Other rules, source C, SA weights on

| Map | Gate | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 | train thr |
|---|---|---:|---:|---:|---:|---:|
| fine-only (V0) | none | 0.695 | 0.704 | 0.713 | 0.717 | 0.575 |
| fine-only (V0) | MK4 | 0.710 | 0.750 | 0.734 | 0.763 | 0.575 |
| fine-only (V0) | SN | 0.690 | 0.733 | 0.713 | 0.741 | 0.600 |
| fine-only (V0) | MK4-LR | 0.722 | 0.722 | 0.741 | 0.741 | 0.550 |
| fine-only (V0) | MK4-RW | 0.731 | 0.731 | 0.750 | 0.750 | 0.550 |
| capped hierarchy (V3) | none | 0.756 | 0.758 | 0.753 | 0.750 | 0.575 |
| capped hierarchy (V3) | MK4 | 0.733 | 0.771 | 0.756 | 0.775 | 0.650 |
| capped hierarchy (V3) | SN | 0.722 | 0.767 | 0.739 | 0.764 | 0.675 |
| capped hierarchy (V3) | MK4-LR | 0.729 | 0.791 | 0.747 | 0.790 | 0.650 |
| capped hierarchy (V3) | MK4-RW | 0.730 | 0.791 | 0.749 | 0.790 | 0.650 |

MK4-LR − MK4 under the other rules (source C, SA weights on):

| Map | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 |
|---|---|---|---|---|
| fine-only (V0) | +0.012 [-0.015, +0.038] | -0.028 [-0.063, +0.008] | +0.007 [-0.013, +0.028] | -0.022 [-0.052, +0.006] |
| capped hierarchy (V3) | -0.004 [-0.041, +0.025] | +0.020 [-0.005, +0.037] | -0.009 [-0.046, +0.021] | +0.014 [-0.007, +0.029] |

## 5. Per class IoU (primary rule, source C, SA weights on)

| Class | items | none V3 | MK4 V3 | SN V3 | LR V3 | RW V3 | MK4 V0 | LR V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 1 | 0.789 | 0.567 | 0.786 | 0.593 | 0.598 | 0.779 | 0.807 |
| bicycle | 1 | 0.292 | 0.347 | 0.331 | 0.341 | 0.338 | 0.444 | 0.427 |
| bus | 1 | 0.836 | 0.826 | 0.823 | 0.835 | 0.834 | 0.866 | 0.866 |
| cat | 1 | 0.979 | 0.960 | 0.956 | 0.959 | 0.960 | 0.960 | 0.969 |
| chair | 1 | 0.286 | 0.210 | 0.236 | 0.348 | 0.331 | 0.204 | 0.242 |
| dog | 2 | 0.944 | 0.866 | 0.804 | 0.870 | 0.865 | 0.947 | 0.947 |
| motorbike | 1 | 0.873 | 0.895 | 0.886 | 0.903 | 0.903 | 0.767 | 0.889 |
| person | 1 | 0.834 | 0.957 | 0.935 | 0.910 | 0.911 | 0.690 | 0.799 |
| potted plant | 1 | 0.933 | 0.904 | 0.904 | 0.892 | 0.897 | 0.688 | 0.644 |
| train | 1 | 0.825 | 0.841 | 0.842 | 0.877 | 0.878 | 0.838 | 0.860 |
| tv | 1 | 0.724 | 0.695 | 0.439 | 0.493 | 0.519 | 0.626 | 0.493 |

Mean gate value over modules and pixels (val): C/MK4 0.472, C/SN 0.499, C/MK4-LR 0.516, C/MK4-RW 0.510, A/MK4 0.469, A/SN 0.500, A/MK4-LR 0.503, A/MK4-RW 0.501.

Run 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, SA weights + uncapped hierarchy 0.610, full stack A 0.646, full stack C 0.645 (uncapped hierarchy + MK4).

