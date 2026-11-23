# Run 38: SA weights × hierarchy × gate on VOC-Sim and COCO-Cap, everything fitted on dev (`PREREG.md`)

**Exploratory.** Test halves: VOC-Sim 3, COCO-Cap 5 images. Fitted on dev (VOC-Sim 5, COCO-Cap 3): thresholds per regime and map, gate strength γ per regime × SA weights × gate × map (grid [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]). RW weights from run 36 (dev, held out). Regime V: profiles and fits from VOC-Sim dev only (COCO-Cap is untuned transfer). Regime X: VOC-Sim + COCO-Cap dev. One generation-time capture for every map. Class-macro mIoU; 95% image-bootstrap intervals (50 samples, paired). DenseCRF at the dev threshold chosen on raw maps (run 37 rule).


## 1. Primary contrasts (test, + DenseCRF, dev-selected threshold and γ, SA weights on)

"Improves" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005 (run 36 rule).

| Regime | # | Contrast | Map | VOC-Sim Δ [95% CI] | COCO-Cap Δ [95% CI] | reading |
|---|---|---|---|---|---|---|
| X | P1 | MK4-LR − MK4 | capped hierarchy (V3) | -0.002 [-0.005, -0.000] (hurts) | -0.002 [-0.006, +0.001] (n.r.) | does not improve |
| X | P2 | MK4-LR − MK4 | fine-only (V0) | -0.045 [-0.136, +0.002] (n.r.) | +0.168 [-0.003, +0.285] (n.r.) | does not improve |
| X | P3 | MK4-RW − MK4 | capped hierarchy (V3) | +0.003 [-0.000, +0.008] (n.r.) | -0.002 [-0.005, +0.001] (n.r.) | does not improve |
| X | P4 | MK4-RW − MK4 | fine-only (V0) | -0.002 [-0.008, +0.002] (n.r.) | +0.160 [-0.009, +0.272] (n.r.) | does not improve |
| X | P5 | SN − MK4 | capped hierarchy (V3) | -0.129 [-0.374, +0.036] (n.r.) | -0.066 [-0.181, +0.010] (n.r.) | does not improve |
| X | P6 | SN − MK4 | fine-only (V0) | -0.043 [-0.126, -0.001] (hurts) | +0.118 [-0.132, +0.284] (n.r.) | does not improve |
| V | P1 | MK4-LR − MK4 | capped hierarchy (V3) | -0.152 [-0.493, +0.107] (n.r.) | -0.069 [-0.185, +0.008] (n.r.) | does not improve |
| V | P2 | MK4-LR − MK4 | fine-only (V0) | +0.083 [-0.147, +0.350] (n.r.) | +0.176 [+0.001, +0.295] (helps) | does not improve |
| V | P3 | MK4-RW − MK4 | capped hierarchy (V3) | -0.095 [-0.265, +0.016] (n.r.) | -0.051 [-0.136, +0.006] (n.r.) | does not improve |
| V | P4 | MK4-RW − MK4 | fine-only (V0) | +0.109 [-0.066, +0.349] (n.r.) | +0.180 [+0.000, +0.302] (helps) | does not improve |
| V | P5 | SN − MK4 | capped hierarchy (V3) | -0.212 [-0.570, +0.013] (n.r.) | -0.063 [-0.168, +0.007] (n.r.) | does not improve |
| V | P6 | SN − MK4 | fine-only (V0) | -0.083 [-0.222, +0.004] (n.r.) | +0.092 [-0.179, +0.280] (n.r.) | does not improve |

- Regime X, MK4-LR: does not hold (improves on V3 or V0: no; hurts somewhere: yes).
- Regime X, MK4-RW: does not hold (improves on V3 or V0: no; hurts somewhere: no).
- Regime X, SN: does not hold (improves on V3 or V0: no; hurts somewhere: yes).
- Regime V, MK4-LR: does not hold (improves on V3 or V0: no; hurts somewhere: no).
- Regime V, MK4-RW: does not hold (improves on V3 or V0: no; hurts somewhere: no).
- Regime V, SN: does not hold (improves on V3 or V0: no; hurts somewhere: no).

## 2. Dev-selected γ

| Regime | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|
| X |  | seed only | 0.75 | 0.75 | 2.0 | 2.0 |
| X |  | fine-only (V0) | 1.0 | 0.75 | 2.0 | 2.0 |
| X |  | capped hierarchy (V3) | 1.5 | 0.75 | 2.0 | 2.0 |
| X | ✓ | seed only | 1.5 | 0.75 | 2.0 | 2.0 |
| X | ✓ | fine-only (V0) | 2.0 | 0.25 | 2.0 | 2.0 |
| X | ✓ | capped hierarchy (V3) | 2.0 | 0.25 | 2.0 | 2.0 |
| V |  | seed only | 1.0 | 0.25 | 1.5 | 1.5 |
| V |  | fine-only (V0) | 1.5 | 0.25 | 1.0 | 2.0 |
| V |  | capped hierarchy (V3) | 0.25 | 0.25 | 0.5 | 0.25 |
| V | ✓ | seed only | 1.0 | 0.25 | 1.5 | 1.5 |
| V | ✓ | fine-only (V0) | 2.0 | 0.25 | 2.0 | 2.0 |
| V | ✓ | capped hierarchy (V3) | 1.5 | 0.25 | 0.25 | 0.25 |

## 3. Regime X (VOC-Sim + COCO-Cap dev profiles)

mIoU, VOC-Sim / COCO-Cap test, primary rule (seed only: raw, dev threshold), dev-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.892 / 0.822 | 0.714 / 0.699 | 0.620 / 0.845 | 0.898 / 0.847 | 0.928 / 0.844 |
|  | capped hierarchy (V3) | 0.736 / 0.833 | 0.726 / 0.708 | 0.660 / 0.834 | 0.890 / 0.877 | 0.896 / 0.877 |
|  | seed only | 0.596 / 0.558 | 0.666 / 0.582 | 0.539 / 0.601 | 0.722 / 0.691 | 0.757 / 0.695 |
| ✓ | fine-only (V0) | 0.932 / 0.673 | 0.922 / 0.707 | 0.879 / 0.825 | 0.877 / 0.876 | 0.921 / 0.867 |
| ✓ | capped hierarchy (V3) | 0.813 / 0.749 | 0.894 / 0.882 | 0.765 / 0.816 | 0.892 / 0.881 | 0.897 / 0.881 |
| ✓ | seed only | 0.598 / 0.526 | 0.690 / 0.637 | 0.537 / 0.612 | 0.702 / 0.680 | 0.725 / 0.686 |

Paired differences, VOC-Sim test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | -0.178 [-0.574, +0.112] | -0.094 [-0.284, +0.001] | +0.183 [-0.117, +0.591] | +0.213 [-0.027, +0.591] |
|  | capped hierarchy (V3) | -0.010 [-0.022, -0.005] | -0.066 [-0.145, -0.020] | +0.164 [-0.016, +0.452] | +0.171 [+0.000, +0.454] |
|  | seed only | +0.070 [-0.034, +0.206] | -0.127 [-0.297, -0.043] | +0.057 [-0.077, +0.232] | +0.092 [-0.058, +0.284] |
| ✓ | fine-only (V0) | -0.010 [-0.032, +0.002] | -0.043 [-0.126, -0.001] | -0.045 [-0.136, +0.002] | -0.002 [-0.008, +0.002] |
| ✓ | capped hierarchy (V3) | +0.082 [-0.060, +0.268] | -0.129 [-0.374, +0.036] | -0.002 [-0.005, -0.000] | +0.003 [-0.000, +0.008] |
| ✓ | seed only | +0.092 [+0.038, +0.177] | -0.153 [-0.254, -0.095] | +0.012 [+0.002, +0.030] | +0.036 [+0.030, +0.046] |

Paired differences, COCO-Cap test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | -0.123 [-0.294, +0.134] | +0.146 [-0.062, +0.295] | +0.147 [-0.006, +0.251] | +0.145 [-0.006, +0.247] |
|  | capped hierarchy (V3) | -0.125 [-0.296, +0.131] | +0.125 [-0.119, +0.289] | +0.169 [-0.008, +0.289] | +0.169 [-0.008, +0.289] |
|  | seed only | +0.024 [-0.101, +0.212] | +0.018 [-0.152, +0.131] | +0.108 [+0.025, +0.174] | +0.112 [+0.035, +0.176] |
| ✓ | fine-only (V0) | +0.034 [-0.068, +0.186] | +0.118 [-0.132, +0.284] | +0.168 [-0.003, +0.285] | +0.160 [-0.009, +0.272] |
| ✓ | capped hierarchy (V3) | +0.133 [+0.007, +0.235] | -0.066 [-0.181, +0.010] | -0.002 [-0.006, +0.001] | -0.002 [-0.005, +0.001] |
| ✓ | seed only | +0.111 [+0.020, +0.217] | -0.025 [-0.133, +0.047] | +0.043 [-0.002, +0.080] | +0.049 [+0.008, +0.084] |

Component effects with the gate fixed, VOC-Sim test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.040 [+0.008, +0.104] | +0.076 [+0.017, +0.177] | -0.156 [-0.424, +0.012] | -0.120 [-0.270, -0.031] |
| MK4 | +0.208 [-0.040, +0.589] | +0.169 [-0.005, +0.454] | +0.011 [-0.121, +0.137] | -0.028 [-0.086, +0.002] |
| SN | +0.259 [+0.058, +0.599] | +0.105 [+0.033, +0.229] | +0.039 [-0.006, +0.125] | -0.114 [-0.370, +0.076] |
| MK4-LR | -0.020 [-0.059, -0.000] | +0.003 [+0.001, +0.005] | -0.008 [-0.020, -0.001] | +0.015 [-0.001, +0.045] |
| MK4-RW | -0.007 [-0.021, +0.000] | +0.001 [+0.000, +0.002] | -0.031 [-0.094, +0.000] | -0.023 [-0.071, +0.000] |

Component effects with the gate fixed, COCO-Cap test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | -0.149 [-0.256, +0.005] | -0.084 [-0.146, -0.004] | +0.011 [-0.010, +0.038] | +0.076 [+0.003, +0.126] |
| MK4 | +0.008 [-0.000, +0.015] | +0.174 [-0.002, +0.294] | +0.009 [-0.006, +0.028] | +0.175 [-0.005, +0.302] |
| SN | -0.020 [-0.063, +0.008] | -0.018 [-0.060, +0.010] | -0.011 [-0.034, +0.017] | -0.009 [-0.026, +0.011] |
| MK4-LR | +0.029 [+0.003, +0.048] | +0.004 [+0.002, +0.007] | +0.030 [-0.004, +0.059] | +0.005 [-0.008, +0.017] |
| MK4-RW | +0.023 [-0.005, +0.042] | +0.004 [+0.002, +0.007] | +0.033 [-0.005, +0.061] | +0.014 [-0.007, +0.034] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule, VOC-Sim / COCO-Cap:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4, VOC-Sim | MK4-LR − MK4, COCO-Cap |
|---|---:|---:|---:|---:|---|---|
| fine-only (V0) | 0.920 / 0.741 | 0.751 / 0.813 | 0.883 / 0.875 | 0.921 / 0.872 | -0.038 [-0.115, +0.003] | +0.135 [-0.004, +0.230] |
| capped hierarchy (V3) | 0.898 / 0.883 | 0.877 / 0.842 | 0.904 / 0.879 | 0.900 / 0.881 | +0.006 [+0.000, +0.018] | -0.005 [-0.015, +0.002] |

## 4. Regime V (VOC-Sim dev profiles; COCO-Cap is transfer)

mIoU, VOC-Sim / COCO-Cap test, primary rule (seed only: raw, dev threshold), dev-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.716 / 0.729 | 0.732 / 0.687 | 0.718 / 0.683 | 0.937 / 0.822 | 0.921 / 0.841 |
|  | capped hierarchy (V3) | 0.699 / 0.827 | 0.702 / 0.855 | 0.686 / 0.813 | 0.849 / 0.880 | 0.703 / 0.874 |
|  | seed only | 0.571 / 0.536 | 0.622 / 0.528 | 0.574 / 0.501 | 0.723 / 0.696 | 0.753 / 0.682 |
| ✓ | fine-only (V0) | 0.943 / 0.638 | 0.812 / 0.684 | 0.730 / 0.776 | 0.895 / 0.860 | 0.921 / 0.864 |
| ✓ | capped hierarchy (V3) | 0.813 / 0.749 | 0.905 / 0.875 | 0.694 / 0.812 | 0.754 / 0.806 | 0.810 / 0.824 |
| ✓ | seed only | 0.573 / 0.520 | 0.641 / 0.615 | 0.577 / 0.557 | 0.699 / 0.676 | 0.718 / 0.667 |

Paired differences, VOC-Sim test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.015 [+0.001, +0.023] | -0.014 [-0.019, -0.004] | +0.206 [-0.044, +0.588] | +0.189 [-0.101, +0.593] |
|  | capped hierarchy (V3) | +0.003 [-0.003, +0.008] | -0.015 [-0.042, -0.001] | +0.147 [-0.049, +0.432] | +0.001 [-0.002, +0.005] |
|  | seed only | +0.051 [-0.003, +0.083] | -0.049 [-0.063, -0.022] | +0.101 [-0.127, +0.364] | +0.131 [-0.078, +0.379] |
| ✓ | fine-only (V0) | -0.130 [-0.339, -0.005] | -0.083 [-0.222, +0.004] | +0.083 [-0.147, +0.350] | +0.109 [-0.066, +0.349] |
| ✓ | capped hierarchy (V3) | +0.093 [-0.016, +0.258] | -0.212 [-0.570, +0.013] | -0.152 [-0.493, +0.107] | -0.095 [-0.265, +0.016] |
| ✓ | seed only | +0.068 [+0.052, +0.099] | -0.064 [-0.111, -0.040] | +0.058 [-0.070, +0.203] | +0.077 [-0.030, +0.210] |

Paired differences, COCO-Cap test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | -0.042 [-0.223, +0.230] | -0.004 [-0.244, +0.156] | +0.134 [-0.010, +0.234] | +0.154 [-0.003, +0.265] |
|  | capped hierarchy (V3) | +0.028 [-0.002, +0.072] | -0.042 [-0.100, -0.003] | +0.024 [-0.003, +0.065] | +0.019 [-0.003, +0.046] |
|  | seed only | -0.007 [-0.102, +0.135] | -0.028 [-0.126, +0.038] | +0.167 [+0.054, +0.245] | +0.154 [+0.047, +0.227] |
| ✓ | fine-only (V0) | +0.046 [-0.038, +0.166] | +0.092 [-0.179, +0.280] | +0.176 [+0.001, +0.295] | +0.180 [+0.000, +0.302] |
| ✓ | capped hierarchy (V3) | +0.126 [-0.003, +0.217] | -0.063 [-0.168, +0.007] | -0.069 [-0.185, +0.008] | -0.051 [-0.136, +0.006] |
| ✓ | seed only | +0.094 [+0.018, +0.186] | -0.057 [-0.155, +0.008] | +0.062 [-0.005, +0.109] | +0.053 [-0.012, +0.099] |

Component effects with the gate fixed, VOC-Sim test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.226 [-0.017, +0.587] | +0.114 [-0.031, +0.324] | -0.017 [-0.086, +0.017] | -0.130 [-0.267, -0.051] |
| MK4 | +0.081 [-0.033, +0.244] | +0.204 [-0.055, +0.585] | -0.030 [-0.084, -0.000] | +0.093 [-0.105, +0.340] |
| SN | +0.012 [-0.024, +0.030] | +0.008 [+0.001, +0.016] | -0.032 [-0.120, +0.013] | -0.036 [-0.096, -0.006] |
| MK4-LR | -0.042 [-0.136, +0.007] | -0.095 [-0.340, +0.101] | -0.088 [-0.166, -0.044] | -0.141 [-0.503, +0.149] |
| MK4-RW | +0.001 [+0.000, +0.001] | +0.108 [-0.037, +0.315] | -0.218 [-0.589, +0.016] | -0.111 [-0.276, -0.012] |

Component effects with the gate fixed, COCO-Cap test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | -0.091 [-0.222, +0.071] | -0.078 [-0.135, +0.005] | +0.098 [+0.024, +0.168] | +0.111 [+0.013, +0.186] |
| MK4 | -0.003 [-0.009, +0.003] | +0.020 [-0.005, +0.058] | +0.168 [-0.036, +0.324] | +0.191 [-0.000, +0.327] |
| SN | +0.093 [+0.011, +0.154] | -0.001 [-0.010, +0.005] | +0.130 [+0.029, +0.198] | +0.036 [+0.011, +0.061] |
| MK4-LR | +0.038 [+0.000, +0.066] | -0.074 [-0.192, +0.006] | +0.058 [+0.002, +0.098] | -0.055 [-0.153, +0.017] |
| MK4-RW | +0.023 [-0.012, +0.048] | -0.050 [-0.124, -0.000] | +0.033 [-0.002, +0.062] | -0.040 [-0.109, +0.030] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule, VOC-Sim / COCO-Cap:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4, VOC-Sim | MK4-LR − MK4, COCO-Cap |
|---|---:|---:|---:|---:|---|---|
| fine-only (V0) | 0.925 / 0.712 | 0.717 / 0.767 | 0.937 / 0.856 | 0.940 / 0.855 | +0.011 [+0.007, +0.019] | +0.143 [-0.030, +0.259] |
| capped hierarchy (V3) | 0.843 / 0.874 | 0.685 / 0.801 | 0.910 / 0.876 | 0.911 / 0.874 | +0.066 [-0.002, +0.177] | +0.002 [-0.000, +0.004] |

## 5. Other rules, SA weights on (dev-selected γ), VOC-Sim / COCO-Cap

| Regime | Map | Gate | + CRF, dev thr (primary) | + CRF, 0.55 | raw, dev thr | raw, 0.55 |
|---|---|---|---:|---:|---:|---:|
| X | fine-only (V0) | none | 0.932 / 0.673 | 0.820 / 0.767 | 0.869 / 0.635 | 0.796 / 0.755 |
| X | fine-only (V0) | MK4 | 0.922 / 0.707 | 0.801 / 0.848 | 0.858 / 0.692 | 0.781 / 0.816 |
| X | fine-only (V0) | SN | 0.879 / 0.825 | 0.806 / 0.815 | 0.824 / 0.782 | 0.783 / 0.812 |
| X | fine-only (V0) | MK4-LR | 0.877 / 0.876 | 0.794 / 0.857 | 0.823 / 0.832 | 0.775 / 0.866 |
| X | fine-only (V0) | MK4-RW | 0.921 / 0.867 | 0.805 / 0.859 | 0.859 / 0.808 | 0.786 / 0.866 |
| X | capped hierarchy (V3) | none | 0.813 / 0.749 | 0.903 / 0.754 | 0.763 / 0.704 | 0.836 / 0.722 |
| X | capped hierarchy (V3) | MK4 | 0.894 / 0.882 | 0.901 / 0.880 | 0.849 / 0.848 | 0.850 / 0.832 |
| X | capped hierarchy (V3) | SN | 0.765 / 0.816 | 0.816 / 0.818 | 0.708 / 0.776 | 0.764 / 0.791 |
| X | capped hierarchy (V3) | MK4-LR | 0.892 / 0.881 | 0.900 / 0.878 | 0.849 / 0.865 | 0.853 / 0.851 |
| X | capped hierarchy (V3) | MK4-RW | 0.897 / 0.881 | 0.903 / 0.878 | 0.854 / 0.865 | 0.857 / 0.851 |
| V | fine-only (V0) | none | 0.943 / 0.638 | 0.820 / 0.767 | 0.865 / 0.599 | 0.796 / 0.755 |
| V | fine-only (V0) | MK4 | 0.812 / 0.684 | 0.790 / 0.706 | 0.747 / 0.655 | 0.770 / 0.719 |
| V | fine-only (V0) | SN | 0.730 / 0.776 | 0.809 / 0.807 | 0.688 / 0.697 | 0.786 / 0.801 |
| V | fine-only (V0) | MK4-LR | 0.895 / 0.860 | 0.780 / 0.853 | 0.828 / 0.801 | 0.763 / 0.862 |
| V | fine-only (V0) | MK4-RW | 0.921 / 0.864 | 0.790 / 0.846 | 0.855 / 0.800 | 0.774 / 0.851 |
| V | capped hierarchy (V3) | none | 0.813 / 0.749 | 0.903 / 0.754 | 0.763 / 0.704 | 0.836 / 0.722 |
| V | capped hierarchy (V3) | MK4 | 0.905 / 0.875 | 0.903 / 0.880 | 0.837 / 0.815 | 0.849 / 0.833 |
| V | capped hierarchy (V3) | SN | 0.694 / 0.812 | 0.812 / 0.818 | 0.655 / 0.758 | 0.762 / 0.789 |
| V | capped hierarchy (V3) | MK4-LR | 0.754 / 0.806 | 0.905 / 0.832 | 0.703 / 0.737 | 0.848 / 0.800 |
| V | capped hierarchy (V3) | MK4-RW | 0.810 / 0.824 | 0.903 / 0.831 | 0.764 / 0.777 | 0.837 / 0.796 |

## 6. Per class IoU (primary rule, SA weights on, dev-selected γ)


Regime X, VOC-Sim test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bird | 1 | 0.642 | 0.943 | 0.522 | 0.943 | 0.943 | 0.939 | 0.942 | 0.939 | 0.944 | 0.944 |
| cat | 1 | 0.841 | 0.781 | 0.817 | 0.776 | 0.789 | 0.899 | 0.867 | 0.741 | 0.731 | 0.860 |
| horse | 1 | 0.954 | 0.959 | 0.956 | 0.959 | 0.959 | 0.959 | 0.958 | 0.957 | 0.957 | 0.958 |

Regime X, COCO-Cap test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 1 | 0.779 | 0.837 | 0.785 | 0.834 | 0.835 | 0.727 | 0.775 | 0.762 | 0.789 | 0.765 |
| cat | 1 | 0.900 | 0.911 | 0.899 | 0.910 | 0.909 | 0.899 | 0.900 | 0.903 | 0.904 | 0.896 |
| cow | 1 | 0.777 | 0.727 | 0.770 | 0.728 | 0.728 | 0.777 | 0.750 | 0.761 | 0.742 | 0.752 |
| person | 1 | 0.624 | 0.960 | 0.959 | 0.963 | 0.963 | 0.298 | 0.121 | 0.959 | 0.961 | 0.940 |
| sofa | 1 | 0.665 | 0.976 | 0.665 | 0.968 | 0.968 | 0.664 | 0.988 | 0.738 | 0.982 | 0.980 |

Regime V, VOC-Sim test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bird | 1 | 0.642 | 0.932 | 0.291 | 0.378 | 0.634 | 0.930 | 0.549 | 0.299 | 0.943 | 0.942 |
| cat | 1 | 0.841 | 0.825 | 0.838 | 0.932 | 0.841 | 0.940 | 0.930 | 0.934 | 0.783 | 0.864 |
| horse | 1 | 0.954 | 0.959 | 0.953 | 0.951 | 0.956 | 0.958 | 0.958 | 0.957 | 0.958 | 0.958 |

Regime V, COCO-Cap test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 1 | 0.779 | 0.791 | 0.761 | 0.727 | 0.781 | 0.680 | 0.688 | 0.653 | 0.712 | 0.721 |
| cat | 1 | 0.900 | 0.903 | 0.895 | 0.889 | 0.900 | 0.867 | 0.888 | 0.833 | 0.892 | 0.886 |
| cow | 1 | 0.777 | 0.753 | 0.780 | 0.788 | 0.773 | 0.786 | 0.768 | 0.789 | 0.768 | 0.771 |
| person | 1 | 0.624 | 0.959 | 0.961 | 0.962 | 0.960 | 0.191 | 0.088 | 0.941 | 0.946 | 0.957 |
| sofa | 1 | 0.665 | 0.970 | 0.664 | 0.663 | 0.707 | 0.665 | 0.989 | 0.665 | 0.984 | 0.984 |

## 7. RW module weights used (run 36, all dev of the regime; MK4 gate AUROC object vs far)

| Regime | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X AUROC | 0.630 | 0.464 | 0.844 | 0.935 | 0.982 | 0.988 | 0.961 | 0.966 | 0.568 | 0.604 | 0.528 |
| X weight | 0.26 | 0.00 | 0.69 | 0.87 | 0.96 | 0.98 | 0.92 | 0.93 | 0.14 | 0.21 | 0.06 |
| V AUROC | 0.650 | 0.460 | 0.902 | 0.947 | 0.987 | 0.994 | 0.970 | 0.964 | 0.594 | 0.683 | 0.593 |
| V weight | 0.30 | 0.00 | 0.80 | 0.89 | 0.97 | 0.99 | 0.94 | 0.93 | 0.19 | 0.37 | 0.19 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Mean gate value at γ = 1 over modules and pixels (test, VOC-Sim / COCO-Cap): X/MK4 0.513 / 0.504, X/SN 0.328 / 0.277, X/MK4-LR 0.495 / 0.430, X/MK4-RW 0.491 / 0.446, V/MK4 0.478 / 0.407, V/SN 0.325 / 0.237, V/MK4-LR 0.445 / 0.331, V/MK4-RW 0.448 / 0.355.

BRACE (W on, no gate, V3 + DenseCRF) is the 'none / capped hierarchy / ✓' cell. BRACE stays frozen; any change to the paper's method is a separate owner decision.

