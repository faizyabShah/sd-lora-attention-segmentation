# New reruns: components on the prompt-corrected generated sets (`PREREG.md`)

`scripts/jc_newrerun.py`. One live generation per image with the frozen runtime, corrected prompts (`configs/new_reruns_prompts.json`), safety checker off. Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold OOF on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap). Δ with image-bootstrap 95% interval (2000). Seeds: **CAu** = cross-attention, uniform layer weights; **CAg** = cross-attention, GOCA-style weights (BRACE seed); **NV** = NERVE CLIP seed (ViT-L/14); **SC** = SCLIP seed (ViT-B/16).

## Data checks

- VOC-Sim: 6 images, 0 with corrected prompt. Generated vs official PNG: median 0.25 grey levels; < 1 on 6, unresolved (≥ 2) 0 (). Capture vs gen step image: max |Δ| 0 on 6/6. Unchanged-prompt images: BRACE map vs stored paper capture median max |Δ| 1.3e-03. NERVE walk vs runs/17 (min–max), images < 1 grey level from official: median max |Δ| 2.0e-02.
- COCO-Cap: 6 images, 0 with corrected prompt. Generated vs official PNG: median 0.40 grey levels; < 1 on 5, unresolved (≥ 2) 0 (). Capture vs gen step image: max |Δ| 0 on 6/6. Unchanged-prompt images: BRACE map vs stored paper capture median max |Δ| 9.3e-03. NERVE walk vs runs/17 (min–max), images < 1 grey level from official: median max |Δ| 4.5e-02.

## Components, VOC-Sim (6 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.845 | 0.897 | 0.904 | 0.842 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.855 | 0.912 | 0.909 | 0.916 |
| NERVE CLIP seed (ViT-L/14) | 0.853 | 0.875 | 0.875 | 0.894 |
| SCLIP seed (ViT-B/16) | 0.756 | 0.771 | 0.856 | 0.779 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.712 | 0.860 | 0.904 | 0.836 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.698 | 0.862 | 0.895 | 0.845 |
| NERVE CLIP seed (ViT-L/14) | 0.802 | 0.877 | 0.878 | 0.888 |
| SCLIP seed (ViT-B/16) | 0.689 | 0.794 | 0.871 | 0.791 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.829 | 0.894 | 0.892 | 0.837 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.804 | 0.876 | 0.875 | 0.905 |
| NERVE CLIP seed (ViT-L/14) | 0.801 | 0.878 | 0.887 | 0.888 |
| SCLIP seed (ViT-B/16) | 0.702 | 0.749 | 0.866 | 0.761 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.696 | 0.861 | 0.897 | 0.845 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.687 | 0.864 | 0.890 | 0.842 |
| NERVE CLIP seed (ViT-L/14) | 0.775 | 0.873 | 0.868 | 0.886 |
| SCLIP seed (ViT-B/16) | 0.640 | 0.794 | 0.862 | 0.786 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.869 | 0.911 | 0.908 | 0.910 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.858 | 0.912 | 0.909 | 0.916 |
| NERVE CLIP seed (ViT-L/14) | 0.853 | 0.882 | 0.889 | 0.894 |
| SCLIP seed (ViT-B/16) | 0.756 | 0.858 | 0.871 | 0.827 |

## Components, COCO-Cap (6 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.625 | 0.708 | 0.746 | 0.729 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.646 | 0.771 | 0.780 | 0.766 |
| NERVE CLIP seed (ViT-L/14) | 0.637 | 0.722 | 0.776 | 0.723 |
| SCLIP seed (ViT-B/16) | 0.444 | 0.563 | 0.663 | 0.554 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.425 | 0.622 | 0.693 | 0.615 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.522 | 0.636 | 0.751 | 0.618 |
| NERVE CLIP seed (ViT-L/14) | 0.639 | 0.731 | 0.847 | 0.708 |
| SCLIP seed (ViT-B/16) | 0.477 | 0.619 | 0.663 | 0.555 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.549 | 0.665 | 0.673 | 0.700 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.560 | 0.723 | 0.744 | 0.742 |
| NERVE CLIP seed (ViT-L/14) | 0.579 | 0.705 | 0.745 | 0.709 |
| SCLIP seed (ViT-B/16) | 0.415 | 0.586 | 0.626 | 0.529 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.412 | 0.610 | 0.673 | 0.613 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.487 | 0.625 | 0.726 | 0.618 |
| NERVE CLIP seed (ViT-L/14) | 0.561 | 0.709 | 0.808 | 0.706 |
| SCLIP seed (ViT-B/16) | 0.427 | 0.601 | 0.634 | 0.546 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.676 | 0.745 | 0.797 | 0.745 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.666 | 0.777 | 0.790 | 0.802 |
| NERVE CLIP seed (ViT-L/14) | 0.646 | 0.748 | 0.849 | 0.753 |
| SCLIP seed (ViT-B/16) | 0.478 | 0.652 | 0.706 | 0.598 |

**Headline:** BRACE (CAg × hierarchy + DenseCRF, 0.55): VOC-Sim 0.895, COCO-Cap 0.751; NERVE (NV × walk, VOC-sel. + DenseCRF): VOC-Sim 0.894, COCO-Cap 0.723.

## Thresholds selected on full VOC-Sim (with DenseCRF)

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| CAu | 0.400 | 0.425 | 0.500 | 0.400 |
| CAg | 0.400 | 0.350 | 0.450 | 0.300 |
| NV | 0.700 | 0.650 | 0.650 | 0.625 |
| SC | 0.300 | 0.325 | 0.550 | 0.300 |
| hA | 0.700 | 0.450 | 0.525 | 0.400 |
| TG | 0.700 | 0.450 | 0.500 | 0.550 |
| TC | 0.700 | 0.450 | 0.500 | 0.550 |
| PW0.5 | 0.625 | 0.425 | 0.500 | 0.425 |
| PW1.5 | 0.300 | 0.325 | 0.425 | 0.300 |
| PW2 | 0.300 | 0.300 | 0.400 | 0.300 |
| PW3 | 0.300 | 0.300 | 0.350 | 0.300 |
| FL50 | 0.300 | 0.300 | 0.375 | 0.300 |
| FL70 | 0.300 | 0.300 | 0.300 | 0.300 |
| FL80 | 0.300 | 0.300 | 0.300 | 0.300 |
| FL90 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK5 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK10 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK20 | 0.300 | 0.300 | 0.325 | 0.300 |
| TK40 | 0.375 | 0.300 | 0.450 | 0.300 |
| FO | 0.300 | 0.300 | 0.350 | 0.300 |

## Registered contrasts

| ID | Contrast | VOC-Sim, VOC-sel. + CRF | COCO-Cap, VOC-sel. + CRF | VOC-Sim, 0.55 + CRF | COCO-Cap, 0.55 + CRF |
|---|---|---|---|---|---|
| F | fine-only − seed, CAu | +0.051 [+0.024, +0.078] | +0.083 [+0.036, +0.131] | +0.148 [+0.116, +0.182] | +0.197 [+0.053, +0.348] |
| F | fine-only − seed, CAg | +0.057 [+0.032, +0.082] | +0.125 [+0.001, +0.249] | +0.164 [+0.137, +0.193] | +0.114 [+0.010, +0.231] |
| F | fine-only − seed, NV | +0.022 [-0.010, +0.054] | +0.084 [+0.001, +0.163] | +0.075 [-0.012, +0.163] | +0.092 [-0.008, +0.191] |
| F | fine-only − seed, SC | +0.015 [-0.017, +0.047] | +0.119 [+0.004, +0.232] | +0.105 [+0.016, +0.195] | +0.142 [+0.036, +0.243] |
| H | hierarchy − fine-only, CAu | +0.007 [-0.001, +0.016] | +0.038 [-0.017, +0.093] | +0.043 [+0.017, +0.070] | +0.071 [+0.007, +0.137] |
| H | hierarchy − fine-only, CAg | -0.003 [-0.006, +0.001] | +0.009 [-0.004, +0.023] | +0.033 [+0.013, +0.052] | +0.116 [+0.010, +0.224] |
| H | hierarchy − fine-only, NV | +0.000 [-0.002, +0.003] | +0.054 [-0.002, +0.110] | +0.001 [-0.001, +0.003] | +0.116 [+0.022, +0.207] |
| H | hierarchy − fine-only, SC | +0.085 [+0.005, +0.166] | +0.100 [+0.018, +0.183] | +0.078 [+0.010, +0.147] | +0.044 [-0.007, +0.096] |
| G | CAg − CAu (GOCA weighting), seed alone | +0.010 [-0.016, +0.036] | +0.021 [-0.014, +0.054] | -0.015 [-0.062, +0.032] | +0.096 [+0.060, +0.139] |
| G | CAg − CAu (GOCA weighting), fine-only | +0.015 [+0.002, +0.027] | +0.063 [-0.070, +0.195] | +0.001 [-0.034, +0.036] | +0.013 [-0.012, +0.039] |
| G | CAg − CAu (GOCA weighting), BRACE hierarchy | +0.005 [-0.006, +0.016] | +0.034 [-0.050, +0.117] | -0.009 [-0.022, +0.005] | +0.058 [-0.006, +0.123] |
| C | NV − CAg, seed alone | -0.002 [-0.029, +0.024] | -0.009 [-0.153, +0.131] | +0.104 [+0.039, +0.170] | +0.117 [-0.120, +0.368] |
| C | NV − CAg, fine-only | -0.037 [-0.072, -0.001] | -0.049 [-0.220, +0.122] | +0.016 [-0.014, +0.046] | +0.095 [-0.016, +0.210] |
| C | NV − CAg, BRACE hierarchy | -0.034 [-0.066, -0.001] | -0.004 [-0.152, +0.143] | -0.017 [-0.044, +0.011] | +0.095 [+0.006, +0.183] |
| C | NV − CAg, NERVE walk | -0.022 [-0.053, +0.008] | -0.043 [-0.254, +0.151] | +0.043 [+0.010, +0.082] | +0.090 [-0.050, +0.254] |
| C | SC − CAg, seed alone | -0.099 [-0.204, +0.006] | -0.202 [-0.421, +0.018] | -0.009 [-0.198, +0.180] | -0.045 [-0.273, +0.193] |
| C | SC − CAg, fine-only | -0.141 [-0.234, -0.047] | -0.208 [-0.410, -0.006] | -0.068 [-0.171, +0.033] | -0.017 [-0.326, +0.230] |
| C | SC − CAg, BRACE hierarchy | -0.053 [-0.095, -0.015] | -0.116 [-0.395, +0.153] | -0.024 [-0.054, +0.007] | -0.088 [-0.370, +0.149] |
| C | SC − CAg, NERVE walk | -0.137 [-0.230, -0.045] | -0.211 [-0.414, -0.009] | -0.054 [-0.168, +0.060] | -0.064 [-0.417, +0.242] |
| R | DenseCRF − raw, hierarchy, CAu | +0.012 [-0.003, +0.029] | +0.073 [+0.031, +0.112] | +0.006 [-0.002, +0.016] | +0.020 [-0.000, +0.041] |
| R | DenseCRF − raw, hierarchy, CAg | +0.034 [+0.007, +0.061] | +0.036 [-0.005, +0.078] | +0.005 [-0.005, +0.017] | +0.025 [+0.005, +0.045] |
| R | DenseCRF − raw, hierarchy, NV | -0.012 [-0.035, +0.017] | +0.030 [+0.014, +0.048] | +0.010 [-0.006, +0.026] | +0.039 [+0.015, +0.062] |
| R | DenseCRF − raw, hierarchy, SC | -0.009 [-0.031, +0.013] | +0.038 [+0.014, +0.058] | +0.009 [-0.004, +0.022] | +0.029 [+0.005, +0.053] |
| N | NERVE (NV × walk) − BRACE (CAg × hierarchy) | -0.015 [-0.045, +0.017] | -0.057 [-0.260, +0.119] | -0.007 [-0.035, +0.022] | -0.043 [-0.199, +0.109] |
| N | BRACE hierarchy on NERVE's seed (NV × hierarchy) − NERVE | -0.019 [-0.023, -0.015] | +0.052 [-0.029, +0.136] | -0.009 [-0.016, -0.003] | +0.139 [+0.059, +0.223] |

Sensitivity, resolved images only (generated within 2 grey levels of the official PNG), VOC-sel. + CRF:

| Contrast | VOC-Sim | COCO-Cap |
|---|---|---|
| H: hierarchy − fine-only, CAu | +0.007 [-0.001, +0.016] | +0.038 [-0.017, +0.093] |
| H: hierarchy − fine-only, CAg | -0.003 [-0.006, +0.001] | +0.009 [-0.004, +0.023] |
| H: hierarchy − fine-only, NV | +0.000 [-0.002, +0.003] | +0.054 [-0.002, +0.110] |
| H: hierarchy − fine-only, SC | +0.085 [+0.005, +0.166] | +0.100 [+0.018, +0.183] |
| G: CAg − CAu (GOCA weighting), seed alone | +0.010 [-0.016, +0.036] | +0.021 [-0.014, +0.054] |
| G: CAg − CAu (GOCA weighting), fine-only | +0.015 [+0.002, +0.027] | +0.063 [-0.070, +0.195] |
| G: CAg − CAu (GOCA weighting), BRACE hierarchy | +0.005 [-0.006, +0.016] | +0.034 [-0.050, +0.117] |
| N: NERVE (NV × walk) − BRACE (CAg × hierarchy) | -0.015 [-0.045, +0.017] | -0.057 [-0.260, +0.119] |
| N: BRACE hierarchy on NERVE's seed (NV × hierarchy) − NERVE | -0.019 [-0.023, -0.015] | +0.052 [-0.029, +0.136] |

Component table on resolved images only (VOC-sel. + CRF):

| Seed | set | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---|---:|---:|---:|---:|
| CAu | VOC-Sim | 0.845 | 0.897 | 0.904 | 0.842 |
| CAg | VOC-Sim | 0.855 | 0.912 | 0.909 | 0.916 |
| NV | VOC-Sim | 0.853 | 0.875 | 0.875 | 0.894 |
| SC | VOC-Sim | 0.756 | 0.771 | 0.856 | 0.779 |
| CAu | COCO-Cap | 0.625 | 0.708 | 0.746 | 0.729 |
| CAg | COCO-Cap | 0.646 | 0.771 | 0.780 | 0.766 |
| NV | COCO-Cap | 0.637 | 0.722 | 0.776 | 0.723 |
| SC | COCO-Cap | 0.444 | 0.563 | 0.663 | 0.554 |

## Calibration arm (exploratory)

- PW: VOC-Sim OOF (hierarchy + CRF) PW0.5 0.908, PW1.5 0.908, PW2 0.909, PW3 0.910 → **PW3**
- FL: VOC-Sim OOF (hierarchy + CRF) FL50 0.909, FL70 0.887, FL80 0.861, FL90 0.826 → **FL50**
- TK: VOC-Sim OOF (hierarchy + CRF) TK5 0.820, TK10 0.851, TK20 0.910, TK40 0.912 → **TK40**

**fine-only, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.912 | 0.771 | 0.636 | 0.777 | +0.003 [-0.001, +0.006] | -0.009 [-0.023, +0.004] | -0.013 |
| hA: CAg ranking + per-image NERVE CLIP values | 0.907 | 0.803 | 0.743 | 0.803 | -0.002 [-0.009, +0.004] | +0.024 [-0.009, +0.057] | +0.013 |
| TG: global VOC template (CLIP offline) | 0.913 | 0.638 | 0.571 | 0.651 | +0.004 [+0.001, +0.007] | -0.142 [-0.301, +0.017] | -0.139 |
| TC: per-class VOC template (CLIP offline) | 0.913 | 0.535 | 0.542 | 0.545 | +0.004 [+0.001, +0.007] | -0.245 [-0.404, -0.038] | -0.245 |
| PW3: power s^3 (SD only) | 0.884 | 0.658 | 0.586 | 0.670 | -0.025 [-0.052, +0.002] | -0.122 [-0.293, +0.055] | -0.120 |
| FL50: percentile floor p50 (SD only) | 0.899 | 0.657 | 0.641 | 0.661 | -0.010 [-0.021, +0.001] | -0.123 [-0.299, +0.056] | -0.129 |
| TK40: top 40% (SD only) | 0.915 | 0.699 | 0.619 | 0.699 | +0.006 [+0.002, +0.009] | -0.081 [-0.194, +0.037] | -0.091 |
| FO: Otsu floor (SD only) | 0.888 | 0.667 | 0.595 | 0.675 | -0.021 [-0.047, +0.006] | -0.113 [-0.316, +0.095] | -0.116 |

**BRACE hierarchy, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.909 | 0.780 | 0.751 | 0.790 | — | — | — |
| hA: CAg ranking + per-image NERVE CLIP values | 0.908 | 0.795 | 0.795 | 0.795 | -0.001 [-0.006, +0.005] | +0.015 [-0.020, +0.050] | +0.005 |
| TG: global VOC template (CLIP offline) | 0.913 | 0.681 | 0.676 | 0.733 | +0.004 [-0.005, +0.014] | -0.099 [-0.245, +0.046] | -0.057 |
| TC: per-class VOC template (CLIP offline) | 0.913 | 0.678 | 0.625 | 0.678 | +0.004 [-0.005, +0.014] | -0.102 [-0.248, +0.042] | -0.112 |
| PW3: power s^3 (SD only) | 0.910 | 0.784 | 0.688 | 0.784 | +0.001 [-0.002, +0.005] | +0.004 [-0.094, +0.104] | -0.006 |
| FL50: percentile floor p50 (SD only) | 0.909 | 0.735 | 0.693 | 0.759 | -0.000 [-0.010, +0.009] | -0.044 [-0.140, +0.053] | -0.031 |
| TK40: top 40% (SD only) | 0.912 | 0.754 | 0.661 | 0.774 | +0.003 [-0.003, +0.010] | -0.025 [-0.101, +0.051] | -0.016 |
| FO: Otsu floor (SD only) | 0.911 | 0.722 | 0.675 | 0.756 | +0.002 [-0.002, +0.007] | -0.058 [-0.251, +0.129] | -0.034 |

**NERVE walk, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.916 | 0.766 | 0.618 | 0.802 | +0.007 [-0.000, +0.015] | -0.014 [-0.028, +0.000] | +0.012 |
| hA: CAg ranking + per-image NERVE CLIP values | 0.901 | 0.778 | 0.719 | 0.793 | -0.008 [-0.031, +0.016] | -0.002 [-0.010, +0.006] | +0.003 |
| TG: global VOC template (CLIP offline) | 0.901 | 0.555 | 0.555 | 0.572 | -0.008 [-0.030, +0.015] | -0.225 [-0.409, +0.003] | -0.218 |
| TC: per-class VOC template (CLIP offline) | 0.901 | 0.508 | 0.508 | 0.516 | -0.008 [-0.030, +0.015] | -0.271 [-0.452, -0.037] | -0.274 |
| PW3: power s^3 (SD only) | 0.811 | 0.651 | 0.410 | 0.651 | -0.098 [-0.148, -0.043] | -0.129 [-0.346, +0.113] | -0.139 |
| FL50: percentile floor p50 (SD only) | 0.804 | 0.651 | 0.602 | 0.674 | -0.105 [-0.185, -0.025] | -0.129 [-0.334, +0.083] | -0.116 |
| TK40: top 40% (SD only) | 0.909 | 0.665 | 0.614 | 0.665 | -0.000 [-0.011, +0.011] | -0.114 [-0.271, +0.050] | -0.125 |
| FO: Otsu floor (SD only) | 0.861 | 0.641 | 0.514 | 0.641 | -0.048 [-0.088, -0.006] | -0.139 [-0.389, +0.131] | -0.150 |

**Reading (hierarchy + CRF vs BRACE; helps = COCO Δ CI > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0):**

- hA (CAg ranking + per-image NERVE CLIP values): COCO +0.015 (lower -0.020), VOC -0.001, COCO best-thr. +0.005 → **does not help**
- TG (global VOC template (CLIP offline)): COCO -0.099 (lower -0.245), VOC +0.004, COCO best-thr. -0.057 → **does not help**
- TC (per-class VOC template (CLIP offline)): COCO -0.102 (lower -0.248), VOC +0.004, COCO best-thr. -0.112 → **does not help**
- PW3 (power s^3 (SD only)): COCO +0.004 (lower -0.094), VOC +0.001, COCO best-thr. -0.006 → **does not help**
- FL50 (percentile floor p50 (SD only)): COCO -0.044 (lower -0.140), VOC -0.000, COCO best-thr. -0.031 → **does not help**
- TK40 (top 40% (SD only)): COCO -0.025 (lower -0.101), VOC +0.003, COCO best-thr. -0.016 → **does not help**
- FO (Otsu floor (SD only)): COCO -0.058 (lower -0.251), VOC +0.002, COCO best-thr. -0.034 → **does not help**

**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**

| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAg | BRACE hierarchy | 0.780 | 0.562 | 0.919 | 0.858 | 0.167 | 0.1069 | 0.795 | 0.982 |
| CAg | NERVE walk | 0.766 | 0.525 | 0.912 | 0.860 | 0.167 | 0.1389 | 0.780 | 0.984 |
| NV | BRACE hierarchy | 0.776 | 0.737 | 0.898 | 0.692 | 0.000 | 0.0307 | 0.896 | 0.855 |
| NV | NERVE walk | 0.723 | 0.750 | 0.699 | 0.721 | 0.000 | 0.0285 | 0.901 | 0.792 |
| SC | BRACE hierarchy | 0.663 | 0.339 | 0.924 | 0.727 | 0.167 | 0.0355 | 0.723 | 0.755 |
| SC | NERVE walk | 0.554 | 0.108 | 0.878 | 0.677 | 0.333 | 0.1178 | 0.610 | 0.758 |
| hA | BRACE hierarchy | 0.795 | 0.603 | 0.924 | 0.859 | 0.000 | 0.0709 | 0.826 | 0.966 |
| hA | NERVE walk | 0.778 | 0.553 | 0.910 | 0.870 | 0.167 | 0.1083 | 0.795 | 0.981 |
| TG | BRACE hierarchy | 0.681 | 0.260 | 0.920 | 0.863 | 0.167 | 0.1402 | 0.689 | 0.988 |
| TG | NERVE walk | 0.555 | 0.278 | 0.870 | 0.517 | 0.167 | 0.1323 | 0.665 | 0.849 |
| TC | BRACE hierarchy | 0.678 | 0.260 | 0.917 | 0.858 | 0.167 | 0.1355 | 0.694 | 0.979 |
| TC | NERVE walk | 0.508 | 0.278 | 0.780 | 0.467 | 0.167 | 0.1439 | 0.631 | 0.831 |
| PW3 | BRACE hierarchy | 0.784 | 0.720 | 0.911 | 0.720 | 0.000 | 0.0494 | 0.843 | 0.928 |
| PW3 | NERVE walk | 0.651 | 0.758 | 0.850 | 0.344 | 0.000 | 0.0272 | 0.884 | 0.713 |
| FL50 | BRACE hierarchy | 0.735 | 0.645 | 0.911 | 0.651 | 0.000 | 0.0601 | 0.818 | 0.901 |
| FL50 | NERVE walk | 0.651 | 0.680 | 0.922 | 0.351 | 0.000 | 0.0440 | 0.827 | 0.771 |
| TK40 | BRACE hierarchy | 0.754 | 0.639 | 0.918 | 0.706 | 0.000 | 0.0588 | 0.829 | 0.913 |
| TK40 | NERVE walk | 0.665 | 0.636 | 0.910 | 0.450 | 0.000 | 0.0582 | 0.806 | 0.814 |
| FO | BRACE hierarchy | 0.722 | 0.769 | 0.911 | 0.484 | 0.000 | 0.0383 | 0.853 | 0.837 |
| FO | NERVE walk | 0.641 | 0.769 | 0.872 | 0.280 | 0.167 | 0.0255 | 0.875 | 0.699 |

## Per class, VOC-sel. + CRF (CAg × hierarchy / NV × walk / SC × hierarchy)

VOC-Sim: aeroplane 0.95 / 0.97 / 0.95, boat 0.95 / 0.97 / 0.97, car 0.92 / 0.83 / 0.76, cow 0.94 / 0.94 / 0.89, motorbike 0.74 / 0.67 / 0.64, sheep 0.96 / 0.98 / 0.92

COCO-Cap: aeroplane 0.87 / 0.44 / 0.88, bird 0.89 / 0.80 / 0.00, car 0.86 / 0.51 / 0.55, cow 0.85 / 0.94 / 0.90, horse 0.23 / 0.70 / 0.68, sheep 0.97 / 0.95 / 0.96

