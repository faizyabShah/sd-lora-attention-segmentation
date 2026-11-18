# Relative-inconsistency pixel gate: 30 + 30 pilot (`PREREG.md`)

`scripts/jc_relgate.py`. Images: crc32 order positions 10–39 per set (VOC-Sim 1, COCO-Cap 1), one live capture each, compared within this capture only. Class-macro mIoU. *VOC30-sel.* = threshold best on these VOC-Sim images (in-sample on VOC-Sim, applied unchanged to COCO-Cap). Paired image-bootstrap 95% intervals (50). Small-sample exploratory pilot.

- VOC-Sim: reconstruction vs runtime seed max |Δ| 8.9e-07.
- COCO-Cap: reconstruction vs runtime seed max |Δ| 1.1e-06.

## Mechanism diagnostics (mean over images)

| quantity | VOC-Sim abs | VOC-Sim rel | COCO-Cap abs | COCO-Cap rel |
|---|---:|---:|---:|---:|
| M1 ρ(r, A), w-weighted | -0.245 | +0.265 | -0.156 | +0.116 |
| M2 8+16-res share inside object (global 0.834 / 0.820) | 0.748 | 0.786 | 0.807 | 0.855 |
| r, 64-res layers, interior | 0.361 | 0.459 | 0.399 | 0.324 |
| r, 64-res layers, near | 0.370 | 0.457 | 0.397 | 0.315 |
| r, 64-res layers, far | 0.435 | 0.511 | 0.455 | 0.342 |
| r, 16-res layers, interior | 0.239 | 0.362 | 0.374 | 0.431 |
| r, 16-res layers, near | 0.353 | 0.307 | 0.494 | 0.487 |
| r, 16-res layers, far | 0.443 | 0.290 | 0.450 | 0.441 |
| r, 8-res layers, interior | 0.221 | 0.341 | 0.381 | 0.414 |
| r, 8-res layers, near | 0.394 | 0.321 | 0.506 | 0.388 |
| r, 8-res layers, far | 0.428 | 0.202 | 0.443 | 0.244 |

| seed (min–max) | VOC-Sim mass on GT | interior | near | far | COCO-Cap mass on GT | interior | near | far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BR | 0.702 | 0.683 | 0.286 | 0.085 | 0.875 | 0.665 | 0.290 | 0.169 |
| Gabs | 0.627 | 0.605 | 0.252 | 0.124 | 0.858 | 0.553 | 0.250 | 0.182 |
| Grel | 0.650 | 0.681 | 0.233 | 0.127 | 0.860 | 0.595 | 0.264 | 0.191 |

**VOC-Sim, VOC30-sel. threshold, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.904 | 0.920 | 0.938 | 0.929 |
| UN: uniform layer weights | 0.907 | 0.921 | 0.938 | 0.925 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.910 | 0.899 | 0.939 | 0.926 |
| Gabsx: absolute gate, shuffled | 0.901 | 0.919 | 0.938 | 0.929 |
| Grel: pixel gate, relative inconsistency | 0.919 | 0.927 | 0.939 | 0.921 |
| Grelx: relative gate, shuffled | 0.892 | 0.919 | 0.937 | 0.929 |

**COCO-Cap, VOC30-sel. threshold, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.939 | 0.869 | 0.867 | 0.930 |
| UN: uniform layer weights | 0.915 | 0.865 | 0.866 | 0.927 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.847 | 0.938 | 0.979 | 0.931 |
| Gabsx: absolute gate, shuffled | 0.903 | 0.887 | 0.868 | 0.970 |
| Grel: pixel gate, relative inconsistency | 0.860 | 0.948 | 0.968 | 0.909 |
| Grelx: relative gate, shuffled | 0.894 | 0.934 | 0.868 | 0.983 |

**VOC-Sim, VOC30-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.887 | 0.905 | 0.923 | 0.923 |
| UN: uniform layer weights | 0.892 | 0.906 | 0.925 | 0.920 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.881 | 0.876 | 0.928 | 0.912 |
| Gabsx: absolute gate, shuffled | 0.880 | 0.906 | 0.923 | 0.925 |
| Grel: pixel gate, relative inconsistency | 0.907 | 0.910 | 0.925 | 0.905 |
| Grelx: relative gate, shuffled | 0.863 | 0.905 | 0.921 | 0.923 |

**COCO-Cap, VOC30-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.927 | 0.952 | 0.869 | 0.966 |
| UN: uniform layer weights | 0.843 | 0.871 | 0.867 | 0.930 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.813 | 0.926 | 0.969 | 0.933 |
| Gabsx: absolute gate, shuffled | 0.762 | 0.960 | 0.869 | 0.974 |
| Grel: pixel gate, relative inconsistency | 0.835 | 0.949 | 0.973 | 0.920 |
| Grelx: relative gate, shuffled | 0.795 | 0.962 | 0.869 | 0.977 |

**VOC-Sim, fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.818 | 0.814 | 0.872 | 0.854 |
| UN: uniform layer weights | 0.744 | 0.776 | 0.857 | 0.796 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.617 | 0.666 | 0.807 | 0.747 |
| Gabsx: absolute gate, shuffled | 0.555 | 0.824 | 0.886 | 0.850 |
| Grel: pixel gate, relative inconsistency | 0.741 | 0.700 | 0.818 | 0.761 |
| Grelx: relative gate, shuffled | 0.202 | 0.838 | 0.883 | 0.905 |

**COCO-Cap, fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.675 | 0.960 | 0.982 | 0.979 |
| UN: uniform layer weights | 0.455 | 0.980 | 0.976 | 0.980 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.483 | 0.936 | 0.945 | 0.926 |
| Gabsx: absolute gate, shuffled | 0.020 | 0.942 | 0.981 | 0.969 |
| Grel: pixel gate, relative inconsistency | 0.619 | 0.935 | 0.953 | 0.927 |
| Grelx: relative gate, shuffled | 0.017 | 0.920 | 0.981 | 0.960 |

**VOC-Sim, fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.808 | 0.805 | 0.863 | 0.858 |
| UN: uniform layer weights | 0.716 | 0.781 | 0.852 | 0.802 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.626 | 0.699 | 0.810 | 0.769 |
| Gabsx: absolute gate, shuffled | 0.529 | 0.814 | 0.865 | 0.855 |
| Grel: pixel gate, relative inconsistency | 0.752 | 0.718 | 0.821 | 0.772 |
| Grelx: relative gate, shuffled | 0.295 | 0.847 | 0.865 | 0.903 |

**COCO-Cap, fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.670 | 0.948 | 0.979 | 0.973 |
| UN: uniform layer weights | 0.466 | 0.978 | 0.973 | 0.978 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.500 | 0.926 | 0.943 | 0.916 |
| Gabsx: absolute gate, shuffled | 0.151 | 0.933 | 0.979 | 0.959 |
| Grel: pixel gate, relative inconsistency | 0.612 | 0.932 | 0.950 | 0.925 |
| Grelx: relative gate, shuffled | 0.128 | 0.913 | 0.977 | 0.941 |

## Contrasts (with DenseCRF)

| Contrast | cell | VOC-Sim, VOC30-sel. | COCO-Cap, VOC30-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---|---|---|---|---|
| Grel − BR | seed alone | +0.016 [+0.016, +0.016] | -0.079 [-0.079, -0.079] | -0.077 [-0.077, -0.077] | -0.056 [-0.056, -0.056] |
| Grel − BR | fine-only | +0.007 [+0.007, +0.007] | +0.079 [+0.079, +0.079] | -0.114 [-0.114, -0.114] | -0.024 [-0.024, -0.024] |
| Grel − BR | BRACE hierarchy | +0.001 [+0.001, +0.001] | +0.101 [+0.101, +0.101] | -0.054 [-0.054, -0.054] | -0.029 [-0.029, -0.029] |
| Grel − BR | NERVE walk | -0.008 [-0.008, -0.008] | -0.020 [-0.020, -0.020] | -0.094 [-0.094, -0.094] | -0.052 [-0.052, -0.052] |
| Grel − Gabs | seed alone | +0.009 [+0.009, +0.009] | +0.013 [+0.013, +0.013] | +0.125 [+0.125, +0.125] | +0.136 [+0.136, +0.136] |
| Grel − Gabs | fine-only | +0.028 [+0.028, +0.028] | +0.010 [+0.010, +0.010] | +0.034 [+0.034, +0.034] | -0.000 [-0.000, -0.000] |
| Grel − Gabs | BRACE hierarchy | -0.000 [-0.000, -0.000] | -0.011 [-0.011, -0.011] | +0.011 [+0.011, +0.011] | +0.007 [+0.007, +0.007] |
| Grel − Gabs | NERVE walk | -0.005 [-0.005, -0.005] | -0.021 [-0.021, -0.021] | +0.014 [+0.014, +0.014] | +0.001 [+0.001, +0.001] |
| Grel − Grelx | seed alone | +0.027 [+0.027, +0.027] | -0.034 [-0.034, -0.034] | +0.539 [+0.539, +0.539] | +0.602 [+0.602, +0.602] |
| Grel − Grelx | fine-only | +0.008 [+0.008, +0.008] | +0.014 [+0.014, +0.014] | -0.138 [-0.138, -0.138] | +0.016 [+0.016, +0.016] |
| Grel − Grelx | BRACE hierarchy | +0.001 [+0.001, +0.001] | +0.101 [+0.101, +0.101] | -0.065 [-0.065, -0.065] | -0.028 [-0.028, -0.028] |
| Grel − Grelx | NERVE walk | -0.008 [-0.008, -0.008] | -0.073 [-0.073, -0.073] | -0.144 [-0.144, -0.144] | -0.034 [-0.034, -0.034] |
| Gabs − BR | seed alone | +0.006 [+0.006, +0.006] | -0.092 [-0.092, -0.092] | -0.202 [-0.202, -0.202] | -0.192 [-0.192, -0.192] |
| Gabs − BR | fine-only | -0.021 [-0.021, -0.021] | +0.069 [+0.069, +0.069] | -0.148 [-0.148, -0.148] | -0.024 [-0.024, -0.024] |
| Gabs − BR | BRACE hierarchy | +0.001 [+0.001, +0.001] | +0.112 [+0.112, +0.112] | -0.065 [-0.065, -0.065] | -0.036 [-0.036, -0.036] |
| Gabs − BR | NERVE walk | -0.003 [-0.003, -0.003] | +0.001 [+0.001, +0.001] | -0.108 [-0.108, -0.108] | -0.053 [-0.053, -0.053] |
| Gabs − Gabsx | seed alone | +0.009 [+0.009, +0.009] | -0.055 [-0.055, -0.055] | +0.062 [+0.062, +0.062] | +0.464 [+0.464, +0.464] |
| Gabs − Gabsx | fine-only | -0.019 [-0.019, -0.019] | +0.051 [+0.051, +0.051] | -0.158 [-0.158, -0.158] | -0.006 [-0.006, -0.006] |
| Gabs − Gabsx | BRACE hierarchy | +0.001 [+0.001, +0.001] | +0.111 [+0.111, +0.111] | -0.079 [-0.079, -0.079] | -0.036 [-0.036, -0.036] |
| Gabs − Gabsx | NERVE walk | -0.003 [-0.003, -0.003] | -0.039 [-0.039, -0.039] | -0.104 [-0.104, -0.104] | -0.043 [-0.043, -0.043] |

## Pre-registered reading

- M1 voc: pass
- M2 voc: fail
- M3 mass voc: fail
- M3 far voc: fail
- M4 voc: pass
- Grel−BR hier+CRF voc sel: pass
- Grel−BR hier+CRF voc 0.55: fail
- M1 coco: pass
- M2 coco: pass
- M3 mass coco: fail
- M3 far coco: fail
- M4 coco: pass
- Grel−BR hier+CRF coco sel: pass
- Grel−BR hier+CRF coco 0.55: fail

**Grel not promising: pixel-gate direction closed.**
