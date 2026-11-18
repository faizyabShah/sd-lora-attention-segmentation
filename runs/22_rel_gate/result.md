# Relative-inconsistency pixel gate: 30 + 30 pilot (`PREREG.md`)

`scripts/jc_relgate.py`. Images: crc32 order positions 10–39 per set (VOC-Sim 30, COCO-Cap 30), one live capture each, compared within this capture only. Class-macro mIoU. *VOC30-sel.* = threshold best on these VOC-Sim images (in-sample on VOC-Sim, applied unchanged to COCO-Cap). Paired image-bootstrap 95% intervals (2000). Small-sample exploratory pilot.

- VOC-Sim: reconstruction vs runtime seed max |Δ| 2.1e-06.
- COCO-Cap: reconstruction vs runtime seed max |Δ| 1.6e-06.

## Mechanism diagnostics (mean over images)

| quantity | VOC-Sim abs | VOC-Sim rel | COCO-Cap abs | COCO-Cap rel |
|---|---:|---:|---:|---:|
| M1 ρ(r, A), w-weighted | -0.105 | +0.244 | -0.143 | +0.135 |
| M2 8+16-res share inside object (global 0.853 / 0.844) | 0.826 | 0.827 | 0.757 | 0.845 |
| r, 64-res layers, interior | 0.339 | 0.407 | 0.362 | 0.330 |
| r, 64-res layers, near | 0.379 | 0.436 | 0.379 | 0.340 |
| r, 64-res layers, far | 0.445 | 0.496 | 0.423 | 0.375 |
| r, 16-res layers, interior | 0.290 | 0.364 | 0.249 | 0.386 |
| r, 16-res layers, near | 0.353 | 0.343 | 0.318 | 0.381 |
| r, 16-res layers, far | 0.438 | 0.348 | 0.436 | 0.436 |
| r, 8-res layers, interior | 0.314 | 0.319 | 0.257 | 0.343 |
| r, 8-res layers, near | 0.361 | 0.249 | 0.321 | 0.309 |
| r, 8-res layers, far | 0.427 | 0.197 | 0.432 | 0.316 |

| seed (min–max) | VOC-Sim mass on GT | interior | near | far | COCO-Cap mass on GT | interior | near | far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BR | 0.505 | 0.730 | 0.412 | 0.156 | 0.439 | 0.649 | 0.356 | 0.138 |
| Gabs | 0.508 | 0.635 | 0.316 | 0.154 | 0.403 | 0.564 | 0.322 | 0.166 |
| Grel | 0.527 | 0.639 | 0.275 | 0.148 | 0.412 | 0.625 | 0.314 | 0.172 |

**VOC-Sim, VOC30-sel. threshold, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.718 | 0.810 | 0.800 | 0.821 |
| UN: uniform layer weights | 0.729 | 0.815 | 0.801 | 0.811 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.796 | 0.804 | 0.798 | 0.821 |
| Gabsx: absolute gate, shuffled | 0.699 | 0.813 | 0.802 | 0.820 |
| Grel: pixel gate, relative inconsistency | 0.797 | 0.809 | 0.801 | 0.824 |
| Grelx: relative gate, shuffled | 0.638 | 0.816 | 0.803 | 0.819 |

**COCO-Cap, VOC30-sel. threshold, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.632 | 0.778 | 0.823 | 0.748 |
| UN: uniform layer weights | 0.595 | 0.731 | 0.796 | 0.686 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.697 | 0.689 | 0.770 | 0.663 |
| Gabsx: absolute gate, shuffled | 0.651 | 0.757 | 0.824 | 0.743 |
| Grel: pixel gate, relative inconsistency | 0.717 | 0.701 | 0.769 | 0.685 |
| Grelx: relative gate, shuffled | 0.683 | 0.758 | 0.816 | 0.757 |

**VOC-Sim, VOC30-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.669 | 0.793 | 0.782 | 0.810 |
| UN: uniform layer weights | 0.688 | 0.795 | 0.784 | 0.802 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.738 | 0.789 | 0.785 | 0.804 |
| Gabsx: absolute gate, shuffled | 0.628 | 0.795 | 0.784 | 0.810 |
| Grel: pixel gate, relative inconsistency | 0.749 | 0.788 | 0.788 | 0.810 |
| Grelx: relative gate, shuffled | 0.575 | 0.793 | 0.786 | 0.806 |

**COCO-Cap, VOC30-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.589 | 0.761 | 0.800 | 0.719 |
| UN: uniform layer weights | 0.557 | 0.709 | 0.779 | 0.685 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.642 | 0.691 | 0.767 | 0.740 |
| Gabsx: absolute gate, shuffled | 0.572 | 0.759 | 0.809 | 0.710 |
| Grel: pixel gate, relative inconsistency | 0.665 | 0.681 | 0.771 | 0.737 |
| Grelx: relative gate, shuffled | 0.588 | 0.736 | 0.804 | 0.708 |

**VOC-Sim, fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.676 | 0.787 | 0.800 | 0.793 |
| UN: uniform layer weights | 0.640 | 0.768 | 0.800 | 0.768 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.608 | 0.772 | 0.794 | 0.777 |
| Gabsx: absolute gate, shuffled | 0.263 | 0.788 | 0.802 | 0.790 |
| Grel: pixel gate, relative inconsistency | 0.607 | 0.771 | 0.798 | 0.775 |
| Grelx: relative gate, shuffled | 0.096 | 0.782 | 0.803 | 0.780 |

**COCO-Cap, fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.505 | 0.735 | 0.823 | 0.675 |
| UN: uniform layer weights | 0.445 | 0.704 | 0.784 | 0.676 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.422 | 0.764 | 0.761 | 0.755 |
| Gabsx: absolute gate, shuffled | 0.212 | 0.731 | 0.824 | 0.674 |
| Grel: pixel gate, relative inconsistency | 0.509 | 0.773 | 0.763 | 0.762 |
| Grelx: relative gate, shuffled | 0.129 | 0.725 | 0.816 | 0.667 |

**VOC-Sim, fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.643 | 0.780 | 0.782 | 0.792 |
| UN: uniform layer weights | 0.616 | 0.764 | 0.783 | 0.769 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.609 | 0.773 | 0.778 | 0.781 |
| Gabsx: absolute gate, shuffled | 0.334 | 0.781 | 0.783 | 0.787 |
| Grel: pixel gate, relative inconsistency | 0.616 | 0.771 | 0.782 | 0.776 |
| Grelx: relative gate, shuffled | 0.196 | 0.779 | 0.784 | 0.777 |

**COCO-Cap, fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global layer weights) | 0.497 | 0.729 | 0.800 | 0.674 |
| UN: uniform layer weights | 0.450 | 0.701 | 0.765 | 0.675 |
| Gabs: pixel gate, absolute inconsistency (runs/21) | 0.447 | 0.752 | 0.742 | 0.750 |
| Gabsx: absolute gate, shuffled | 0.262 | 0.726 | 0.800 | 0.673 |
| Grel: pixel gate, relative inconsistency | 0.520 | 0.765 | 0.747 | 0.761 |
| Grelx: relative gate, shuffled | 0.199 | 0.716 | 0.794 | 0.667 |

## Contrasts (with DenseCRF)

| Contrast | cell | VOC-Sim, VOC30-sel. | COCO-Cap, VOC30-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---|---|---|---|---|
| Grel − BR | seed alone | +0.079 [+0.056, +0.118] | +0.085 [-0.000, +0.142] | -0.069 [-0.121, -0.020] | +0.004 [-0.055, +0.059] |
| Grel − BR | fine-only | -0.000 [-0.024, +0.030] | -0.077 [-0.126, -0.035] | -0.017 [-0.038, +0.002] | +0.038 [-0.013, +0.096] |
| Grel − BR | BRACE hierarchy | +0.000 [-0.011, +0.015] | -0.054 [-0.085, -0.013] | -0.002 [-0.009, +0.009] | -0.060 [-0.091, -0.016] |
| Grel − BR | NERVE walk | +0.003 [-0.027, +0.044] | -0.063 [-0.129, -0.005] | -0.018 [-0.036, +0.005] | +0.088 [+0.012, +0.130] |
| Grel − Gabs | seed alone | +0.001 [-0.007, +0.012] | +0.020 [-0.019, +0.073] | -0.001 [-0.033, +0.022] | +0.088 [+0.058, +0.157] |
| Grel − Gabs | fine-only | +0.005 [-0.005, +0.011] | +0.012 [-0.008, +0.024] | -0.001 [-0.020, +0.010] | +0.010 [-0.036, +0.078] |
| Grel − Gabs | BRACE hierarchy | +0.002 [-0.004, +0.006] | -0.000 [-0.010, +0.015] | +0.004 [-0.003, +0.011] | +0.002 [-0.009, +0.015] |
| Grel − Gabs | NERVE walk | +0.003 [-0.009, +0.015] | +0.022 [+0.003, +0.049] | -0.002 [-0.025, +0.012] | +0.007 [-0.044, +0.074] |
| Grel − Grelx | seed alone | +0.159 [+0.125, +0.235] | +0.034 [-0.006, +0.077] | +0.510 [+0.420, +0.565] | +0.380 [+0.298, +0.422] |
| Grel − Grelx | fine-only | -0.006 [-0.025, +0.013] | -0.056 [-0.102, -0.025] | -0.011 [-0.032, +0.009] | +0.048 [-0.011, +0.095] |
| Grel − Grelx | BRACE hierarchy | -0.002 [-0.012, +0.012] | -0.047 [-0.080, -0.007] | -0.004 [-0.010, +0.006] | -0.053 [-0.086, -0.012] |
| Grel − Grelx | NERVE walk | +0.005 [-0.023, +0.042] | -0.072 [-0.133, -0.020] | -0.006 [-0.030, +0.030] | +0.095 [+0.019, +0.147] |
| Gabs − BR | seed alone | +0.078 [+0.056, +0.114] | +0.065 [-0.032, +0.119] | -0.067 [-0.116, -0.015] | -0.084 [-0.179, -0.029] |
| Gabs − BR | fine-only | -0.006 [-0.027, +0.028] | -0.088 [-0.138, -0.040] | -0.016 [-0.031, +0.004] | +0.028 [-0.044, +0.081] |
| Gabs − BR | BRACE hierarchy | -0.002 [-0.017, +0.018] | -0.054 [-0.091, -0.011] | -0.006 [-0.015, +0.008] | -0.062 [-0.100, -0.015] |
| Gabs − BR | NERVE walk | +0.000 [-0.030, +0.043] | -0.085 [-0.155, -0.019] | -0.016 [-0.042, +0.023] | +0.081 [-0.026, +0.138] |
| Gabs − Gabsx | seed alone | +0.097 [+0.074, +0.140] | +0.047 [-0.028, +0.090] | +0.345 [+0.259, +0.410] | +0.210 [+0.098, +0.262] |
| Gabs − Gabsx | fine-only | -0.009 [-0.026, +0.015] | -0.067 [-0.104, -0.024] | -0.016 [-0.032, +0.005] | +0.032 [-0.042, +0.076] |
| Gabs − Gabsx | BRACE hierarchy | -0.003 [-0.017, +0.017] | -0.054 [-0.092, -0.011] | -0.008 [-0.016, +0.007] | -0.063 [-0.100, -0.015] |
| Gabs − Gabsx | NERVE walk | +0.002 [-0.029, +0.046] | -0.080 [-0.154, -0.018] | -0.013 [-0.038, +0.026] | +0.081 [-0.023, +0.138] |

## Pre-registered reading

- M1 voc: pass
- M2 voc: fail
- M3 mass voc: pass
- M3 far voc: pass
- M4 voc: fail
- Grel−BR hier+CRF voc sel: pass
- Grel−BR hier+CRF voc 0.55: fail
- M1 coco: pass
- M2 coco: pass
- M3 mass coco: fail
- M3 far coco: fail
- M4 coco: fail
- Grel−BR hier+CRF coco sel: fail
- Grel−BR hier+CRF coco 0.55: fail

**Grel not promising: pixel-gate direction closed.**
