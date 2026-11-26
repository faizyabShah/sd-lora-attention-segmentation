# Seed alone: rank–distribution swap (runs/19 amendment 1, CPU part)

Seeds from the runs/17 caches (CLIP seed on the stored image; CA seed from the runs/17 live generation). No propagation, no DenseCRF. Class-macro mIoU at the VOC-selected threshold (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold on COCO-Cap). Δ with image-bootstrap 95% interval (2000).

| Seed | VOC-Sim | COCO-Cap | VOC-best threshold |
|---|---:|---:|---:|
| CA | 0.705 | 0.584 | 0.400 |
| CLIP | 0.727 | 0.650 | 0.625 |
| CA ranking + CLIP values (hA) | 0.706 | 0.591 | 0.600 |
| CLIP ranking + CA values (hB) | 0.711 | 0.611 | 0.400 |

| Contrast | VOC-Sim | COCO-Cap |
|---|---|---|
| values: hA − CA | +0.002 [-0.007, +0.010] | +0.007 [-0.009, +0.023] |
| ranking: CLIP − hA | +0.020 [+0.015, +0.027] | +0.060 [+0.050, +0.071] |
| values: hB − CLIP | -0.016 [-0.026, -0.006] | -0.039 [-0.058, -0.022] |
| ranking: hB − CA | +0.006 [+0.000, +0.012] | +0.028 [+0.019, +0.038] |

COCO-Cap per-image IoU by object-size third (small / middle / large) and share below 0.3 IoU:

- CA: 0.408 / 0.618 / 0.682; below 0.3: 0.151
- CLIP: 0.532 / 0.697 / 0.734; below 0.3: 0.076
- CA ranking + CLIP values (hA): 0.428 / 0.610 / 0.682; below 0.3: 0.132
- CLIP ranking + CA values (hB): 0.454 / 0.680 / 0.711; below 0.3: 0.123
