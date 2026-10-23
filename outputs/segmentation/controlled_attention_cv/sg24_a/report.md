# SG24-A results

VOC-only thresholds; COCO frozen transfer.

| Cell | VOC OOF | OOF delta | Fold wins | Full VOC | Threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| A0/CA | 0.7135 | +0.0000 | 0/5 | 0.7135 | 0.400 | 0.5627 |
| A0/FINAL | 0.8169 | +0.0000 | 0/5 | 0.8185 | 0.450 | 0.6437 |
| A1/CA | 0.6406 | -0.0730 | 0/5 | 0.6406 | 0.700 | 0.0000 |
| A1/FINAL | 0.8005 | -0.0164 | 0/5 | 0.8022 | 0.625 | 0.0000 |
| A2/CA | 0.6897 | -0.0239 | 0/5 | 0.6897 | 0.300 | 0.5559 |
| A2/FINAL | 0.8033 | -0.0136 | 0/5 | 0.8039 | 0.375 | 0.6366 |
| A3/CA | 0.7049 | -0.0086 | 1/5 | 0.7049 | 0.400 | 0.5836 |
| A3/FINAL | 0.8230 | +0.0061 | 4/5 | 0.8243 | 0.425 | 0.6620 |
| A4/CA | 0.6288 | -0.0848 | 0/5 | 0.6288 | 0.700 | 0.0000 |
| A4/FINAL | 0.8004 | -0.0165 | 0/5 | 0.8027 | 0.600 | 0.0000 |

Per-class IoUs: `per_class_iou.csv`. Full metrics, 20k paired intervals, CA diagnostics and layer weights: `results.json`.
