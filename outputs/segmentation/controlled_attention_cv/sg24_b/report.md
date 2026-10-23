# SG24-B results

VOC-only thresholds; COCO frozen transfer.

| Cell | VOC OOF | OOF delta | Fold wins | Full VOC | Threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| B0-A3/CA | 0.7050 | +0.0000 | 0/5 | 0.7050 | 0.400 | 0.5836 |
| B0-A3/FINAL | 0.8230 | +0.0000 | 0/5 | 0.8244 | 0.425 | 0.6622 |
| B1-A3/CA | 0.6932 | -0.0117 | 1/5 | 0.6932 | 0.350 | 0.5799 |
| B1-A3/FINAL | 0.8152 | -0.0079 | 0/5 | 0.8152 | 0.425 | 0.6703 |

Per-class IoUs: `per_class_iou.csv`. Full metrics, 20k paired intervals, CA diagnostics and layer weights: `results.json`.
