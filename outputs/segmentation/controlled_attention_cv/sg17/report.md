# SG17 locked screen

VOC fold winners: ['W/P1', 'W/P1', 'W/P1', 'W/P1', 'W/P1']
Frozen: W/P1; promoted: False
OOF reference 0.7998, selection 0.7998, gain +0.0000

| Method | VOC full mIoU | COCO frozen mIoU | VOC OOF forced gain |
|---|---:|---:|---:|
| W/P1 | 0.8007 | 0.6249 | +0.0000 |
| W/SG17-Q70 | 0.8001 | 0.6196 | -0.0007 |
| W/SG17-Q80 | 0.8001 | 0.6196 | -0.0007 |
| W/SG17-Q90 | 0.8001 | 0.6196 | -0.0007 |

## Mechanism: rejected P1 additions

| Dataset | Method | TP retention | FP rejection | Rejected FP >16 px | Rejected FP >64 px |
|---|---|---:|---:|---:|---:|
| voc | W/SG17-Q70 | 0.810 | 0.232 | 94098 | 46568 |
| voc | W/SG17-Q80 | 0.810 | 0.232 | 94098 | 46568 |
| voc | W/SG17-Q90 | 0.810 | 0.232 | 94098 | 46568 |
| coco | W/SG17-Q70 | 0.902 | 0.120 | 255550 | 164640 |
| coco | W/SG17-Q80 | 0.902 | 0.120 | 255550 | 164640 |
| coco | W/SG17-Q90 | 0.902 | 0.120 | 255550 | 164640 |

All five folds, candidate thresholds, bootstrap intervals, per-class IoUs, retained/rejected events, and VOC-frozen COCO reliability bins are in `results.json`.
