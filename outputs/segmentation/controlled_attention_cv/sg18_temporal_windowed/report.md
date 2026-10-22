# SG18 locked temporal-windowed propagation

VOC fold winners: ['W/T1', 'W/T1', 'W/T1', 'W/T1', 'W/T1']
Frozen: W/T1; promoted: True
OOF selection 0.7939; REF 0.7873; gain +0.0066

| Cell | VOC forced OOF | OOF Δ REF | Fold wins | VOC full | VOC threshold | COCO frozen |
|---|---:|---:|---:|---:|---:|---:|
| W/REF | 0.7873 | +0.0000 | 0/5 | 0.7886 | 0.425 | 0.6172 |
| W/T1 | 0.7939 | +0.0066 | 5/5 | 0.7959 | 0.425 | 0.6285 |
| W/T2 | 0.7936 | +0.0063 | 0/5 | 0.7936 | 0.425 | 0.6240 |
| W/T3 | 0.7916 | +0.0043 | 0/5 | 0.7935 | 0.425 | 0.6227 |
| W/P1 | 0.7997 | +0.0124 | 0/5 | 0.8006 | 0.450 | 0.6245 |

Mechanism at VOC-frozen thresholds and controlled fixed 0.45, all 20 per-class values, bootstrap intervals and image-level rows are in `results.json` and `per_image_selected.csv`.
