# SG18 T1 + unchanged DenseCRF

VOC fold votes: {'W/P1+R2': 5}; frozen: W/P1+R2
Forced OOF T1+CRF 0.8106 versus P1+CRF 0.8135 (-0.0029)

| Method | VOC threshold | VOC mIoU | COCO frozen mIoU | VOC @0.45 | COCO @0.45 |
|---|---:|---:|---:|---:|---:|
| W/T1 | 0.425 | 0.7960 | 0.6286 | 0.7942 | 0.6346 |
| W/T1+R2 | 0.425 | 0.8122 | 0.6465 | 0.8080 | 0.6510 |
| W/P1 | 0.450 | 0.8007 | 0.6249 | 0.8007 | 0.6249 |
| W/P1+R2 | 0.450 | 0.8135 | 0.6375 | 0.8135 | 0.6375 |

Full precision, recall, per-class values and paired intervals are in results.json. The fixed-0.45 column includes the locked final-pipeline P1+CRF reference.
