# Locked SG2 + DenseCRF confirmation

AUDITED RECOVERY: historical pixel-level reproduction failed across hardware; these are internally consistent V100 comparisons. See validation_recovery and count_audit.json; independent review required.

No parameter or threshold chosen on COCO. Existing FTF at its locked .45 threshold is reported alongside the framework-selected thresholds. CRF cells have no continuous-map AP.

| Comparison | OOF reference | Locked candidate OOF | Delta | Protocol frozen | Candidate fold votes |
|---|---:|---:|---:|---|---:|
| primary_vs_FTF | 0.8132 | 0.8111 | -0.0021 | W/P1+R2 | 0/5 |
| secondary_vs_one_pass_CRF | 0.8047 | 0.8111 | +0.0064 | W/REF+R2 | 2/5 |

| Cell | VOC threshold | VOC mIoU | COCO mIoU | VOC @.45 | COCO @.45 |
|---|---:|---:|---:|---:|---:|
| W/REF | 0.425 | 0.7886 | 0.6172 | 0.7884 | 0.6259 |
| W/P1 | 0.450 | 0.8006 | 0.6245 | 0.8006 | 0.6245 |
| W/SG2 | 0.450 | 0.7984 | 0.6235 | 0.7984 | 0.6235 |
| W/REF+R2 | 0.425 | 0.8047 | 0.6348 | 0.8024 | 0.6415 |
| W/P1+R2 | 0.450 | 0.8132 | 0.6371 | 0.8132 | 0.6371 |
| W/SG2+R2 | 0.425 | 0.8111 | 0.6261 | 0.8099 | 0.6350 |

OOF rows evaluate the locked candidate in every held-out fold with training-fold-selected thresholds. Protocol selection is separately reported and can retain FTF for a marginal gain. All class IoUs/deltas, recall/precision and paired intervals are in summary.json and per_class_results.csv.
