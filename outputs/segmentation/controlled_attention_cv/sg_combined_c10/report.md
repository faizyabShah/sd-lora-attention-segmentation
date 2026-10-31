# SG-combined-v1 C10 add-on

Three selectable Markov-source cells, 40 complete class shards, 490 VOC and 707 COCO images.
VOC-only selection; COCO thresholds frozen. S0 is a same-generation reference only.

| Cell | VOC forced OOF | Δ vs S0 | Direct fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.8098 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| S0+R2 | 0.8231 | +0.0000 | 0/5 | 0.8244 | 0.425 | 0.6623 |
| C10a | 0.8216 | +0.0118 | 5/5 | 0.8226 | 0.675 | 0.6465 |
| C10a+R2 | 0.8313 | +0.0083 | 3/5 | 0.8313 | 0.675 | 0.6529 |
| C10b | 0.6444 | -0.1654 | 0/5 | 0.6473 | 0.425 | 0.5150 |
| C10b+R2 | 0.6632 | -0.1599 | 0/5 | 0.6665 | 0.425 | 0.5343 |
| C10c | 0.7622 | -0.0476 | 0/5 | 0.7622 | 0.700 | 0.6177 |
| C10c+R2 | 0.7852 | -0.0378 | 0/5 | 0.7852 | 0.700 | 0.6334 |

Raw selector: C10a {'C10a': 4, 'S0': 1}
R2 selector: S0+R2 {'S0+R2': 5}

Scale AUC, M2 coarse retention, TP/FP distance bands, completion efficiency, and all per-class values are in results.json and per_class_iou.csv.
