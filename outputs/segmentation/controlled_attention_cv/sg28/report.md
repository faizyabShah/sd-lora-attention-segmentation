# SG28 PEARL-inspired independent components

40 complete class shards; 490 VOC and 707 COCO images; VOC-only selection; COCO frozen.

A3 and R2 remain as implemented historically; PA and Laplacian are not combined.

## Raw

| Cell | VOC forced OOF | Δ vs P0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.8097 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| P1 | 0.8129 | +0.0032 | 4/5 | 0.8129 | 0.425 | 0.6443 |
| P2 | 0.7961 | -0.0136 | 0/5 | 0.7973 | 0.475 | 0.6555 |
| P3 | 0.8129 | +0.0032 | 4/5 | 0.8129 | 0.425 | 0.6444 |
| B1 | 0.8031 | -0.0066 | 0/5 | 0.8031 | 0.475 | 0.6537 |
| B2 | 0.8140 | +0.0043 | 5/5 | 0.8140 | 0.475 | 0.6654 |
| C1 | 0.8021 | -0.0077 | 0/5 | 0.8021 | 0.475 | 0.6530 |
| C2 | 0.8027 | -0.0070 | 0/5 | 0.8027 | 0.475 | 0.6534 |
| C3 | 0.8024 | -0.0073 | 0/5 | 0.8024 | 0.475 | 0.6532 |
| D1 | 0.8139 | +0.0042 | 5/5 | 0.8139 | 0.475 | 0.6652 |
| D2 | 0.8140 | +0.0042 | 5/5 | 0.8140 | 0.475 | 0.6653 |
| D3 | 0.8140 | +0.0042 | 5/5 | 0.8140 | 0.475 | 0.6653 |

## After R2

| Cell | VOC forced OOF | Δ vs P0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| P0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 | 0.425 | 0.6621 |
| P1+R2 | 0.8220 | -0.0008 | 2/5 | 0.8234 | 0.425 | 0.6543 |
| P2+R2 | 0.8155 | -0.0074 | 0/5 | 0.8167 | 0.450 | 0.6640 |
| P3+R2 | 0.8220 | -0.0008 | 2/5 | 0.8234 | 0.425 | 0.6543 |
| B1+R2 | 0.8138 | -0.0090 | 0/5 | 0.8155 | 0.450 | 0.6585 |
| B2+R2 | 0.8230 | +0.0001 | 1/5 | 0.8230 | 0.450 | 0.6661 |
| C1+R2 | 0.8129 | -0.0099 | 0/5 | 0.8146 | 0.450 | 0.6580 |
| C2+R2 | 0.8135 | -0.0093 | 0/5 | 0.8152 | 0.450 | 0.6583 |
| C3+R2 | 0.8132 | -0.0096 | 0/5 | 0.8149 | 0.450 | 0.6581 |
| D1+R2 | 0.8227 | -0.0002 | 1/5 | 0.8227 | 0.450 | 0.6658 |
| D2+R2 | 0.8228 | +0.0000 | 1/5 | 0.8228 | 0.450 | 0.6659 |
| D3+R2 | 0.8228 | -0.0001 | 1/5 | 0.8228 | 0.450 | 0.6659 |

## Separate VOC selectors

| Stage | Branch | Fold votes | Frozen cell | OOF gain |
|---|---|---|---|---:|
| raw | all | {'P0': 5} | P0 | +0.0000 |
| raw | PA | {'P0': 5} | P0 | +0.0000 |
| raw | plain_and_boundary | {'P0': 5} | P0 | +0.0000 |
| raw | confidence | {'P0': 5} | P0 | +0.0000 |
| raw | confidence_and_boundary | {'P0': 5} | P0 | +0.0000 |
| final | all | {'P0+R2': 5} | P0+R2 | +0.0000 |
| final | PA | {'P0+R2': 5} | P0+R2 | +0.0000 |
| final | plain_and_boundary | {'P0+R2': 5} | P0+R2 | +0.0000 |
| final | confidence | {'P0+R2': 5} | P0+R2 | +0.0000 |
| final | confidence_and_boundary | {'P0+R2': 5} | P0+R2 | +0.0000 |

Full per-class results, precision/recall, paired intervals, fixed-threshold mechanism, PA alignment/entropy correlations, CG accuracy, and confidence bins are in `results.json`; per-class and PA summary tables are also exported as CSV.
