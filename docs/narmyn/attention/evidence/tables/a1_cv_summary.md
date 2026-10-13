**Controlled CA package B: cross-validated selection (VOC-Sim, 5 folds, threshold 0.45)**

| Screen | Reference | Vote winner (votes) | OOF mIoU selection | OOF mIoU reference | OOF gain | Paired Δ image IoU [95% CI] | Calibration-dependent | Outcome |
|---|---|---|---:|---:|---:|---|---|---|
| CA-L | L/all_16 | L/leave2_10_11 (5/5) | 0.6923 | 0.6661 | 0.0262 | +0.0209 [+0.0155, +0.0265] | yes | fold vote and OOF guard passed |
| CA-H | H/0_1_2_3_4_5_6_7 | H/0_2_3_4_5_6_7 (4/5) | 0.6741 | 0.6661 | 0.0080 | +0.0089 [+0.0047, +0.0131] | yes | fold vote and OOF guard passed |
| CA-A | A/cond_padded_all | A/cond_padded_all (5/5) | 0.6661 | 0.6661 | 0.0000 | +0.0000 [+0.0000, +0.0000] | no | reference won the fold vote |
