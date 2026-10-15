**Controlled SA package B (frozen CA seed; VOC-Sim, 5 folds, selection at 0.45). SA-C combines these with SA-T/SA-R.**

| Screen | Reference | Vote winner (votes) | OOF mIoU reference | OOF mIoU choice | OOF gain | Paired Δ image IoU [95% CI] | Δ mIoU @0.35 / 0.45 / 0.55 | Outcome |
|---|---|---|---:|---:|---:|---|---|---|
| SA-L | SL/all_16 | SL/single_15 (5/5) | 0.7553 | 0.7884 | 0.0331 | +0.0417 [+0.0365, +0.0469] | +0.0477 / +0.0331 / +0.0068 | fold vote and OOF guard passed |
| SA-H | SH/0_1_2_3_4_5_6_7 | SH/1_4_6 (5/5) | 0.7553 | 0.7610 | 0.0057 | +0.0065 [+0.0056, +0.0074] | +0.0083 / +0.0057 / +0.0004 | fold vote and OOF guard passed |
