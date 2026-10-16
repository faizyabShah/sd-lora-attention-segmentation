**Phase 3 fusion and calibration package (VOC-Sim, 5 folds; method parameter and threshold chosen on the training folds)**

| Screen | Fold choice (votes) | OOF mIoU REF | OOF mIoU choice | OOF gain | Paired Δ image IoU [95% CI]; better/worse | Δ mIoU @0.35 / 0.45 / 0.55 | Candidates | Outcome |
|---|---|---:|---:|---:|---|---|---:|---|
| P4 residual mix | REF/one_pass (5/5) | 0.7875 | 0.7875 | 0.0000 | — | — | 4 | reference won the fold vote |
| F3/F4 gating | REF/one_pass (5/5) | 0.7875 | 0.7875 | 0.0000 | — | — | 2 | reference won the fold vote |
| N2-N4 calibration | REF/one_pass (5/5) | 0.7875 | 0.7875 | 0.0000 | — | — | 3 | reference won the fold vote |
| R2 DenseCRF | R2/crf (5/5) | 0.7875 | 0.8049 | 0.0175 | +0.0222 [+0.0185, +0.0260]; 361/129 | +0.0215 / +0.0141 / +0.0063 | 1 | fold vote and OOF guard passed |
