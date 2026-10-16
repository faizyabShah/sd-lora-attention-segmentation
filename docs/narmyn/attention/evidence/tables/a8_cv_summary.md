**Phase 3 propagation package (Codex) (VOC-Sim, 5 folds; method parameter and threshold chosen on the training folds)**

| Screen | Fold choice (votes) | OOF mIoU REF | OOF mIoU choice | OOF gain | Paired Δ image IoU [95% CI]; better/worse | Δ mIoU @0.35 / 0.45 / 0.55 | Candidates | Outcome |
|---|---|---:|---:|---:|---|---|---:|---|
| P1 observation-wise powers | P1/obs_n2 (4/5) | 0.7875 | 0.7991 | 0.0116 | +0.0140 [+0.0097, +0.0180]; 359/131 | +0.0087 / +0.0121 / +0.0243 | 3 | fold vote and OOF guard passed |
| W3 average-before powers | REF/one_pass (4/5) | 0.7875 | 0.7875 | -0.0000 | — | — | 4 | reference won the fold vote |
| P2 observation-wise restart | REF/one_pass (4/5) | 0.7875 | 0.7877 | 0.0002 | — | — | 12 | reference won the fold vote |
| P2 average-before restart | REF/one_pass (5/5) | 0.7875 | 0.7875 | 0.0000 | — | — | 12 | reference won the fold vote |
| P6 sparse affinity | REF/one_pass (5/5) | 0.7875 | 0.7875 | 0.0000 | — | — | 7 | reference won the fold vote |
