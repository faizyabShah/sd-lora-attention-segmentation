**Token track B1 (released token vs class word on REF) (VOC-Sim, 5 folds; method parameter and threshold chosen on the training folds)**

| Screen | Fold choice (votes) | OOF mIoU REF | OOF mIoU choice | OOF gain | Paired Δ image IoU [95% CI]; better/worse | Δ mIoU @0.35 / 0.45 / 0.55 | Candidates | Outcome |
|---|---|---:|---:|---:|---|---|---:|---|
| B1 released token, OVAM form (unconditional query) | TU/REF (5/5) | 0.7873 | 0.8428 | 0.0555 | +0.0470 [+0.0380, +0.0560]; 369/121 | +0.0324 / +0.0518 / +0.0819 | 1 | fold vote and OOF guard passed |
| B1 released token, conditional query | TC/REF (5/5) | 0.7873 | 0.8507 | 0.0634 | +0.0529 [+0.0440, +0.0621]; 375/115 | +0.0414 / +0.0596 / +0.0852 | 1 | fold vote and OOF guard passed |
