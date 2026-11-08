# Pre-registration: CLIP seed + SD propagation, confirmation on fresh real VOC photos (F)

Written 2026-11-08, before any F capture or score exists. Committed before the run.

**Set F.** The next 200 VOC2012 train ids in the seed-137 shuffle that produced S (`scripts/prep_sg_fresh.py`). It is disjoint from D and S by construction and has never been captured or inspected. 200 photos, 294 (photo, class) pairs, 5 photo-grouped class-stratified folds.

**Frozen candidate.** `clip C16`:
- The SCLIP (ViT-B/16, HF port reproducing the repo on VOC val) VOC21 class probability of the target, at 64×64, min-max normalised.
- It is used as the seed of the frozen C16 path through the target run's own SD1.5 DDIM self-attention.
- Thresholded at **τ = 0.575**, fitted on all of D.
- Nothing about it is changed after this note.

**Comparators.** All thresholds were fitted on D, and nothing is fitted on F:

| comparator | definition | τ |
|---|---|---|
| SCLIP standalone + R2 | SCLIP's own VOC21 prediction of the class, DenseCRF | none |
| SCLIP standalone | SCLIP's own VOC21 prediction of the class | none |
| comp C16 | generic-word SD competition (marg.5/mm/gen) → C16 | 0.50 |
| base V3+R2 | BRACE on real photos | 0.70 |

**Primary test (pass/fail).** `clip C16 @0.575` − `SCLIP standalone + R2` in class-macro IoU over F's pairs. The photo-bootstrap 95% CI (1000 resamples) must be > 0.

**Secondary tests (reported, not pass/fail).**
- `clip C16` vs SCLIP standalone, vs comp C16 and vs base V3+R2, all at the D thresholds.
- The same comparisons with τ chosen OOF within F.
- Per class and by object size (<1%, 1–5%, 5–20%, >20%).
- Seed-level AUROC (target vs other objects, target vs background) for the CLIP and A3 seeds.

**If the primary test fails,** the D result is treated as not confirmed. No variant is re-selected on F.
