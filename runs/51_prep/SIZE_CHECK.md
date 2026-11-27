# Run 51 preparation: can a label-free statistic of our maps estimate object size? (CPU check; rule fixed before running)

**Purpose.** Candidate B (refinement chosen by estimated object size) needs a label-free size estimate. Run 50 found
ERS helps small objects (+0.081) and hurts larger ones (−0.047). If no estimate sorts objects by size, B is dropped.

**Data.** Run 45's saved maps (cluster): PASCAL Context **train** (development), 4,238 kept primary items, arm A|ip
(our frozen prompt), SD 1.5, 64 × 64 maps. Saved seeds are already min-maxed, so the gate's own mass is not
recoverable; the statistics below come from our maps only. Ground truth is used for scoring only.

**Statistics (label-free, per item):**
- `area55`: fraction of the 64 × 64 map with min-max(uL|V0) ≥ 0.55 (our output's own area);
- `seed50`: fraction with uL|seed ≥ 0.5;
- `mean`: mean of min-max(uL|V0) (soft area);
- `gatefrac`: fraction with uL|seed ≥ u-|seed, a crude gate-effect proxy.

**Truth.** Object area as a fraction of valid pixels; small = < 5%.

**Reported.** Spearman ρ with true area; AUROC for small vs not small; for things, stuff and all; and separately for
items where the map is on the object (IoU at 0.55 > 0.3) and where it is not.

**Rule (fixed now).** B is viable if, on things, one statistic has **ρ ≥ 0.6 and AUROC ≥ 0.80**. Otherwise B is
dropped from run 51, or kept only with a gate-mass statistic that the run 51 capture saves, then checked the same way.
Exploratory; nothing is selected from this beyond that yes / no.
