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

## Result (2026-11-27; `/home/shafa/sg_comp/tmp/r51/sizecheck.py`): FAILS the rule

| Items | n (small) | area55 ρ / AUROC | seed50 ρ / AUROC | mean ρ / AUROC | gatefrac ρ / AUROC |
|---|---|---|---|---|---|
| things | 2,479 (697) | 0.512 / 0.698 | 0.488 / **0.736** | 0.390 / 0.619 | 0.247 / 0.571 |
| stuff | 1,759 (358) | 0.088 / 0.528 | 0.157 / 0.607 | 0.069 / 0.525 | 0.118 / 0.533 |
| things, map on object (IoU > 0.3) | 1,638 (245) | **0.897 / 0.974** | 0.619 / 0.888 | 0.786 / 0.873 | 0.556 / 0.791 |
| things, map off object (IoU ≤ 0.3) | 841 (452) | 0.085 / 0.565 | 0.110 / 0.559 | 0.074 / 0.550 | 0.040 / 0.513 |

Best on things: ρ 0.512 (< 0.6), AUROC 0.736 (< 0.80). **B with map statistics is dropped.** The estimate is circular:
when the map sits on the object, its area tracks the object's almost perfectly. But 452 of the 697 small things have
their map on another object, and then its area says nothing about the target's size. Those are exactly the items B was
meant to help. Per the rule, B survives only if a gate-mass statistic saved by the run 51 capture passes this check.
