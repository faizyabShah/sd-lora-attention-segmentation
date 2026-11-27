# Run 51: gate-guided refinements, and run 50's leads on fresh random photos (pre-registration)

**Status: exploratory, development photos only.** Nothing is promoted, and the paper 2 method stays frozen. Driver:
`scripts/jc_gate51.py`. Runs on the VPU-lab machine; every compared map comes from that machine.

## 1. Why this run

Run 50 found that iSeg's edge is label-free and comes from steps that suppress attention on the wrong objects:
- sink words in the prompt;
- its aggregation, whose clustering helps larger objects (+0.077) but hurts small ones (−0.021);
- ERS, which helps small objects (+0.081) but hurts larger ones (−0.047).

Our gate helps everywhere. The run 51 prep check (`runs/51_prep/SIZE_CHECK.md`) ruled out choosing a refinement by a
size estimate taken from our maps: 65% of small objects have their map on another object.

This run therefore tests two refinements of our own that use the gate where iSeg uses fixed rules. It also gives run 50's
seven leads the confirmation on random photos that run 50's pre-registration requires.

## 2. Photos (fixed now; computed on the cluster, CPU)

| Half | Source | Photos | Items | Small (< 5%) | Classes |
|---|---|---:|---:|---:|---:|
| **random** (primary) | run 49's COCO train2017 development pool, run 49's permutation, positions 1,000–1,100, every photo with ≥ 1 primary item (area ≥ 1%) | 100 | 221 | 88 | 60 |
| hard | run 50's 100 photos, unchanged | 100 | 364 | 178 | 59 |

- **No shared photos:** run 49's rounds use the first ~70 positions, and run 50 uses positions ≥ 2,000.
- **The random half is fresh;** nothing has been scored on it.
- **The hard half is not fresh:** run 50's leads were found on it. It is reported, but no confirmation and no screen
  decision uses it.

## 3. Cells

**Passes.** Our pass code, as in run 50:
- SD 1.5: prompts T, S1, S2, S2n, B, F (run 50's definitions; B uses new BLIP-large captions of all 200 photos);
- SD 2.1-base: prompts T and S2;
- iSeg's own pass on SD 2.1, once per photo.

**Seeds:** U- (ours, no gate), UL (ours, MK4-LR gate), IL (iSeg's aggregation × the 64 × 64 gate map).

**Refinements** (all from the same pass; every setting below is fixed now and is not tuned):

| Name | What it does |
|---|---|
| seed | none |
| V0 | ours (frozen): two head-wise steps x ← H x through the 64 × 64 self-attention H; head mean |
| ERS | iSeg's, COCO settings (run 50) |
| **A2 / A5** | **gated propagation:** 2 / 5 head-wise steps x ← H (g ⊙ x), with g = the 64 × 64 MK4-LR gate map (floor 1e-3). Mass flows only out of locations the gate keeps, and the gate is applied again at every step. No row renormalisation; head mean. With g = 1 it equals V0. |
| **C** | **region blend, then V0.** Regions = spherical k-means of the gate's own image queries (every gate module, each L2-normalised, upsampled to 64 × 64, concatenated; 6 × 6 grid start; 20 iterations). seed' = ½ seed + ½ mean of the seed over its region. |
| **CA** | region blend, then A2 |

**Gate mass** (T pass, saved per item): the mean of the 64 × 64 gate map, and its fraction ≥ 0.5.

**Read-out:** 0.55 (ours) for every decision. iSeg's competition read-out is used only for its own pass, as a reference.

**Built-in checks** (the `check` step; any failure stops the run):
1. gated propagation with g = 1 equals V0;
2. region blend with one-pixel regions equals the seed;
3. on run 50's photos, the new capture's `T|UL|V0` and `S2|IL|ERS` maps equal run 50's saved maps (same machine,
   |diff| < 2e-3).

## 4. Registered decisions (class-macro mIoU at 0.55; 2,000-sample paired photo bootstrap)

**4.1 Run 50's leads on the random half.** The leads, each vs `15|T|UL|V0`:
- `15|S1|UL|ERS`, `15|S2|UL|ERS`, `15|S2|IL|ERS`;
- `15|S2n|IL|V0`, `15|B|IL|V0`;
- `15|F|UL|V0`, `15|F|IL|V0`.

A lead is **confirmed** if all of these hold:
- Δ ≥ +0.02;
- the lower end of its Bonferroni interval (7 tests, 99.29%) is > 0;
- Δ on not-small items is ≥ −0.01.

A confirmed lead still needs the run 44 test columns before any claim.

**4.2 New refinements vs V0** (8 contrasts; same prompt and seed). `15|{T, F}|UL|{A2, A5, C, CA}` vs
`15|{T, F}|UL|V0`. A contrast is a **lead** if all of these hold on the random half:
- Δ ≥ +0.02;
- the 95% interval is above 0;
- Δ on not-small items is ≥ −0.01.

A lead then needs its own confirmation: fresh random photos, then the run 44 test columns.

**4.3 Size check with the gate's mass** (rule of `runs/51_prep/SIZE_CHECK.md`, applied to all 585 items). It passes
if either gate-mass statistic reaches Spearman ρ ≥ 0.6 with true area **and** AUROC ≥ 0.80 for small vs not small.
Only a pass brings candidate B (refinement chosen by size) back, in a later run.

## 5. Descriptive (no decisions)

- Each refinement vs V0, averaged over prompts and seeds, per backbone (random small / not small, hard).
- Label-free cells vs iSeg's own pass on the random half (`21|S2|IL|ERS`, A5 variants, `15|F|UL|V0`).
- Run 49's error measures for the main cells.
- Every cell.
- Panels: the 5 random-half items A5 helps most and the 5 it hurts most.
