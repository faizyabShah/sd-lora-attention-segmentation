# Run 41: median and trimmed low-resolution consensus, end to end (pre-registration)

Committed before any full capture or scoring of this run. The smoke test touches only training photos (VOC train and the
run 40 COCO train threshold-fit items). It also builds masks for 40 items of the new COCO set, which checks the item
builder; no model is run on those 40 items. BRACE stays frozen. **Development on VOC train, then a frozen confirmation on
new COCO photos.** Nothing is promoted; any change to the paper's method is a separate owner decision.

## 1. Question

Run 39 (stage 1) found that two robust versions of the low-resolution consensus gate beat the plain mean (MK4-LR) on
object retention at matched far-background retention:

| Variant | Real VOC train | COCO-Cap dev |
|---|---|---|
| LR-med | +0.032 | +0.028 |
| LR-trim | +0.020 | +0.020 |

Their AUROC was no higher than LR's, so the gain may not reach mIoU. **Does one of them beat MK4-LR in mIoU, end to end?**

## 2. Candidates (all built from the same six low-resolution MK4 gates as MK4-LR)

Each gate is upsampled bilinearly to 64 × 64 and combined per pixel. The result is area-resized back to each module's
grid and clamped to [0, 1], exactly as MK4-LR (`jc_gate36.consensus`).

| Code | Name | Per-pixel combination |
|---|---|---|
| L | MK4-LR (reference) | mean of the six gates |
| D | LR-med | median of the six (mean of the two middle values; `torch.quantile` 0.5, as run 39) |
| T | LR-trim | mean after dropping the highest and the lowest |
| O | LR without module 2 | mean of five, leaving out `down_blocks.2.attentions.0` |

Run 39 found the O variant on the same VOC train photos. It can win only by a margin (§4). If an item lacks one of the six
gates, D and T fall back to the mean of those present; the number of such items is reported.

**Also captured:** no gate and MK4 (per-module gates).

**Everything else is as run 40:**
- one SD1.5 pass at t = 100;
- source C profiles (run 33 `C_all`, 490 COCO train photos and captions, no labels);
- SA-guided weights on;
- maps: seed only, fine-only (V0), capped hierarchy (V3);
- γ grid {0.25, 0.5, 0.75, 1, 1.5, 2}.

## 3. Data

| Role | Set | Size | Use |
|---|---|---|---|
| Development | **VOC train** (runs 32–37 items) | 2,170 items | Choose γ, thresholds and the winner. |
| Thresholds for COCO | Run 40's COCO train2017 threshold-fit items | 1,200 items | Thresholds only, as in run 40. |
| **Confirmation** | **New COCO train2017 set** | 3,500 photos (seed 41) | Primary test. |
| Reuse, secondary | COCO val2017 (run 40's set) | 6,624 items | Agreement only; it has been evaluated once. |
| Reuse, secondary | VOC val (runs 32–37) | 2,148 items | Agreement only; it has been evaluated several times. |

**Confirmation set:**
- drawn from COCO train2017 photos with at least one non-crowd instance of a VOC class;
- excludes the 490 profile photos and the run 40 threshold photos (asserted);
- items, masks, crowd-ignore regions and the ≥ 1% primary rule are exactly as in run 40;
- no earlier run has used these photos.

## 4. Development rule on VOC train (fixed now)

**Step 1: γ and thresholds.**
- γ is chosen per (gate, map) as in run 37: the best raw-map VOC-train mIoU, each map at its own best threshold.
- Each map's threshold is its best on VOC train (raw).
- COCO thresholds come from the run 40 threshold-fit items, per map (raw).
- VOC val uses the VOC-train thresholds.

**Step 2: score.** For each candidate X in {D, T, O}:

> d_X = ½ [(X − LR) at V0 + (X − LR) at V3]

- Setting: VOC train, SA weights on, each gate at its own selected γ, + DenseCRF at the VOC-train threshold, class-macro
  mIoU over all items.
- The 95% interval comes from a 2,000-sample photo bootstrap (`jc_realvoc.Boot`).

**Step 3: choose the winner.**
1. Between D and T: if |d_D − d_T| < 0.005, take **T**, the smaller change from LR (it keeps four of six gates). Otherwise
   take the larger.
2. O replaces that choice only if d_O ≥ d_choice + 0.005.

**Step 4: development passes** if the winner's d has a 95% CI above 0.

The winner is decided only from VOC train scores. The COCO scoring jobs run alongside it, but the rule never reads them.

## 5. Primary contrasts (fixed now)

**Setting:**
- the new COCO train2017 set, primary items;
- SA weights on;
- each gate at its VOC-train-selected γ;
- + DenseCRF at the COCO-fit threshold.

| # | Contrast | Map |
|---|---|---|
| P1 | winner − MK4-LR | capped hierarchy (V3) |
| P2 | winner − MK4-LR | fine-only (V0) |

- Intervals: 95% from a 2,000-sample photo bootstrap, the same resamples for every map (paired).
- Reading: CI > 0 = helps, CI < 0 = hurts, otherwise not resolved.

**Confirmation rule.** The winner is **confirmed over MK4-LR** if:
- development passed; and
- P1 or P2 helps; and
- neither hurts.

If development fails, P1 and P2 are still reported, as descriptive, and the verdict is "not confirmed (failed
development)".

**Secondary, also reported (descriptive):**
- the other two candidates against LR;
- γ = 1 for every gate;
- MK4-LR − MK4;
- candidates − no gate;
- the hierarchy effect (V3 − V0) per gate;
- seed-only maps;
- person excluded;
- all items, including those below 1%;
- every threshold on the grid (raw);
- the 0.55 rules;
- per class, with counts of classes better and worse;
- the two reuse sets (COCO val2017 primary items; VOC val all items).

The generated VOC-Sim and COCO-Cap halves are **not** included: their pipeline (`jc_gen38.py`) is separate, and run 38 used
them repeatedly.

## 6. Checks (smoke)

- MK4-LR equals `jc_gate36.consensus` with LR weights: max |Δ| < 1e-6.
- V0 and V3 from the seed with no gate match `sg_combined_v2.aggregate`: max |Δ| < 1e-5 after min-max.
- At every pixel, D and T lie between the minimum and maximum of the six gates.
- All gates are in [0, 1]; all maps are finite.
- Every confirmation photo is outside the profile and threshold-fit sets (asserted).

## 7. Files and cost

- **Script:** `scripts/jc_cons41.py`.
- **Report:** `runs/41_robust_consensus/result.md`.
- **Data:** `<CACHE>/jc_cons41/`.

| Part | Cost |
|---|---|
| capture | about 17,000 items; about 5 GPU-hours over up to 4 GPUs |
| DenseCRF scoring | about 22 maps per item, in parallel CPU shards |

- Smoke outputs are tagged `_smoke` and kept.
