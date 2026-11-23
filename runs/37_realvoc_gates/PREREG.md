# Run 37: new absent-class gates on real PASCAL VOC photos (pre-registration, amended)

Committed before any scoring of this run. BRACE stays frozen. **Exploratory.**

**Third look at VOC val.** Runs 32 and 33 already reported VOC 2012 val. Nothing here is chosen on val:
- every threshold, gate strength and RW weight is fitted on VOC train;
- the contrasts below are fixed now.

**Amendment (2026-11-23, before any scoring; owner's request).** The first version (commit 7926e3ba2) took the gate
strengths γ and MK4-RW's module weights from run 36 (generated images). It is replaced by fitting both **on real VOC
train photos**, because runs 32–33 showed that fitting on real photos is what matters.
- The first version's chain was cancelled during capture; nothing was scored.
- Its partial capture (`<CACHE>/jc_realvoc37/`) is not used.

## 1. Question

Run 36 (generated VOC-Sim / COCO-Cap) found the following:
- **MK4-LR improves on MK4.** MK4-LR is MK4's gates at the six low-resolution modules averaged into one map and
  applied at every module. MK4-RW (reliability-weighted) improves on MK4 in regime V only.
- **The gain is already present in the seed.**
- **SN** (self-negative) is worse than MK4 at every stage.

Do these hold on real photos, when everything is fitted on real photos? Each gate is read with and without the
SA-guided layer weights, and with and without the hierarchy.

## 2. Setting (as run 33, t = 100 only)

- **Items, prompts, recorder:** as runs 32 and 33.
  - 2,170 VOC train items and 2,148 val items.
  - Prompt `A photograph of a {w}`.
  - `scripts/jc_realvoc.py` is imported unchanged.
- **Capture:** one VAE encode and one conditional SD1.5 call per item at t = 100, with the run 32 noise draw.
- **Profiles:** run 33's t = 100 fit (`<CACHE>/jc_realvoc_t/fit_t100.pt`), which is already fitted on real photos.

| Source | Fit images | "Absent" means | Role |
|---|---|---|---|
| **C** | 490 real COCO train2017 photos | none of the photo's 5 captions mentions c | **primary (label-free profiles)** |
| A | VOC train items: cross-fitted by fold on train, all of train for val | not in the image labels | reference (uses labels) |

## 3. Factors

| Factor | Off | On |
|---|---|---|
| **W** SA-guided layer weights | uniform over the 11 CA modules | A3 weights (`jc_realvoc.layer_weights`) |
| **H** hierarchy | fine-only (V0) | **capped hierarchy V3** (BRACE's 16 → 32 → 64) |
| **G** gate | none | MK4, SN, MK4-LR or MK4-RW |

Gates. The definitions are run 36's.
- **MK4:** run 31.
- **SN:** MK4's four absent centroids plus the item's own self-negative: the mean q̂ over the bottom 30% of t̄_c, where
  t̄_c is attention to the class-word key alone.
- **MK4-LR:** the consensus of MK4's gates at the six modules below 64 × 64, applied at every module.
- **MK4-RW:** the reliability-weighted consensus of MK4's gates.

### Fitted on VOC train photos (fixed now)

**RW module weights:**
- w_m = clip(2·(AUROC_m − 0.5), 0, 1).
- AUROC_m is the class-balanced AUROC (object vs far background, as run 35) of the MK4 gate at module m, over VOC
  train items, with that source's profiles. Source A uses cross-fitted profiles.
- **Train items** use weights from the other fold; **val** uses all of train.
- Weights are fitted separately for sources C and A.

**Gate strength γ:**
- The gate is applied as g^γ, γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}.
- **Every gate, MK4 included,** gets its γ the same way: the γ with the best VOC train mIoU on raw maps, each map at
  its own best train threshold.
- γ is chosen per gate family and map: SA weights × source × gate × map (seed, V0, V3).
- γ = 1 (untuned) is also reported for every gate, as a descriptive result.

**Use of labels.** RW weights and γ use VOC train masks, the same status as the thresholds, which are already chosen
on VOC train. Source C's profiles stay label-free. No val image or val mask is used for any fit.

A module without a profile gets g = 1.

## 4. Thresholds, scoring, reading

- **Thresholds:** grid 0.30–0.70, chosen per map on VOC train (raw maps). On val, DenseCRF is applied at that
  threshold and at a fixed 0.55. DenseCRF is applied to the train-selected-γ maps, the γ = 1 maps and the ungated
  maps (V0 and V3).
- **Metric:** class-macro mIoU over items.
- **Intervals:** 95% from a 2000-sample photo bootstrap. The same resamples are used for every map, so differences
  are paired.
- **Reading:** CI above 0 = **helps**; CI below 0 = **hurts**; otherwise **not resolved**.

**Primary contrasts (fixed now).** Val, + DenseCRF, train-selected threshold and γ, source C, SA weights on:

| # | Contrast | Hierarchy |
|---|---|---|
| P1 | MK4-LR − MK4 | capped (V3) |
| P2 | MK4-LR − MK4 | fine-only (V0) |
| P3 | MK4-RW − MK4 | V3 |
| P4 | MK4-RW − MK4 | V0 |
| P5 | SN − MK4 | V3 |
| P6 | SN − MK4 | V0 |

**Reading of the run:** a gate "holds on real photos" if it helps in P(V3) or P(V0) and hurts in neither. This is
not a promotion: BRACE stays frozen, and any change to the paper's method is a separate owner decision.

**Secondary, also reported:**
- the full W × H × G table for sources C and A, with paired differences;
- the SA-weight and hierarchy effects for each gate;
- the selected γ table;
- γ = 1 results;
- the raw and 0.55 rules;
- seed-only maps;
- per-class IoU;
- the RW weights and per-module AUROC on real photos;
- the mean gate values.

## 5. Checks (smoke, before the full run)

- The single-call V0 / V3 match `sg_combined_v2.aggregate` fed that call 11 times: max |Δ| < 1e-5 after min-max.
- SN with the self-negative replaced by a copy of an existing centroid equals MK4: max |Δ| < 1e-5.
- The CA module names in the RW weights equal the recorder's.
- Every map is finite, and every gate is in [0, 1].

## 6. Files and cost

- **Script:** `scripts/jc_realvoc37.py` (imports `jc_realvoc`, `jc_gate36`).
- **Report:** `runs/37_realvoc_gates/result.md`.
- **Data:** `<CACHE>/jc_realvoc37b/`.
- **Cost (estimate):**

  | Part | Cost |
  |---|---|
  | RW weight pass | about 15 GPU-minutes |
  | capture | 98 seeds × 3 maps per item; about 1 GPU-hour per split over 2 shards |
  | maps on disk | about 10 GB |
  | scoring | about 1–2 CPU-hours |

- Smoke outputs are tagged `_smoke` and kept.
