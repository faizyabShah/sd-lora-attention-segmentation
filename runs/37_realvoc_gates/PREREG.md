# Run 37: new absent-class gates on real PASCAL VOC photos (pre-registration)

Committed before any capture or scoring of this run. BRACE stays frozen. **Exploratory.**

**Third look at VOC val.** Runs 32 and 33 already reported VOC 2012 val. Nothing here is chosen on val:
- the gate definitions and strengths come from run 36 (generated images);
- every threshold is chosen on VOC train.

## 1. Question

Run 36 (generated VOC-Sim / COCO-Cap) found the following:
- **MK4-LR improves on MK4.** MK4-LR is MK4's gates at the six low-resolution modules averaged into one map and
  applied at every module. MK4-RW (reliability-weighted) improves on MK4 in regime V only.
- **The gain is already present in the seed**, before the hierarchy.
- **SN** (self-negative) is worse than MK4 at every stage.

Do these hold on real photos? Each gate is read with and without the SA-guided layer weights, and with and without
the hierarchy.

## 2. Setting (as run 33, t = 100 only)

- **Items, prompts, recorder:** as runs 32 and 33.
  - 2,170 VOC train items and 2,148 val items (one item = one photo × one present class).
  - Prompt `A photograph of a {w}`.
  - `scripts/jc_realvoc.py` is imported unchanged.
- **Capture:** one VAE encode and one conditional SD1.5 call per item at t = 100, with the run 32 noise draw
  (`jc_realvoc.one_pass`).
- **Profiles:** run 33's t = 100 fit (`<CACHE>/jc_realvoc_t/fit_t100.pt`). It is not refitted.

| Source | Fit images | "Absent" means | Role |
|---|---|---|---|
| **C** | 490 real COCO train2017 photos | none of the photo's 5 captions mentions c | **primary (label-free)** |
| A | VOC train items: cross-fitted by fold on train, all of train for val | not in the image labels | reference (uses labels) |

## 3. Factors

| Factor | Off | On |
|---|---|---|
| **W** SA-guided layer weights | uniform over the 11 CA modules | A3 weights (`jc_realvoc.layer_weights`) |
| **H** hierarchy | fine-only (V0) | **capped hierarchy V3** (BRACE's 16 → 32 → 64); the uncapped V1 is also reported |
| **G** gate | none | MK4, SN, MK4-LR or MK4-RW |

Gates. All are run 36 definitions, with nothing refitted on real photos.
- **MK4:** run 31, γ = 1.
- **SN:** MK4's four absent centroids plus the item's own self-negative: the mean q̂ over the bottom 30% of t̄_c,
  where t̄_c is attention to the class-word key alone (`jc_realvoc.tbar`).
- **MK4-LR:** the consensus of MK4's gates at the six modules below 64 × 64, applied at every module (run 36 §2).
- **MK4-RW:** the reliability-weighted consensus of MK4's gates. The weights are run 36's regime X weights
  (`runs/36_sn_consensus/stage1.json`, `weights.X.MK4.all`), fitted on generated dev images.

**Strengths: fixed from run 36 regime X, not refitted.**

| Gate | γ |
|---|---|
| MK4 | 1 |
| SN | 0.532 |
| MK4-LR | 0.716 |
| MK4-RW | 0.765 |

The values are read from `runs/36_sn_consensus/stage1.json`, `gstar.X`. On real photos these strengths are a
transfer: they are not tuned for VOC, which is a stated limitation.

Each gate multiplies the head-mean target attention of each CA module before the weighted sum, as in runs 32 and 33.
A module without a profile gets g = 1.

**Seeds:**
- u-, w- (no gate);
- {u, w} × {A, C} × {MK4, SN, LR, RW}.

That is 18 seeds, each read as seed, V0, V1 and V3. DenseCRF R2 is applied to every V0, V1 and V3 map.

## 4. Thresholds, scoring, reading

- **Thresholds:** grid 0.30–0.70, chosen per map on VOC train (raw maps). On val, DenseCRF is applied at that
  threshold and at a fixed 0.55.
- **Metric:** class-macro mIoU over items.
- **Intervals:** 95% from a 2000-sample photo bootstrap. The same resamples are used for every map, so differences
  are paired.
- **Reading:** CI above 0 = **helps**; CI below 0 = **hurts**; otherwise **not resolved**.

**Primary contrasts (fixed now).** Val, + DenseCRF, train-selected threshold, source C, SA weights on:

| # | Contrast | Hierarchy |
|---|---|---|
| P1 | MK4-LR − MK4 | capped (V3) |
| P2 | MK4-LR − MK4 | fine-only (V0) |
| P3 | MK4-RW − MK4 | V3 |
| P4 | MK4-RW − MK4 | V0 |
| P5 | SN − MK4 | V3 |
| P6 | SN − MK4 | V0 |

**Reading of the run:** MK4-LR "holds on real photos" if P1 or P2 helps and neither hurts. The same rule applies to
MK4-RW (P3, P4) and SN (P5, P6). This is not a promotion: BRACE stays frozen, and any change to the paper's
method is a separate owner decision.

**Secondary, also reported:**
- the full W × H × G table for sources C and A: mIoU of every cell;
- every gate − MK4 and every gate − none, per W and H;
- the uncapped hierarchy V1;
- the raw and 0.55 rules;
- seed-only maps;
- per-class IoU (person especially);
- the mean gate value per gate.

## 5. Checks (smoke, before the full run)

- The single-call V0 / V1 / V3 match `sg_combined_v2.aggregate` fed that call 11 times: max |Δ| < 1e-5 after
  min-max (as run 33).
- SN with the self-negative replaced by a copy of an existing centroid equals MK4: max |Δ| < 1e-5.
- The CA module names of the real-photo recorder equal the module names in run 36's RW weights.
- Every map is finite, and every gate is in [0, 1].

## 6. Files and cost

- **Script:** `scripts/jc_realvoc37.py` (imports `jc_realvoc`, `jc_gate36`).
- **Report:** `runs/37_realvoc_gates/result.md`.
- **Data:** `<CACHE>/jc_realvoc37/`.
- **Cost (run 32 / 33 timings):** capture about 20 GPU-minutes per split over 2 shards each; scoring under 1
  CPU-hour per split over 6 shards. The whole chain takes about 2 h.
- Smoke outputs are tagged `_smoke` and kept.
