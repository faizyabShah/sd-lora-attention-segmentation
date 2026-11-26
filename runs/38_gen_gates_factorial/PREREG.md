# Run 38: run 37's factorial on the generated benchmarks (pre-registration)

Committed before any capture or scoring of this run. BRACE stays frozen. **Exploratory.**

## 1. Question

Run 37 ran the following factorial on real VOC photos:
- SA-guided layer weights: off / on;
- hierarchy: off (fine-only) / on (capped);
- gate: none, MK4, SN, MK4-LR, MK4-RW.

On real VOC it found:
- MK4-LR and MK4-RW improve on MK4, with and without the hierarchy;
- SN hurts;
- once LR or RW is on, the hierarchy hurts.

This run asks the same question on the generated benchmarks, **VOC-Sim and COCO-Cap**, with the same factors and
tables. Everything tunable is fitted on held-in **dev** images of the generated sets (the analogue of VOC train in
run 37), never on test images.

## 2. Setting (as runs 31, 34–36)

- **Benchmarks:** VOC-Sim (490) and COCO-Cap (707). One new generation-time capture of all 1,197 images with the
  runs/20 prompts. Run 31's `split.json`: dev folds dA / dB and test halves.
- **Regimes:**

  | Regime | Profiles fitted on | COCO-Cap is |
  |---|---|---|
  | **V** | VOC-Sim dev only | untuned transfer |
  | **X** | VOC-Sim dev + COCO-Cap dev ("half of each") | in-domain (label-free adaptation) |

- **Profiles:** run 35's fit (`<CACHE>/jc_gate35/fit.pt`): MK4 object means and 4 absent centroids. It is not refitted.
- **Held-out rules (as runs 34–36):**
  - a dev image uses the other fold's profiles;
  - a test image uses all dev;
  - regime V on COCO-Cap uses all VOC-Sim dev.
- **Gates:** run 36's definitions (`jc_gate36.gate_set`): MK4, SN, MK4-LR, MK4-RW.
- **RW module weights:** run 36's (`runs/36_sn_consensus/stage1.json`). They come from run 35's held-out per-module
  AUROC on **dev** images, with the same fold rule as the profiles. These are the generated-domain analogue of
  run 37's VOC train weights.

## 3. Factors

| Factor | Off | On |
|---|---|---|
| **W** SA-guided layer weights | uniform over the 11 CA modules (and all denoising calls) | A3 weights (BRACE seed) |
| **H** hierarchy | fine-only (V0) | **capped hierarchy V3** (BRACE's 16 → 32 → 64) |
| **G** gate | none | MK4, SN, MK4-LR or MK4-RW |

- The seed with W on and no gate is BRACE's A3 seed. V3 with W on and no gate is BRACE before DenseCRF.
- Each gate is applied as g^γ, multiplying each CA module's head-mean target attention (as run 36).

## 4. Fitted on dev (fixed now)

- **Thresholds:**
  - per regime and map, from the grid 0.30–0.70, on dev raw maps;
  - regime V uses the VOC-Sim dev curve; regime X uses the mean of the VOC-Sim dev and COCO-Cap dev curves (run 36 rule).
- **Gate strength γ:**
  - γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}, for **every** gate, MK4 included;
  - chosen per regime × W × gate × map (seed, V0, V3);
  - the γ with the best dev raw mIoU, each map at its own dev threshold, using the regime's dev rule.
- **γ = 1 (untuned)** is also reported for every gate, as a descriptive result.
- **No test image or test mask** is used for any fit.

## 5. Scoring and reading

- **DenseCRF R2** is applied to V0 and V3 maps on test images, at the dev-selected threshold and at a fixed 0.55. It
  covers the γ-selected maps, the γ = 1 maps, and the ungated maps. This is run 37's rule: the threshold is chosen on
  raw dev maps. Run 36 instead chose thresholds on dev DenseCRF curves, so the numbers are not identical to run 36's.
- **Metric:** class-macro mIoU on the **test halves**, reported separately for VOC-Sim and COCO-Cap.
- **Intervals:** 95% from a 2000-sample image bootstrap. The same resamples are used for every map, so differences are
  paired.
- **Per-set reading:** CI above 0 = **helps**; CI below 0 = **hurts**; otherwise **not resolved**.
- **"Improves on" (run 36 rule):** COCO-Cap CI lower bound > 0 **and** VOC-Sim CI lower bound ≥ −0.005.

**Primary contrasts (fixed now).** In each regime: test, + DenseCRF, dev-selected threshold and γ, SA weights on.

| # | Contrast | Hierarchy |
|---|---|---|
| P1 | MK4-LR − MK4 | capped (V3) |
| P2 | MK4-LR − MK4 | fine-only (V0) |
| P3 | MK4-RW − MK4 | V3 |
| P4 | MK4-RW − MK4 | V0 |
| P5 | SN − MK4 | V3 |
| P6 | SN − MK4 | V0 |

**Reading:** in a regime, a gate "holds" if it improves on MK4 in P(V3) or P(V0), and in neither map does it hurt on
either benchmark. Both regimes are read; neither is promoted over the other. This is not a promotion: BRACE stays
frozen, and any change to the paper's method is a separate owner decision.

**Secondary, also reported (per regime, VOC-Sim and COCO-Cap test):**
- the full W × H × G table, with paired differences (MK4 − none, SN / LR / RW − MK4);
- the SA-weight and hierarchy effects for each gate;
- the selected γ table;
- γ = 1 results;
- the raw and 0.55 rules;
- seed-only maps;
- per-class IoU;
- the RW weights used.

## 6. Checks (smoke, before the full run)

- The ungated W-on seed equals `jc_trio.ca_seeds` with the recorded target, max |Δ| < 1e-5. A gated seed equals
  `ca_seeds` with that gate, max |Δ| < 1e-5. The A3 seed matches the runtime's `selected/A3` within 1e-2 (run 36's
  fidelity check).
- The batched V0 / V3 match `sg_combined_v2.aggregate` for the BRACE seed: max |Δ| < 1e-5 after min-max.
- SN with the self-negative replaced by a copy of an existing centroid equals MK4: max |Δ| < 1e-5.
- Every map is finite, and every gate is in [0, 1].

## 7. Files and cost

- **Script:** `scripts/jc_gen38.py` (imports `jc_gate36` and its imports).
- **Report:** `runs/38_gen_gates_factorial/result.md`.
- **Data:** `<CACHE>/jc_gen38/`, about 3 GB.

| Part | Cost estimate |
|---|---|
| capture | about 6 GPU-hours, over 4 shards |
| raw scoring | about 1 CPU-hour |
| DenseCRF on test | about 2–4 CPU-hours |

- Smoke outputs are tagged `_smoke` and kept.
