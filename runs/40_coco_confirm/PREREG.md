# Run 40: confirmation of the low-resolution consensus gate on untouched real COCO val2017 photos (pre-registration)

Committed before the full capture or any full scoring of this run. A smoke test (40 COCO train items, 41 val items) ran
just before this commit to check the code. Its numbers are not used, and nothing in the design was changed after it. BRACE stays frozen. **Confirmatory in intent.** The method and
the contrasts are frozen below, and nothing is chosen on the evaluation images.

## 1. Question

Runs 36–38 found MK4-LR > MK4 > no gate on generated images and on real VOC val. VOC val has been used three times,
and the generated test halves several times. Does **MK4-LR**, frozen now, beat MK4 and no gate on real photos that no
earlier run has looked at?

## 2. Data

**Evaluation set: COCO val2017, minus the 200 images of the Section 136 set**
(`datasets/coco2017/sg_real_data_labels/`). Those 200 are the only val2017 images any earlier run used.
- Classes: the 20 PASCAL VOC classes. COCO → VOC names: airplane → aeroplane, motorcycle → motorbike, couch → sofa.
  The other 17 names are the same.
- An **item** is an (image, class) pair, with ≥ 1 non-crowd instance of the class.
- **Truth:** the union of the class's non-crowd instance masks, polygons rasterised.
- **Ignore:** the class's crowd regions.
- **Primary items:** the class covers ≥ 1% of the image, about 41 latent pixels. Smaller objects are below what a
  64 × 64 attention map can resolve. **All items** are reported as secondary.

**Threshold-fit set: COCO train2017 photos** that are not among the 490 profile-fit photos. Up to 60 items per class,
drawn at random with seed 0, with the same ≥ 1% rule. Only thresholds are fitted here (the fit-on-real-photos rule).

## 3. Method (frozen now)

- **Capture:** one SD1.5 pass at t = 100 per item, as runs 32–37: prompt `A photograph of a {w}`, the run 32 noise
  draw, image resized to 512.
- **Profiles:** source **C**, run 33 `C_all`. These are 490 real COCO **train2017** photos and their captions, with
  no labels. No profile is refitted.

| Factor | Levels |
|---|---|
| SA-guided weights W | on (A3) / off (uniform) |
| Gate | none, MK4, **MK4-LR**, MK4-RW |
| Map | seed only, fine-only (V0), capped hierarchy (V3) |

- **Gate strength (primary): γ = 1, untuned.**
- **Secondary arms:** run 37's VOC-train-selected γ for source C with W on: MK4 0.25 / 0.5 / 0.75 and LR, RW 1.5
  (seed / V0 / V3).
- **RW weights:** run 37's (VOC train, source C).
- **Thresholds:** per map, from the grid 0.30–0.70, chosen on the COCO train threshold-fit items (raw maps). DenseCRF
  R2 is applied at that threshold and at a fixed 0.55.

## 4. Primary contrasts (fixed now)

**Setting:** primary items, W on, γ = 1, + DenseCRF at the train threshold.

| # | Contrast | Map |
|---|---|---|
| C1 | MK4-LR − MK4 | capped hierarchy (V3) |
| C2 | MK4-LR − MK4 | fine-only (V0) |
| C3 | MK4-LR − none | V3 |
| C4 | MK4-LR − none | V0 |

- **Metric:** class-macro mIoU.
- **Intervals:** 95% from a 2000-sample **photo** bootstrap. The same resamples are used for every map, so
  differences are paired.
- **Reading:** CI > 0 = helps, CI < 0 = hurts, otherwise not resolved.

**Confirmation rule:**
- MK4-LR is **confirmed over MK4** if C1 or C2 helps and neither hurts.
- It is **confirmed over no gate** if C3 or C4 helps and neither hurts.

This is not a promotion. BRACE stays frozen, and any change to the paper's method is a separate owner decision.

**Secondary, also reported:**
- seed only (raw maps, train threshold): MK4-LR − MK4 and MK4-LR − none;
- MK4-RW, the selected-γ arms, and W off;
- the hierarchy effect (V3 − V0) for each gate;
- person excluded;
- per class, with counts of classes better and worse;
- every threshold on the grid (raw);
- the 0.55 rules;
- all items, including those below 1%.

## 5. Checks (smoke)

- The single-call V0 / V3 match `sg_combined_v2.aggregate`: max |Δ| < 1e-5 after min-max.
- The MK4-LR map equals `jc_gate36.consensus` with LR weights.
- The masks are non-empty, and every evaluation image is outside the Section 136 set and every fit image outside the
  490 profile photos (asserted).
- All maps are finite, and every gate is in [0, 1].

## 6. Files and cost

- **Script:** `scripts/jc_coco40.py`.
- **Report:** `runs/40_coco_confirm/result.md`.
- **Data:** `<CACHE>/jc_coco40/`.

| Part | Cost |
|---|---|
| capture | about 1 GPU-hour |
| scoring with DenseCRF | a few CPU-hours, in parallel |

- Smoke outputs are tagged `_smoke` and kept.
