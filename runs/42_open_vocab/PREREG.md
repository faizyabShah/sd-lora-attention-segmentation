# Run 42: does the low-resolution consensus gate generalise to classes never used in development? (pre-registration)

Committed before any full profile dump, capture or scoring of this run. The smoke test uses only COCO train2017 photos (no
val2017 photo is passed through the model), and its numbers are not used. BRACE stays frozen. Nothing is promoted.

**Confirmatory for the unseen-class question** (O2): no mask of the 60 unseen classes has been scored by any earlier run,
and nothing below is tuned on them. The COCO val2017 photos were used before for the 20 VOC classes (runs 40 and 41), and
this is stated next to every result. **Profile size (O4) is descriptive.**

## 1. Questions

- **O2.** MK4-LR was developed only on the 20 PASCAL VOC classes. For COCO's other 60 classes, with profiles built the same
  way from captioned photos, does the gate still improve over no gate, and over MK4?
- **O4.** How does this change when a new class has 5, 20 or 100 caption-matched reference photos?

## 2. Frozen pipeline (chosen on seen classes only)

| Part | Setting |
|---|---|
| Capture | One SD1.5 pass at t = 100 (run 32 noise), prompt `A photograph of a {w}`, image resized to 512 |
| Seed | Cross-attention to the class word, **uniform** module weights (no SA-guided weights) |
| Gate | **MK4-LR**: MK4 gates at the six modules below 64 × 64, averaged at 64 × 64 (`jc_gate36.consensus`), applied at every module, **γ = 1** |
| Propagation | **Fine-only (V0)**: two 64 × 64 self-attention steps (`jc_realvoc.propagate`); no hierarchy |
| Read-out | **Raw map, primary.** DenseCRF R2 secondary. Threshold per map from seen-class COCO train items (§4). |

**Comparison arms:** no gate (`u-`) and MK4 (per-module gates, γ = 1), in the same capture.

## 3. Classes, words and profiles

**Classes.**
- **Seen (20):** the VOC classes, with COCO names mapped (airplane → aeroplane, motorcycle → motorbike, couch → sofa).
- **Unseen (60):** the other 60 COCO categories.

**Words.**
- Seen classes keep their earlier prompt words and caption rules (`jc_trio.SYN`).
- Unseen classes use the **COCO category name verbatim** as the prompt word.
- Caption rules for the unseen classes are fixed in `scripts/jc_ov42.py` (`SYN_NEW`), written before any capture. Each rule
  is the name, obvious synonyms and listed irregular plurals, with an optional `s` / `es`; `bear` excludes `teddy bear` and `stuffed bear`. The `stuffed bear` exclusion was added after the smoke printed caption examples; no capture or score had run.
- Known ambiguous words (for example `orange`, `ball`, `glove`, `remote`) are kept as written, not tuned.

**Profile photos (captions only, no labels).**
- From COCO train2017, excluding the run 40 threshold-fit photos.
- A photo is a positive for word *w* if any of its captions matches *w*'s rule.
- Each class gets up to **100 positives**: shuffled with seed 42, assigned rarest class first, one photo per class (as run
  33). Classes with fewer available photos take what exists; the counts are reported.
- **Sizes 5, 20, 100** are nested prefixes of that list. **Size 20 is primary**, close to run 33's average of about 25.

**Object profile** (unchanged rule, run 33):
- per module, the mean of the top 5% of queries by the class word's attention on its positive photos;
- each photo is prompted with its own class word, and attention uses keys from the word alone (`jc_trio.class_keys`).

**Negative centroids** (run 33 rule, with a fixed pool):
- **Pool:** the size-20 positives of all 80 classes.
- For word *w*: 480 pool photos whose captions do **not** match *w* (seed 42). This is about the size of run 33's negative
  set, about 470.
- Take *w*'s top 1% attended queries on each, then 4 spherical k-means centroids (`jc_mneg.cluster`).
- The negatives do not change with profile size, so O4 varies only the positives.

## 4. Evaluation data

- **Evaluation:** COCO val2017 minus the 200 Section 136 images, all 80 classes.
  - An item is an (image, class) pair with at least one non-crowd instance.
  - Truth is the union of instance polygons; crowd regions are ignored (run 40 code).
  - **Primary items:** class area ≥ 1%. All items are secondary.
- **Thresholds:** run 40's 1,200 **seen-class** COCO train2017 items, recaptured with this run's maps. Each map's threshold
  (grid 0.30–0.70) is its best raw mIoU there. **No unseen-class mask is used for any choice.**
- **Disjointness (asserted):** profile photos ∩ threshold photos = ∅; all are train2017; evaluation is val2017 minus Section
  136.

## 5. Primary contrasts (fixed now)

**Setting:** primary items, raw V0 map at the seen-class threshold, profile size 20, γ = 1. Class-macro mIoU over the
**60 unseen** classes. 95% intervals from a 2,000-sample photo bootstrap, the same resamples for every map (paired).

| # | Contrast | Classes |
|---|---|---|
| **U1** | MK4-LR − no gate | unseen |
| **U2** | MK4-LR − MK4 | unseen |
| S1 | MK4-LR − no gate | seen (anchor, same capture) |
| S2 | MK4-LR − MK4 | seen (anchor) |

**Reading:**
- The **gate generalises to unseen classes** if U1's CI is above 0.
- **LR's advantage over MK4 generalises** if U2's CI is above 0.
- Either CI below 0 = hurts; otherwise not resolved.

**Secondary (descriptive):**
- seed-only maps;
- DenseCRF;
- all items;
- every threshold;
- unseen classes better / worse (LR vs none, LR vs MK4);
- per-class IoU with each class's number of positive photos;
- the ratio of unseen to seen gains;
- the share of modules with no profile;
- the seen classes again with run 33's `C_all` profiles, to check that the new fitter reproduces the old one;
- person excluded for the seen anchor.

**O4 (descriptive):** U1, U2, S1 and S2 at profile sizes 5, 20 and 100.

## 6. Checks (smoke, COCO train photos only)

- V0 from this script matches `jc_realvoc.propagate`: max |Δ| < 1e-5.
- MK4-LR equals `jc_gate36.consensus` with LR weights: max |Δ| < 1e-6.
- Caption rules: example matches are printed for every class. `teddy bear` must not match `bear`.
- All gates are in [0, 1]; all maps are finite; the disjointness asserts hold.

## 7. Files and cost

- **Script:** `scripts/jc_ov42.py`.
- **Report:** `runs/42_open_vocab/result.md`.
- **Data:** `<CACHE>/jc_ov42/`.

| Part | Cost |
|---|---|
| Profile dump | about 8,000 photos; about 1 GPU-hour over 2 GPUs |
| Fit | CPU |
| Capture | about 15,000 val items plus 1,200 threshold items; about 2 GPU-hours over 4 GPUs |
| Scoring | CPU shards |

- Smoke outputs are tagged `_smoke` and kept.
