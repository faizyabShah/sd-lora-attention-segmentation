# Pre-registration: (A) green-screen tokens, (B) depth–time signature, (C) caption-competitive attribution

Written 2026-11-16, before any exemplar is generated or any replay is run. Code: `scripts/gs_tokens.py` (A, stage 1) and `scripts/replay_abc.py` (A evaluation, B, C). BRACE stays frozen. Ground truth only scores.

## Shared evaluation

- **Images:** the 150 VOC-Sim + 150 COCO-Cap screening images (`CACHE/ana_dump`). These are development images, disclosed.
- **Replay:** the stored inputs of all 31 UNet calls, conditional prompt only, with the exact A3 seed weighting (as in `runs/13_value_purify`). Every arm comes from the same replay. Any GPU.
- **Mask:** min–max map, bilinear to 512, τ 0.55, DenseCRF R2.
- **Metric:** class-macro mIoU; Δ against `base` with an image-bootstrap 95% CI (2000 resamples).
- **`base`** is the frozen attribution convention: the caption on VOC-Sim ("A photograph of a {class}"), the class name alone on COCO-Cap.
- **Fidelity arm `live`:** the stored live seed with replayed self-attention.

## (A) Green-screen tokens

**Exemplars (stage 1, GPU):**
- **Model and settings:** plain SD1.5 with BRACE's live generation settings.
- **Prompt:** "a photo of a {class}, isolated on a plain white background", for the 20 classes, with hyphens written as spaces.
- **Seeds:** 16 per class, 50000 + 100·(class index) + j.
- **Colour key:**
  - a pixel is *white* if its smallest RGB channel is ≥ 235;
  - the backdrop is the union of white connected components touching the image border;
  - the object is everything else.
- **Label-free filter.** An exemplar is kept if:
  - at least 90% of the 2-pixel border frame is backdrop; and
  - the object covers 5–70% of the image.
- **Kept per class:** the first 10 kept exemplars, in seed order. A class with fewer than 3 keeps its original token (δ = 0), and this is reported.

**Training data per kept exemplar:**
- **Queries:** the cross-attention queries of the 11 A3 modules at UNet calls 8, 15 and 22, from the exemplar's own generation trajectory (conditional replay).
- **Cells:**
  - a cell is *object* if pooled object ≥ 0.9 and *backdrop* if ≤ 0.1; mixed cells are dropped;
  - at 64 × 64, up to 192 object and 192 backdrop cells are sampled (seeded);
  - at 16 × 16 and 8 × 8, all pure cells are used.

**Token (one offset δ per class):**
- δ is added to every class-piece position of the class-only attribution context, the text-encoder output of the class name padded to 77.
- **Score per module and call:** s = log of the head-mean class probability under the 77-token softmax.
- **Loss:** class-balanced binary cross-entropy on a·(s − c), with one learned scale a and offset c per module, discarded afterwards.
  - **Positives:** object cells of the class's own exemplars.
  - **Negatives:** their backdrop cells, and all sampled cells of other classes' exemplars.
- **Batch:** 4 own-class and 4 other-class exemplars per step, with replacement.
- **Optimiser:** Adam, 300 steps; learning rate 0.005 on θ = δ/‖e‖ and 0.05 on a and c; penalty 0.1·‖θ‖². Torch seed 0.
- Nothing here is tuned; there is no validation loop.

**Arms:**
- `gs` (**primary**): the class-only context with the learned δ.
- `gs_lin` (secondary): one closed-form step of size 0.1·‖e‖ along the gradient of [mean class logit on own-class object cells − mean class logit on negatives], from the same exemplars.
- `clsonly`: the class-only context without δ. This isolates the token from the change of prompt on VOC-Sim.

**Gate (`gs` is promising only if both hold):**
- COCO-Cap Δ vs `base` ≥ **+0.010** with CI lower bound **> 0**.
- VOC-Sim Δ vs `base` ≥ **−0.005**.

**Reported:** `gs − clsonly`, the kept-exemplar counts per class, and the training loss at the first and last step.

## (B) Depth–time signature (diagnostic)

**Profile:**
- For each 64 × 64 cell, the head-mean class probability at each of the 11 A3 modules and 31 calls (341 values), under `base` attribution, bilinear to 64 × 64.
- Take logs, then double-centre: subtract the mean over cells for each (module, call), and the mean over (module, call) for each cell. This removes each layer's scale and each cell's overall magnitude.

**Test features:**
- **`sig`:** cos(profile, mean profile of core F) − cos(profile, mean profile of clear background B). F and B are the label-free cores of `runs/08_feat_bleed`, from the stored live V3.
- **`fl`:** the mean centred value over the 64-px modules at calls 16–30, minus the mean over the 8/16-px modules at calls 0–15 ("fine-late minus coarse-early").

**Task and baseline:** bleed vs correct on the cells Q, with the baseline of `runs/09_equiv_noise` (RGB, contour, distance, BRACE confidence, depth, area, RGB classifier, mean CA and mean V3 maps).

**Gate (decisive set COCO-Cap):** out-of-fold logistic regression, baseline vs baseline + `sig` + `fl`.
- **PASS:** ΔAUROC ≥ **+0.05** with image-bootstrap 95% CI lower bound **> 0**.
- A pass only licenses a pre-registered refinement test.

**Reported:** single-feature AUROCs, and the VOC-Sim gate.

## (C) Caption-competitive attribution

**Arms:**
- `incap`: the class word scored inside the caption. If the class name's tokens occur in the caption, the caption is used as is (first occurrence); otherwise "{caption}, {class name}".
- `tmpl` (control): "A photograph of a {class name}", i.e. a sentence context without competing nouns.
- `clsonly`: the class name alone.
- On VOC-Sim `incap` and `tmpl` equal `base`; on COCO-Cap `clsonly` equals `base`.

**Gate (COCO-Cap):** Δ(`incap` − `base`) ≥ **+0.010** with CI lower bound **> 0**.

**Reported:**
- Δ(`incap` − `tmpl`) on COCO-Cap: does competition matter beyond a sentence context?
- Δ(`clsonly` − `base`) on VOC-Sim.
- The COCO-Cap split by whether the class word is in the caption.

## Fixed

Nothing is tuned after the run: not the prompts, the key thresholds, the filter, the optimiser settings or the bars. There is no follow-up sweep after a negative result.

## Amendment 1 (2026-11-16, after the first smoke test, before any token was learned or any arm scored)

**What the smoke test showed.**
- All 32 exemplars for 2 classes were rejected: SD1.5 does not draw white backdrops for this prompt. It draws plain grey or coloured ones, often with a soft shadow, so the "white pixel" key cannot work.
- The replay and the (B) features ran correctly at about 11 s per image.
- No token existed, so no (A) result was seen.

**Changes to (A), stage 1 only.** The prompt, seeds and everything downstream are unchanged.
- **Colour key, replaced:**
  - Convert to Lab and smooth with σ 1.5. A pixel is *smooth* if its Sobel gradient magnitude is below 8.
  - The backdrop is the union of smooth connected components touching the border, plus any smooth component of at least 64 pixels whose mean Lab colour is within 10 of the backdrop's median colour (for example, the wall seen through a wheel).
  - The object is the rest, after a 2-iteration binary opening and removal of components under 0.5% of the image.
- **Filter:** at least **95%** of the 2-pixel border frame is backdrop (was 90%); the object covers 5–70% (unchanged).
- **Up to 40 seeds per class** (was 16), stopping at 10 kept, because textured backdrops are rejected and the yield is lower than assumed.

**How these values were chosen.** The gradient threshold (8, from 4 / 8 / 14) and the filter were chosen by eye on the 32 smoke exemplars of aeroplane and bicycle. No ground truth, no evaluation image and no segmentation score was involved.

**Run order.**
- (B) and (C) do not depend on the tokens. They are scored from a token-free replay (tag `bc`) submitted now; that replay is their registered result.
- (A) is scored from a second replay, run after the tokens exist, with its own `base` arm from the same replay.
