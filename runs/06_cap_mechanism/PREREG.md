# Pre-registration: does BRACE's cap help most where competing objects are present? (mechanism diagnostic)

Written 2026-11-15, before any competition score or any statistic below has been computed. Code: `scripts/cap_mechanism.py`.

This is a **mechanism analysis** for the BRACE paper, not a method test:
- No method is selected, tuned or built from it.
- BRACE and every reported result stay frozen.
- COCO-Cap is used post hoc to explain an existing, already reported gain, never to choose anything.
- Ground truth only measures, and in the diagnostic controls (object area).

## Data (one capture)

- **Images.** The stored paper captures for all **490 VOC-Sim and 707 COCO-Cap** images (`CACHE/vocsim_maps`, `CACHE/cococap_maps`).
- **Same capture.** Each file holds the generated image, the ground truth and `map/base` = [seed, P1, **V0**, **V3**, C16] from a single live capture. V0 (FINE-P2) and V3 come from the same `sg_combined_v2.aggregate` call. Nothing is regenerated.
- **SCLIP.** Run once on exactly these stored images (same pixels):
  - ViT-B/16, VOC21 vocabulary, logit scale 65;
  - softmax over all 21 classes, max over synonyms;
  - **no** probability or area thresholds;
  - stored at 128 × 128 and upsampled bilinearly to 512 for pixel statistics.

## Masks

- **V3 (BRACE):** min–max(V3), bilinear to 512, τ = 0.55, frozen DenseCRF R2.
- **V0:** the same path with τ₀.
  - τ₀ is the value on `cc.THR_CRF` (0.30–0.90 in steps of 0.05) that maximises **full VOC-Sim** class-macro mIoU of V0 + R2. Ties go to the lower τ.
  - The same rule applied to V3 gives 0.55. τ₀ is reported.
- **Gate.** The recomputed V3 + R2 counts must match the stored `base|V3+R2|raw` counts at 0.55 on every image, within 0.1% of pixels. Otherwise the run stops.
- **Secondary (no CRF).** Raw V3 and raw V0, each at its own full-VOC-best τ on `cc.THR`.

## Competition score (label-free; fixed now)

- **Per pixel:** q(x) = max over k ∉ {background, target class y} of P_SCLIP(k | x).
- **Competitor region:** R_η = {x : q(x) > η}.
- **Per image:** C_i = |R_η| / (H·W).
- **Primary:** η = 0.5. **Sensitivity:** η = 0.3. η is not chosen against any outcome.
- **Degeneracy fallback (fixed before the full run).** A 6 + 6-image code test found competitor probabilities rarely above 0.5. If C at η = 0.5 is zero on more than 50% of COCO-Cap images, the primary region becomes the **argmax region**: pixels whose SCLIP argmax is a class k ∉ {bg, y}. It has no η.
  - The verdict then uses this region.
  - The argmax region is reported in every case, alongside η = 0.5 and η = 0.3.
  - Only the score distribution decides the switch, never Δ or any outcome.
- **Secondary count score:** K_i = the number of classes k ∉ {bg, y} that are the SCLIP argmax on more than 1% of the pixels.

## H1 (image level): does competition predict the V3 gain?

- **Outcome:** Δ_i = IoU_i(V3 + R2) − IoU_i(V0 + R2).
- **Model:** ordinary least squares Δ_i ~ z(C_i) + log(GT area_i) + class fixed effects, within each dataset. z is standardised within the dataset. Report β_C, the change in Δ per standard deviation of C.
- **Intervals:**
  - **Primary:** image bootstrap stratified by class, 2000 resamples, 95% percentile CI.
  - **Sensitivity:** class-cluster bootstrap, resampling the 20 classes, 2000 resamples.
- **Also reported:**
  - the model without the area term;
  - the mean within-class Spearman correlation of Δ with C;
  - the distribution of C (mean, quantiles, fraction zero);
  - the same model for the raw (no-CRF) masks, for η = 0.3 and for K_i.

## H2 (pixel level): does the cap preferentially remove false positives on competing objects?

- **Pixels.** Among V0's false-positive pixels, FP₀ = M₀ ∧ ¬GT. A pixel is **removed** if it is not in M₃.
- **Distance bands.** Both removal and competitor presence depend on distance from the target object, so pixels are stratified by Euclidean distance to the ground-truth object: (0, 8], (8, 16], (16, 32], (32, 64] and > 64 px.
  - Ground truth only measures here.
  - Distance to M₃ cannot be used: a removed pixel lies outside M₃ by definition, so that distance would determine the outcome.
- **Per image and band:** a 2 × 2 table of {on R_η, off R_η} × {removed, retained}.
- **Statistic:** the Mantel–Haenszel common odds ratio of removal for on-R vs off-R pixels, pooled over images and bands within a dataset. Report log OR with a class-stratified image bootstrap 95% CI (2000 resamples).
- **Also reported:**
  - the unstratified OR;
  - the per-band ORs;
  - the fraction of FP₀ on R_η;
  - for η = 0.3.

## H3 (descriptive): what does the cap change?

Per dataset, for V3 + R2 against V0 + R2:
- pooled and class-macro precision and recall;
- total FP and FN pixel changes, and the share of the IoU change attributable to each;
- the FP change split into on-R_η and off-R_η pixels;
- the fraction of images where IoU rises, falls or stays the same.

## Decision categories (fixed now)

The verdict is decided on **COCO-Cap**, at η = 0.5, with R2:

| category | condition |
|---|---|
| **Supported** | H1 β_C > 0 with CI lower bound > 0, **and** H2 log OR > 0 with CI lower bound > 0 |
| **Partial** | exactly one of the two holds |
| **Not supported** | neither holds |

VOC-Sim is reported but does not decide. Its single-object prompts give little variation in C, so a null result there is expected and uninformative.

**What follows from each outcome:**
- **Not supported:** the cap is described as reducing general over-expansion, and the clutter-adaptation hypothesis is closed.
- **Supported:** this is a paper explanation only. Any adaptive method would need its own pre-registration and a genuinely unused multi-object confirmation set.
- **In every case:** no method is built from these numbers.

## Limitations stated in advance

- SCLIP's VOC21 vocabulary misses competitors outside the 20 VOC classes, which are common on COCO-Cap.
- SCLIP is a separate model and can be wrong on generated images.
- Controlling for ground-truth area is valid for a diagnostic but does not show that a label-free rule could make the same distinction.
- The comparison V3 vs V0 bundles the 16→32→64 hierarchy and the cap. V1 (the uncapped hierarchy) is not in the stored capture, so the cap alone is not isolated here.
