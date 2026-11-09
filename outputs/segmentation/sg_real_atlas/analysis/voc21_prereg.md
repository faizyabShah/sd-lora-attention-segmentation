# Pre-registration: SCLIP + SD self-attention on the standard VOC2012 val protocol

Written 2026-11-09, before any VOC val capture or score of this pipeline exists. Committed before the val capture is submitted.

**Data.** The official VOC2012 val split (1449 photos). Two protocols:
- **VOC21:** 21 classes, background included.
- **VOC20:** 20 classes, background pixels ignored.

Scores use dataset-level confusion, as in mmseg IoUMetric. No class-presence information is used: every class competes in every photo.

**Fixed parts** (`scripts/sg_voc21.py`, `scripts/sg_voc21_dev.py`):
1. **SCLIP probabilities.** SCLIP ViT-B/16 (our port, which reproduces the repo on VOC val: 59.64 / 81.49).
   - Per-class probabilities p use the repo's logit scale (65 for VOC21, 40 for VOC20), with synonyms max-pooled.
2. **One SD1.5 run per photo.**
   - Empty-prompt DDIM inversion (30 steps).
   - Then the frozen live pipeline runs on that trajectory with "a photograph of a {w}".
   - w is the foreground class with the largest total SCLIP probability, so the choice is label-free.
   - D check (`sa_prompt_dep_D.md`): swapping the prompt word changes C16 by −0.008 on average.
3. **Propagation.** The frozen C16 path, applied to every class map at once:
   - 64 → 16 → A16 (head mean) → 16→64 → A64 (head sum), averaged over 11 calls, divided by 88.
   - Numerically equal to `sg27_fusion.base_paths` (rel. err 2e-7).
4. **Post-processing.** SCLIP's own post-processing, unchanged:
   - VOC21: area_thd 0.1, then prob_thd 0.1.
   - VOC20: argmax.
   - DenseCRF where used is multi-label, with the R2 parameters (10, 80, 13, 3, 3, 5) on the renormalised probabilities.

**Variant chosen on D, by rule.** A 12-photo smoke test on D showed that the raw C16 maps lose to SCLIP under the multi-class protocol. So the way the propagation is combined with p is chosen on D (200 VOC train photos, disjoint from val) by this rule, fixed now:
- **Candidates** (all parameter-free):

  | name | definition |
  |---|---|
  | c16 | bilinear(C16(p64)) |
  | resid | p + bilinear(C16(p64) − p64) |
  | geo | sqrt(p · bilinear(C16(p64))) |

  Each is taken with and without DenseCRF.
- **Selection:** the candidate with the highest D VOC21 mIoU. It is written to `voc21_choice.json` by `sg_voc21_dev.py dev` before val is scored.
- `pool` (no SA) is reported as a control and is not a candidate.

**Comparators (same protocol, same photos).**
- SCLIP standalone.
- SCLIP + DenseCRF.

**Primary test.** chosen − SCLIP + DenseCRF, VOC21 mIoU on val. The photo-bootstrap 95% CI (1000 resamples) must be > 0.

**Secondary (reported, not pass/fail).**
- The chosen variant vs SCLIP standalone.
- Every other variant vs both comparators.
- VOC20 versions.
- Per-class IoU.
- Background error rates.
- Runtime per photo.

**Published methods**, as context only (different backbones and settings):
- NERVE (WACV 2026): ViT-L/14, VOC21 69.7 / VOC20 90.1.
- CLIPer (ICCV 2025): ViT-L, VOC21 69.8 (72.2 with sliding window).

Our ViT-B/16 run is not a like-for-like comparison with these.

**If the primary test fails,** SD self-attention propagation does not beat CRF on the standard protocol. Nothing is re-selected on val.
