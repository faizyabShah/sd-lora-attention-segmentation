# SG29 — NERVE-inspired semantic head weighting and stopped walks

**Status:** complete. GPU array 237909 and CPU merge 237914 completed successfully. The merger validated **490 VOC-Sim** and **707 COCO-Cap** images, all 40 class shards, and compact diagnostics. The first array failed because an exact-count comparison against a separate regeneration was made fatal; this was corrected before the complete run. All candidate comparisons use **the same newly generated images**. N0's raw map was checked against FINE-P2 from the same attention capture for every newly processed image.

**Give ChatGPT this document together with** `results.json` and `per_class_iou.csv` in the package archive. The JSON has every threshold, forced OOF fold, VOC selector, paired 20,000-resample mean-image bootstrap interval, precision/recall, 20 per-class IoUs, mechanism counts, and replay audit. The CSV is an easy-to-read all-class table.

## Locked conditions and interpretation boundary

The historical A3 class-word CA seed, 11 selected CA modules, calls 0–30, SD1.5, 64×64 SA module 15, SA calls 10–20, eight heads, 17-threshold VOC grid, and R2 DenseCRF were fixed. N0 is observation-wise `mean(A²c)`; N1 weights its eight heads within each call by one-step **single-class spatial** semantic-output entropy. N2 uses the 10-step geometrically stopped walk over ordinary SD softmax SA. N3 uses only the local eight-neighbor Q/K graph. N4 fuses global and local transitions. N5=N2+weights; N6=N4+weights. N2 lengths 1, 2, and 5 were registered **diagnostics only**. α=.9, β=.8, c=3, local self weight=.01 are fixed from differing released NERVE scripts, without a sweep. SG29 **adapts**, and does not reproduce, [NERVE](https://arxiv.org/abs/2511.08248). It also retains A3's documented affinity-orientation caveat.

All model and threshold selection used the five VOC folds. COCO was transfer-only. The registered multi-candidate selector retained **N0 in 5/5 folds for raw maps** and **N0+R2 in 5/5 for final masks**. Pairwise fold wins below are separate from that selector.

## Main results

| Cell | VOC forced OOF | Δ OOF vs same-stage N0 | Pairwise VOC fold wins | Full VOC | COCO at VOC threshold | COCO Δ vs same-stage N0 |
|---|---:|---:|---:|---:|---:|---:|
| N0 raw | .8098 | — | — | .8114 | .6598 | — |
| N1 entropy P2 raw | .8098 | +.0000 | 3/5 | .8114 | .6602 | +.0004 |
| N2 global walk raw | .7994 | −.0104 | 0/5 | .7994 | .6610 | +.0011 |
| N3 local walk raw | .6988 | −.1109 | 0/5 | .6999 | .5784 | −.0814 |
| N4 fused walk raw | .7949 | −.0148 | 0/5 | .7949 | .6600 | +.0002 |
| N5 weighted global walk raw | .7994 | −.0103 | 0/5 | .7994 | .6611 | +.0013 |
| N6 weighted fused walk raw | .7950 | −.0148 | 0/5 | .7950 | .6601 | +.0003 |
| N0+R2 | .8229 | — | — | .8243 | .6623 | — |
| N1+R2 | .8229 | +.0001 | 5/5 | .8243 | .6627 | +.0004 |
| N2+R2 | .8135 | −.0094 | 0/5 | .8147 | .6696 | +.0073 |
| N3+R2 | .7278 | −.0951 | 0/5 | .7278 | .6118 | −.0505 |
| N4+R2 | .8103 | −.0126 | 0/5 | .8110 | .6704 | +.0081 |
| N5+R2 | .8136 | −.0093 | 0/5 | .8147 | .6697 | +.0074 |
| N6+R2 | .8104 | −.0125 | 0/5 | .8110 | .6705 | +.0082 |

The **best COCO macro mIoU** here is N6+R2 at **.6705**, but it loses **.0125 forced OOF on VOC**, wins **0/5 pairwise VOC folds**, and is **not promoted**. N4+R2 reaches .6704 without effective entropy weighting. N1+R2 wins all five *pairwise* folds but its OOF gain is only +.0001, below the registered selection floor; the multi-candidate selector still votes N0+R2 5/5. All COCO improvements are transfer observations, not COCO-selected systems.

### Macro mIoU versus image-level evidence

The reported 20k paired bootstrap is on **mean-image IoU**, whereas the primary metric is **20-class pooled macro mIoU**. The two can disagree because class prevalence and image size differ. On COCO, relative to N0+R2:

| Cell | Macro mIoU Δ | Mean-image IoU Δ, paired 95% interval | Better / worse / tied images |
|---|---:|---:|---:|
| N1+R2 | +.0004 | +.00035 [.00027, .00044] | 437 / 237 / 33 |
| N2+R2 | +.0073 | −.00059 [−.00467, .00341] | 361 / 343 / 3 |
| N4+R2 | +.0081 | −.00184 [−.00630, .00255] | 348 / 357 / 2 |
| N6+R2 | +.0082 | −.00168 [−.00613, .00272] | 349 / 356 / 2 |

Thus the N2/N4/N6 COCO macro gains are **not** broad paired image-level improvements. For VOC, N2+R2 and N4+R2 have mean-image losses of −.0100 [−.0141, −.0061] and −.0143 [−.0189, −.0098], respectively.

## Mechanism diagnostics

### Head weighting: almost equal weights, no reliable within-call quality signal

Uniform head weight is .125. The eight mean N1 weights range only **.1223–.1270 on VOC** and **.1214–.1277 on COCO**; effective head count is **7.998/8** and **7.997/8**. Fused N6 weights are similarly uniform. N1 and N5/N6 accordingly differ from their equal-head counterparts by at most a few ten-thousandths of mIoU.

Post-hoc correlation of **negative entropy** with each head's one-step CA-propagation AP, computed *within the eight heads of the same image and call*, is **negative** on average: Spearman **−.140 VOC**, **−.065 COCO**. Pooled correlations across all images/calls are positive (+.131 VOC, +.328 COCO), but pooling confounds image/call and is not evidence that the within-call weight rule picks the better head. Entropy head ranking is very stable between calls of the same image (mean Spearman .977 VOC, .982 COCO); it is simply not a strong quality ranking under this adaptation. No GT was used to form weights.

### Stopped walk: substantial mass outside depth two

With α=.9 and L=10, the normalized walk places **39.5%** of its mass at depths 0–2 and **60.5%** at depths 3–10; depth two alone receives **11.8%**. This is very different from N0's pure depth-two map. On VOC, N2+R2 loses .0094 forced OOF; at raw threshold .45 the walk gains 366k TP but **loses 730k TP** and gains **900k FP** relative to N0. On COCO, N2+R2 raises precision **.7033→.7269** but lowers recall **.8892→.8758**. At raw .45 it removes 2.87M FP while also losing 1.13M TP. These are diagnostic pixel counts, not an alternate mask-selection criterion.

The diagnostic N2 walk lengths L=1, 2, 5 show VOC forced OOF+R2 of **.7794, .7958, .8108**, versus **.8135** at locked L=10, all below N0+R2 .8229. On COCO the L=5 diagnostic reaches **.6748+R2**, higher than L=10 .6696, but it was **not eligible for selection** and loses .0121 VOC OOF versus N0. Do not promote it based on COCO.

### Local graph: spatially compact, but not useful alone

On fixed sampled transition rows, mean transition distance is **21.61 px global**, **1.06 px local**, **17.50 px fused** for VOC; COCO is **20.74, 1.06, 16.80 px**. Eight-neighbor transition mass is **.047 global** versus **.238 fused** on VOC (.049 versus .239 COCO). The graph is genuinely more local. But N3 local-only loses .0951 VOC OOF+R2 and .0505 COCO macro mIoU, so local coherence alone does not preserve the semantic seed. N4 is worse than N2 on VOC, and only +.0008 macro mIoU over N2+R2 on COCO. N6 is virtually N4 with near-uniform weights. This does not support adding the local graph to the selected pipeline.

## Class audit: the COCO gain has a costly person tradeoff

Values are **per-class IoU changes after R2 versus N0+R2**, not absolute scores.

| Class | VOC N2 | VOC N4 | COCO N2 | COCO N4 |
|---|---:|---:|---:|---:|
| person | −.0306 | −.0362 | −.0399 | −.0356 |
| chair | −.1015 | −.1337 | +.0069 | +.0038 |
| bicycle | −.0291 | −.0411 | +.0112 | +.0105 |
| potted plant | −.0197 | −.0314 | −.0092 | −.0093 |
| sheep | +.0079 | +.0068 | −.0047 | −.0052 |
| cow | −.0009 | −.0030 | +.0065 | +.0023 |
| dining table | −.0065 | −.0046 | +.0285 | +.0315 |
| car | −.0015 | −.0009 | +.0288 | +.0412 |
| train | +.0063 | +.0110 | +.0027 | +.0027 |
| aeroplane | −.0170 | −.0256 | +.0008 | −.0111 |

N2/N4 improve only **4/20 VOC classes**, but **16/20 and 15/20 COCO classes**. The conspicuous COCO losses include person and potted plant. COCO gains for car, dining table, bus, and some other classes offset them in the macro mean. The all-class values are in `per_class_iou.csv`.

## Regeneration and baseline audit

SG29 N0+R2 is **.8243 VOC / .6623 COCO**, close to SG28 P0+R2 **.8242 / .6621**. Exact per-image raw/R2 counts match SG28 for **295/490 VOC** and **360/707 COCO** images; differences arise across separate regenerations, with mean absolute raw image-IoU difference at threshold .45 only **.00047 VOC / .00062 COCO**. A few individual pixel-count differences are larger (maximum 9,676 VOC and 38,356 COCO), so one must compare SG29 cells against **same-run N0**, not use SG28 pixel counts as a fatal equality condition. The SG29 same-generation raw-map identity check passed during the corrected production run.

## Conclusion for the next decision

Under the locked VOC-selection protocol, **retain N0+R2**. The one-class entropy rule scarcely changes head weights and fails to rank head utility within a call; the stopped walk loses on VOC; local-only propagation fails badly; adding local affinity or entropy to the walk does not rescue VOC. The higher COCO macro scores for stopped walks are exploratory transfer signals with class-specific losses and no positive paired mean-image interval. They do not justify replacing the selected system. If exploring this direction further, the evidence points to the *precision/recall and person tradeoff*, rather than to stacking all three NERVE-inspired components.
