# SG30 — SPARK-inspired nonlinear SA affinity sharpening

**Status:** complete. Smoke 237945, 40-task GPU array 237946, and CPU merger
237947 all completed `0:0`. The merger validated 490 VOC-Sim and 707 COCO-Cap
images, all 20 class shards in each dataset, complete compact diagnostics,
and fixed VOC fold IDs. Give ChatGPT this file with the accompanying
`results.json` and `per_class_iou.csv`; the JSON contains every threshold,
five-fold selector and forced OOF result, paired 20,000-resample *mean-image*
interval, all 20 class IoUs, and graph/transition/iteration diagnostics.

## Locked design and provenance

SG30 adapts [SPARK: Stochastic Propagation via Affinity-guided Random walK
for Training-free Unsupervised Segmentation](https://arxiv.org/abs/2602.00516)
to the historical A3 class-word SD1.5 seed. It is **not a SPARK reproduction**:
SPARK clusters an unlabeled diffusion-feature graph; SG30 uses original
SD-softmax module-15 SA and a target-class CA seed. A3's previously documented
affinity-orientation caveat remains. The authors' project page listed code
as coming soon when SG30 was locked. The feature-local S7 construction and
negative-cosine clamp are therefore explicit adaptations, not recovered
author defaults.

Fixed throughout: 11 A3 CA modules, CA calls 0–30, 64×64 SA module 15,
SA calls 10–20, eight heads, observation-wise graph operations, 17 VOC-only
thresholds, unchanged R2 DenseCRF, and no COCO tuning. Expansion ℓ=2 and
pruning τ=10⁻⁷ came from the paper setup; inflation r=2.6 and local/global
β=.6 were single paper-ablation peaks, not tuned here.

| Cell | Operator before optional R2 |
|---|---|
| S0 | Historical `mean_i(A_i² c)` |
| S1 | Inflate A, then use it for both passes |
| S2 | Ordinary first pass, inflated A on second pass |
| S3 | Form A², inflate, apply to c |
| S4 | Ordinary A first, then inflated A² |
| S5 | S3 plus τ pruning and row normalization |
| S6 | S4 plus τ pruning and row normalization |
| S7 | Feature-local/global affinity fusion and one SPARK-like update |

S5-I2/I3 are fixed two/three-update diagnostics, **not selectable**. Every
main cell has raw and +R2 scores. Candidate comparisons use the **same
generation** and same-run S0. A smoke assertion checked S0's map against
historical FINE-P2; production ran with that assertion for every image.

## Result: no SG30 promotion under the registered VOC rule

The multi-candidate VOC selector chose **S0 in 5/5 raw folds** and **S0+R2
in 5/5 final folds**. Forced OOF and pairwise fold wins below are separate
from that selector. The reference is S0 within each stage.

| Cell | VOC forced OOF | Δ OOF | Pairwise VOC wins | Full VOC | COCO at frozen VOC threshold | COCO Δ |
|---|---:|---:|---:|---:|---:|---:|
| S0 | .8098 | — | — | .8115 | .6600 | — |
| S1 | .7634 | −.0464 | 0/5 | .7634 | .6378 | −.0222 |
| S2 | .7997 | −.0101 | 0/5 | .7997 | .6674 | +.0074 |
| S3 | .7880 | −.0219 | 0/5 | .7892 | .6587 | −.0013 |
| S4 | .8032 | −.0066 | 0/5 | .8048 | .6664 | +.0064 |
| S5 | .7620 | −.0478 | 0/5 | .7632 | .6402 | −.0198 |
| S6 | .7964 | −.0135 | 0/5 | .7978 | .6638 | +.0037 |
| S7 | .7069 | −.1029 | 0/5 | .7069 | .5866 | −.0734 |
| S0+R2 | .8231 | — | — | .8244 | .6623 | — |
| S1+R2 | .7841 | −.0390 | 0/5 | .7841 | .6623 | +.0000 |
| S2+R2 | .8125 | −.0106 | 1/5 | .8156 | **.6855** | **+.0232** |
| S3+R2 | .8056 | −.0174 | 1/5 | .8069 | .6697 | +.0074 |
| S4+R2 | .8187 | −.0044 | 1/5 | .8200 | .6732 | +.0109 |
| S5+R2 | .7795 | −.0436 | 0/5 | .7806 | .6619 | −.0004 |
| S6+R2 | .8144 | −.0087 | 1/5 | .8144 | .6836 | +.0212 |
| S7+R2 | .7356 | −.0875 | 0/5 | .7356 | .6206 | −.0417 |

S2+R2 has the largest **exploratory COCO macro mIoU gain**, but loses .0106
forced OOF on VOC and wins only one pairwise VOC fold. S6+R2 shows a similar
COCO transfer signal (+.0212) with a VOC OOF loss (−.0087). Neither passes
the registered selection rule; the frozen pipeline remains S0+R2.

### Paired image-level evidence

Primary mIoU is the mean of 20 **pooled class IoUs**. The bootstrap below is
for **mean per-image IoU**, a different estimand; its interval must not be
presented as a 95% interval for the macro mIoU delta.

| Cell vs S0+R2 | VOC mean-image Δ [95% paired interval] | COCO mean-image Δ [95% paired interval] | COCO better/worse/tied images |
|---|---:|---:|---:|
| S2+R2 | −.0073 [−.0120, −.0025] | +.0131 [+.0083, +.0179] | 440/265/2 |
| S4+R2 | −.0042 [−.0066, −.0018] | +.0043 [+.0017, +.0073] | 380/324/3 |
| S6+R2 | −.0083 [−.0129, −.0036] | +.0111 [+.0067, +.0155] | 406/299/2 |

The COCO gains are broader than a class-macro artifact, especially for S2,
but they do **not** overturn VOC-only model selection.

## What the graph and masks show

- **Inflation is real and strong.** On VOC sampled graph rows, S0 mean row
  entropy is 6.79, effective neighbors 1,299, and top-10 mass .112. S2's
  inflated graph has entropy 4.68, effective neighbors 292, and top-10 mass
  .411. Original top-quartile edges carry .825 of row mass; after inflation
  they carry .976. On COCO, the corresponding top-quartile shift is
  .845→.984. These are graph descriptors, not segmentation scores.
- **Graph purity alone does not guarantee better masks.** Post-hoc VOC sampled
  foreground→foreground transition mass rises from .795 (S0) to .919 (S2).
  Yet S2+R2 loses .0106 forced VOC OOF. We infer that stronger within-object
  transitions can also suppress useful object extent or change calibration;
  this diagnostic does not establish a causal explanation by itself.
- **S2 trades recall for precision.** Full VOC S0+R2 precision/recall is
  .8829/.9162; S2+R2 is .8926/.8912. COCO changes from .7034/.8893 to
  .7462/.8683. At fixed raw diagnostic threshold .45, S2 removes 2.19M COCO
  FP, including 1.26M >64 px from GT, but loses 686k TP and gains 771k
  other FP. On VOC it loses 818k TP while removing 533k FP. Those counts
  help explain the opposite dataset-level outcomes.
- **Pruning is too aggressive under this fixed operator.** S6 prunes about
  99.17% of inflated entries on VOC; 32.07% of rows hit the specified
  pre-prune fallback. On COCO the corresponding figures are 99.07% and
  27.45%. S5+R2 falls to .7795 VOC OOF; S6+R2 is .8144. Repeating S5's
  update worsens raw VOC OOF to .7216 (two updates) and .7031 (three).
  Those iterations were diagnostics, not candidate methods.
- **The feature-local fusion fails.** S7+R2 scores .7356 VOC OOF and .6206
  COCO, far below S0+R2. This only rejects SG30's particular feature-local
  adaptation and fixed parameters.

## Class tradeoffs worth asking about

These are **per-class IoU deltas after R2** relative to the same-run S0+R2.

| Class | VOC S2 | COCO S2 | VOC S6 | COCO S6 |
|---|---:|---:|---:|---:|
| person | −.0340 | −.0163 | −.0153 | −.0093 |
| chair | −.0447 | +.0424 | −.0325 | +.0467 |
| car | +.0106 | +.0725 | +.0053 | +.0642 |
| potted plant | −.0512 | +.0200 | −.0576 | +.0225 |
| cat | −.0686 | +.0001 | −.0547 | +.0028 |
| train | +.0661 | +.0213 | +.0592 | +.0224 |

The pattern is especially clear for S2: COCO car +.0725, motorbike +.0642,
chair +.0424, and TV +.0422, while dining table −.0327 and person −.0163.
VOC loses on cat −.0686, potted plant −.0512, chair −.0447, cow −.0443,
and sheep −.0358. All absolute scores and deltas for every class and cell
are in the CSV/JSON. The class table above is illustrative, not a selection
criterion.

## Baseline replay and interpretation cautions

SG30 and SG29 were separate regenerations. SG30 S0 exactly matches SG29 N0's
all-threshold per-image counts for 367/490 VOC and 289/707 COCO images.
Mean absolute raw per-image IoU difference at .45 is only .00029 VOC and
.00077 COCO, but maxima in individual pixel counts are larger. Therefore
all SG30 candidate comparisons must use SG30's **same-run S0**. The
cross-run mismatch is an audit observation, not an SG30 candidate failure.

GT was used only for post-hoc sampled-row and foreground/background
diagnostics, never to construct an affinity or mask. Graph statistics sample
32 uniform rows plus up to 16 GT foreground and 16 GT background rows per
image; they are not whole-graph summaries. The 20k paired intervals concern
mean-image IoU; VOC selector decisions use class-macro mIoU and registered
fold rules. Do not choose S2/S6 on COCO after viewing transfer results.

**Current decision:** retain S0+R2. SG30 exposes a credible transfer-specific
precision/leakage signal in S2/S6, together with a VOC object-completion
cost. Any follow-up should explain that tradeoff under VOC-only selection,
without treating the COCO outcome as a tuning set.
