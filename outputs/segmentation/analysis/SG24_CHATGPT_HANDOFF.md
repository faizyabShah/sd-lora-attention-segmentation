# SG24-A/B handoff for ChatGPT — 24 October 2026

## Setup and validity

Both experiments completed: **490 VOC-Sim** and **707 COCO-Cap** images per
experiment, with all 40 class shards and both mergers successful. Each cell
was computed from the same regenerated image within its experiment. The
class-word W attribution, SD 1.5, 31 CA calls, MID-P2 (SA module 15, calls
10–20, all eight heads, per-observation two-pass propagation), R2 DenseCRF,
five fixed VOC folds, 17 thresholds, and COCO frozen-transfer rules were
unchanged. The A0 final control matched the previous SG22
MID-P2+R2 **17-threshold counts exactly** on the VOC and COCO smoke images.

All scores below are **macro mean IoU over 20 classes** unless explicitly
called mean-image IoU. The paired 20,000-resample intervals apply to the
*mean per-image IoU difference*, not the macro mIoU difference. VOC OOF
uses training-fold-selected thresholds; full VOC chooses one threshold on
all VOC and freezes it for COCO. COCO was never used for selection.
The A3 and B0 numbers differ by at most a few ten-thousandths after
independent regeneration; use each experiment's same-run reference for
its comparisons.

## SG24-A: change CA construction within the selected 11 modules

| Variant | VOC CA OOF | Full VOC CA | COCO CA | VOC final OOF | Full VOC final | COCO final | VOC final threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 equal CA | .7135 | .7135 | .5627 | .8169 | .8185 | .6437 | .450 |
| A1 content rescaling | .6406 | .6406 | .0000 | .8005 | .8022 | .0000 | .625 |
| A2 projected-output head weights | .6897 | .6897 | .5559 | .8033 | .8039 | .6366 | .375 |
| A3 pseudo-global layer weights | .7049 | .7049 | .5836 | .8230 | .8243 | .6620 | .425 |
| A4 A1+A2+A3 | .6288 | .6288 | .0000 | .8004 | .8027 | .0000 | .600 |

The **CA-only five-fold screen retained A0 in all five folds**. The
**final-pipeline multi-candidate screen also retained A0**: A3 was selected
in 3 folds, A0 in 2, below the registered 4/5 promotion requirement. The
screen's selected-path OOF was .81616 versus .81689 for A0. This is the
formal result: **no SG24-A variant was promoted**.

A useful secondary result is the *forced* A3 comparison. If A3 is applied
to every held-out fold, it wins **4/5**, with OOF **.82297 versus .81689**
(+ .00608). Full VOC final is **.8243 versus .8185** (+ .0058), and frozen
COCO final is **.6620 versus .6437** (+ .0183). Its paired mean-image IoU
gain is +.00507 on VOC (95% bootstrap interval +.00030 to +.01003) and
+.01165 on COCO (+.00371 to +.01979). These are encouraging observations,
but the preregistered multi-candidate selector did not promote A3. Please
preserve both statements rather than calling A3 the new final system.

| VOC fold | Forced A3 held-out mIoU | A0 held-out mIoU | Multi-candidate screen chose |
|---|---:|---:|---|
| 0 | .8210 | .8259 | A3 (training-eligible; held-out loss) |
| 1 | .8575 | .8416 | A0 (A3 training-ineligible) |
| 2 | .8220 | .8175 | A3 |
| 3 | .8003 | .7997 | A3 |
| 4 | .8227 | .8061 | A0 (A3 training-ineligible) |

Thus the forced 4/5 held-out wins and the screen's 3/5 A3 votes measure
different things. The latter, plus its negative selected-path OOF gain,
controls promotion.

**Why A3 is interesting:** CA-only VOC worsens by .0086 even while the
full pipeline improves by .0058; COCO improves at both levels (+.0209 CA,
+.0183 final). This suggests an interaction with propagation/CRF, rather
than a universally better standalone semantic seed. A3 shifts layer mass
from the 64×64 CA modules toward coarser structure. On VOC the average
automatic weights are **24.3% at 8×8**, **59.7% at 16×16**, and **16.0%
at 64×64**, versus equal-per-module mass **9.1% / 45.5% / 45.5%**. The
single 8×8 middle module is top-weighted in **93.6%** of image-call
observations. COCO has essentially the same pattern (24.5% / 60.1% /
15.5%). This is consistent with broader connected target regions, but is
not a causal proof: at a fixed diagnostic threshold of .45, A3 VOC CA
gains **0.593M TP** and **0.937M FP** versus A0, while separate-component
FP falls **97.3k→34.5k**. On COCO it gains **2.010M TP** and **0.828M FP**,
while separate-component FP falls **1.972M→1.163M**. The final VOC
precision/recall moves **.8914/.8970→.8828/.9162**; COCO moves
**.6889/.8654→.7031/.8892**.

Per-class A3 final gains on VOC are largest for **cow +.060, person +.053,
sheep +.034**; losses include **bottle −.041, chair −.021**. On COCO,
gains include **dining table +.115, car +.070, aeroplane +.068, train
+.051**; losses include **chair −.025, bicycle −.020**. Complete
20-class IoUs for every A variant and both evaluation levels follow below.

**A1/A4 limitation:** They are substantially worse on VOC and give zero
IoU on all 20 COCO classes at the frozen thresholds. COCO's unchanged
attribution prompt contains only the class name. Once special/padding and
stop words are removed, the content-token set equals the target-token
set, so the rescaled target map is mathematically spatially constant.
It was explicitly held constant in code to prevent min-max normalization
from amplifying floating-point roundoff. This is a structural incompatibility
with the current COCO attribution protocol, not evidence that content
rescaling can never work with another attribution design. On VOC, content
tokens account for only **2.82%** of the 77-token softmax mass on average;
at fixed .45, A1 foreground coverage expands **44.2M→105.4M pixels**, with
FP **7.0M→58.2M**. A2 also loses mIoU at both levels and on both datasets.

## SG24-B: restore all five 32×32 CA modules under the same A3 weighting

A3 was the preregistered diagnostic fallback for B because SG24-A did not
formally promote an automatic method. B0 uses the same 11 selected modules;
B1 uses all 16. There is no resolution-subset search.

| CA set | VOC CA OOF | Full VOC CA | COCO CA | VOC final OOF | Full VOC final | COCO final | VOC final threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0: selected 11 | .7050 | .7050 | .5836 | .8230 | .8244 | .6622 | .425 |
| B1: all 16 | .6932 | .6932 | .5799 | .8152 | .8152 | .6703 | .425 |

The VOC five-fold CA-only and final screens chose **B0 in all five folds**.
Forced B1 final loses **.00786 OOF** and all five held-out folds. Full VOC
loses **.00928**; CA-only VOC loses **.01173**. Thus B1 is **not promoted**.
However, COCO final improves **.6622→.6703** (+.00815) even while its
CA-only mIoU falls **.5836→.5799** (−.00366). On COCO the paired
mean-image IoU gain is +.00644 (95% bootstrap interval +.00146 to
+.01167); on VOC it is −.00762 (−.01147 to −.00400). COCO final
precision/recall both rise **.7034/.8892→.7124/.9001**, whereas VOC
falls **.8830/.9162→.8786/.9055**.

B1 COCO gains are concentrated in **dining table +.101, car +.054,
person +.047, horse +.033**; losses include **sofa −.040, potted plant
−.023, chair −.015**. VOC loses especially **dog −.064, cat −.044,
dining-table −.024, cow −.024**. These class tradeoffs explain why a COCO
gain should not override the VOC-only selection rule.

The added 32×32 group receives **31.5%** of B1's average layer weight on
VOC and **31.1%** on COCO, close to its 5/16 module-count share of
31.25%. The automatic weighting therefore **does not suppress** the
previously excluded 32×32 group. The 8×8 middle module still gets the
top individual weight in **73.9% VOC / 76.9% COCO** image-call observations.
Weight on restored modules has weak per-image correlation with final IoU
change (**−.192 VOC, +.079 COCO**). That limits any claim that high 32×32
weight itself predicts improvement. At fixed .45 on VOC, B1 reduces both
TP (**37.83M→34.99M**) and FP (**7.90M→7.07M**); more useful foreground
is lost than harmful background is removed. On COCO, fixed-.45 TP and FP
both fall too, while the *full final* VOC-frozen-threshold result improves;
the CA diagnostic threshold and final CRF threshold are different stages.

## Interpretation and decision for ChatGPT

1. **Keep A0→MID-P2→R2 as the formally selected final pipeline** under
   the existing multi-candidate VOC policy. The A3 forced comparison is a
   strong follow-up candidate, not a formal promotion. Its benefit seems
   tied to how a coarser CA seed interacts with SA/CRF; this is an inference
   from the observed CA-only/final sign reversal, not a proven mechanism.
2. **Do not promote B1 using COCO.** Its COCO transfer gain is observed in the
   frozen evaluation, but VOC worsens in every held-out fold. The 32×32
   weights are substantial rather than automatically pruned.
3. **Retire A1/A4 for this attribution protocol.** COCO's class-only
   content set makes their seed constant by algebra. A2 projected-output
   head weights alone did not improve these segmentation metrics.
4. The most interesting methodological question is whether a *locked,
   separately preregistered A3-versus-A0 confirmation* would satisfy the
   selection rule, or whether its gains are sensitive to the current
   multi-candidate eligibility screen. Such a rerun would be a new
   experiment, not a reinterpretation of this one.

## Raw result files

- [SG24-A JSON](../controlled_attention_cv/sg24_a/results.json), [all A per-class rows](../controlled_attention_cv/sg24_a/per_class_iou.csv), [A report](../controlled_attention_cv/sg24_a/report.md), [A-to-B selection manifest](../controlled_attention_cv/sg24_a/selection_for_b.json)
- [SG24-B JSON](../controlled_attention_cv/sg24_b/results.json), [all B per-class rows](../controlled_attention_cv/sg24_b/per_class_iou.csv), [B report](../controlled_attention_cv/sg24_b/report.md)

The JSON files contain all fold selections, forced OOF and full-dataset
metrics, 20-class IoUs and deltas, precision/recall, pixel/mean-image IoU,
paired intervals and better/worse/tied counts, fixed-threshold CA changes,
module weights by class and diffusion call, and B restored-weight correlations.

## Additional operating-point metrics

Each row uses its full-VOC-selected threshold on both datasets. The paired interval is for mean per-image IoU versus the same-level A0 or B0 reference.

### SG24-A

| Cell | VOC threshold | VOC P/R | COCO P/R | VOC pixel / mean-image IoU | COCO pixel / mean-image IoU | VOC paired mean (95% CI) | COCO paired mean (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0/CA | 0.400 | 0.807/0.854 | 0.687/0.759 | 0.709/0.703 | 0.564/0.563 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| A0/FINAL | 0.450 | 0.891/0.897 | 0.689/0.865 | 0.809/0.823 | 0.622/0.681 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| A1/CA | 0.700 | 0.719/0.875 | 0.000/0.000 | 0.652/0.633 | 0.000/0.000 | -0.0706 [-0.0846, -0.0568] | -0.5627 [-0.5797, -0.5453] |
| A1/FINAL | 0.625 | 0.871/0.900 | 0.000/0.000 | 0.794/0.808 | 0.000/0.000 | -0.0157 [-0.0290, -0.0027] | -0.6809 [-0.7015, -0.6597] |
| A2/CA | 0.300 | 0.771/0.863 | 0.682/0.767 | 0.687/0.677 | 0.565/0.548 | -0.0265 [-0.0336, -0.0194] | -0.0142 [-0.0205, -0.0079] |
| A2/FINAL | 0.375 | 0.857/0.916 | 0.703/0.861 | 0.795/0.806 | 0.632/0.678 | -0.0172 [-0.0258, -0.0085] | -0.0026 [-0.0127, +0.0078] |
| A3/CA | 0.400 | 0.794/0.859 | 0.696/0.790 | 0.703/0.690 | 0.588/0.569 | -0.0131 [-0.0186, -0.0074] | +0.0065 [+0.0009, +0.0123] |
| A3/FINAL | 0.425 | 0.883/0.916 | 0.703/0.889 | 0.817/0.828 | 0.647/0.693 | +0.0051 [+0.0003, +0.0100] | +0.0116 [+0.0037, +0.0198] |
| A4/CA | 0.700 | 0.698/0.879 | 0.000/0.000 | 0.637/0.617 | 0.000/0.000 | -0.0867 [-0.1009, -0.0726] | -0.5627 [-0.5797, -0.5453] |
| A4/FINAL | 0.600 | 0.870/0.903 | 0.000/0.000 | 0.796/0.809 | 0.000/0.000 | -0.0144 [-0.0270, -0.0020] | -0.6809 [-0.7015, -0.6597] |

### SG24-B

| Cell | VOC threshold | VOC P/R | COCO P/R | VOC pixel / mean-image IoU | COCO pixel / mean-image IoU | VOC paired mean (95% CI) | COCO paired mean (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-A3/CA | 0.400 | 0.794/0.859 | 0.697/0.790 | 0.703/0.690 | 0.588/0.569 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| B0-A3/FINAL | 0.425 | 0.883/0.916 | 0.703/0.889 | 0.817/0.829 | 0.647/0.693 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| B1-A3/CA | 0.350 | 0.766/0.872 | 0.656/0.840 | 0.688/0.681 | 0.583/0.564 | -0.0097 [-0.0155, -0.0039] | -0.0058 [-0.0109, -0.0006] |
| B1-A3/FINAL | 0.425 | 0.879/0.905 | 0.712/0.900 | 0.805/0.821 | 0.660/0.699 | -0.0076 [-0.0115, -0.0040] | +0.0064 [+0.0015, +0.0117] |

## Complete per-class IoU tables

### VOC-Sim: SG24-A CA-only IoU

| Class | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.6944 | 0.5783 | 0.6308 | 0.6185 | 0.4132 |
| bicycle | 0.6476 | 0.5286 | 0.6619 | 0.6095 | 0.5266 |
| bird | 0.7413 | 0.6198 | 0.7149 | 0.7191 | 0.6442 |
| boat | 0.7781 | 0.6062 | 0.6747 | 0.7566 | 0.5213 |
| bottle | 0.7540 | 0.5565 | 0.7188 | 0.7140 | 0.6537 |
| bus | 0.8954 | 0.6931 | 0.8639 | 0.8877 | 0.7075 |
| car | 0.8001 | 0.6203 | 0.7787 | 0.7841 | 0.6314 |
| cat | 0.7147 | 0.7962 | 0.7164 | 0.7095 | 0.7698 |
| chair | 0.4815 | 0.5413 | 0.4582 | 0.4428 | 0.5023 |
| cow | 0.8166 | 0.7734 | 0.8446 | 0.8218 | 0.7470 |
| dining-table | 0.4422 | 0.3899 | 0.3510 | 0.4348 | 0.3697 |
| dog | 0.8214 | 0.7921 | 0.8104 | 0.7826 | 0.7880 |
| horse | 0.7845 | 0.7282 | 0.7765 | 0.7585 | 0.6806 |
| motorbike | 0.7950 | 0.6601 | 0.7740 | 0.7754 | 0.5743 |
| person | 0.4352 | 0.6340 | 0.3899 | 0.5897 | 0.6352 |
| potted-plant | 0.6389 | 0.7074 | 0.6612 | 0.5932 | 0.6904 |
| sheep | 0.7798 | 0.7043 | 0.8595 | 0.7900 | 0.7030 |
| sofa | 0.8612 | 0.7885 | 0.8541 | 0.8702 | 0.8480 |
| train | 0.6417 | 0.5663 | 0.4615 | 0.6704 | 0.4395 |
| tv | 0.7472 | 0.5269 | 0.7927 | 0.7703 | 0.7295 |

### VOC-Sim: SG24-A final IoU

| Class | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.8970 | 0.8885 | 0.8902 | 0.8840 | 0.8782 |
| bicycle | 0.7606 | 0.7669 | 0.7751 | 0.7630 | 0.7571 |
| bird | 0.8977 | 0.8446 | 0.8753 | 0.9130 | 0.9145 |
| boat | 0.8543 | 0.7643 | 0.7606 | 0.8391 | 0.7282 |
| bottle | 0.8935 | 0.8087 | 0.8772 | 0.8528 | 0.8354 |
| bus | 0.9098 | 0.8278 | 0.8679 | 0.8970 | 0.8693 |
| car | 0.8316 | 0.7678 | 0.8326 | 0.8461 | 0.7477 |
| cat | 0.8274 | 0.8195 | 0.8514 | 0.8390 | 0.7993 |
| chair | 0.8008 | 0.8637 | 0.7473 | 0.7794 | 0.8297 |
| cow | 0.8621 | 0.9380 | 0.9499 | 0.9221 | 0.9392 |
| dining-table | 0.4640 | 0.3976 | 0.3548 | 0.4556 | 0.3978 |
| dog | 0.9183 | 0.9229 | 0.9273 | 0.9315 | 0.9213 |
| horse | 0.8983 | 0.9318 | 0.8971 | 0.8981 | 0.9143 |
| motorbike | 0.8570 | 0.7770 | 0.8324 | 0.8565 | 0.7613 |
| person | 0.7217 | 0.7977 | 0.6466 | 0.7748 | 0.8424 |
| potted-plant | 0.7535 | 0.7505 | 0.7792 | 0.7393 | 0.7533 |
| sheep | 0.8066 | 0.8536 | 0.9106 | 0.8404 | 0.8672 |
| sofa | 0.9602 | 0.9426 | 0.9598 | 0.9640 | 0.9430 |
| train | 0.6774 | 0.7272 | 0.5287 | 0.6889 | 0.6208 |
| tv | 0.7785 | 0.6526 | 0.8143 | 0.8020 | 0.7337 |

### COCO-Cap: SG24-A CA-only IoU

| Class | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.5465 | 0.0000 | 0.4918 | 0.5319 | 0.0000 |
| bicycle | 0.5825 | 0.0000 | 0.5707 | 0.5333 | 0.0000 |
| bird | 0.6117 | 0.0000 | 0.6342 | 0.6132 | 0.0000 |
| boat | 0.5344 | 0.0000 | 0.5397 | 0.5638 | 0.0000 |
| bottle | 0.4600 | 0.0000 | 0.4644 | 0.4660 | 0.0000 |
| bus | 0.8869 | 0.0000 | 0.8901 | 0.8898 | 0.0000 |
| car | 0.5374 | 0.0000 | 0.5450 | 0.6134 | 0.0000 |
| cat | 0.7041 | 0.0000 | 0.7357 | 0.6895 | 0.0000 |
| chair | 0.3965 | 0.0000 | 0.3923 | 0.3792 | 0.0000 |
| cow | 0.7078 | 0.0000 | 0.7281 | 0.7091 | 0.0000 |
| dining table | 0.3768 | 0.0000 | 0.3360 | 0.5485 | 0.0000 |
| dog | 0.6854 | 0.0000 | 0.6811 | 0.6780 | 0.0000 |
| horse | 0.6190 | 0.0000 | 0.5875 | 0.5885 | 0.0000 |
| motorbike | 0.6255 | 0.0000 | 0.5840 | 0.6415 | 0.0000 |
| person | 0.1192 | 0.0000 | 0.0682 | 0.1729 | 0.0000 |
| potted plant | 0.5228 | 0.0000 | 0.5856 | 0.5298 | 0.0000 |
| sheep | 0.7265 | 0.0000 | 0.7102 | 0.7105 | 0.0000 |
| sofa | 0.5045 | 0.0000 | 0.5846 | 0.5682 | 0.0000 |
| train | 0.5627 | 0.0000 | 0.4227 | 0.6409 | 0.0000 |
| tv | 0.5436 | 0.0000 | 0.5650 | 0.6031 | 0.0000 |

### COCO-Cap: SG24-A final IoU

| Class | A0 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.6494 | 0.0000 | 0.5862 | 0.7171 | 0.0000 |
| bicycle | 0.5907 | 0.0000 | 0.5369 | 0.5707 | 0.0000 |
| bird | 0.7578 | 0.0000 | 0.7772 | 0.7786 | 0.0000 |
| boat | 0.5336 | 0.0000 | 0.4974 | 0.5627 | 0.0000 |
| bottle | 0.4892 | 0.0000 | 0.5020 | 0.4963 | 0.0000 |
| bus | 0.9100 | 0.0000 | 0.8790 | 0.9027 | 0.0000 |
| car | 0.5626 | 0.0000 | 0.6180 | 0.6326 | 0.0000 |
| cat | 0.8553 | 0.0000 | 0.8618 | 0.8541 | 0.0000 |
| chair | 0.4711 | 0.0000 | 0.4740 | 0.4459 | 0.0000 |
| cow | 0.8474 | 0.0000 | 0.8885 | 0.8550 | 0.0000 |
| dining table | 0.3559 | 0.0000 | 0.3542 | 0.4704 | 0.0000 |
| dog | 0.7558 | 0.0000 | 0.7592 | 0.7556 | 0.0000 |
| horse | 0.6797 | 0.0000 | 0.6436 | 0.6743 | 0.0000 |
| motorbike | 0.6391 | 0.0000 | 0.5903 | 0.6235 | 0.0000 |
| person | 0.2388 | 0.0000 | 0.2832 | 0.2658 | 0.0000 |
| potted plant | 0.6975 | 0.0000 | 0.7035 | 0.6969 | 0.0000 |
| sheep | 0.8843 | 0.0000 | 0.8939 | 0.8733 | 0.0000 |
| sofa | 0.6178 | 0.0000 | 0.7410 | 0.6621 | 0.0000 |
| train | 0.6366 | 0.0000 | 0.4348 | 0.6871 | 0.0000 |
| tv | 0.7014 | 0.0000 | 0.7081 | 0.7152 | 0.0000 |

### VOC-Sim: SG24-B CA-only IoU

| Class | B0 selected 11 | B1 all 16 | B1 − B0 |
|---|---:|---:|---:|
| aeroplane | 0.6185 | 0.5774 | -0.0411 |
| bicycle | 0.6095 | 0.6076 | -0.0019 |
| bird | 0.7191 | 0.6999 | -0.0192 |
| boat | 0.7566 | 0.7009 | -0.0557 |
| bottle | 0.7140 | 0.6569 | -0.0571 |
| bus | 0.8877 | 0.8633 | -0.0244 |
| car | 0.7841 | 0.7889 | +0.0048 |
| cat | 0.7095 | 0.6281 | -0.0814 |
| chair | 0.4428 | 0.4095 | -0.0334 |
| cow | 0.8213 | 0.8339 | +0.0126 |
| dining-table | 0.4355 | 0.3823 | -0.0531 |
| dog | 0.7827 | 0.7938 | +0.0111 |
| horse | 0.7577 | 0.7491 | -0.0086 |
| motorbike | 0.7754 | 0.8122 | +0.0368 |
| person | 0.5900 | 0.6566 | +0.0666 |
| potted-plant | 0.5932 | 0.6505 | +0.0573 |
| sheep | 0.7901 | 0.7881 | -0.0020 |
| sofa | 0.8702 | 0.8605 | -0.0097 |
| train | 0.6706 | 0.5995 | -0.0711 |
| tv | 0.7705 | 0.8053 | +0.0348 |

### VOC-Sim: SG24-B final IoU

| Class | B0 selected 11 | B1 all 16 | B1 − B0 |
|---|---:|---:|---:|
| aeroplane | 0.8840 | 0.8690 | -0.0150 |
| bicycle | 0.7630 | 0.7769 | +0.0139 |
| bird | 0.9130 | 0.9134 | +0.0004 |
| boat | 0.8391 | 0.8345 | -0.0046 |
| bottle | 0.8528 | 0.8479 | -0.0049 |
| bus | 0.8970 | 0.8859 | -0.0111 |
| car | 0.8461 | 0.8448 | -0.0013 |
| cat | 0.8390 | 0.7947 | -0.0444 |
| chair | 0.7799 | 0.7788 | -0.0011 |
| cow | 0.9219 | 0.8980 | -0.0239 |
| dining-table | 0.4553 | 0.4312 | -0.0241 |
| dog | 0.9312 | 0.8670 | -0.0642 |
| horse | 0.9007 | 0.8992 | -0.0014 |
| motorbike | 0.8565 | 0.8591 | +0.0026 |
| person | 0.7745 | 0.7779 | +0.0034 |
| potted-plant | 0.7392 | 0.7482 | +0.0089 |
| sheep | 0.8404 | 0.8282 | -0.0122 |
| sofa | 0.9640 | 0.9570 | -0.0070 |
| train | 0.6888 | 0.6880 | -0.0008 |
| tv | 0.8024 | 0.8036 | +0.0012 |

### COCO-Cap: SG24-B CA-only IoU

| Class | B0 selected 11 | B1 all 16 | B1 − B0 |
|---|---:|---:|---:|
| aeroplane | 0.5319 | 0.4832 | -0.0487 |
| bicycle | 0.5333 | 0.5152 | -0.0181 |
| bird | 0.6132 | 0.6041 | -0.0092 |
| boat | 0.5637 | 0.5135 | -0.0502 |
| bottle | 0.4667 | 0.4087 | -0.0580 |
| bus | 0.8898 | 0.9000 | +0.0102 |
| car | 0.6135 | 0.6197 | +0.0062 |
| cat | 0.6894 | 0.7097 | +0.0203 |
| chair | 0.3792 | 0.3269 | -0.0523 |
| cow | 0.7091 | 0.7059 | -0.0032 |
| dining table | 0.5485 | 0.6884 | +0.1399 |
| dog | 0.6780 | 0.6529 | -0.0251 |
| horse | 0.5885 | 0.6343 | +0.0458 |
| motorbike | 0.6417 | 0.6325 | -0.0093 |
| person | 0.1736 | 0.2621 | +0.0885 |
| potted plant | 0.5298 | 0.5206 | -0.0092 |
| sheep | 0.7105 | 0.6852 | -0.0253 |
| sofa | 0.5682 | 0.5334 | -0.0348 |
| train | 0.6409 | 0.5997 | -0.0413 |
| tv | 0.6025 | 0.6030 | +0.0006 |

### COCO-Cap: SG24-B final IoU

| Class | B0 selected 11 | B1 all 16 | B1 − B0 |
|---|---:|---:|---:|
| aeroplane | 0.7171 | 0.7434 | +0.0262 |
| bicycle | 0.5707 | 0.5722 | +0.0016 |
| bird | 0.7786 | 0.7766 | -0.0020 |
| boat | 0.5627 | 0.5569 | -0.0059 |
| bottle | 0.4984 | 0.4885 | -0.0098 |
| bus | 0.9027 | 0.8962 | -0.0065 |
| car | 0.6325 | 0.6865 | +0.0540 |
| cat | 0.8546 | 0.8603 | +0.0057 |
| chair | 0.4459 | 0.4313 | -0.0146 |
| cow | 0.8550 | 0.8496 | -0.0054 |
| dining table | 0.4704 | 0.5710 | +0.1006 |
| dog | 0.7556 | 0.7496 | -0.0060 |
| horse | 0.6743 | 0.7070 | +0.0327 |
| motorbike | 0.6248 | 0.6416 | +0.0168 |
| person | 0.2659 | 0.3129 | +0.0470 |
| potted plant | 0.6969 | 0.6735 | -0.0234 |
| sheep | 0.8733 | 0.8706 | -0.0028 |
| sofa | 0.6621 | 0.6217 | -0.0404 |
| train | 0.6871 | 0.6904 | +0.0033 |
| tv | 0.7152 | 0.7072 | -0.0079 |
