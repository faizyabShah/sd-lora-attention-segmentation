# SG25 and Pre-SG26 handoff — 24 October 2026

## What ran

SG25 evaluated multi-resolution self-attention propagation with **A3 CA
frozen**. CA used the selected 11 modules, calls 0–30, all heads, and SG24's
pseudo-global layer weights. SA used calls **10–20**, all eight heads, and
fixed final upsampling-path modules at 16×16, 32×32, or 64×64. Each cell used
two propagation operations and the unchanged **R2 DenseCRF** for its final
score. Cross-resolution cells paired operators at the same diffusion call,
averaged the first operator's heads, resized, applied all second-operator
heads, and averaged calls and second heads. No CA, SA time window, head,
reliability, or CRF parameter was retuned. The established FINE-P2 is the
64×64 `mean_i(A_i² c)` control.

The GPU array **236164** and CPU merge **236165** completed successfully.
Coverage is **490 VOC-Sim** and **707 COCO-Cap** images (20 classes each).
The final smoke job **236162** passed live VOC and COCO examples and directly
matched the original SG24 MID-P2 operator numerically. The recomputed
A3/FINE-P2+R2 control is **0.82425 VOC / 0.66210 COCO**, matching SG24-A3
within regeneration drift (previously 0.8243 / 0.6620).

The existing **five fixed VOC folds** and 17-threshold grid selected methods
and thresholds. COCO used the **full-VOC-selected thresholds unchanged**.
All mIoU values below are macro averages over the 20 target classes.

## SG25: all cells

| SA cell | Forced VOC OOF | Δ OOF vs same-stage fine | Held-out fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| FINE-P2 | .8097 | — | — | .8114 | .450 | .6599 |
| FINE-P2+R2 | .8228 | — | — | .8242 | .425 | .6621 |
| COARSE16-P2 | .7225 | −.0873 | 0/5 | .7231 | .575 | .6282 |
| COARSE16-P2+R2 | .7538 | −.0690 | 0/5 | .7538 | .575 | .6652 |
| COARSE32-P2 | .7924 | −.0173 | 0/5 | .7924 | .475 | .6693 |
| COARSE32-P2+R2 | .8138 | −.0090 | 0/5 | .8148 | .475 | .6948 |
| 16→64 | .8020 | −.0077 | 0/5 | .8032 | .525 | .6869 |
| 16→64+R2 | .8170 | −.0058 | 1/5 | .8183 | .500 | **.6975** |
| 32→64 | .8089 | −.0008 | 1/5 | .8106 | .475 | .6695 |
| 32→64+R2 | .8217 | −.0012 | 3/5 | .8228 | .450 | .6717 |
| 64→16 | .7505 | −.0592 | 0/5 | .7513 | .600 | .6403 |
| 64→16+R2 | .7825 | −.0403 | 0/5 | .7825 | .575 | .6705 |

**Registered VOC decision:** both raw and R2 multi-candidate screens chose
their respective FINE-P2 references in **5/5 folds**; no candidate met the
training-fold eligibility rule. Therefore **none is promoted** under the
VOC tuning protocol. The closest final candidate, 32→64+R2, won 3/5 in a
*forced held-out comparison* but had OOF −.00118 and was selected by the
registered screen in 0/5 folds. The forced wins and selector votes answer
different questions, as in SG24.

**Frozen transfer is interesting:** 16→64+R2 scores **.69750 COCO**, a
**+.03540** macro mIoU gain over FINE-P2+R2. COARSE32-P2+R2 reaches
**.69481** (+.03271). They lose on VOC by .00598 and .00950 full-set mIoU,
respectively. These COCO gains are observations after VOC-only selection,
not grounds to retroactively choose a new pipeline. On COCO, 16→64+R2
improves 19/20 class IoUs; dining table is the exception (−.080). Its
precision/recall changes from **.7032/.8893** to **.7514/.8803**. This
suggests less leakage with a modest recall cost. COARSE32-P2+R2 changes
COCO precision/recall to **.7537/.8732**.

Paired **mean-image IoU** (distinct from macro mIoU) for 16→64+R2 versus
FINE-P2+R2 is **+.02716** on COCO (20k-resample 95% interval
[+.02000,+.03459], 426 better / 278 worse / 3 tied images). On VOC it is
−.00526 [−.01106,+.00039], 213 better / 277 worse. COARSE32-P2+R2's
paired mean-image delta is +.01852 [+.01244,+.02477] on COCO and
−.00937 [−.01514,−.00366] on VOC.

### Per-class change for the strongest COCO transfer cell

IoU values below are rounded; the exact 20-class data for **every** cell are
in [`per_class_iou.csv`](../controlled_attention_cv/sg25/per_class_iou.csv).

| Class | VOC fine | VOC 16→64 | VOC Δ | COCO fine | COCO 16→64 | COCO Δ |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | .884 | .870 | −.014 | .717 | .774 | +.057 |
| bicycle | .762 | .760 | −.001 | .570 | .615 | +.045 |
| bird | .913 | .920 | +.007 | .779 | .813 | +.034 |
| boat | .839 | .851 | +.012 | .563 | .601 | +.038 |
| bottle | .853 | .884 | +.031 | .496 | .522 | +.026 |
| bus | .897 | .919 | +.022 | .903 | .921 | +.018 |
| car | .846 | .853 | +.007 | .633 | .766 | +.134 |
| cat | .839 | .831 | −.008 | .855 | .863 | +.008 |
| chair | .779 | .753 | −.027 | .446 | .487 | +.041 |
| cow | .922 | .873 | −.049 | .855 | .856 | +.001 |
| dining table | .456 | .438 | −.017 | .470 | .390 | −.080 |
| dog | .931 | .929 | −.002 | .756 | .756 | +.001 |
| horse | .898 | .880 | −.018 | .674 | .715 | +.041 |
| motorbike | .857 | .862 | +.006 | .625 | .679 | +.054 |
| person | .775 | .782 | +.007 | .266 | .282 | +.016 |
| potted plant | .739 | .704 | −.035 | .697 | .780 | +.083 |
| sheep | .840 | .725 | −.115 | .874 | .890 | +.017 |
| sofa | .964 | .957 | −.007 | .662 | .706 | +.043 |
| train | .689 | .753 | +.064 | .687 | .758 | +.071 |
| TV | .802 | .822 | +.019 | .715 | .775 | +.060 |

VOC class names in the raw file use `dining-table` / `potted-plant`, while
COCO uses spaces; they refer to the same target categories.

## Pre-SG26: first positive propagation depth

Using the same A3 seed and 64×64 SA observations, we formed mean maps
`mean_i(A_i^k c)` for **k=1..5**. Each depth map was separately min-max
normalized, resized to annotation resolution, and thresholded at the fixed
VOC A3 threshold **.425**. A pixel belongs to the *first* depth at which it
crosses; later persistence was not required. Counts are diagnostic only.

| Dataset | First depth | Newly positive pixels | TP rate | TP/FP | FP >16 px | FP >64 px | Separate-component FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOC | 1 | 49,901,885 | .865 | 6.38 | 2,364,466 | 669,234 | 52,862 |
| VOC | 2 | 1,582,733 | .542 | 1.18 | 442,837 | 213,805 | 36,651 |
| VOC | 3 | 881,016 | .393 | .65 | 356,482 | 207,231 | 31,315 |
| VOC | 4 | 676,019 | .289 | .41 | 340,317 | 186,537 | 26,414 |
| VOC | 5 | 569,448 | .223 | .29 | 329,548 | 182,022 | 23,175 |
| COCO | 1 | 53,361,823 | .708 | 2.43 | 9,746,486 | 5,755,204 | 1,503,240 |
| COCO | 2 | 3,644,935 | .225 | .29 | 2,317,217 | 1,609,379 | 482,378 |
| COCO | 3 | 2,008,303 | .140 | .16 | 1,434,108 | 932,021 | 331,744 |
| COCO | 4 | 1,737,106 | .115 | .13 | 1,308,110 | 850,403 | 289,736 |
| COCO | 5 | 1,535,243 | .117 | .13 | 1,170,046 | 735,231 | 184,855 |

The overall trend strongly favors early arrivals. On VOC, the fraction of
FP more than 16 pixels from GT rises from **35% at depth 1** to **74% at
depth 5**; on COCO it rises from **63%** to **86%**. Later additions are
more often distant leakage. This is consistent with P2 beating deeper
unrestricted propagation, and suggests arrival depth could be a confidence
feature. It does **not** yet show that an arrival-based veto improves IoU.

The pattern has class exceptions. VOC person first-arrival TP rates are
**.930, .639, .648, .679, .586** across depths 1–5; a hard depth-2 cutoff
could reject useful person pixels. COCO person is already difficult at depth
1 (.317 TP rate). COCO dining-table arrival pixels are >.98 TP at every
depth, but its GT masks cover **93% of image pixels on average** across 14
samples (median 100%); this class must not be used as evidence that late
propagation is universally reliable. All class/depth counts are in
[`first_arrival.csv`](../controlled_attention_cv/sg25/first_arrival.csv).

**Interpretation limits:** Separately normalizing each depth means the
threshold measures each depth map's *relative* spatial contrast rather than
a common absolute score. First crossing can be followed by disappearance.
These summaries are pixel-count weighted; images/classes with large masks
contribute more. No arrival threshold or SG26 mask was tuned or evaluated.

## Decision for the next discussion

1. Keep **A3→FINE-P2→R2** as the VOC-selected working pipeline.
2. Treat **16→64+R2** and **COARSE32-P2+R2** as strong COCO-transfer leads
   whose VOC losses need explanation or a locked follow-up; avoid choosing
   either using the COCO result alone.
3. The first-arrival signal is promising as a *diagnostic* because late
   additions have substantially worse TP/FP ratios, but class exceptions
   and per-depth normalization call for a carefully controlled SG26 design.

Machine-readable full folds, thresholds, precision/recall, bootstrap
intervals, and per-class metrics are in
[`results.json`](../controlled_attention_cv/sg25/results.json). The compact
[`report.md`](../controlled_attention_cv/sg25/report.md) reproduces the
headline tables. Resumable per-image shards were not committed.

**Post-hoc caption-presence check:** both COCO caption-present and
caption-absent groups gain from 16→64+R2; the absent group's macro mIoU
gain is larger, but paired mean-image and class-balanced interaction
intervals include zero. See the separate
[caption-presence analysis](SG25_CAPTION_PRESENCE.md) and its reduced
per-image CSV. COCO's class attribution prompt always contains the queried
word even when the generation caption does not.
