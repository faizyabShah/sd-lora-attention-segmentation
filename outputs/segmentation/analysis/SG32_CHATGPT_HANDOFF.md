# SG32 — reliability-adaptive second self-attention pass

**Status:** complete. The 40 class shards in Slurm array 238102 and merger
238103 completed successfully. Merger validation found all **490 VOC-Sim**
and **707 COCO-Cap** images, their fixed VOC folds, all 24 raw/R2 cells,
and complete compact diagnostic sidecars. Give ChatGPT this file together
with `outputs/segmentation/controlled_attention_cv/sg32/results.json` and
`per_class_iou.csv`. The JSON has every cell, forced OOF fold, five-fold
registered selector, all class IoUs, paired 20,000-resample mean-image
bootstrap, caption split, and mechanism diagnostics. The CSV provides all
class/cell deltas. Production per-image compact shards are also committed
for Claude's independent audit.

## Locked design

A3 class-word CA is frozen. SA uses the same 16/32/64 modules, matched calls
10–20 and eight heads as SG25. Each cell leaves the first fine SA pass
`u=A c` intact and changes only the second pass. S0 is historical ordinary
fine P2; S1 is historical SG30 fixed second-pass sharpening with power 2.6.
S2–S6 use a convex blend of ordinary and sharpened second-pass outputs;
S7–S11 use a row-wise adaptive exponent between 1 and 2.6. The reliability
signals are coarse/fine agreement (`c16`), all-three-scale pair agreement
(`pair`), three-scale range agreement (`range`), selected-row background
support (`bg`), and products with background. Cell assignments:

| Cell | Second-pass rule |
|---|---|
| S0 | Ordinary P2 control |
| S1 | Fixed power-2.6 sharpening control |
| S2 / S3 / S4 / S5 / S6 | Convex blend using c16 / pair / bg / pair×bg / range×bg |
| S7 / S8 / S9 / S10 / S11 | Row temperature using c16 / pair / bg / pair×bg / range×bg |

All cells are evaluated raw and with the unchanged R2 DenseCRF. VOC uses the
same five fixed folds and 17 thresholds. Full-VOC thresholds are frozen for
COCO. COCO is never used for selection. The reliability statistics below use
GT **only after inference**. A3's existing attention-orientation caveat
remains; SG32 was not an orientation correction.

## Main results

The registered multi-candidate selector retained **S0 in all 5 raw folds**
and **S0+R2 in all 5 final folds**. The best new final forced-OOF cell is
S7+R2, but its gain is only **+0.0005** and it wins only 2/5 pairwise folds.
This is not a promotion. Forced OOF, pairwise wins and the registered
selector are distinct statistics.

| Cell | Raw VOC OOF | +R2 VOC OOF | +R2 Δ vs S0 | +R2 wins | Raw COCO | +R2 COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | .8097 | **.8229** | — | — | .6598 | .6622 |
| S1, fixed sharp control | .7996 | .8122 | −.0107 | 1/5 | .6673 | **.6854** |
| S2, c16 convex | .8088 | .8223 | −.0006 | 2/5 | .6693 | .6750 |
| S3, pair convex | .8090 | .8223 | −.0006 | 2/5 | .6598 | .6747 |
| S4, bg convex | .8091 | .8220 | −.0009 | 1/5 | .6639 | .6788 |
| S5, pair×bg convex | .8081 | .8214 | −.0014 | 1/5 | .6641 | .6796 |
| S6, range×bg convex | .8076 | .8212 | −.0016 | 1/5 | .6642 | .6800 |
| S7, c16 temperature | .8094 | **.8234** | **+.0005** | 2/5 | .6692 | .6773 |
| S8, pair temperature | .8094 | .8231 | +.0002 | 2/5 | .6700 | .6772 |
| S9, bg temperature | .8067 | .8206 | −.0022 | 1/5 | .6714 | .6788 |
| S10, pair×bg temperature | .8054 | .8202 | −.0027 | 1/5 | .6641 | .6811 |
| S11, range×bg temperature | .8051 | .8198 | −.0030 | 1/5 | .6649 | .6820 |

The full-VOC S0+R2 score is **.8243**; its frozen COCO score is **.6622**.
S7+R2 scores .8248 full VOC and .6773 COCO, but the full-VOC difference is
not the five-fold selection result. S1+R2 is the strongest COCO SG32 cell
at .6854 (+.0232 over this run's fine-P2 reference), yet it is the **known
SG30 fixed-sharpening control**, not a new reliability method. The separate
SG31 historical 16→64+R2 cell scored .6977 COCO; cross-run comparisons are
descriptive because regenerated images and masks can differ. SG32 therefore
does not establish a new overall best pipeline.

### Transfer and paired-image evidence

These intervals are on **paired mean-image IoU differences**, not on
20-class macro mIoU differences. The same-generation S0+R2 is the reference.

| Cell | COCO macro Δ | COCO mean-image Δ [95% paired interval] | Better/worse/tied images |
|---|---:|---:|---:|
| S1+R2 | +.0232 | +.0129 [.0082, .0177] | 442/263/2 |
| S2+R2 | +.0128 | +.0097 [.0072, .0122] | 493/211/3 |
| S7+R2 | +.0151 | +.0106 [.0075, .0138] | 471/234/2 |
| S8+R2 | +.0150 | +.0105 [.0077, .0137] | 491/213/3 |
| S11+R2 | +.0198 | +.0128 [.0090, .0168] | 450/254/3 |

The 465 COCO images with the class word in the generation caption and 242
without it both improve for S7+R2 against S0+R2: macro Δ **+.0138** and
**+.0168**, respectively. For S8+R2 they are +.0138 and +.0171. These
are post-hoc splits and do not justify choosing a cell on COCO.

## Mechanism: why the adaptive cells did not win VOC

The three-scale agreement signals are nearly saturated. On VOC, mean `pair`
reliability is .937 on GT foreground and .964 on background more than 64
pixels from GT; COCO is .932 and .963. The `pair<.5` group is essentially
empty. Thus simple cross-scale agreement is **not a useful foreground vs
background discriminator** here. It also assigns *higher* ordinary-pass
reliability to distant background than to target foreground, so it cannot
by itself veto leakage.

The selected-row background signal is more discriminative: mean `bg`
reliability is .880 on VOC GT foreground versus .545 on distant background
(.872 vs .578 on COCO). Its product with `pair` gives .827 vs .525 on VOC
and .815 vs .556 on COCO. This is a real spatial distinction, but coupling
it to stronger row sharpening did not improve VOC OOF; S10/S11 lose .0027/
.0030 after R2. For S10, the mean adaptive exponent on VOC is **1.245 on
GT foreground versus 1.782 on distant background**; on COCO it is 1.270
versus 1.732. The constructed mechanism does sharpen background rows more.

Row entropy confirms that sharpening was applied: on VOC, S7 changes sampled
SA-row entropy from 6.828 to 6.582, and S10 to 6.061; on COCO, S7 changes
6.749 to 6.518, and S10 to 5.857. More aggressive sharpening does not
translate into VOC improvement. At fixed raw diagnostic threshold .45,
S10 compared with S1 gains 1.23M VOC TP but also 0.98M FP; on COCO it gains
0.88M TP but 2.89M FP, including 1.45M FP more than 64 pixels from GT.
That diagnostic comparison is against S1 and is **not** an R2 performance
comparison. It indicates that the adaptive map partly undoes the fixed
sharpening's precision benefit.

Per-class examples show the tradeoff. Relative to S0+R2 on VOC, S7+R2 gains
train +.0365 and dining table +.0170, but loses cow −.0222, cat −.0209,
potted plant −.0177, and chair −.0151. On COCO, S7+R2 gains motorbike
+.0435, TV +.0349, bicycle +.0338, chair +.0323, while losing dining
table −.0284 and person −.0117. The complete 20-class audit for all cells
and both datasets is in the CSV.

## Interpretation for ChatGPT

1. **No SG32 promotion.** VOC five-fold selection kept S0/S0+R2 in every
   fold. The small S7+R2 OOF gain lacks fold support.
2. **COCO gains are interesting but not a selection criterion.** The
   strongest COCO score belongs to S1, an existing fixed-sharpening control.
   Adaptive cells transfer positively relative to fine P2, yet fail the
   locked VOC selection rule.
3. **Background reliability has signal; scale agreement is saturated.**
   This explains why the combined signal changes where sharpening occurs,
   but the present second-pass mapping does not convert that distinction
   into robust VOC gains.
4. **Mechanism claim is limited.** Pixel-count and entropy diagnostics show
   the intended operations ran, not that adaptive sharpening is causally
   better. No new hyperparameter was tuned on COCO.

Questions worth considering after the already scheduled SG33 result: whether
the background signal should gate *propagation extent* rather than row
temperature, and whether foreground completion versus distant leakage can be
balanced without losing VOC fold robustness. Those are hypotheses, not SG32
results or pre-registered follow-up cells.
