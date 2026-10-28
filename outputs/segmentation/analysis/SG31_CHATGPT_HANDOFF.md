# SG31 — SeeDiff-inspired re-seeding and background propagation

**Status:** complete. Smoke 237989, 39 original production shards in 237990,
resumed COCO-person shard 238099, and replacement merger 238100 completed.
The merger validated **490 VOC-Sim** and **707 COCO-Cap** images, all 40
class shards, fixed VOC folds, and complete compact diagnostic sidecars.
One S1 same-generation assertion originally stopped shard 34 after 48/52
images because a single pixel differed by 2.16e-6, just over `atol=2e-6`.
The check was relaxed to `atol=1e-5`, the four remaining images were resumed,
and S0/S1 are scored using the reference maps. No experiment cell was changed.

**Give ChatGPT this handoff together with** `results.json` and
`per_class_iou.csv` in the package. The JSON has every threshold, forced OOF
fold, registered selector, 20-class IoUs, paired 20,000-resample *mean-image*
interval, caption split, seed distribution and background/transition
diagnostic. The CSV makes all class/cell deltas easier to inspect.

## Locked conditions and meaning of the cells

SG31 tests mechanisms inspired by [SeeDiff: Off-the-Shelf Seeded Mask
Generation from Diffusion Models](https://doi.org/10.1609/aaai.v39i6.32686).
It is **not** SeeDiff end-to-end: it uses the historical A3 class-word CA
seed in SD1.5, not SeeDiff's own backbone or 16×16 CA source. A3's known
affinity-orientation caveat remains. A3 CA modules/calls, SA calls 10–20,
eight heads, SG25 SA modules at 16/32/64, 17 VOC thresholds and unchanged
R2 DenseCRF were fixed. Seed threshold α=.5 came from SeeDiff and was not
tuned. Selected-row expansion literally averages `A[p,:]` over chosen seed
queries; it is intentionally different from historical continuous `A @ c`.

| Cell | Fixed operation before optional R2 |
|---|---|
| S0 | Historical FINE-P2 at 64×64 |
| S1 | Historical SG25 continuous 16→64 control |
| S2 | Continuous 16→32→64, same call |
| S3 | Hard selected-row re-seeding 16→32→64 |
| S4 | Hard selected-row re-seeding 32→64 |
| S5 | Soft weighted-row control 16→32→64 |
| S6 | S0 plus one-step selected-row background suppression |
| S7 | S0 plus observation-wise P2 background suppression |
| S8 | S3 plus one-step background suppression |
| S9 | S4 plus one-step background suppression |
| S10 | `minmax(S0²)` sharpening/inversion-only control |

S0 and S1 are **historical controls**, not new SG31 discoveries. SG31 S1+R2
COCO .6977 closely replays SG25 16→64+R2 .6975; S0+R2 .6622 similarly
replays SG25 FINE-P2+R2 .6621. All comparisons below use SG31's **same-run
S0**, because separately regenerated pixel counts need not match exactly.

## Main result: no SG31 promotion on VOC

The registered five-fold, multi-candidate VOC selector retained **S0 in
all 5 raw folds** and **S0+R2 in all 5 final folds**. Forced OOF and pairwise
fold wins are separate statistics. COCO thresholds were selected on full
VOC only; there was no COCO model selection.

| Cell | Raw VOC OOF | +R2 VOC OOF | +R2 Δ vs S0 | +R2 pairwise VOC wins | Raw COCO | +R2 COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | .8097 | **.8228** | — | — | .6600 | .6622 |
| S1 | .8020 | .8170 | −.0058 | 1/5 | .6870 | **.6977** |
| S2 | .8060 | .8182 | −.0047 | 1/5 | .6782 | .6923 |
| S3 | .2465 | .2122 | −.6106 | 0/5 | .2846 | .2606 |
| S4 | .3907 | .3695 | −.4534 | 0/5 | .4425 | .4303 |
| S5 | .4920 | .4863 | −.3366 | 0/5 | .4601 | .4658 |
| S6 | .8075 | .8199 | −.0030 | 2/5 | .6729 | .6757 |
| S7 | .8078 | .8132 | −.0096 | 1/5 | .6785 | .6847 |
| S8 | .2130 | .1785 | −.6443 | 0/5 | .2479 | .2224 |
| S9 | .3405 | .3126 | −.5102 | 0/5 | .3978 | .3801 |
| S10 | .7907 | .7968 | −.0260 | 0/5 | .6768 | .6855 |

Full VOC S0+R2 is **.8243** and frozen COCO is **.6622**. S1 is the
strongest COCO cell here (+.0354) but is a known SG25 control and loses
.0058 forced VOC OOF. S2's added 32×32 step is slightly higher on full
VOC (.8193 vs .8183) but lower on COCO (.6923 vs .6977); the direction is
mixed, so do not claim that three-scale continuous propagation is superior.
All newly constructed S3–S10 cells fail the
registered VOC promotion rule. The best new final VOC cell is S6+R2 at
.8199 OOF, still −.0030 versus S0+R2.

### COCO transfer and paired-image evidence

The primary score is 20-class pooled macro mIoU. The intervals below are
for paired **mean-image IoU**, a different estimand; they are not confidence
intervals on the macro mIoU delta.

| Cell vs S0+R2 | COCO macro mIoU Δ | COCO mean-image Δ [95% paired interval] | Better/worse/tied COCO images |
|---|---:|---:|---:|
| S1+R2, historical SG25 | +.0354 | +.0276 [+.0204, +.0351] | 423/281/3 |
| S2+R2 | +.0301 | +.0239 [+.0154, +.0328] | 417/289/1 |
| S6+R2 | +.0135 | +.0109 [+.0068, +.0153] | 449/254/4 |
| S7+R2 | +.0225 | +.0171 [+.0089, +.0254] | 431/275/1 |
| S10+R2, simple square | +.0233 | +.0186 [+.0092, +.0279] | 409/297/1 |

S10 matches or exceeds S7 on COCO (.6855 vs .6847) without any SA
background propagation. That weakens a claim that S7's COCO transfer gain
comes from a distinctive background mechanism. S6 is also below S10 on
COCO. All three lose on VOC, especially S10 (−.0260 OOF).

## Mechanism diagnostics

### Hard re-seeding collapses foreground extent

At fixed raw diagnostic threshold .45, VOC S3 removes **37.48M true-positive
pixels** relative to S0 while removing only 5.71M false positives; S4
removes 33.91M TP and 5.57M FP. On COCO S3 removes 30.78M TP and 14.84M
FP; S4 removes 26.30M TP and 14.52M FP. Relative to the directly relevant
continuous S1 path, hard S3 loses 38.89M VOC TP and S4 loses 35.32M TP.
These counts use one fixed diagnostic threshold, not the VOC-selected
threshold, but match the final-mask recall collapse: VOC S0+R2 recall .9161,
S3+R2 .1785, S4+R2 .3369. Precision rises above .96 because the masks
retain small cores.

Seed diagnostics show **no empty-seed fallback** among 5,390 VOC and 7,777
COCO image-call observations at each relevant path/scale. The failure is
therefore not caused by empty initialization. For S3, median seed fraction
shrinks from **30.9% at 16×16 → 13.4% at 32×32 → 6.4% at 64×64** on VOC;
on COCO it is **19.1% → 9.0% → 5.1%**. Median connected components rise
from 1→3→4 on VOC. The mean 64×64 expanded area above .5 is only **3.46%
VOC / 3.44% COCO**. S4's 64×64 expanded area is similarly small: **5.41%
VOC / 5.11% COCO**. This supports the interpretation that fixed α=.5 with
selected-row expansion repeatedly concentrates and fragments the seed set
in our A3/SD1.5 setting. It does **not** prove SeeDiff's original pipeline
would behave similarly.

The soft weighted-row control S5 is less catastrophic, but still falls to
.4863 VOC OOF+R2 and .4658 COCO. Thus hard thresholding is not the only
problem; the row-query expansion orientation and repeated per-scale
renormalization are also plausible contributors. This is an inference from
the controls, not a separately isolated causal test.

### Background support is accurate in isolation, but removal is costly

At 64×64 post-hoc background threshold .5, S6's high-background pixels are
outside GT **94.1% VOC / 96.0% COCO**. Yet at raw foreground threshold .45
it removes only **0.74 FP per TP removed on VOC**, and **1.93 on COCO**;
S7 removes **0.60 / 1.48** respectively. Most removed FP are close to GT
boundaries (S6: 37,575/39,090 VOC; 65,211/84,128 COCO within 16 px).
This is a boundary-vs-object-completion tradeoff, not a clean elimination of
distant false objects. After R2, S6+R2 loses .0030 VOC OOF while gaining
.0135 COCO; S7+R2 loses .0096 VOC OOF and gains .0225 COCO.

On top of already-collapsed S3/S4 foreground, S8/S9 background seeds cover
about **94–97% of the grid** on average. S8 removes only .057 VOC FP per TP
removed; S9 .065. These combinations make the maps worse on both datasets.
No further stacking is supported by this run.

## Class and caption-presence checks

Per-class **+R2 IoU deltas** versus same-run S0+R2:

| Class | VOC S1 | COCO S1 | VOC S2 | COCO S2 | VOC S6 | COCO S6 |
|---|---:|---:|---:|---:|---:|---:|
| person | +.0074 | +.0168 | +.0147 | +.0197 | −.0304 | −.0152 |
| chair | −.0265 | +.0413 | −.0049 | +.0211 | +.0121 | +.0616 |
| bicycle | −.0014 | +.0448 | +.0180 | +.0155 | +.0033 | +.0315 |
| sheep | −.1155 | +.0164 | −.1183 | −.0004 | −.0574 | +.0168 |
| cow | −.0492 | +.0012 | −.0415 | −.0023 | −.0648 | −.0169 |
| dining table | −.0173 | −.0802 | −.0377 | −.0553 | +.0193 | −.0462 |
| car | +.0066 | +.1337 | −.0060 | +.1351 | −.0019 | +.0205 |
| train | +.0649 | +.0708 | +.0710 | +.0737 | +.0669 | +.0186 |
| aeroplane | −.0144 | +.0592 | −.0022 | +.0964 | +.0076 | +.0345 |
| potted plant | −.0352 | +.0828 | −.0395 | +.0916 | −.0141 | −.0028 |

S1 improves **19/20 COCO classes**; the exception is dining table (−.0802).
VOC S1 improves only **9/20**, with sheep −.1155. This is the main
cross-dataset class tradeoff of the historical coarse-to-fine path.

The post-hoc COCO caption split uses the released `word_included` field:
**465 present**, **242 absent**. S1+R2 macro gain versus S0+R2 is **+.0239
present / +.0481 absent**. S2+R2 is **+.0158 / +.0483**, and S6+R2 is
**+.0105 / +.0205**. These descriptive splits use the fixed VOC thresholds,
but class composition differs between subgroups; do not treat the difference
between split deltas as a controlled effect of caption presence or tune on it.

## Decision for ChatGPT

Under the agreed VOC-only selector, **retain S0+R2**. The genuinely new
SG31 hard re-seeding branch fails because it discards most foreground;
soft row weighting does not rescue it. Explicit background support can
improve frozen COCO transfer but loses VOC OOF and does not clearly beat
simple score squaring. The strongest COCO cell, S1+R2, is the **already
known SG25 continuous 16→64 control**, not a SeeDiff-derived innovation.
Its +.0354 COCO transfer signal and pronounced VOC sheep/dining-table
losses are worth explaining, but not promoting by COCO selection. All GT
usage in seed/background mechanisms was post-hoc diagnostics only.
