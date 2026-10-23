# SG22 and SG23: complete results for external review

**Run status (23 October 2026):** SG22 and SG23 both completed on all 490
VOC-Sim and 707 COCO-Cap images. Their CPU mergers exited 0:0 and checked
the coverage. This is a training-free, class-word W-track analysis. COCO
was transfer-only: every threshold and promotion decision came from VOC.

## Read this first

- **SG22's selected final pipeline is MID-P2+R2.** Calls 10–20, two
  observation-wise SA propagation passes, then unchanged DenseCRF. VOC
  fold selection voted for it **4/5**; selected OOF mIoU was **0.8158**
  versus **0.8132** for the same-generation ALL-P2+R2 reference. Its
  independently forced OOF mIoU was **0.8168** versus **0.8132**. Full
  VOC: **0.8185** versus **0.8132** (+0.0053). Frozen COCO: **0.6438**
  versus **0.6372** (+0.0066). Mean per-image IoU differences had paired
  20k-bootstrap 95% intervals **+0.0045 [0.0022, 0.0069]** on VOC and
  **+0.0045 [0.0025, 0.0065]** on COCO. These intervals are for mean
  *per-image* IoU, not macro 20-class mIoU.
- **SG22 raw-map selection did not promote a change.** MID-P2 raw scored
  0.8035 OOF versus 0.7997 for ALL-P2, but the normal promotion screen
  voted for it only 2/5 folds. After R2, MID-P2+R2 got 4/5 votes. This
  is a specific propagation-depth × postprocessing result, not a reason
  to select raw MID-P2 separately.
- **SG23 selected raw MID-SG2, but no SG23 CRF cell.** Raw MID-SG2 won
  5/5 VOC folds, OOF **0.7999** versus MID-R0/T1 **0.7939** (+0.0060).
  Full VOC was 0.8020 versus 0.7959. Frozen COCO was **0.6292** versus
  **0.6287**, only +0.0006. Its CRF version scored VOC 0.8168 and COCO
  0.6453, but only 1/5 fold votes in the final-pipeline screen. The
  SG23 final reference ALL-P2+R2 remained frozen.
- **The strongest SG22 COCO transfer cell is still MID-P1+R2/T1+R2**:
  SG22 replay COCO **0.6464**, though its VOC OOF
  0.8104 did not displace P1+R2 in the locked selection. SG22's newly
  selected system scored 0.6438 on COCO. The equivalent SG23 replay
  scored 0.6467. Do not use COCO to revise the VOC-frozen choice.
- **SG23 interactions are predominantly negative.** On raw VOC OOF,
  adding SG2/SG11/SG12 improves ALL by +0.0110/+0.0110/+0.0094, but
  MID by only +0.0060/+0.0055/+0.0025. The corresponding interactions
  are −0.0050/−0.0055/−0.0068. Raw frozen COCO interactions are also
  negative (−0.0056/−0.0020/−0.0114). Temporal filtering and these
  reliability rules overlap more than they complement each other.

## Frozen methodology and terminology

SD 1.5, frozen ordinary class-word CA seed `c`, SA module 15 at 64×64,
eight SA heads, 31 diffusion calls, min-max normalize each map at 64×64,
bilinear resize to annotation size, and threshold from the existing
17-point grid 0.30–0.70. The existing five VOC folds choose thresholds
on four folds at a time. Normal magnitude/class-stability promotion
guards, 4/5 fold votes, and a positive selected-OOF guard decide the
frozen method. R2 is the unchanged OVAM SimpleCRF wrapper and parameters,
applied to each thresholded mask with the regenerated RGB image. No
optimized tokens, learned parameters, COCO tuning, or new CRF settings.

`ALL` = calls 0–30, `MID` = calls 10–20 inclusive. In SG22 `P1/P2/P3`
mean average of **per-observation** `A_i c`, `A_i² c`, `A_i³ c`; powers of
an averaged matrix were never used. Historical REF = ALL-P1, historical
P1/final reference = ALL-P2, and SG18 T1 = MID-P1. In SG23 R0 is the
one-pass mean; SG2 is exact locked `G_cv_g20_n1` (gamma 2); SG11 is 20%
trimmed observations gated by directional consensus rho .80; SG12 keeps
the top 50% under leave-one-out cosine agreement. MID versions use only
88 MID observations; SG12 keeps exactly 44. ALL uses 248 and keeps 124.
ALL-P2(+R2) is regenerated inside SG23 only as a same-generation
reference, outside the eight-cell reliability factorial.

**Metric definitions:** each dataset mIoU is the mean of the 20 pooled
foreground class IoUs. “VOC OOF” below is the *forced per-cell* five-fold
score, with a threshold selected independently for that cell in each
training fold. The “selected OOF” above is the fold-voted mixed policy,
so it can differ from a candidate's forced OOF. Precision/recall and
pixel IoU pool all pixels. Paired bootstrap intervals use mean per-image
IoU differences. All classes and intervals for every cell are in the
attached JSON/CSV files.

## SG22: all 12 cells

| Cell | VOC forced OOF | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|
| ALL-P1 / REF | 0.7873 | 0.7886 | .425 | 0.6172 |
| ALL-P1+R2 | 0.8047 | 0.8047 | .425 | 0.6348 |
| ALL-P2 / historical P1 | 0.7997 | 0.8006 | .450 | 0.6246 |
| **ALL-P2+R2 / final reference** | **0.8132** | **0.8132** | **.450** | **0.6372** |
| ALL-P3 | 0.8002 | 0.8017 | .475 | 0.6282 |
| ALL-P3+R2 | 0.8122 | 0.8134 | .450 | 0.6287 |
| MID-P1 / T1 | 0.7939 | 0.7959 | .425 | 0.6284 |
| MID-P1+R2 | 0.8104 | 0.8121 | .425 | **0.6464** |
| MID-P2 | 0.8035 | 0.8055 | .450 | 0.6308 |
| **MID-P2+R2 / VOC-selected** | **0.8168** | **0.8185** | **.450** | **0.6438** |
| MID-P3 | 0.8024 | 0.8044 | .450 | 0.6214 |
| MID-P3+R2 | 0.8146 | 0.8168 | .425 | 0.6216 |

**Class effects of MID-P2+R2 versus ALL-P2+R2.** On VOC, chair +0.0248,
bottle +0.017, bird +0.0139 and dining table +0.0136 improve; person
−0.0159, cat −0.0057 and bicycle −0.0028 worsen. On COCO, chair
+0.0208, aeroplane +0.015, bicycle +0.015 and sofa +0.012 improve;
dining table −0.009 and cat −0.004 worsen. Thus two MID passes recover
much of T1's VOC chair and person loss: relative to MID-P1+R2, VOC chair
gains +0.0811 and person +0.0447. The person score still trails
ALL-P2+R2 by −0.0159. SG22 was selected by the cross-class rule, not
universal gains.

**Depth mechanism, fixed .45 before CRF.** On VOC, MID depth 1→2
gained 1.180M TP and 0.720M FP while losing 0.257M TP and removing
0.492M FP. MID depth 2→3 gained only 0.450M TP but 0.513M FP and lost
0.206M TP. COCO depth 2→3 gained 0.479M TP versus 1.721M FP,
including 0.932M newly gained FP farther than 64 pixels from GT. This
supports the observed plateau/transfer drop at P3. These are pixel
counts across datasets, not independent images or a causal proof; raw
threshold crossings and selected-threshold IoUs answer different
questions. The JSON includes all ALL/MID TP/FP gained/lost, distant and
separate-component FP, plus CA-relative events.

## SG23: full reliability factorial

| Cell | VOC forced OOF | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|
| ALL-R0 / REF | 0.7874 | 0.7887 | .425 | 0.6175 |
| ALL-R0+R2 | 0.8049 | 0.8049 | .425 | 0.6351 |
| ALL-SG2 | 0.7984 | 0.7984 | .450 | 0.6237 |
| ALL-SG2+R2 | 0.8112 | 0.8112 | .425 | 0.6263 |
| ALL-SG11 | 0.7984 | 0.7984 | .425 | 0.6219 |
| ALL-SG11+R2 | 0.8153 | 0.8153 | .400 | 0.6295 |
| ALL-SG12 | 0.7968 | 0.7987 | .450 | 0.6274 |
| ALL-SG12+R2 | 0.8117 | 0.8133 | .425 | 0.6317 |
| MID-R0 / T1 | 0.7939 | 0.7959 | .425 | 0.6287 |
| MID-R0+R2 | 0.8105 | 0.8122 | .425 | **0.6467** |
| **MID-SG2 / raw VOC-selected** | **0.7999** | **0.8020** | **.425** | **0.6292** |
| MID-SG2+R2 | 0.8155 | 0.8168 | .425 | 0.6453 |
| MID-SG11 | 0.7994 | 0.8012 | .425 | 0.6311 |
| MID-SG11+R2 | 0.8158 | 0.8178 | .400 | 0.6405 |
| MID-SG12 | 0.7964 | 0.7982 | .425 | 0.6271 |
| MID-SG12+R2 | 0.8146 | 0.8146 | .425 | 0.6439 |
| ALL-P2 / auxiliary | 0.7997 | 0.8006 | .450 | 0.6247 |
| **ALL-P2+R2 / final reference** | **0.8133** | **0.8133** | **.450** | **0.6373** |

**SG23 selection:** raw MID-SG2 received 5/5 fold votes. The CRF
screen voted 4/5 for the ALL-P2+R2 reference and 1/5 for MID-SG2+R2,
so no SG23 final-pipeline variant was promoted. Each forced MID
reliability+R2 cell is at most +0.0024 OOF versus P1+R2; that is below
the normal promotion size guard in most folds. Compare all CRF cells to
P1+R2, not to raw T1.

**Class and mechanism effects.** Raw MID-SG2 versus MID-R0 on VOC gives
person +0.032 and chair +0.026 class IoU; on COCO person +0.009 and chair
+0.007, but car −0.016 and sofa −0.011 help cancel the macro gain. At
fixed .45, MID-SG2 gains 315k TP and 173k FP, removes 62k TP and 77k FP
on VOC relative to T1's CA-background additions. On COCO it gains 327k
TP and **751k FP**, of which 437k are >64 pixels from GT; it removes
127k TP and 252k FP. This is a possible reason for the weak COCO mIoU
gain despite useful raw VOC person/chair gains. MID-SG11 removes more
FP but also loses 414k TP on VOC and 613k TP on COCO; its fixed-threshold
FP-removed/TP-lost ratio is only 0.87 VOC / 1.84 COCO. It is not a
clean suppression of leakage. All retained/removed and distant/separate
counts are in SG23 `results.json`.

| Raw interaction | VOC OOF | Full VOC | Frozen COCO |
|---|---:|---:|---:|
| SG2 | −0.0050 | −0.0036 | −0.0056 |
| SG11 | −0.0055 | −0.0044 | −0.0020 |
| SG12 | −0.0068 | −0.0077 | −0.0114 |

The interaction is `(MID-method−MID-R0)−(ALL-method−ALL-R0)`.
CRF interactions and exact unrounded values are in SG23 `results.json`.
The occasional positive COCO CRF interaction reflects less loss than
the corresponding ALL reliability+CRF cell; it does **not** imply a
winning MID reliability final pipeline.

**Replay check:** common control cells agree closely in aggregate across
the two independent experiments (for example, full VOC ALL-P2+R2 is
0.8132 in SG22 and 0.8133 in SG23; COCO 0.6372/0.6373). Individual
per-image confusion counts are not identical across the two GPU replays,
so paired intervals and class deltas should be read *within* each
experiment. The independently regenerated control difference is too
small to change the qualitative fold decisions, but SG22 and SG23
candidate cells were not compared as one paired run.

## How to use the attached files

The bundle contains this Markdown file, both full `results.json` files,
both `per_class_iou.csv` files, both `report.md` files, and the two locked
configuration JSONs. In each `results.json`, inspect `selection` for
fold votes and the frozen choice; `forced_oof` for every cell's independent
fold result and paired bootstrap; `cells` for VOC/COCO precision, recall,
pixel/mean-image IoU and all class deltas; `diagnostics_fixed_0p45` for
CA-relative and mask-change counts; and `interactions` for SG23.

**Questions for ChatGPT:** Does the 4/5 VOC-selected SG22 MID-P2+R2 gain
justify updating the formal final pipeline, given its positive but
smaller COCO gain than T1+R2? Does SG23's negative temporal×reliability
interaction and its COCO leakage rule out further simple reliability
stacking? Which specific class tradeoffs deserve a visual audit before
paper wording is frozen?
