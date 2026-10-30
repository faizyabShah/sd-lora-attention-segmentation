# SG-combined-v1: completed C0–C9 results and analysis for ChatGPT

**Scope/status.** The main ten-cell SG-combined-v1 study completed: all 40 class shards, 490 VOC-Sim and 707 COCO-Cap images, 20 visual sheets, and merger job 238285 exit 0. The separate SG33-M2-source C10a/b/c add-on is still running (production array 238473; merger 238477 pending). This document contains no C10 outcome.

## Exact question and frozen protocol

The hypothesis was that 16→32→64 self-attention should conserve coarse class-conditioned semantic mass during refinement, and that a second fine-scale completion step should add foreground only under an explicit background-aware budget. The A3 class-word CA seed, matched SA calls 10–20, all heads, SD 1.5, five VOC folds, 17 candidate thresholds, and DenseCRF R2 were frozen. Candidate and threshold selection use VOC only; each full-VOC threshold is transferred unchanged to COCO. GT is used only for VOC selection and post-hoc diagnostics. All comparisons within this study use the same generated images and same-generation controls.

### Cells

- **C0:** reference fine 64×64 two-pass self-attention (FINE-P2).
- **C1:** ordinary sequential 16→32→64 hierarchy, no conservation. It is the corrected SG31 S2 same-generation replay.
- **C2/C3:** exact parentwise KL projection at 16→32 only / 32→64 only.
- **C4:** exact KL projection at both transitions.
- **C5:** 0.5-relaxed projection at both transitions.
- **C6:** C4 followed by unconditional second fine pass.
- **C7/C8:** C4 followed by positive-residual completion gated by background-only (1−B) / foreground-background odds P1/(P1+B).
- **C9:** C5 followed by the odds completion rule. Each cell is evaluated raw and with unchanged R2.

## All outcomes

mIoU is the mean of 20 per-class IoUs. Forced OOF evaluates each fixed candidate with thresholds chosen on the other four VOC folds. Direct fold wins are pairwise against the matching C0 stage; the registered multi-candidate selector is reported separately.

| Cell | VOC forced OOF | Δ OOF vs C0 | Direct fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.8097 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| C0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 | 0.425 | 0.6622 |
| C1 | 0.8059 | -0.0038 | 1/5 | 0.8075 | 0.525 | 0.6782 |
| C1+R2 | 0.8181 | -0.0047 | 1/5 | 0.8192 | 0.525 | 0.6923 |
| C2 | 0.8034 | -0.0064 | 0/5 | 0.8049 | 0.525 | 0.6921 |
| C2+R2 | 0.8185 | -0.0043 | 2/5 | 0.8208 | 0.500 | 0.7038 |
| C3 | 0.7894 | -0.0203 | 0/5 | 0.7911 | 0.525 | 0.6843 |
| C3+R2 | 0.8135 | -0.0093 | 0/5 | 0.8147 | 0.500 | 0.7022 |
| C4 | 0.7592 | -0.0505 | 0/5 | 0.7592 | 0.475 | 0.6499 |
| C4+R2 | 0.7917 | -0.0311 | 0/5 | 0.7917 | 0.475 | 0.6851 |
| C5 | 0.7926 | -0.0172 | 0/5 | 0.7940 | 0.525 | 0.6861 |
| C5+R2 | 0.8149 | -0.0079 | 1/5 | 0.8161 | 0.500 | 0.7059 |
| C6 | 0.8035 | -0.0063 | 0/5 | 0.8044 | 0.525 | 0.6933 |
| C6+R2 | 0.8201 | -0.0027 | 2/5 | 0.8221 | 0.500 | 0.7063 |
| C7 | 0.7785 | -0.0313 | 0/5 | 0.7785 | 0.550 | 0.6825 |
| C7+R2 | 0.8047 | -0.0182 | 0/5 | 0.8057 | 0.525 | 0.7054 |
| C8 | 0.7774 | -0.0323 | 0/5 | 0.7774 | 0.550 | 0.6823 |
| C8+R2 | 0.8031 | -0.0197 | 0/5 | 0.8046 | 0.525 | 0.7061 |
| C9 | 0.7951 | -0.0146 | 0/5 | 0.7966 | 0.550 | 0.6887 |
| C9+R2 | 0.8140 | -0.0088 | 1/5 | 0.8164 | 0.525 | 0.7057 |

## Selection and what improved or failed

- **No method promoted.** The registered selector retained C0 in **5/5 VOC folds** for raw maps (OOF 0.8097), and C0+R2 in **5/5 folds** for the final stage (OOF 0.8228). Full VOC references were 0.8114 raw and 0.8242 with R2. This remains the defensible selected pipeline.
- **Exact conservation fails on VOC.** C4 is 0.7592 forced OOF (−0.0505 versus C0), and C4+R2 is 0.7917 (−0.0311). Both transition constraints together are worse than either isolated constraint. Against ordinary C1, C2 loses −0.0025 OOF, C3 −0.0165, and C4 −0.0467. The precise KL operator worked numerically; the segmentation hypothesis did not.
- **Relaxation and second propagation partly rescue C4.** C5 gains +0.0334 forced OOF versus C4, and C6 gains +0.0443 versus C4, each in 5/5 direct folds. Neither reaches C0. The relaxed C9 also beats exact C8 by +0.0177 OOF, confirming that strict conservation is the damaging part of this combination.
- **Budgeting improves addition purity, not segmentation mIoU.** Against C6, C7 loses −0.0250 forced OOF and C8 loses −0.0261 (0/5 fold wins each). C8 is only −0.0011 versus C7: adding P1 to background-only gating does not rescue the final mask. C8+R2 is 0.8031 OOF versus C0+R2 0.8228 (−0.0197).
- **COCO transfer is different but not selectable.** C6+R2 reaches 0.7063 and C8+R2 0.7061, versus same-generation C0+R2 0.6622; C5+R2 is 0.7059. The gain is descriptive because every one of these cells lost VOC OOF. In particular C6+R2 and C8+R2 are nearly tied on COCO, so the budget provides no clear final-IoU advantage there.

## Mechanism diagnostics and causal limits

**Projection validity.** Exact C4 parent-mass errors are at floating-point scale: mean 16→32 error 1.41×10⁻¹⁰ on VOC and 1.36×10⁻¹⁰ on COCO; mean 32→64 error about 3.5×10⁻¹¹. Zero fallback parents occurred. Mean absolute within-parent log-ratio error is about 2.9×10⁻⁸. C5 deliberately relaxes conservation, so its nonzero parent error is expected. A numerical implementation error does not explain C4’s loss.

**Why exact conservation can hurt.** At the fixed 0.45 diagnostic threshold, 2,209 VOC 16×16 parents were high-scoring despite containing <25% target GT. C4 yielded 1.697M FP pixels inside those parents versus 1.203M for ordinary C1; C5 was intermediate at 1.428M. On COCO the corresponding counts were 4,578 parents and 3.739M C4 FP versus 3.404M C1 FP. This is consistent with preserving incorrect coarse mass. It is a post-hoc association, not proof that every such FP was caused by the constraint.

**Fine within-parent preference is weak.** The entropy of a uniform 2×2 child distribution is log(4)=1.3863. The mean SA proposal entropies are 1.3854 (16→32) and 1.3858 (32→64) on VOC, similarly 1.3852/1.3857 on COCO. Thus this particular fine proposal is almost uniform within each parent; projection mostly fixes coarse mass and contributes little discriminative redistribution inside a 2×2 parent. This is a plausible reason strict conservation has limited upside here, not a universal statement about SA.

**Completion efficiency at fixed 0.45.** Relative to the C4 semantic core, the following pooled additions were measured at full image resolution. The fixed-threshold diagnostics do not use each cell’s VOC-selected threshold.

| Dataset | Cell | TP added | FP added | FP >64 px added | TP/FP | TP/FP >64 |
|---|---|---:|---:|---:|---:|---:|
| VOC | C6 | 972,850 | 1,133,208 | 291,143 | 0.858 | 3.341 |
| VOC | C7 | 738,166 | 732,841 | 137,657 | 1.007 | 5.362 |
| VOC | C8 | 696,751 | 665,390 | 121,904 | 1.047 | 5.716 |
| COCO | C6 | 1,940,396 | 3,980,228 | 1,754,225 | 0.488 | 1.106 |
| COCO | C7 | 1,561,948 | 2,626,980 | 1,041,946 | 0.595 | 1.499 |
| COCO | C8 | 1,472,060 | 2,277,297 | 879,283 | 0.646 | 1.674 |

- C8 accepts a mean fraction β=0.754 of positive completion mass on VOC and 0.632 on COCO (C7: 0.843 and 0.782). It suppresses distant FP more efficiently than C6, but also discards many useful TP additions. On VOC, C8 retains 0.697M of C6’s 0.973M TP additions; on COCO, 1.472M of 1.940M. The full mIoU confirms that better TP/FP addition ratio alone is insufficient.
- The odds G bins rank candidate pixels well: among positive-residual pixels, GT-foreground fraction rises from 0.0065 to 0.943 across low→high G quintile bins on VOC, and 0.0093 to 0.836 on COCO. This is a strong descriptive ranking signal. It does not imply the chosen multiplicative budget or later min-max/CRF step improves segmentation. The positive-residual guarantee applies before across-call aggregation and final normalization, not necessarily to final binary masks.

## Class and caption patterns

- On VOC with R2, C8 loses most versus C0 for sheep (−0.138 IoU), chair (−0.118), cow (−0.062), potted plant (−0.050), while train gains +0.104. C4’s exact constraint particularly hurts chair (−0.178) and sheep (−0.096).
- On COCO, C8+R2 gains most for car (+0.130), aeroplane (+0.106), train (+0.098), chair (+0.089), bicycle (+0.085), but hurts dining table (−0.095). C6+R2 gains +0.132 car and +0.100 aeroplane but loses −0.084 dining table. Full 20-class values for all 20 cells are in `per_class_iou.csv`.
- In the descriptive COCO caption split, C8+R2 gains +0.0347 mIoU versus C0+R2 on 465 images where the queried class appears in the caption and +0.0474 on 242 images where it is absent. These are subgroup macro scores with different class composition, not evidence that absence causes a larger gain; no cell was selected on this split.

## Implications for next discussion

1. The original full C8 mechanism did **not** validate as a selected method. Future work should not describe SG-combined-v1 as a performance improvement on VOC.
2. Two separable signals remain worth understanding: coarse-to-fine hierarchy reduces COCO leakage, and odds G ranks proposed completion pixels by correctness. The locked way of using them (exact conservation plus multiplicative budget) damages VOC sufficiently to fail selection.
3. The C10 add-on, currently running, tests whether replacing the A3 coarse start with SG33’s class-seeded M2 Markov source changes this conclusion. Do not infer C10 results from the C0–C9 data.
4. An especially useful falsification check is whether C10b preserves coarse M2 foreground without also preserving coarse-supported background, and whether C10c gains mIoU beyond C10b after R2. The registered VOC rule still governs promotion.

## Files, figures, and caveats

- `results.json`: all cells, forced OOF, selector folds, pairwise 20k bootstrap summaries, transport/completion mechanisms and caption split.
- `per_class_iou.csv`: every per-class value and delta; `report.md`: compact summary; `completed.json`: coverage.
- `configs/sg_combined_v1.json`, `src/segmentation/sg_combined_v1.py`, `scripts/run_sg_combined_v1.py`, `scripts/finalize_sg_combined_v1.py`: locked settings and code.
- `visuals/`: 20 post-hoc VOC sheets, one per class, with image, CA, hierarchy states, completion residual, B/G, C6/C8 maps and GT. They were selected for best or worst C8−C6 examples by class and are diagnostic illustrations, not random or unbiased samples.
- Historical SG31 replay was a separate regeneration and shows small numeric drift. C0 and C1 are same-generation controls within this experiment. The results are for generated OVAM VOC-Sim/COCO-Cap masks with class-word W attribution; no token optimization or trained adapter.

**Question for ChatGPT:** Given these registered outcomes and failure diagnostics, what can we infer about coarse-mass conservation and completion budgeting without choosing methods on COCO? Please wait for the C10 add-on before making a final judgment about the Markov-source variant.
