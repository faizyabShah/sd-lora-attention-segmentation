# SG33 handoff for ChatGPT

**Question.** Can hierarchical residual self-attention or class-seeded coarse Markov reachability improve the frozen A3 → fine two-pass system?

**Run status.** All 40 class shards and merger completed. Coverage: 490 VOC-Sim images and 707 COCO-Cap images. Five frozen VOC folds select methods and thresholds; COCO uses those thresholds without tuning. Every cell is run raw and with unchanged DenseCRF R2. Results below are mIoU (mean over 20 class IoUs), not mean image IoU.

## Methods

- **S0:** same-generation A3 CA → fine 64×64 self-attention two-pass control; **S0+R2** is the final-stage reference.
- **H0:** ordinary sequential 16→32→64 SA hierarchy, replaying SG31 S2. **H1–H3:** add fixed intermediate-stage residuals with weights equal, 0.2/0.3/0.5, and 1/9–5/9–3/9. **H4:** max of fine, half middle, quarter coarse stage.
- **M1:** replace coarse 16×16 start with class-seeded Markov first-arrival support (relative threshold 0.3, cap 256); **M2:** add iterative proportional fitting (IPF) to approximately balance the 16×16 transition; **M3:** sharpen rows at T=0.65 before IPF.
- **C1–C3:** M1–M3 plus equal residuals from the three hierarchy stages.
- Same A3 seed and same-call 16/32/64 SA at calls 10–20, all eight heads. The earlier backlog cell also named M2 is unrelated.

## Complete result table

Forced OOF is the score for each fixed candidate with its own threshold selected on the other four VOC folds. The fold vote uses the registered multi-candidate selector and can differ from direct pairwise fold wins.

| Cell | VOC forced OOF | Δ OOF vs S0 at same stage | Direct fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.8098 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| S0+R2 | 0.8229 | +0.0000 | 0/5 | 0.8243 | 0.425 | 0.6623 |
| H0 | 0.8061 | -0.0036 | 1/5 | 0.8077 | 0.525 | 0.6782 |
| H0+R2 | 0.8184 | -0.0045 | 1/5 | 0.8196 | 0.525 | 0.6924 |
| H1 | 0.7879 | -0.0218 | 0/5 | 0.7892 | 0.550 | 0.6770 |
| H1+R2 | 0.8083 | -0.0146 | 0/5 | 0.8096 | 0.525 | 0.6961 |
| H2 | 0.7968 | -0.0130 | 0/5 | 0.7983 | 0.550 | 0.6840 |
| H2+R2 | 0.8141 | -0.0088 | 1/5 | 0.8153 | 0.525 | 0.6973 |
| H3 | 0.7926 | -0.0172 | 0/5 | 0.7941 | 0.550 | 0.6822 |
| H3+R2 | 0.8106 | -0.0123 | 0/5 | 0.8126 | 0.525 | 0.6976 |
| H4 | 0.8061 | -0.0037 | 1/5 | 0.8072 | 0.525 | 0.6774 |
| H4+R2 | 0.8180 | -0.0049 | 1/5 | 0.8191 | 0.525 | 0.6916 |
| M1 | 0.7961 | -0.0137 | 0/5 | 0.7967 | 0.700 | 0.5986 |
| M1+R2 | 0.8039 | -0.0191 | 0/5 | 0.8039 | 0.675 | 0.5994 |
| M2 | 0.8216 | +0.0119 | 5/5 | 0.8226 | 0.675 | 0.6463 |
| M2+R2 | 0.8313 | +0.0084 | 3/5 | 0.8313 | 0.675 | 0.6528 |
| M3 | 0.8093 | -0.0004 | 1/5 | 0.8093 | 0.700 | 0.6191 |
| M3+R2 | 0.8167 | -0.0062 | 1/5 | 0.8167 | 0.700 | 0.6250 |
| C1 | 0.7567 | -0.0530 | 0/5 | 0.7567 | 0.700 | 0.5581 |
| C1+R2 | 0.7761 | -0.0468 | 0/5 | 0.7761 | 0.700 | 0.5725 |
| C2 | 0.7824 | -0.0273 | 0/5 | 0.7824 | 0.700 | 0.6186 |
| C2+R2 | 0.8038 | -0.0191 | 0/5 | 0.8038 | 0.700 | 0.6372 |
| C3 | 0.7642 | -0.0456 | 0/5 | 0.7642 | 0.700 | 0.5890 |
| C3+R2 | 0.7840 | -0.0390 | 0/5 | 0.7840 | 0.700 | 0.6056 |

## Registered selection and transfer

- **Raw VOC:** M2 is the frozen winner by **4/5 selector votes** (folds 0, 1, 3, 4); fold 2 retained S0 because no candidate passed the training-fold eligibility rule. The actual selected out-of-fold mixture scores **0.8159** versus S0 **0.8098**, gain **+0.0062**. The fixed M2 candidate wins all five direct pairwise held-out folds and scores **0.8216** forced OOF, gain **+0.0119**. Its full VOC threshold is 0.675.
- **Raw COCO transfer:** M2 scores **0.6463**, versus same-generation S0 **0.6599**, a **-0.0136** loss. Its paired mean-image IoU difference is -0.0103 (20,000-bootstrap 95% interval [-0.0191, -0.0018]). Thus the raw VOC gain does **not** transfer.
- **R2 final stage:** S0+R2 wins **5/5 selector votes**; no SG33 cell is promoted. Direct M2+R2 forced OOF is 0.8313 versus 0.8229 (+0.0084), but it wins only 3/5 direct folds and zero selector votes. On COCO it is 0.6528 versus reference 0.6623 (-0.0095).
- **COCO-only descriptive pattern:** H2+R2 = 0.6973 and H3+R2 = 0.6976 versus S0+R2 = 0.6623. These do not qualify for promotion because their VOC forced OOF losses are −0.0088 and −0.0123, with 1/5 and 0/5 direct fold wins. This is a cross-dataset contrast, not a COCO-selected result.

## Per-class differences (IoU points against same-stage S0)

| Class | VOC M2 raw | COCO M2 raw | VOC M2+R2 | COCO M2+R2 | COCO H3+R2 |
|---|---:|---:|---:|---:|---:|
| aeroplane | +1.17 | -5.26 | +0.09 | -5.56 | +8.09 |
| bicycle | +0.30 | -3.60 | -0.03 | -2.08 | +3.95 |
| bird | -3.99 | +1.72 | -4.91 | +0.62 | +2.25 |
| boat | -1.36 | -0.31 | -0.55 | +1.29 | +2.67 |
| bottle | +1.17 | -4.70 | +1.06 | -4.57 | +2.76 |
| bus | +0.48 | +0.75 | +1.28 | +1.70 | +2.19 |
| car | -0.37 | -0.00 | -0.68 | +2.02 | +14.44 |
| cat | +6.69 | -0.89 | +4.05 | -1.03 | +0.59 |
| chair | +5.74 | -2.51 | +6.33 | -0.34 | +2.84 |
| cow | +6.72 | +3.15 | +2.61 | +2.24 | -0.46 |
| dining-table | -2.62 | +10.62 | -1.16 | +8.52 | -8.27 |
| dog | +1.39 | -3.31 | +0.75 | -3.93 | -0.06 |
| horse | +0.99 | -3.24 | -0.05 | -3.75 | +4.16 |
| motorbike | +0.43 | -2.50 | +0.71 | -1.28 | +4.19 |
| person | +1.95 | +3.28 | +0.82 | +2.51 | +2.27 |
| potted-plant | -3.91 | -3.76 | -5.16 | -3.41 | +9.09 |
| sheep | +9.34 | -3.46 | +7.65 | -3.36 | -0.21 |
| sofa | +0.57 | -1.15 | +0.19 | +1.17 | +4.25 |
| train | -10.24 | -10.95 | -6.78 | -9.62 | +7.60 |
| tv | +8.07 | -1.14 | +7.72 | -0.07 | +8.25 |

## Mechanism and interpretation

- The ordinary hierarchy improves as it refines spatial scale: stage IoU at fixed diagnostic threshold 0.45 is VOC 0.7133 → 0.7640 → 0.7927, COCO 0.5958 → 0.6299 → 0.6346 for 16→32→64. These stage maps are not the selected final masks.
- IPF behaved numerically as specified: M2 mean maximum column error fell from 2.84 to 3.78×10⁻⁶ on VOC and from 2.40 to 3.90×10⁻⁶ on COCO; row error remained near 1.5×10⁻⁷. Mean convergence took about 10.5 iterations on each dataset. This validates balancing, not segmentation benefit.
- First-arrival rank has strong post-hoc coarse GT discrimination (ROC AUC of negative arrival time for M2: VOC 0.844, COCO 0.850), yet the full M2 mask transfers poorly. A coarse diagnostic signal can be informative without producing a robust 64×64 segmentation after further propagation.
- At fixed 0.45, M2 adds both foreground and background relative to S0. On COCO it gains 2.85M TP pixels but introduces 20.23M FP pixels, including 10.09M beyond 64 px from GT. Those counts are diagnostics at a fixed threshold, not changes at each method’s frozen selected threshold.
- H3+R2’s COCO gain comes with higher precision (0.7531 vs 0.7034) and slightly lower recall (0.8774 vs 0.8892). Its VOC loss and COCO dining-table class loss (−8.27 IoU points) argue against adopting it globally on these data.
- Separate-generation historical SG31 replay differed slightly because the images were regenerated. Same-generation S0/H0 controls were validated in the live smoke, and all within-SG33 comparisons use those controls.
- Caption presence is descriptive only: COCO H4+R2 gains +0.0141 mIoU on 465 present-word images and +0.0481 on 242 absent-word images versus S0+R2. M2+R2 loses −0.0103 and −0.0009 respectively. Do not use this split for selection.

## Direct conclusion for next-idea discussion

**No SG33 final-stage candidate is promoted.** Raw M2 shows a real VOC screening signal but fails COCO transfer, while fixed residual hierarchies are much stronger on COCO than VOC. The unresolved question is why hierarchy suppresses COCO leakage yet damages VOC classes such as sheep, chair and dining table; the registered result does not support choosing a COCO-favorable hierarchy after seeing COCO.

## Files and provenance

- `results.json`: all cell metrics, five-fold selection, paired 20k intervals, diagnostics, historical replay and caption split.
- `per_class_iou.csv`: full 20-class table for all 24 cells and both datasets.
- `report.md`: short machine-generated summary.
- `configs/sg33.json`: frozen configuration; `src/segmentation/sg33_hierarchy.py` and `scripts/finalize_sg33.py`: implementation and merger.
- Slurm smoke 238147, array 238148 (40/40 successful), merger 238149 all completed exit 0.
- This is training-free class-word W track with frozen A3 CA and unchanged R2; no optimized tokens, no COCO threshold selection.
