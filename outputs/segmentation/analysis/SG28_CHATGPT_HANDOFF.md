# SG28 — ChatGPT handoff (complete)

**Scope.** 490 VOC-Sim and 707 COCO-Cap images; all 20 classes in each dataset. Five fixed VOC folds select method and threshold; COCO is frozen transfer only. All 40 class shards per dataset (20 metric + 20 diagnostics) passed coverage checks. The GPU array completed; the first CPU merger hit a tuple-unpacking typo, which was fixed. The completed result below was merged from the saved shards, with no generation rerun.

## Exact experimental setup

- **P0** is the existing A3 class-word CA seed (11 selected CA modules, calls 0–30, all heads), 64×64 SA module 15, calls 10–20, all eight heads, two observation-wise propagation passes. Its final variant adds the unchanged OVAM DenseCRF **R2**. A3 has the previously documented transposed-affinity implementation caveat; SG28 keeps that historical baseline exactly.
- **P1:** weighted-centering Procrustes rotation of SA keys toward queries before the two passes. **P2:** weighted-centered Q/K only, no rotation. **P3:** uniform-centering Procrustes. These are PEARL-inspired adaptations to SD1.5, not a reproduction of PEARL's CLIP system.
- **B1:** plain four-neighbor Laplacian solve on the normalized FINE-P2 map. **B2:** the same with image-boundary edge weights. **C1/C2/C3:** plain graph plus FINE–C16/FINE–C32/three-scale agreement as data trust. **D1/D2/D3:** the corresponding confidence terms plus image-boundary edges. Fixed 64×64 grid, tau=1, kappa=5, CG tolerance 1e-8. No PA+Laplacian combination was tested.
- Each cell was evaluated as a raw map and after unchanged R2. Candidate thresholds come from the fixed 17-point VOC grid. The reported forced OOF score means the named candidate is evaluated on all five held-out folds even if the registered selector rejects it.

## Main decision

**No SG28 component is promoted.** In the broad, PA-only, plain/boundary, confidence-only, and confidence-plus-boundary screens, the registered selector chooses **P0 in 5/5 folds** for raw maps and **P0+R2 in 5/5 folds** for final masks. Pairwise fold wins in the table below are not the registered selector vote. B2 raw wins 5/5 pairwise folds but its forced OOF gain is only +0.0043, below the +0.005 conditional promotion floor; after R2 its forced OOF gain is +0.0001 with only 1/5 pairwise wins.

## Every variant

| Cell | VOC forced OOF | Δ OOF | Pairwise wins | Full VOC | VOC threshold | Frozen COCO | VOC P/R | COCO P/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.8097 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 | 0.880/0.904 | 0.708/0.882 |
| P1 | 0.8129 | +0.0032 | 4/5 | 0.8129 | 0.425 | 0.6443 | 0.886/0.900 | 0.690/0.878 |
| P2 | 0.7961 | -0.0136 | 0/5 | 0.7973 | 0.475 | 0.6555 | 0.882/0.888 | 0.720/0.861 |
| P3 | 0.8129 | +0.0032 | 4/5 | 0.8129 | 0.425 | 0.6444 | 0.886/0.900 | 0.690/0.878 |
| B1 | 0.8031 | -0.0066 | 0/5 | 0.8031 | 0.475 | 0.6537 | 0.881/0.891 | 0.709/0.872 |
| B2 | 0.8140 | +0.0043 | 5/5 | 0.8140 | 0.475 | 0.6654 | 0.892/0.891 | 0.721/0.872 |
| C1 | 0.8021 | -0.0077 | 0/5 | 0.8021 | 0.475 | 0.6530 | 0.880/0.891 | 0.708/0.872 |
| C2 | 0.8027 | -0.0070 | 0/5 | 0.8027 | 0.475 | 0.6534 | 0.880/0.891 | 0.708/0.872 |
| C3 | 0.8024 | -0.0073 | 0/5 | 0.8024 | 0.475 | 0.6532 | 0.880/0.891 | 0.708/0.872 |
| D1 | 0.8139 | +0.0042 | 5/5 | 0.8139 | 0.475 | 0.6652 | 0.892/0.891 | 0.721/0.872 |
| D2 | 0.8140 | +0.0042 | 5/5 | 0.8140 | 0.475 | 0.6653 | 0.892/0.891 | 0.721/0.872 |
| D3 | 0.8140 | +0.0042 | 5/5 | 0.8140 | 0.475 | 0.6653 | 0.892/0.891 | 0.721/0.872 |
| **Final cells below** | | | | | | | | |
| P0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 | 0.425 | 0.6621 | 0.883/0.916 | 0.703/0.889 |
| P1+R2 | 0.8220 | -0.0008 | 2/5 | 0.8234 | 0.425 | 0.6543 | 0.901/0.896 | 0.703/0.871 |
| P2+R2 | 0.8155 | -0.0074 | 0/5 | 0.8167 | 0.450 | 0.6640 | 0.891/0.900 | 0.721/0.869 |
| P3+R2 | 0.8220 | -0.0008 | 2/5 | 0.8234 | 0.425 | 0.6543 | 0.901/0.896 | 0.703/0.871 |
| B1+R2 | 0.8138 | -0.0090 | 0/5 | 0.8155 | 0.450 | 0.6585 | 0.881/0.906 | 0.705/0.882 |
| B2+R2 | 0.8230 | +0.0001 | 1/5 | 0.8230 | 0.450 | 0.6661 | 0.890/0.905 | 0.714/0.881 |
| C1+R2 | 0.8129 | -0.0099 | 0/5 | 0.8146 | 0.450 | 0.6580 | 0.880/0.906 | 0.704/0.882 |
| C2+R2 | 0.8135 | -0.0093 | 0/5 | 0.8152 | 0.450 | 0.6583 | 0.881/0.906 | 0.704/0.882 |
| C3+R2 | 0.8132 | -0.0096 | 0/5 | 0.8149 | 0.450 | 0.6581 | 0.880/0.906 | 0.704/0.882 |
| D1+R2 | 0.8227 | -0.0002 | 1/5 | 0.8227 | 0.450 | 0.6658 | 0.889/0.905 | 0.713/0.881 |
| D2+R2 | 0.8228 | +0.0000 | 1/5 | 0.8228 | 0.450 | 0.6659 | 0.890/0.905 | 0.713/0.881 |
| D3+R2 | 0.8228 | -0.0001 | 1/5 | 0.8228 | 0.450 | 0.6659 | 0.890/0.905 | 0.713/0.881 |

**Read the three VOC columns separately.** Forced OOF is held-out performance for a named candidate with fold-specific VOC thresholds. Full VOC is the in-sample score at one threshold selected on all VOC; it is the threshold frozen for COCO. Thus B2+R2 is +0.0001 forced OOF but -0.0013 on full VOC against P0+R2. Do not mix these deltas.

## Mechanism evidence

### PA: alignment changes without transferable segmentation gain

| Dataset/cell | Q/K RMS error before→after | SA entropy before→after | Effective neighbors before→after | Pearson / Spearman: error reduction vs raw image-IoU gain |
|---|---:|---:|---:|---:|
| VOC P1 | 0.678→0.513 | 6.882→6.817 | 1377→1366 | -0.118 / -0.107 |
| VOC P2 | 1.129→0.678 | 6.882→6.889 | 1377→1522 | -0.170 / -0.203 |
| VOC P3 | 0.678→0.513 | 6.882→6.818 | 1377→1366 | -0.118 / -0.107 |
| COCO P1 | 0.687→0.522 | 6.799→6.724 | 1261→1242 | -0.063 / -0.023 |
| COCO P2 | 1.125→0.687 | 6.799→6.804 | 1261→1382 | -0.147 / -0.238 |
| COCO P3 | 0.687→0.522 | 6.799→6.724 | 1261→1242 | -0.063 / -0.023 |

P1/P3 are nearly identical in mIoU and diagnostics; query-norm versus uniform centering contributes little here. Both gain +0.0032 forced VOC OOF raw but lose about 0.016 COCO mIoU raw, and both are below P0 after R2. Centering alone (P2) loses 0.0136 VOC OOF raw and 0.0074 after R2. The alignment-error reduction does **not** positively correlate with segmentation improvement (P1 Pearson -0.118 VOC, -0.063 COCO). For P2, the logged before/after errors use different centering conventions, so its large numeric reduction is **not** evidence of a successful Procrustes correction.

### Laplacian: boundaries help more than confidence

- B1 plain smoothing loses 0.0066 VOC forced OOF raw and 0.0090 after R2. B2 boundary-aware smoothing gains 0.0043 VOC forced OOF raw and transfers from 0.6599 to 0.6654 COCO raw. This is mainly a precision–recall shift: on COCO raw, precision 0.708→0.721 while recall 0.882→0.872.
- After R2, B2 goes from 0.8228 to 0.8230 VOC forced OOF (+0.0001 after rounding) and 0.6621 to 0.6661 COCO. Its full-VOC score is **lower** than P0+R2 (0.8230 vs 0.8242). B2 is interesting as an independent boundary-aware refinement but does not replace the final pipeline.
- D1/D2/D3 are almost indistinguishable from B2. C1/C2/C3 (confidence without boundaries) all lose about 0.007–0.008 VOC forced OOF raw and about 0.009–0.010 after R2. This does not support the proposed cross-scale confidence mechanism at these fixed settings.
- The Laplacian solves converged: mean 23–26 CG iterations and maximum recorded relative residual about 1.0×10⁻⁸. No solver-accuracy failure was reported.

### Why confidence had little effect

| Dataset | Trust source | Low (<0.5) | Medium (0.5–0.75) | High (≥0.75) |
|---|---|---:|---:|---:|
| VOC | C16 | 0.0000% | 0.1362% | 99.8638% |
| VOC | C32 | 0.0000% | 0.0001% | 99.9999% |
| VOC | three_scale | 0.0000% | 0.0020% | 99.9980% |
| COCO | C16 | 0.0100% | 0.5951% | 99.3949% |
| COCO | C32 | 0.0000% | 0.0085% | 99.9915% |
| COCO | three_scale | 0.0000% | 0.0470% | 99.9530% |

Nearly every pixel is in the high-trust bin: >99.39% even for C16 on COCO, and >99.95% for the other COCO confidence variants. The trust coefficient therefore stays near 1 almost everywhere, so the confidence cells behave much like their plain/boundary controls. The tiny COCO C16 low-trust group (290 pixels) is 99.7% GT-positive, which even runs counter to treating disagreement as predominantly erroneous background. This is a post-hoc diagnostic, not a tuned rule.

### Pixel changes and class effects

At a **fixed raw-map threshold of 0.45** (different from the selected thresholds in the score table), B2 versus P0 on VOC gains 346,218 TP, loses 175,281 TP, gains 507,069 FP and removes 362,257 FP. On COCO it gains 313,115 TP, loses 188,539 TP, gains 1,189,248 FP and removes 464,620 FP. The operation is not simply removing leakage; it shifts both foreground coverage and calibration. See `results.json` for distant FP (>16/>64 pixels), separate-component FP and every class.

Selected-threshold raw B2 class changes versus P0: VOC train +0.033, bottle +0.031, bird +0.016, but cat -0.022, sheep -0.022, potted plant -0.019; COCO motorbike +0.026, aeroplane +0.020, boat +0.013, but dining table -0.024. After R2, VOC train still gains +0.034 while cow loses -0.031; COCO dining table loses -0.028. Full 20-class IoUs and deltas for **all 24 cells** are in `per_class_iou.csv`.

## Interpretation for deciding what to do next

1. **Q/K alignment (PA): no go as currently adapted.** Geometry metrics improve, but the gains are small on VOC, reverse on COCO and disappear after R2. An improved alignment error is not a surrogate for mask quality.
2. **Boundary-aware Laplacian (B2): mild exploratory signal, not a promotion.** It beats plain smoothing and helps COCO slightly, but VOC forced gain is below the registered floor and final-pipeline gain is negligible. DenseCRF already supplies much of its benefit.
3. **Cross-scale confidence: no go at the fixed definition.** Almost all weights saturate in the high-trust bin, so this formulation does not express the intended reliability variation. Do not claim that cross-scale agreement itself was disproven; only this specific trust map was tested.
4. **No SG29 combination is justified by the registered SG28 results.** PA and Laplacian were not combined in SG28. Any combination would be a new, explicitly exploratory experiment.

## Provenance and files

- GoCA: *Making Training-Free Diffusion Segmentors Scale with the Generative Power* (https://arxiv.org/abs/2603.06178) inspired the historical A3 layer weighting; A3 is **not** an exact GoCA implementation because of the documented transpose orientation.
- PEARL: *Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation* (https://arxiv.org/abs/2603.21528) inspired PA and the confidence-weighted Laplacian, adapted here to one binary class score and SD1.5.
- Full machine-readable results: `outputs/segmentation/controlled_attention_cv/sg28/results.json`.
- Compact tables: `outputs/segmentation/controlled_attention_cv/sg28/report.md`, `per_class_iou.csv`, `pa_diagnostics.csv`.
- Reproducible merge from compact per-image shards: `scripts/finalize_sg28.py`; run specification: `configs/sg28.json`.
