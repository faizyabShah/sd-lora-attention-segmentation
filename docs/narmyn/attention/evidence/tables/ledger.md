**Every experiment run so far**

| ID | Experiment | Dataset | Status | Configurations | Best mIoU | Decision / role |
|---|---|---|---|---:|---:|---|
| B1 | CA time windows and resolutions | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 11 | 0.7095 | first look at time and resolution |
| B2 | Single and leave-out CA layers | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 24 | 0.7232 | layer screening |
| B3 | Joint time x resolution | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 21 | 0.7095 | interaction check |
| B4 | Exhaustive CA layer subsets | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 17 | 0.7409 | chose BEST_12 = layers {0, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15} |
| B5 | Threshold x normalization | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 153 | 0.7439 | chose percentile 1-99 and threshold 0.45 |
| B6 | Stage-1 suite (heads, time-layer, fusion, cleanup) | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 590 | 0.7691 | chose the fixed top-4 CA heads per layer |
| B7 | Self-attention screens and progressive SA | VOC-Sim dev subset (99 images, 4 classes; inside the 490) | exploratory, in-sample | 5320 | 0.8355 | first evidence that SA propagation helps |
| B8 | Full streaming ablations | VOC-Sim (490 images, 20 classes) | exploratory, in-sample (choices from B1-B7 frozen) | 6261 | 0.7777 | SA at U3.2 (64x64), 4 passes: 0.7777 |
| B9 | Focused SA follow-up | VOC-Sim (490 images, 20 classes) | exploratory, in-sample | 1632 | 0.7922 | top-4-head seed, 2 passes, t=0.45: 0.7922 |
| B10 | Joint SA head x time x passes x threshold | VOC-Sim (490 images, 20 classes) | exploratory, in-sample | 4680 | 0.7997 | final method: 4 passes, heads {0,1,2,4,5,6}, calls 15-30, t=0.45, 1% cleanup: 0.7997 |
| B11 | Component ablation of the final method | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) | exploratory (COCO untuned) | 18 | 0.7999 | SA propagation is the largest component |
| B12 | COCO-Cap generalization of the final method | COCO-Cap (707 images, 20 classes) | untuned transfer | 6 | 0.6080 | 0.5719 -> 0.6080 |
| B13 | Revised system (1 pass, all 16 CA layers) and its ablation | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) | exploratory; informed by COCO-Cap | 14 | 0.7837 | revised system: VOC 0.7824, COCO 0.6322 |
| B14 | CA call windows with all SA heads | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) | exploratory | 4 | 0.7811 | no gain over all calls |
| B15 | Clean CA-only screens (pre-sync) | VOC-Sim (490 images, 20 classes) | exploratory, in-sample; audited, superseded by A1 | 1573 | 0.6968 | ratified as correct; re-run with folds as A1 |
| B16 | OVAM released masks | VOC-Sim (490 images, 20 classes) | reference (authors' masks) | 3 | 0.8247 | 0.7036 / 0.8247 / 0.6616 |
| B17 | OVAM optimized token and DenseCRF on the final method | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) | exploratory; nothing selected | 32 | 0.8553 | token + DenseCRF: VOC 0.8525, COCO 0.6612 |
| A1 | Controlled CA package B: CA-L, CA-H, CA-A | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 65961 | 0.6923 | CA-L drop modules 10-11; CA-H drop head 1; CA-A keep reference (combined in A3) |
| A2 | Controlled CA package A: CA-T, CA-R (Codex) | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 1279 | 0.6957 | CA-T keep all calls; CA-R keep 8+16+64 (drop 32x32) (combined in A3) |
| A3 | Controlled CA confirmation (CA-C) and the frozen CA | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 6 | 0.6956 | frozen CA: CA-R alone (drop the 32x32 modules), all heads |
| A4 | Controlled SA package B: SA-L, SA-H, operator check | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 65951 | 0.7884 | SA-L only module 15 (U3.2, 64x64); SA-H heads {1, 4, 6}; A x kept (combined in A6) |
| A5 | Controlled SA package A: SA-T, SA-R (Codex) | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 1279 | 0.7710 | SA-T call 30 only; SA-R resolutions 8+64; both calibration-dependent (combined in A6) |
| A6 | Controlled SA confirmation (SA-C, Codex) and the frozen SA | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 13 | 0.7887 | frozen SA: module 15 (U3.2) only, all calls, all heads |
| A7 | Joint CA x SA interaction audit (Codex) | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated | 78 | 0.7887 | sequential pair kept 5/5; Section 11 unchanged |
| A8 | Phase 3 propagation package (Codex): P1, W3, P2, P6 | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated; threshold chosen in the folds | 40 | 0.7991 | two passes mean_i(A_i² c) promoted to P3-C |
| A9 | Phase 3 fusion and calibration package: P4, F3, F4, N2-N4, R2 | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated; threshold chosen in the folds | 12 | 0.8049 | DenseCRF promoted to P3-C; the rest kept at REF |
| A10 | Released baselines scored with OVAM's procedures | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) (each method's own images) | reproduction of published values | 19 | — | every printed OVAM-notebook value reproduced |
| AN1 | Why a second SA pass helps (analysis, nothing selected) | VOC-Sim (490 images, 20 classes); COCO-Cap (707 images, 20 classes) | analysis of committed counts | — | — | second pass fills under-covered objects; the token seed already covers them; COCO-Cap leakage cancels the gain |
| A11 | Token track B1/B1b: OVAM's optimized tokens on the frozen pipeline | VOC-Sim (490 images, 20 classes) | controlled, 5-fold cross-validated; few-shot track (2-3 annotated images per class) | 27 | 0.8507 | token promoted on every pipeline: REF 0.787 -> 0.851 (conditional query) |
