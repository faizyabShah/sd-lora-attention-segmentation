# SG12 + DenseCRF, SG14 and SG15: completed results and interpretation

All three production jobs completed successfully (exit 0:0). Each evaluated 490 VOC-Sim images using the same five folds and 707 frozen-transfer COCO-Cap images. These are class-word W methods; no optimized tokens or training.

| Package | Job | Elapsed |
|---|---:|---:|
| SG12 + fixed DenseCRF | 235021 | 6:00:27 |
| SG14 | 235022 | 4:15:21 |
| SG15 | 235023 | 4:56:52 |

## Decision

**Keep the existing final training-free system: ordinary two-pass SA + fixed DenseCRF.** No new component passes the registered promotion rule. Its VOC OOF mIoU is **81.321%** and frozen COCO mIoU **63.714%**. No COCO score was used to choose a method or threshold.

OOF is the primary comparison. Full-VOC scores use a threshold fitted on all VOC; they can differ from OOF. Percentage-point changes below are differences of macro class mIoU unless explicitly labelled mean-image IoU. Paired bootstrap intervals are for **mean per-image IoU**, not macro class mIoU.

## 1. Frozen SG12 top50 + fixed DenseCRF

SG12 retains 124/248 observations by the registered leave-one-out raw-vector cosine reliability. Compare its one-pass output + the existing fixed CRF with ordinary two-pass + the same CRF. The original threshold grid and folds are unchanged.

| Method | VOC OOF | Full VOC | VOC threshold | COCO transfer |
|---|---:|---:|---:|---:|
| W/REF | 78.728% | 78.861% | 0.425 | 61.716% |
| W/P1 | 79.965% | 80.058% | 0.450 | 62.447% |
| W/SG12/top50 | 79.666% | 79.856% | 0.450 | 62.717% |
| W/P1+R2 | 81.321% | 81.321% | 0.450 | 63.714% |
| W/SG12/top50+R2 | 81.153% | 81.313% | 0.425 | 63.150% |

Locked candidate OOF change versus final reference: **-0.168 pp**. Candidate protocol votes: **0/5**; final reference retained in all folds. Forced OOF mean-image-IoU change -0.209 pp, 95% paired 20k CI [-0.559 pp, +0.180 pp], 178 images improve / 312 worsen. The OOF interval includes zero; do not call this statistically proven inferiority on VOC.

Full-VOC scores almost tie, but the OOF and COCO comparisons do not favor replacement. On COCO, macro mIoU falls by 0.564 pp; the paired mean-image-IoU change is -0.685 pp, CI [-0.931, -0.445] pp, with 228 improved / 477 worsened / 2 tied images.

### What CRF did to SG12

CRF does improve SG12 itself. At individually VOC-selected thresholds, full VOC increases 79.856% -> 81.313%, and COCO 62.717% -> 63.150%. Same-input-threshold diagnostics at .425 separate the CRF effect from threshold changes:

| Dataset | FN -> TP | FP -> TN | TP -> FN | TN -> FP | Boundary errors before -> after |
|---|---:|---:|---:|---:|---|
| voc | 232,075 | 1,325,589 | 381,867 | 273,015 | 2,749,953 -> 2,154,661 |
| coco | 185,784 | 2,256,274 | 478,735 | 508,152 | 3,147,160 -> 2,446,771 |

The dominant correction is FP removal, with fewer boundary errors. This supports useful CRF cleanup rather than a blanket claim that SG12 and CRF interfere. The cleaned SG12 system still fails to improve the stronger two-pass+CRF reference. A CRF-cleaned map can have higher recall but lower precision than that reference.

Class examples at each method’s VOC-selected full-data threshold: VOC person +2.235 pp and cow +3.090 pp, but chair -5.716 pp and train -1.788 pp versus final FTF. COCO dining table +2.607 pp and person +1.296 pp, but sofa -2.475 pp and train -2.092 pp.

Runtime diagnostics charge shared generation/CA in full to each pipeline: approximately 6.65 s/image for the reference versus 6.67 s/image for candidate (VOC/COCO similar). Ranking is about 0.9–1.1 ms/image. These are synchronized component timings plus shared generation, excluding model-loading/report overhead, **not independent end-to-end job benchmarks**. No speed advantage is claimed.

## 2. SG14: selected-observation consensus propagation

Exactly top50 + trim20 per tail + rho=.80. Both directional consensus and trimming use the same 124 observations. No CRF.

| Method | VOC OOF | Full VOC | VOC threshold | COCO transfer |
|---|---:|---:|---:|---:|
| W/REF | 78.728% | 78.861% | 0.425 | 61.716% |
| W/SG11/trim20/rho0p80 | 79.840% | 79.840% | 0.425 | 62.192% |
| W/SG12/top50 | 79.666% | 79.856% | 0.450 | 62.717% |
| W/SG14 | 79.865% | 79.865% | 0.450 | 62.168% |

SG14 improves REF in 5/5 protocol folds. Against either frozen parent, however, it gets **0/5 promotion votes**; the descriptive joint screen retains SG11 in 5/5 folds. Its OOF advantage is only **+0.025 pp versus SG11** and **+0.199 pp versus SG12**. On COCO it is **-0.024 pp versus SG11** and **-0.550 pp versus SG12**. The preregistered complementarity criterion is not met.

Raw held-out fold improvements and protocol votes are distinct: SG14 has 3/5 positive held-out deltas versus SG11 and 4/5 versus SG12, but 0/5 eligible promotion votes versus either. SG12+CRF has 2/5 positive held-out deltas versus FTF despite 0/5 promotion votes.

The paired OOF mean-image interval versus SG11 is [+0.014, +0.737] pp, while the macro class gain remains only +0.025 pp. This distinction matters: a positive mean-image interval is not evidence of a substantial macro class gain. Versus SG12 the OOF mean-image interval spans zero.

### Mechanism

| Dataset | All-observation gate rejects | Selected-observation gate rejects | Gate membership changes |
|---|---:|---:|---:|
| voc | 23.521% | 14.392% | 9.998% |
| coco | 21.888% | 13.860% | 8.974% |

Filtering makes the retained observations more self-consistent, so more pixels pass the gate. It does not establish greater correctness: correlated observations can reinforce the same error. Reliability correlates with agreement on useful changes, but also harmful changes; on COCO the mean per-image correlations with agreement on useful/harmful REF-vs-CA flips are about .589/.532. These are descriptive, GT-only diagnostics, not calibration or causal evidence.

Compared with SG12 on COCO, SG14 improves sheep (+0.737 pp) and chair (+0.657 pp), but loses TV (-5.930 pp), potted plant (-1.706 pp), person (-1.215 pp), and car (-1.018 pp). The stacking changes class trade-offs without reliable transfer gains.

Binary-mask change groups and continuous-map direction groups are separate. Opposite binary changes from the same binary REF are mathematically impossible; their zero count is expected. Continuous opposite changes can occur and are recorded. All groups and GT-labelled gains/removals are in the stage handoff/summary.

Interpretation: no useful complementarity is established by this locked combination. It is unnecessary to keep stacking these two components on the basis of these results. This is not proof that all conceivable combinations are equivalent.

## 3. SG15: SPEC32 refinement of validated priors

A025/A050/A075 project an entire region to foreground when prior occupancy exceeds q=.25/.50/.75. B snaps >=.75 to FG, <=.25 to BG, and preserves original pixels in mixed regions. Every refinement inherits its prior’s training-fold-selected threshold; it cannot retune that threshold. No CRF.

All six family screens (three priors x A/B) retained the unrefined prior in **5/5 folds**. No region refinement passes the promotion rule.

| Prior | Variant | VOC OOF | OOF delta vs own prior | Full VOC | COCO transfer | COCO delta vs own prior |
|---|---|---:|---:|---:|---:|---:|
| W/P1 | A025 | 77.182% | -2.783 pp | 77.234% | 59.010% | -3.437 pp |
| W/P1 | A050 | 78.575% | -1.390 pp | 78.695% | 61.096% | -1.352 pp |
| W/P1 | A075 | 78.121% | -1.844 pp | 78.303% | 61.707% | -0.740 pp |
| W/P1 | B | 79.648% | -0.318 pp | 79.745% | 62.291% | -0.156 pp |
| W/SG11/trim20/rho0p80 | A025 | 77.484% | -2.356 pp | 77.484% | 59.176% | -3.015 pp |
| W/SG11/trim20/rho0p80 | A050 | 78.820% | -1.020 pp | 78.820% | 61.017% | -1.174 pp |
| W/SG11/trim20/rho0p80 | A075 | 78.617% | -1.223 pp | 78.617% | 61.340% | -0.852 pp |
| W/SG11/trim20/rho0p80 | B | 79.893% | +0.053 pp | 79.893% | 62.202% | +0.011 pp |
| W/SG12/top50 | A025 | 77.265% | -2.401 pp | 77.435% | 59.259% | -3.458 pp |
| W/SG12/top50 | A050 | 78.366% | -1.300 pp | 78.627% | 61.560% | -1.157 pp |
| W/SG12/top50 | A075 | 78.017% | -1.649 pp | 78.219% | 61.877% | -0.840 pp |
| W/SG12/top50 | B | 79.490% | -0.175 pp | 79.711% | 62.599% | -0.119 pp |

Only SG11+B has a positive OOF change: **+0.053 pp**, far below the requested ~+0.5 pp target, with a mean-image-IoU CI spanning zero ([-0.100, +0.164] pp). Its COCO macro change is +0.011 pp, while COCO mean-image IoU decreases by 0.084 pp. Treat this as marginal, not successful region refinement.

### Oracle gap and changed pixels

The matching annotation-resolution SPEC32 majority oracle mean-image IoU is **88.588% VOC / 80.523% COCO**. These are not macro class mIoUs. Legacy SG13 nearest64-GT oracle values are also recorded separately (87.519% / 79.460% for these generations). Different oracle definitions/hardware must not be silently mixed.

SG11+B closes only **0.348% of the VOC oracle gap**, and widens the COCO mean-image oracle gap. Its VOC useful/harmful changed-pixel totals are 913,348 / 912,553: almost exactly balanced. P1+B and SG12+B harm more VOC pixels than they fix. Majority projection is much more destructive: q=.25 overfills regions; q=.75 trades FP removal for target erosion. Strong priors do not make all region memberships reliable, especially when a region crosses target boundaries.

Pixel totals, mean-image IoU and macro class IoU weight errors differently. For example some high-q COCO variants fix more pixels in total than they harm, yet reduce both mean-image and macro class IoU. Do not infer metric gains from aggregate pixel-change totals alone.

### Person check

Confidence snapping avoids SG13’s catastrophic person collapse, but this alone does not make the method useful overall.

| Prior | Person VOC before -> B | Person COCO before -> B |
|---|---|---|
| W/P1 | 73.363% -> 73.262% | 23.540% -> 23.244% |
| W/SG11/trim20/rho0p80 | 69.719% -> 69.551% | 22.128% -> 21.843% |
| W/SG12/top50 | 72.259% -> 72.303% | 23.774% -> 23.720% |

Stop this tested hard SPEC32 projection/snapping direction under the registered criterion. Oracle headroom alone does not show it can be recovered from the available semantic priors. The result does not rule out every possible use of regions.

## Validation and handoff contents

All 20 VOC and 20 COCO metric shards per package completed. Streaming audits cover the original parent identities, exact stable top50 ranks, image/GT area conservation, and equality between changed-pixel accounting and observed confusion-count differences. Independent run controls and ranks are checked again during result review. Historical cross-hardware mask differences remain diagnostic; these are new internally controlled runs, not an unlabelled recovery of historical masks.

The result review validated all 120 metric shards and their 120 sidecars. The maximum raw same-generation audit discrepancy is 2.980232238769531e-7. REF confusion-count curves match exactly between all three independent runs on both datasets (maximum difference 0 pixels), and every per-image top50 index list is identical across them. The saved validation record includes raw-shard SHA256 fingerprints.

The accompanying archive contains this analysis, the original locked request and config, and for each stage: CHATGPT_RESULTS.md, summary.json, fold_results.json, per_class_results.csv and per_image_results.csv. It also contains all seven SG15 visual sheets. Summary JSON includes full precision/recall, pixel and mean-image IoU, every class, paired 20k-bootstrap intervals, image win/loss/tie counts, runtime, and mechanism diagnostics. Raw attention/masks are not needed for the ChatGPT handoff.

Claude independently reviews the completed packages before any paper or pipeline change. No new method run is authorized or submitted by this results report.
