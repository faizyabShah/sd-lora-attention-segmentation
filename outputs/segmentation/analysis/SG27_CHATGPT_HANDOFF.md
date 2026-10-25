# SG27: completed multi-resolution SA late-fusion study — 26 October 2026

## Setup and completion

SG27 asked whether the **completed** FINE-P2, 16→64, and 32→64 SA score
maps can be combined after propagation. It did not alter the SA graph.
All 46 fixed formulas were checked on four generated VOC/COCO samples;
S11=S7, S12=S8, S15=S7, and S24=S20 collapsed, leaving **42 unique
numeric maps**. Every retained map was evaluated raw and after unchanged
R2 DenseCRF. Base maps S0/S1/S2 matched SG25 on the same observations,
and their full-set final scores match historical SG25 results to displayed
precision.

The locked setup was class-word W attribution, A3 CA (11 selected modules,
calls 0–30), SA calls 10–20 with all eight heads, five fixed VOC folds,
17 thresholds from .300 to .700, and VOC-only threshold/method selection.
VOC-Sim has 490 images; COCO-Cap has 707. All **40/40** GPU class shards
and CPU merge completed with exit code zero. Each of the three completed
base maps was independently spatially min-max normalized before fusion;
every fused map was spatially min-max normalized again before scoring.
COCO used each cell's full-VOC-selected threshold unchanged. Macro mIoU
is the mean of the 20 pooled class IoUs. The protocol and code are in
`docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` Section 106,
`configs/sg27.json`, and `src/segmentation/sg27_fusion.py`.

**Interpretation caveat already identified by Claude:** A3's CA layer
similarity uses a transposed attention orientation relative to the GoCA
formula. SG27 consistently uses the previously measured A3 operator,
so the within-SG27 comparisons are valid for that operator. Results
must not be described as a corrected GoCA weighting experiment.

## Main result

The registered multi-candidate selector kept **S0 FINE-P2** in **5/5
VOC folds** both raw and after R2. No SG27 fusion passed the registered
VOC promotion rule, so **the working default remains S0+R2**:
**.8242 VOC / .6621 frozen COCO**. This decision is different from
ranking each cell on the full VOC data or selecting on COCO.

| Cell + R2 | Meaning | VOC forced OOF | OOF Δ vs S0 | Pairwise fold wins | Full VOC | VOC threshold | Frozen COCO | COCO Δ vs S0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | FINE-P2 reference | .8228 | — | — | .8242 | .425 | .6621 | — |
| S1 | 16→64 control | .8170 | −.0058 | 1/5 | .8183 | .500 | .6975 | +.0354 |
| S3 | FINE/C16 arithmetic | .8241 | +.0013 | 4/5 | .8241 | .475 | .6881 | +.0260 |
| S5 | FINE/C16 harmonic | .8236 | +.0007 | 3/5 | .8247 | .475 | .6882 | +.0261 |
| S8 | FINE/C16 minimum | .8248 | +.0020 | 4/5 | **.8258** | .425 | .6743 | +.0122 |
| S9 | FINE×C16 product | .8123 | −.0106 | 1/5 | .8123 | **.300** | **.7054** | **+.0433** |
| S18 | FINE/C32 harmonic | .8247 | +.0019 | 5/5 | .8247 | .450 | .6732 | +.0111 |
| S23 | FINE/C32 soft OR | .8245 | +.0016 | 4/5 | .8257 | **.700** | .6750 | +.0129 |
| S30 | C16/C32 maximum | .8170 | −.0058 | 1/5 | .8194 | .525 | .6986 | +.0365 |
| S38 | Three-way minimum | **.8256** | **+.0028** | 5/5 | .8256 | .425 | .6760 | +.0139 |
| S40 | Three-way median | .8230 | +.0002 | 3/5 | .8239 | .450 | .6746 | +.0125 |
| S41 | Three-way top-two mean | .8211 | −.0018 | 2/5 | .8224 | .500 | .6913 | +.0292 |
| S42 | Three-way bottom-two mean | .8229 | +.0000 | 4/5 | .8247 | .450 | .6808 | +.0187 |

The **best full-VOC mIoU** is S8+R2 at .8258 (+.0016), whereas the
**highest forced OOF** is S38+R2 at .8256 (+.0028). Neither clears the
registered promotion gate. The **highest frozen COCO result** is S9+R2
at .7054: +.0433 over FINE and +.0079 over historical 16→64+R2 (.6975).
It is a *transfer-only observation*, not a method selected on VOC; on
VOC it loses .0120 full mIoU and .0106 forced OOF. S5 has the best
descriptive mean rank across VOC and COCO, but that rank was never a
selection criterion.

## Precision, recall, and paired-image evidence

| Cell + R2 | VOC precision / recall | COCO precision / recall | COCO paired mean-image IoU Δ vs S0 (20k bootstrap 95% CI) | COCO better / worse / tied images |
|---|---:|---:|---:|---:|
| S0 | .883 / .916 | .703 / .889 | reference | — |
| S1 | .882 / .904 | .751 / .880 | +.0272 [+.0200,+.0346] | 426 / 278 / 3 |
| S3 | .890 / .905 | .741 / .879 | +.0205 [+.0154,+.0258] | 479 / 223 / 5 |
| S8 | .885 / .915 | .724 / .882 | +.0104 [+.0061,+.0149] | 373 / 218 / 116 |
| S9 | .920 / .853 | **.787 / .842** | **+.0347 [+.0261,+.0436]** | 454 / 252 / 1 |
| S38 | .886 / .914 | .726 / .881 | +.0115 [+.0072,+.0161] | 434 / 226 / 47 |
| S40 | .882 / .915 | .719 / .888 | +.0093 [+.0059,+.0128] | 429 / 274 / 4 |

These intervals are for **paired mean-image IoU**, not a confidence
interval for 20-class macro mIoU. S9's COCO increase is mostly a
precision/recall trade: precision rises from .703 to .787 while recall
falls from .889 to .842. Its VOC precision rises, but recall falls much
more and class mIoU drops.

## Class-level trade-offs

At final VOC-selected thresholds, S9's largest COCO gains versus S0
are aeroplane +.121, car +.120, potted plant +.100, chair +.099,
bicycle +.087, motorbike +.080, boat +.076, and sofa +.077. COCO
dining table loses **−.130**; cow −.025 and person −.014. On VOC,
S9 gains train +.146 but loses sheep **−.157**, cat −.101, cow −.095,
person −.087, and potted plant −.078. This is why its COCO result
cannot be treated as a successful VOC-selected system.

S8 is much more conservative: VOC car +.015, train +.011 and person
+.008, with sheep −.016 and potted plant −.010. On COCO it gains car
+.051, potted plant +.033 and aeroplane +.031, but dining table
loses −.046. S38 behaves similarly to S8. The full 20-class table for
every cell is in `sg27/per_class_iou.csv`; the compact tables below
include all 42 final cells.

## Why these operators behave differently

At the **fixed .45 raw-map diagnostic threshold** (before R2, not each
cell's VOC-selected threshold), S8 minimum removes a net 1.36M COCO
false-positive pixels while losing 257k net true-positive pixels; on
VOC it removes 96k net FP but loses 53k net TP. S38 three-way minimum
removes 1.50M net COCO FP and loses 291k net TP. This supports a
precision-oriented cross-scale consensus mechanism. The detailed
gained/removed TP, FP, distant FP, and separate-component FP counts are
in `sg27/mechanism.csv`.

S9 product is much more suppressive at the same fixed .45 diagnostic
threshold (COCO: 9.81M FP removed, 7.02M TP lost); **those counts must
not be used to explain its final .300-threshold+R2 mask directly**.
They show strong calibration compression. S9's selected threshold is the
*lower edge* of the locked .300–.700 grid, so its absolute score is
sensitive to the permitted threshold range.

At fixed .5 on the 64×64 normalized base maps, the FINE/C16
high/low bins give this GT-positive fraction:

| FINE,C16 | VOC | COCO |
|---|---:|---:|
| both low (00) | .054 | .036 |
| only C16 high (01) | .473 | .399 |
| only FINE high (10) | .591 | .222 |
| both high (11) | .897 | .757 |

The COCO-only-FINE bin has low GT purity, so demanding agreement or
using coarse confidence can remove leakage there. On VOC the only-FINE
bin is much purer; suppressing it sacrifices useful foreground.
These are pooled 64×64 pixels and descriptive GT analysis, not an
inference-time rule or class-balanced estimate. All two- and
three-scale bin counts are in `sg27/disagreement.csv`.

COCO caption-presence diagnostics were post hoc: 465 queried class words
occur in the generation caption and 242 do not. S9 improves over S0 by
+.0346 macro mIoU in the present subset and +.0511 in the absent
subset, using the same VOC threshold. S1's corresponding gains are
+.0239 and +.0480. The subset macro mIoUs are not additive and do not
need to average to the full-dataset macro mIoU, because each subset
re-pools class counts. No COCO result influenced selection.

**Important mathematical limitation:** geometric mean S4 and product
S9 are strictly monotone transforms of the same pixelwise product
before final min-max normalization. With a continuous free threshold,
they can represent the same binary masks. Their different observed
scores arise from the fixed discrete threshold grid and its interaction
with R2. The four-image duplicate scan removed numerically equal maps,
but it did not collapse monotone-equivalent score rankings. Therefore
S9's improvement should be called a strong *calibration/thresholded
consensus* observation, not proof of a new spatial ordering. S23 also
selects the upper threshold-grid endpoint (.700); endpoint-selected
cells deserve the same caution.

## Files and questions for follow-up

The fully auditable SG27 package is
`outputs/segmentation/controlled_attention_cv/sg27/`:
`results.json` (fold screens, full metrics, intervals, class deltas,
caption split), `report.md` (all 42 raw/final tables),
`per_class_iou.csv`, `mechanism.csv`, `disagreement.csv`, the
four-sample `duplicate_scan.json`, and 40 VOC + 40 COCO compact
metric/diagnostic shard files. No images, masks, or attention matrices
are in those shards. The runner/manifest are `scripts/run_sg27.py` and
`configs/sg27_unique.json`.

Questions worth discussing: whether to keep FINE+R2 as the formal
reference (the locked protocol says yes); whether the COCO-only
precision gain of product consensus warrants a new preregistered VOC
threshold-calibration study; and whether the VOC/COCO difference in
FINE-only pixel purity is an interpretable transfer effect. The A3
orientation audit should be resolved before writing a paper claim about
GoCA-style CA weighting.

## Complete locked cell tables

The following two tables reproduce all unique cells from the merged
report. Forced per-cell OOF, pairwise fold wins, and the **separate**
multi-candidate selector must remain distinct.

## Raw

| Fusion | VOC forced OOF | Δ vs FINE | Fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.8097 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| S1 | 0.8020 | -0.0077 | 0/5 | 0.8032 | 0.525 | 0.6869 |
| S2 | 0.8089 | -0.0008 | 1/5 | 0.8106 | 0.475 | 0.6695 |
| S3 | 0.8091 | -0.0006 | 2/5 | 0.8103 | 0.500 | 0.6821 |
| S4 | 0.8088 | -0.0009 | 2/5 | 0.8106 | 0.500 | 0.6820 |
| S5 | 0.8091 | -0.0006 | 2/5 | 0.8108 | 0.500 | 0.6823 |
| S6 | 0.8088 | -0.0010 | 2/5 | 0.8099 | 0.500 | 0.6820 |
| S7 | 0.8037 | -0.0061 | 0/5 | 0.8050 | 0.550 | 0.6881 |
| S8 | 0.8125 | +0.0028 | 5/5 | 0.8125 | 0.450 | 0.6706 |
| S9 | 0.8022 | -0.0075 | 1/5 | 0.8022 | 0.300 | 0.6927 |
| S10 | 0.8096 | -0.0002 | 3/5 | 0.8096 | 0.700 | 0.6622 |
| S13 | 0.8091 | -0.0006 | 2/5 | 0.8091 | 0.425 | 0.6525 |
| S14 | 0.8119 | +0.0022 | 4/5 | 0.8119 | 0.450 | 0.6694 |
| S16 | 0.8099 | +0.0002 | 2/5 | 0.8116 | 0.475 | 0.6705 |
| S17 | 0.8100 | +0.0003 | 2/5 | 0.8116 | 0.475 | 0.6703 |
| S18 | 0.8101 | +0.0003 | 2/5 | 0.8117 | 0.475 | 0.6705 |
| S19 | 0.8099 | +0.0001 | 2/5 | 0.8116 | 0.475 | 0.6704 |
| S20 | 0.8094 | -0.0004 | 2/5 | 0.8109 | 0.475 | 0.6641 |
| S21 | 0.8117 | +0.0019 | 5/5 | 0.8117 | 0.450 | 0.6655 |
| S22 | 0.7967 | -0.0130 | 0/5 | 0.7967 | 0.300 | 0.6875 |
| S23 | 0.8143 | +0.0045 | 4/5 | 0.8143 | 0.700 | 0.6626 |
| S25 | 0.8107 | +0.0010 | 4/5 | 0.8107 | 0.450 | 0.6626 |
| S26 | 0.8106 | +0.0008 | 4/5 | 0.8115 | 0.450 | 0.6650 |
| S27 | 0.8074 | -0.0023 | 2/5 | 0.8089 | 0.500 | 0.6810 |
| S28 | 0.8079 | -0.0018 | 2/5 | 0.8092 | 0.500 | 0.6812 |
| S29 | 0.8081 | -0.0017 | 2/5 | 0.8094 | 0.500 | 0.6814 |
| S30 | 0.8048 | -0.0050 | 1/5 | 0.8057 | 0.550 | 0.6893 |
| S31 | 0.8071 | -0.0026 | 1/5 | 0.8108 | 0.475 | 0.6783 |
| S32 | 0.8070 | -0.0027 | 2/5 | 0.8070 | 0.700 | 0.6591 |
| S33 | 0.8090 | -0.0007 | 2/5 | 0.8108 | 0.500 | 0.6817 |
| S34 | 0.8093 | -0.0004 | 2/5 | 0.8110 | 0.500 | 0.6817 |
| S35 | 0.8094 | -0.0003 | 2/5 | 0.8112 | 0.475 | 0.6720 |
| S36 | 0.8094 | -0.0004 | 3/5 | 0.8106 | 0.500 | 0.6815 |
| S37 | 0.8050 | -0.0047 | 1/5 | 0.8060 | 0.550 | 0.6867 |
| S38 | 0.8112 | +0.0015 | 4/5 | 0.8124 | 0.450 | 0.6718 |
| S39 | 0.7672 | -0.0425 | 0/5 | 0.7672 | 0.700 | 0.5749 |
| S40 | 0.8094 | -0.0003 | 1/5 | 0.8110 | 0.475 | 0.6718 |
| S41 | 0.8074 | -0.0023 | 1/5 | 0.8090 | 0.525 | 0.6852 |
| S42 | 0.8103 | +0.0006 | 3/5 | 0.8119 | 0.475 | 0.6773 |
| S43 | 0.8080 | -0.0017 | 0/5 | 0.8095 | 0.425 | 0.6510 |
| S44 | 0.8083 | -0.0015 | 0/5 | 0.8098 | 0.425 | 0.6520 |
| S45 | 0.8074 | -0.0023 | 2/5 | 0.8089 | 0.500 | 0.6746 |

## After R2

| Fusion | VOC forced OOF | Δ vs FINE | Fold wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 | 0.425 | 0.6621 |
| S1+R2 | 0.8170 | -0.0058 | 1/5 | 0.8183 | 0.500 | 0.6975 |
| S2+R2 | 0.8217 | -0.0012 | 3/5 | 0.8228 | 0.450 | 0.6717 |
| S3+R2 | 0.8241 | +0.0013 | 4/5 | 0.8241 | 0.475 | 0.6881 |
| S4+R2 | 0.8233 | +0.0005 | 3/5 | 0.8245 | 0.475 | 0.6879 |
| S5+R2 | 0.8236 | +0.0007 | 3/5 | 0.8247 | 0.475 | 0.6882 |
| S6+R2 | 0.8236 | +0.0008 | 3/5 | 0.8236 | 0.475 | 0.6881 |
| S7+R2 | 0.8162 | -0.0067 | 1/5 | 0.8189 | 0.500 | 0.6885 |
| S8+R2 | 0.8248 | +0.0020 | 4/5 | 0.8258 | 0.425 | 0.6743 |
| S9+R2 | 0.8123 | -0.0106 | 1/5 | 0.8123 | 0.300 | 0.7054 |
| S10+R2 | 0.8239 | +0.0011 | 3/5 | 0.8239 | 0.700 | 0.6786 |
| S13+R2 | 0.8211 | -0.0018 | 2/5 | 0.8211 | 0.400 | 0.6538 |
| S14+R2 | 0.8242 | +0.0014 | 4/5 | 0.8253 | 0.425 | 0.6730 |
| S16+R2 | 0.8246 | +0.0018 | 4/5 | 0.8246 | 0.450 | 0.6732 |
| S17+R2 | 0.8247 | +0.0018 | 4/5 | 0.8247 | 0.450 | 0.6730 |
| S18+R2 | 0.8247 | +0.0019 | 5/5 | 0.8247 | 0.450 | 0.6732 |
| S19+R2 | 0.8246 | +0.0018 | 4/5 | 0.8246 | 0.450 | 0.6731 |
| S20+R2 | 0.8220 | -0.0008 | 3/5 | 0.8235 | 0.475 | 0.6772 |
| S21+R2 | 0.8237 | +0.0009 | 4/5 | 0.8246 | 0.425 | 0.6682 |
| S22+R2 | 0.8037 | -0.0191 | 0/5 | 0.8037 | 0.300 | 0.6971 |
| S23+R2 | 0.8245 | +0.0016 | 4/5 | 0.8257 | 0.700 | 0.6750 |
| S25+R2 | 0.8233 | +0.0004 | 3/5 | 0.8233 | 0.425 | 0.6657 |
| S26+R2 | 0.8236 | +0.0008 | 3/5 | 0.8247 | 0.425 | 0.6673 |
| S27+R2 | 0.8211 | -0.0017 | 2/5 | 0.8225 | 0.475 | 0.6869 |
| S28+R2 | 0.8213 | -0.0015 | 2/5 | 0.8228 | 0.475 | 0.6871 |
| S29+R2 | 0.8199 | -0.0029 | 2/5 | 0.8229 | 0.475 | 0.6872 |
| S30+R2 | 0.8170 | -0.0058 | 1/5 | 0.8194 | 0.525 | 0.6986 |
| S31+R2 | 0.8243 | +0.0014 | 4/5 | 0.8243 | 0.450 | 0.6832 |
| S32+R2 | 0.8222 | -0.0006 | 2/5 | 0.8222 | 0.700 | 0.6764 |
| S33+R2 | 0.8225 | -0.0003 | 3/5 | 0.8241 | 0.475 | 0.6871 |
| S34+R2 | 0.8228 | -0.0000 | 3/5 | 0.8243 | 0.475 | 0.6871 |
| S35+R2 | 0.8230 | +0.0002 | 3/5 | 0.8244 | 0.475 | 0.6872 |
| S36+R2 | 0.8228 | -0.0001 | 3/5 | 0.8238 | 0.475 | 0.6869 |
| S37+R2 | 0.8171 | -0.0057 | 1/5 | 0.8195 | 0.525 | 0.6951 |
| S38+R2 | 0.8256 | +0.0028 | 5/5 | 0.8256 | 0.425 | 0.6760 |
| S39+R2 | 0.7883 | -0.0345 | 0/5 | 0.7883 | 0.700 | 0.5937 |
| S40+R2 | 0.8230 | +0.0002 | 3/5 | 0.8239 | 0.450 | 0.6746 |
| S41+R2 | 0.8211 | -0.0018 | 2/5 | 0.8224 | 0.500 | 0.6913 |
| S42+R2 | 0.8229 | +0.0000 | 4/5 | 0.8247 | 0.450 | 0.6808 |
| S43+R2 | 0.8209 | -0.0019 | 0/5 | 0.8222 | 0.425 | 0.6641 |
| S44+R2 | 0.8207 | -0.0021 | 0/5 | 0.8223 | 0.425 | 0.6651 |
| S45+R2 | 0.8209 | -0.0019 | 2/5 | 0.8225 | 0.475 | 0.6789 |


## Full 20-class audit for key final cells

All entries are IoU; candidate columns are changes from S0+R2. VOC and COCO use the same selected VOC threshold for each cell.

### VOC-Sim

| Class | S0+R2 IoU | S1 Δ | S8 Δ | S9 Δ | S38 Δ |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.884 | -0.014 | -0.008 | +0.002 | -0.008 |
| bicycle | 0.762 | -0.001 | +0.002 | +0.016 | +0.001 |
| bird | 0.913 | +0.007 | +0.002 | +0.010 | +0.002 |
| boat | 0.839 | +0.012 | +0.002 | +0.028 | +0.002 |
| bottle | 0.853 | +0.031 | +0.009 | +0.045 | +0.009 |
| bus | 0.897 | +0.022 | +0.005 | +0.033 | +0.006 |
| car | 0.846 | +0.007 | +0.015 | +0.001 | +0.016 |
| cat | 0.839 | -0.008 | +0.002 | -0.101 | -0.002 |
| chair | 0.779 | -0.027 | +0.001 | +0.023 | +0.002 |
| cow | 0.922 | -0.049 | +0.000 | -0.095 | +0.000 |
| dining-table | 0.456 | -0.017 | +0.003 | +0.055 | +0.004 |
| dog | 0.931 | -0.002 | +0.000 | -0.022 | -0.001 |
| horse | 0.898 | -0.018 | +0.001 | -0.027 | +0.000 |
| motorbike | 0.857 | +0.006 | +0.001 | +0.011 | +0.002 |
| person | 0.775 | +0.007 | +0.008 | -0.087 | +0.007 |
| potted-plant | 0.739 | -0.035 | -0.010 | -0.078 | -0.010 |
| sheep | 0.840 | -0.115 | -0.016 | -0.157 | -0.020 |
| sofa | 0.964 | -0.007 | -0.001 | -0.008 | -0.000 |
| train | 0.689 | +0.064 | +0.011 | +0.146 | +0.014 |
| tv | 0.802 | +0.019 | +0.003 | -0.034 | +0.002 |

### Frozen COCO-Cap

| Class | S0+R2 IoU | S1 Δ | S8 Δ | S9 Δ | S38 Δ |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.717 | +0.057 | +0.031 | +0.121 | +0.038 |
| bicycle | 0.570 | +0.045 | +0.012 | +0.087 | +0.014 |
| bird | 0.779 | +0.034 | +0.012 | +0.017 | +0.016 |
| boat | 0.563 | +0.038 | +0.015 | +0.076 | +0.018 |
| bottle | 0.496 | +0.026 | +0.007 | +0.040 | +0.010 |
| bus | 0.903 | +0.018 | +0.008 | +0.032 | +0.008 |
| car | 0.633 | +0.134 | +0.051 | +0.120 | +0.053 |
| cat | 0.855 | +0.008 | +0.005 | +0.004 | +0.005 |
| chair | 0.446 | +0.041 | +0.022 | +0.099 | +0.021 |
| cow | 0.855 | +0.001 | +0.006 | -0.025 | +0.007 |
| dining table | 0.470 | -0.080 | -0.046 | -0.130 | -0.047 |
| dog | 0.756 | +0.001 | -0.000 | -0.000 | -0.000 |
| horse | 0.674 | +0.041 | +0.009 | +0.054 | +0.010 |
| motorbike | 0.625 | +0.054 | +0.028 | +0.080 | +0.029 |
| person | 0.266 | +0.016 | -0.004 | -0.014 | -0.004 |
| potted plant | 0.697 | +0.083 | +0.033 | +0.100 | +0.041 |
| sheep | 0.874 | +0.017 | +0.005 | -0.000 | +0.003 |
| sofa | 0.662 | +0.043 | +0.027 | +0.077 | +0.031 |
| train | 0.687 | +0.071 | +0.015 | +0.057 | +0.016 |
| tv | 0.715 | +0.060 | +0.009 | +0.072 | +0.009 |
