# SG21 structural self-attention head weighting: results for ChatGPT

## Locked question

Can intrinsic structure choose reliable SA heads **within each diffusion call** and improve SG18/T1? All cells use the W class-word CA seed, SD 1.5, SA module 15, eight heads, calls 10–20, one-pass propagation, and the unchanged VOC-fold/COCO-transfer protocol. The only change is the within-call head weights; calls remain equally weighted. The four preregistered, parameter-free rules are raw radius-1 local mass (LM), its stable ordinal rank (LM-RANK), inverse entropy (ENT), and inverse mean spatial distance (DIST). No CRF, second pass, learned weight, head pruning, COCO tuning, or extra window was used. **T1 is the same-generation reference.** Historical W/P1 is external context only.

## Selection and transfer

| Method | VOC forced OOF mIoU | Δ versus T1 | VOC fold votes | Full VOC mIoU | VOC threshold | Frozen COCO mIoU |
|---|---:|---:|---:|---:|---:|---:|
| SG18/T1 | **0.7939** | — | **5/5** | **0.7959** | 0.425 | 0.6285 |
| SG21-LM | 0.7884 | −0.0055 | 0/5 | 0.7903 | 0.400 | 0.6193 |
| SG21-LM-RANK | 0.7901 | −0.0039 | 0/5 | 0.7918 | 0.400 | 0.6200 |
| SG21-ENT | 0.7872 | −0.0067 | 0/5 | 0.7892 | 0.400 | 0.6201 |
| SG21-DIST | 0.7941 | +0.0002 | 0/5 | 0.7941 | 0.425 | 0.6287 |
| W/P1, historical | 0.7997 | not comparable here | — | 0.8006 | 0.450 | 0.6245 |

**No SG21 candidate was promoted.** T1 won every VOC fold. DIST's forced OOF gain is only **+0.00017**, below the preregistered magnitude guard, while its full-VOC mIoU is **−0.00185** versus T1. Its frozen COCO advantage is **+0.00024**; paired COCO mean-image IoU change **+0.00027**, 95% CI **[−0.00082, +0.00123]**, so it is not convincing transfer evidence. LM, LM-RANK, and ENT lose on both datasets. The experiment is closed under its locked rule; there is no follow-up structural sweep in this package.

## What changed in the masks?

At the fixed **0.45 diagnostic threshold**, the structural rules generally suppress both useful and false CA-relative additions. Percent changes below are versus T1 on the same generated images:

| Dataset | Rule | TP additions | FP additions | FP >16 px | TP/FP ratio (T1 → rule) |
|---|---|---:|---:|---:|---:|
| VOC | LM | −18.2% | −23.2% | −25.3% | 3.49 → 3.71 |
| VOC | LM-RANK | −13.9% | −18.6% | −20.2% | 3.49 → 3.69 |
| VOC | ENT | −18.9% | −23.3% | −25.7% | 3.49 → 3.69 |
| VOC | DIST | −5.2% | −7.4% | −8.8% | 3.49 → 3.57 |
| COCO | LM | −13.8% | −19.2% | −19.8% | 0.94 → 1.00 |
| COCO | LM-RANK | −10.8% | −15.6% | −16.0% | 0.94 → 0.99 |
| COCO | ENT | −13.5% | −19.2% | −19.9% | 0.94 → 1.01 |
| COCO | DIST | −4.0% | −5.9% | −6.1% | 0.94 → 0.96 |

The **higher TP/FP ratio does not translate into higher mIoU** for LM/ENT: they remove too many true-positive additions and reduce object coverage. LM's full-VOC precision falls from **0.8646 to 0.8560** even though its *CA-relative addition ratio* improves. The ratio uses fixed 0.45 masks, while full-VOC precision uses VOC-selected thresholds (0.425 for T1, 0.400 for LM); the denominators and masks therefore differ. Full raw event counts, recall, pixel IoU, distant and separate FP, useful/harmful removals are in `results.json`.

## Head-weight behavior

| Rule | Mean effective heads, VOC / COCO | Most often top-weighted physical heads | Images whose top head changes across calls, VOC / COCO |
|---|---:|---|---:|
| LM | 5.72 / 5.83 | head 0: 59%/61%; head 5: 34%/32% | 60% / 63% |
| LM-RANK | 6.35 / 6.35 | same top-head pattern as LM | 60% / 63% |
| ENT | 5.49 / 5.56 | heads 0 and 6 account for >99% | 52% / 62% |
| DIST | 7.79 / 7.78 | head 6: 88%/95% | 14% / 8% |

LM and ENT are moderately concentrated and mostly favor a small set of physical heads. DIST is almost uniform by effective head count, but head 6 receives the largest weight at most calls. Thus these locked rules do change weights, yet their top-head identity is strongly structured rather than freely varying by image and call. This **suggests** that the earlier within-call correlations may partly reflect persistent head identity; it does not prove that explanation. `weight_diagnostics.json` reports all eight top-head frequencies, per-call effective counts, adjacent-call changes, and mean/SD weight by head.

## Class pattern and verification

- VOC LM improves only **3/20** classes; chair **−0.0349** is its largest loss. ENT also improves **3/20**; chair **−0.0374**. DIST improves **8/20**, but loses person **−0.0122** and cat **−0.0150**.
- COCO LM improves **1/20** classes; person **−0.0204** and potted plant **−0.0237** are large losses. DIST improves **12/20** by mostly small amounts, while losing dining table **−0.0092** and person **−0.0083**.
- The regenerated T1 maps match the saved SG18 T1 maps to mean absolute difference about **2×10⁻⁸ on VOC** and **1.6×10⁻⁸ on COCO**. This supports a same-pipeline comparison. GPU array **235322** produced all 490 VOC and 707 COCO image checkpoints; CPU merger **235326** completed 0:0 and verified coverage.

No weight-quartile outcome table was produced because the preregistered rule required it **only for a promoted candidate**, and none was promoted. The full five-fold decisions, per-class IoUs/deltas, 20,000-resample paired intervals and better/worse/tied counts are in `results.json`; `per_class_iou.csv` provides the 20-class table.

## Interpretation for ChatGPT

The within-call correlation of low entropy/locality with good diagnostic head behavior **did not carry through to improved end-to-end segmentation under these four simple weightings**. LM and ENT reduce leakage but also useful extent. DIST preserves performance approximately, with negligible and uncertain transfer gain. The result rejects **these locked weighting rules**, not the broader possibility that structural information matters. It offers no reason to promote SG21 or tune another structural-weight grid within this protocol.
