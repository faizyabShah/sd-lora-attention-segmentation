# SG26 coarse-guided fine SA: results and analysis — 24 October 2026

## Setup and validity

SG26 tested whether a coarse self-attention affinity can **gate each fine
64×64 SA edge** without replacing fine propagation. A3 class-word CA,
selected 11 CA modules, CA calls 0–30, SA calls 10–20, all heads, equal
observation averaging, the five fixed VOC folds, 17 thresholds, and R2
DenseCRF were frozen. Each coarse affinity is the mean of eight heads at
the *same* diffusion call. It is block-lifted by natural coarse-cell
correspondence, multiplied elementwise with each fine head matrix, and
row-normalized. There is no gate exponent, threshold, learned weight, or
extra interpolation choice. All seven cells were scored raw and after R2.

**40/40 GPU shards** and the CPU merge completed successfully: **490
VOC-Sim** and **707 COCO-Cap** images. Live VOC/COCO smoke tests checked
S0–S2 against SG25 on the *same* captured observations. Their completed
full-set scores reproduce SG25 at 0–1.2×10⁻⁷ mIoU drift. Results below
use full-VOC-selected thresholds unchanged on COCO. Macro mIoU averages
pooled foreground IoU across 20 classes.

| Cell | Pass 1 | Pass 2 |
|---|---|---|
| S0 | fine 64 | same-head fine 64 |
| S1 | mean-head coarse 16, then upsample | fine 64 (SG25 control) |
| S2 | mean-head coarse 32, then upsample | fine 64 (SG25 control) |
| S3 | fine gated by coarse 16 | fine gated by coarse 16 |
| S4 | fine gated by coarse 32 | fine gated by coarse 32 |
| S5 | ordinary fine 64 | fine gated by coarse 16 |
| S6 | ordinary fine 64 | fine gated by coarse 32 |

## All scores

| Cell, raw | Forced VOC OOF | Δ vs S0 | Held-out wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | .8097 | — | — | .8114 | .450 | .6599 |
| S1 | .8020 | −.0077 | 0/5 | .8032 | .525 | **.6869** |
| S2 | .8089 | −.0008 | 1/5 | .8106 | .475 | .6695 |
| S3 | .7711 | −.0387 | 0/5 | .7725 | .450 | .6502 |
| S4 | .7748 | −.0349 | 0/5 | .7755 | .400 | .6474 |
| S5 | .7999 | −.0099 | 0/5 | .8009 | .475 | .6701 |
| S6 | .8021 | −.0077 | 0/5 | .8021 | .450 | .6672 |

| Cell, after R2 | Forced VOC OOF | Δ vs S0 | Held-out wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| **S0+R2** | **.8228** | — | — | **.8242** | .425 | .6621 |
| S1+R2 | .8170 | −.0058 | 1/5 | .8183 | .500 | **.6975** |
| S2+R2 | .8217 | −.0012 | 3/5 | .8228 | .450 | .6717 |
| S3+R2 | .7983 | −.0245 | 0/5 | .7992 | .450 | .6817 |
| S4+R2 | .7993 | −.0235 | 0/5 | .7993 | .400 | .6746 |
| S5+R2 | .8185 | −.0043 | 1/5 | .8201 | .450 | .6821 |
| S6+R2 | .8160 | −.0068 | 1/5 | .8193 | .425 | .6773 |

**Registered selection:** the raw and final multi-candidate VOC screens
each selected S0 in **5/5 folds** because no candidate cleared the
training-fold eligibility rule. No SG26 candidate is promoted. The
strongest frozen COCO score remains historical SG25 **S1+R2 at .6975**;
every new gate scores below it. The gates do transfer above S0+R2 on
COCO, but all lose on VOC. S5+R2 gives the closest 16-gate VOC result
(.8201 versus .8242); relative to historical S1+R2 it gains only .0018
VOC and loses .0154 COCO.

Final-mask precision/recall illustrates the trade-off:

| Cell | VOC precision / recall | COCO precision / recall | COCO paired mean-image Δ vs S0 (95% CI) |
|---|---:|---:|---:|
| S0+R2 | .8828 / .9162 | .7032 / .8893 | reference |
| S1+R2 | .8822 / .9037 | .7514 / .8803 | +.0272 [+.0200,+.0346] |
| S3+R2 | .8689 / .8994 | .7547 / .8543 | +.0021 [−.0073,+.0113] |
| S4+R2 | .8664 / .9001 | .7362 / .8644 | −.0033 [−.0108,+.0040] |
| S5+R2 | .8857 / .9069 | .7362 / .8772 | +.0120 [+.0078,+.0163] |
| S6+R2 | .8819 / .9106 | .7237 / .8851 | +.0067 [+.0032,+.0103] |

These 20k-resample intervals concern **paired mean-image IoU**, not
20-class macro mIoU. VOC paired mean-image deltas for S3/S4/S5/S6 are
−.0279 [−.0355,−.0204], −.0272 [−.0346,−.0196],
−.0045 [−.0084,−.0005], and −.0054 [−.0089,−.0018], respectively.
Their VOC better/worse image counts are 156/334, 153/337, 192/298,
and 193/297. Complete precision, recall, pixel IoU, mean-image IoU,
20-class IoUs, and bootstrap counts are in [`results.json`](../controlled_attention_cv/sg26/results.json).

## Did coarse gating remove harmful edges?

**Yes at the graph level, but it also changes too much useful structure.**
The table uses a deterministic post-hoc sample of 64 GT foreground and
64 background fine-grid positions per image; GT never affects inference.
“Suppressed” means the gated/original fine edge probability is below 0.5;
“nearly unchanged” means the ratio lies in [0.9,1.1]. Entropy and
effective-neighbor counts are measured on these sampled query rows.

| Dataset | Gate | Edges suppressed | Nearly unchanged | Mean row entropy | Effective neighbors |
|---|---:|---:|---:|---:|---:|
| VOC | 16 | 64.8% | 4.9% | 6.698→5.954 | 1220→655 |
| VOC | 32 | 81.1% | 2.5% | 6.698→5.535 | 1220→505 |
| COCO | 16 | 67.0% | 4.4% | 6.570→5.707 | 1098→531 |
| COCO | 32 | 82.0% | 2.3% | 6.570→5.294 | 1098→399 |

Mean **after/before fine-edge probability** by GT edge type:

| Dataset | Gate | Object→object | Object→background | Background→object | Background→background |
|---|---:|---:|---:|---:|---:|
| VOC | 16 | 1.415× | .519× | .797× | 1.298× |
| VOC | 32 | 1.768× | .432× | .554× | 1.559× |
| COCO | 16 | 1.515× | .493× | .784× | 1.392× |
| COCO | 32 | 1.764× | .429× | .558× | 1.754× |

The gate therefore suppresses crossing edges and strengthens
within-object edges, supporting the intended structural mechanism.
It also strengthens **background→background** edges. It is a spatial
coherence prior rather than a target-specific filter. **32-gating is
stronger than 16-gating** by these measures, contrary to the simple idea
that the 32 gate would preserve more fine connectivity. This is an
inference from sampled graph statistics, not a causal decomposition of IoU.

At a **fixed .45 raw-map threshold**, the mask-change counts reveal why
graph selectivity does not guarantee a better final mask. Negative net FP
means more FP removed than added; negative net TP means more TP lost than
gained.

| Dataset | Cell | Net TP | Net FP | Net distant FP >16 | Net separate FP |
|---|---|---:|---:|---:|---:|
| VOC | S3 | −199k | **+2,066k** | +211k | −32k |
| VOC | S4 | −2,365k | +165k | −333k | −37k |
| VOC | S5 | +233k | **+857k** | −36k | −20k |
| VOC | S6 | −376k | +165k | −179k | −19k |
| COCO | S3 | −1,053k | −1,251k | −2,708k | −619k |
| COCO | S4 | −2,278k | −3,658k | −3,698k | −676k |
| COCO | S5 | −11k | **−312k** | −1,099k | −347k |
| COCO | S6 | −275k | −1,188k | −1,394k | −343k |

For **VOC S5**, removal of some distant/separate FP is overwhelmed by
new, mainly nearer FP. On **COCO S5**, the gate removes 1.743M FP and
adds 1.431M FP, for net −.312M FP, with almost balanced TP gains/losses.
This plausibly explains its COCO precision gain. The FP-removed/TP-lost
ratio alone is misleading unless FP gained is reported too. Detailed
gained/removed counts, including >64-pixel FP, are in
[`mechanism.csv`](../controlled_attention_cv/sg26/mechanism.csv).

## Class trade-offs

Selected per-class **final IoU deltas versus S0+R2** show why neither
double-gating nor gated pass 2 can replace the reference. Exact all-cell
20-class values are in [`per_class_iou.csv`](../controlled_attention_cv/sg26/per_class_iou.csv).

| Class | VOC S1 | VOC S5 | VOC S6 | COCO S1 | COCO S5 | COCO S6 |
|---|---:|---:|---:|---:|---:|---:|
| sheep | −.115 | −.010 | +.005 | +.017 | +.022 | +.016 |
| chair | −.027 | −.048 | −.049 | +.041 | +.035 | +.019 |
| cow | −.049 | −.034 | −.016 | +.001 | +.002 | +.005 |
| potted plant | −.035 | −.029 | −.024 | +.083 | +.000 | +.002 |
| dining table | −.017 | +.021 | +.010 | −.080 | −.040 | +.005 |
| car | +.007 | +.006 | +.009 | +.134 | +.063 | +.044 |
| train | +.064 | +.054 | +.026 | +.071 | +.024 | +.014 |
| person | +.007 | −.011 | −.003 | +.016 | −.010 | −.010 |
| bicycle | −.001 | −.008 | −.015 | +.045 | +.040 | +.020 |

Gating only the second pass **does reduce** S1's large VOC sheep/cow
losses, but worsens VOC chair and forfeits much of S1's COCO car, train,
and potted-plant gains. Gating both passes is worse: VOC S3/S4 lose
about .025 macro mIoU after R2; VOC chair alone falls .164/.169.
The 32 gate partly rescues sheep (+.022 VOC with S4) but does not restore
overall performance.

## Decision and limits

Keep **A3→S0/FINE-P2→R2** as the VOC-selected working system. The exact
multiplicative coarse gate is **not promoted**. SG25's historical
16→64+R2 remains the strongest COCO-only transfer observation (.6975),
but VOC selection still does not promote it. SG26 shows that coarse SA
contains spatial boundary information; using it as a strong multiplicative
prior suppresses too many useful fine connections or creates new nearby
foreground. This mechanism reading is supported by the graph and mask
diagnostics but does not prove a unique causal explanation.

GT appears only in post-hoc diagnostics and evaluation. The edge sample
balances foreground/background per image, so its percentages describe
that sample, not all fine graph edges. For all-foreground GT masks,
unavailable background-edge categories were excluded. Raw-mask change
counts are at the fixed .45 diagnostic threshold, not the individually
selected final thresholds. No weights, exponents, gate strengths, or
class-specific rules were tuned in this run.

The compact [`report.md`](../controlled_attention_cv/sg26/report.md),
machine-readable [`results.json`](../controlled_attention_cv/sg26/results.json),
and two CSVs above are shareable. Resumable per-image shards remain local.
