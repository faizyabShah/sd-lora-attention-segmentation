# BRACE experiments, evening of 16 Nov to 17 Nov 2026: handoff for review

Prepared by Claude (Faizyab's side) for an independent review. Every number below is copied from a committed result file; the file is named in each section. Please check the reasoning, the conventions and the conclusions, and say where you disagree.

## 0. Context

**Method under study (BRACE, frozen).** Training-free class-word segmentation from Stable Diffusion 1.5 attention, read during generation:

1. Cross-attention seed for the class word, from 11 cross-attention layers over 31 calls, layers weighted by a self-attention-guided rule. The weights do not depend on the text.
2. Self-attention propagation as a capped 16→32→64 hierarchy, with a one-sided cap only at 16→32.
3. DenseCRF, then a fixed threshold of 0.55.

It scores 0.830 on VOC-Sim and 0.696 on COCO-Cap (class-macro mIoU). BRACE is frozen; nothing below changes it.

**Benchmarks.**

| | VOC-Sim | COCO-Cap |
|---|---|---|
| Images | 490 generated (single-object prompts) | 707 generated from COCO captions |
| Role | development set | untuned transfer |

Both sets were used while developing BRACE, so BRACE's own numbers are in-sample for the method design.

**Rules we follow.**
- Pre-register and commit before scoring.
- Select only on VOC-Sim; never tune on COCO-Cap.
- Compare variants only within one capture.
- Report failures as failures.

**Conventions used below.**
- **Fixed 0.55:** every cell at threshold 0.55.
- **VOC-selected:** on VOC-Sim, the threshold is chosen 5-fold out-of-fold (fold = crc32(image id) mod 5); on COCO-Cap, each cell gets the threshold that is best on full VOC-Sim, unchanged.
- **Intervals:** image-bootstrap 95% intervals, 2000 resamples.
- **Regeneration:** every run regenerates the images live. Regeneration is pixel-identical to the stored benchmark image on only part of the set (counts given per run), but the maps of the reference cell match the stored capture closely, and the paper's numbers reproduce.

**Machine.** A SLURM cluster (V100 and A100 nodes); the paper's numbers were produced there too.

---

## 1. Green-screen tokens (`runs/14_abc`, part A): STOP

**Commit** b112b079b. Pre-registration 202326b4a, with Amendment 1 made before any token was learned.

**Idea.** Learn an OVAM-style class token without labels. Generate the class on a plain backdrop, key out the backdrop to get a pseudo-mask, and optimise the token on those exemplars.

**Setup.** Replayed trajectories of 150 + 150 screening images, exact seed weighting, DenseCRF at 0.55. These are development images.

| Arm | VOC-Sim | Δ vs base | COCO-Cap | Δ vs base |
|---|---:|---|---:|---|
| base (frozen attribution, replay) | 0.7856 | — | 0.6447 | — |
| green-screen token (primary) | 0.7780 | −0.0076 [−0.0276, +0.0109] | 0.6633 | +0.0185 [−0.0147, +0.0547] |
| closed-form step (secondary) | 0.7907 | +0.0050 [−0.0033, +0.0129] | 0.6459 | +0.0012 [−0.0143, +0.0192] |

**Verdict: STOP.** The COCO-Cap interval crosses zero and VOC-Sim goes down. Train had only 2 usable exemplars and kept the original word. Parts B and C of the same run had closed earlier the same day.

---

## 2. Component ladder: class word vs OVAM's optimized token (`runs/15_token_ladder`)

**Commits:** pre-registration 95288dc09 (empty-fold guard 95381a493), result 812405e1c.

**Question** (supervisor's comments #4 and #7). Does each component of BRACE still help when the seed comes from OVAM's optimized token instead of the class word?

**Design.**
- One live generation per image and two query columns from that generation:
  - **W:** the class word, frozen attribution.
  - **T:** OVAM's released VOC-Sim token (mean of 2–3 per-image tokens), keys [start token, token], softmax over the two, conditional query.
- The same VOC-Sim tokens are used on COCO-Cap, as OVAM did.
- The token is few-shot: it needs annotated training images. No evaluation seed appears among the token seeds (checked by both sides).

**VOC-Sim (490), fixed 0.55, Δ to the row above**

| Row | Step | W | Δ W | T | Δ T | T − W | W VOC-sel. | T VOC-sel. |
|---|---|---:|---|---:|---|---|---:|---:|
| 1 | CA seed, uniform layer weights | 0.588 | — | 0.766 | — | +0.178 [+0.166, +0.189] | 0.713 | 0.796 |
| 2 | CA seed, guided weights | 0.611 | +0.023 [+0.016, +0.030] | 0.754 | −0.012 [−0.016, −0.009] | +0.143 | 0.705 | 0.775 |
| 3 | + fine-only SA (two passes) | 0.791 | +0.180 [+0.168, +0.193] | 0.845 | +0.091 [+0.082, +0.100] | +0.054 | 0.810 | 0.858 |
| 4 | + hierarchy, no cap | 0.807 | +0.016 [+0.008, +0.024] | 0.835 | −0.010 [−0.016, −0.002] | +0.028 | 0.807 | 0.835 |
| 5 | + one-sided cap | 0.817 | +0.010 [+0.004, +0.016] | 0.847 | +0.012 [+0.007, +0.016] | +0.030 | 0.816 | 0.848 |
| 6 | + DenseCRF (full) | **0.830** | +0.013 | 0.861 | +0.014 | +0.031 [+0.022, +0.039] | 0.830 | 0.861 |
| 3c | fine-only + DenseCRF | 0.796 | — | 0.851 | — | +0.055 | 0.824 | 0.870 |
| 6u | full, uniform weights | 0.826 | — | 0.857 | — | +0.031 | 0.826 | 0.857 |

**COCO-Cap (707), fixed 0.55**

| Row | Step | W | Δ W | T | Δ T | T − W | W VOC-sel. | T VOC-sel. |
|---|---|---:|---|---:|---|---|---:|---:|
| 1 | CA seed, uniform | 0.444 | — | 0.625 | — | +0.181 [+0.164, +0.197] | 0.563 | 0.638 |
| 2 | CA seed, guided | 0.501 | +0.056 [+0.046, +0.065] | 0.617 | −0.008 | +0.116 | 0.584 | 0.624 |
| 3 | + fine-only SA | 0.677 | +0.176 | 0.689 | +0.072 | +0.012 [−0.002, +0.029] | 0.660 | 0.673 |
| 4 | + hierarchy, no cap | 0.687 | +0.010 [−0.001, +0.023] | 0.669 | −0.020 [−0.029, −0.012] | −0.018 | 0.678 | 0.669 |
| 5 | + cap | 0.682 | −0.005 [−0.011, −0.001] | 0.667 | −0.002 | −0.014 | 0.682 | 0.674 |
| 6 | + DenseCRF (full) | **0.696** | +0.014 | 0.681 | +0.013 | −0.015 [−0.030, +0.000] | 0.696 | 0.681 |
| 3c | fine-only + DenseCRF | 0.684 | — | 0.698 | — | +0.014 | 0.674 | 0.678 |
| 6u | full, uniform weights | 0.672 | — | 0.677 | — | +0.004 | 0.672 | 0.677 |

**Registered comparisons (token column, fixed 0.55).** Rule: a component holds if its COCO-Cap Δ has an interval above zero and its VOC-Sim Δ is at least −0.005.

| ID | Comparison | Token: VOC / COCO | Verdict | Word (reference): VOC / COCO |
|---|---|---|---|---|
| H | hierarchy vs fine-only, both + DenseCRF | +0.010 / −0.017 [−0.028, −0.007] | does not hold | +0.034 / +0.012 [−0.002, +0.026] |
| L | guided vs uniform layer weights | +0.003 / +0.004 [−0.002, +0.011] | does not hold | +0.004 / +0.024 [+0.013, +0.036] |
| C | cap vs no cap (raw) | +0.012 / −0.002 | does not hold | +0.010 / −0.005 [−0.011, −0.001] |

**Per class, COCO-Cap, full pipeline (W → T).** Person 0.28 → 0.62; sofa 0.70 → 0.34; car 0.76 → 0.66; train 0.75 → 0.85.

**Reading.**
- The token is a much better seed (+0.18 on both benchmarks) and gives the best VOC-Sim result (0.861).
- On COCO-Cap, with the token, none of the three components helps, and the best token cell is fine-only + DenseCRF (0.698).
- With the class word, the layer weighting holds on COCO-Cap at 0.55 and the hierarchy's COCO-Cap gain over fine-only at 0.55 is only +0.012 (interval through zero).

---

## 3. Where the draft's "+0.034 on COCO-Cap" for the hierarchy comes from (`runs/16_trace_034`)

**Commit** cc2d732c2. Post hoc diagnostic; nothing selected.

**Source.** In the earlier experiment (`sg_combined_v3`), +0.0341 = capped hierarchy + DenseCRF at threshold 0.55 (0.6964) minus fine-only + DenseCRF at threshold 0.425 (0.6623). Each map is at its own full-VOC-Sim-best threshold, frozen for COCO-Cap. The paper's table caption says "VOC-frozen threshold" but not that the two thresholds differ.

**Recomputed from the runs/15 capture, COCO-Cap, both with DenseCRF**

| Convention | hierarchy (thr) | fine-only (thr) | Δ [95% interval] |
|---|---:|---:|---|
| Each at its own VOC-Sim-best threshold (draft) | 0.6962 (0.550) | 0.6621 (0.425) | **+0.0341 [+0.0227, +0.0450]** |
| Both at 0.55 | 0.6962 | 0.6841 | +0.0121 [−0.0018, +0.0264] |
| Both at 0.425 | 0.6234 | 0.6621 | −0.0387 [−0.0502, −0.0265] |
| Each at its COCO-Cap-best threshold (uses COCO-Cap labels; diagnostic) | 0.7066 (0.600) | 0.6890 (0.525) | +0.0176 [+0.0063, +0.0294] |
| Raw, own VOC thresholds (0.55 / 0.45) | 0.6818 | 0.6599 | +0.0219 [+0.0117, +0.0321] |
| Raw, both at 0.55 | 0.6818 | 0.6769 | +0.0049 [−0.0077, +0.0186] |

**VOC-Sim, same comparison.**
- Each at its own threshold: +0.0057 [−0.0009, +0.0122].
- Out-of-fold threshold per map: +0.0074 [+0.0009, +0.0142].

**Reading.**
- The fine-only map's best threshold moves 0.425 → 0.525 from VOC-Sim to COCO-Cap; the hierarchy's moves 0.55 → 0.60.
- About half of the +0.034 (+0.018) remains when each map gets its best threshold. The rest is that the hierarchy's map keeps its threshold better across datasets.
- This matches an earlier protocol finding ("cross-image calibration, not better localisation").
- Proposed wording: the hierarchy's map transfers to cluttered scenes at its VOC-Sim threshold, where the fine-only map does not.

---

## 4. Seed × propagation swap between BRACE and NERVE (`runs/17_seed_walk_swap`)

**Commits:** pre-registration 6a445fc0e, result d921b1d2f. Two code changes after the pre-registration, neither in scoring: an empty-subset guard and an output-folder suffix (150586fa3).

**Question.** NERVE (a CLIP-seeded method with a closed-form random walk over SD self-attention) beats BRACE on COCO-Cap. Is that because of its seed or its propagation?

**Design.**

| | Option 1 | Option 2 |
|---|---|---|
| Seeds | BRACE's cross-attention seed (live generation) | NERVE's CLIP ViT-L/14 seed on the stored image: {class word} + NERVE's VOC background words, from NERVE's released code with the other side's minimal patch |
| Propagations | none; BRACE fine-only; BRACE capped hierarchy (frozen code) | NERVE's released walk, unchanged (one noised pass of the stored image, timestep about 101, two 64-px layers, 0.8 global + 0.2 local affinity, closed-form walk δ = 0.9) |

- DenseCRF is applied to every cell.
- **Primary number:** VOC-selected threshold with DenseCRF.

**VOC-selected threshold, with DenseCRF**

| Propagation | VOC-Sim, CA seed | VOC-Sim, CLIP seed | COCO-Cap, CA seed | COCO-Cap, CLIP seed |
|---|---:|---:|---:|---:|
| none | 0.735 | 0.759 | 0.619 | 0.693 |
| BRACE fine-only | 0.823 | 0.830 | 0.662 | 0.704 |
| BRACE capped hierarchy | **0.830 (BRACE)** | 0.836 | **0.696 (BRACE)** | 0.704 |
| NERVE walk | 0.828 | **0.842 (NERVE)** | 0.678 | **0.737 (NERVE)** |

Raw and fixed-0.5 tables are in the result file and agree in direction.

**Registered contrasts (VOC-selected, DenseCRF)**

| ID | Contrast | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images only (243) |
|---|---|---|---|---|
| P-CLIP | walk − hierarchy, CLIP seed | +0.006 [−0.002, +0.015] | +0.033 [+0.019, +0.049] | +0.030 [+0.006, +0.058] |
| P-CA | walk − hierarchy, CA seed | −0.002 [−0.013, +0.008] | −0.019 [−0.031, −0.005] | −0.013 [−0.035, +0.008] |
| S-H | CLIP − CA seed, hierarchy | +0.006 [−0.007, +0.018] | +0.008 [−0.012, +0.029] | +0.005 [−0.024, +0.038] |
| S-W | CLIP − CA seed, walk | +0.014 [+0.001, +0.029] | +0.060 [+0.035, +0.083] | +0.048 [+0.012, +0.088] |
| I | interaction | +0.008 [−0.004, +0.021] | +0.052 [+0.034, +0.070] | +0.043 [+0.015, +0.072] |

**Registered reading rule (COCO-Cap): mixed.**
- "The seed explains NERVE's lead" (S-H and S-W both above zero): no.
- "The walk explains NERVE's lead" (P-CLIP and P-CA both above zero): no.
- NERVE's lead appears only when its walk runs on its CLIP seed.

**Fidelity.**
- BRACE cell at 0.55 with DenseCRF: 0.830 / 0.696 (paper: same).
- NERVE cell at the other side's convention (0.5, raw): 0.8302 / 0.7295, equal to her independent rerun to four decimals.
- Per-image pixel counts are identical on only 40/490 and 94/707 images (max |ΔIoU| 0.0016 / 0.0021). The cause is likely hardware sensitivity in CLIP's half precision and NERVE's 20% seed cut-off; both NERVE stages ran on the A100 node, as hers did. Unexplained residual.

**Caveats.**
- The CLIP seed and NERVE's walk read the stored image, while BRACE reads its regeneration.
- NERVE's constants were not retuned for the CA seed.
- CLIP-seeded cells are not SD-only.

**Per class, COCO-Cap (CA × hierarchy / CA × walk / CLIP × hierarchy / CLIP × walk).**

| Class | CA × hierarchy | CA × walk | CLIP × hierarchy | CLIP × walk |
|---|---:|---:|---:|---:|
| person | 0.28 | 0.26 | 0.64 | 0.71 |
| chair | 0.47 | 0.46 | 0.59 | 0.68 |
| tv | 0.79 | 0.74 | 0.26 | 0.25 |
| aeroplane | 0.80 | 0.68 | 0.67 | 0.64 |

---

## 5. Post hoc breakdown of the swap: where NERVE's +0.041 sits (exploratory, not committed)

**Scripts:** `outputs/claude_scratch/swap_diag.py`, `swap_diag2.py`, reading the runs/17 scores. Nothing was selected from this. COCO-Cap unless stated.

**By object size (thirds of images by ground-truth area; per-image mean IoU, full pipelines)**

| Size third | BRACE | NERVE | CA seed alone | CLIP seed alone |
|---|---:|---:|---:|---:|
| small (< 13% of image) | 0.584 | 0.699 | 0.501 | 0.666 |
| middle | 0.764 | 0.792 | 0.670 | 0.762 |
| large | 0.814 | 0.817 | 0.704 | 0.751 |

**Misses.**
- BRACE scores below 0.3 IoU on 79/707 images (11.2%), against 6.1% for NERVE.
- 58 of the 79 are small objects; 29 are person (of 52 person images).
- In these 79, BRACE's recall is 0.75 but its false-positive area is about 16 times the object's: the map floods.
- The CA seed scores 0.18 IoU on them; the CLIP seed scores 0.56.

**Other observations.**
- At equal recall (0.905 vs 0.886), BRACE's precision is 0.756 against NERVE's 0.810.
- Person contributes +0.021 of the +0.041 and tv −0.027; without both, NERVE leads by +0.052 and wins 14 of 20 classes.
- Whether the class word appears in the caption makes no difference to the gap.
- On images neither method misses (607), BRACE 0.776 vs NERVE 0.798.
- On the CLIP seed, the walk beats our hierarchy on small objects (0.699 vs 0.650) but ties it on large ones (0.817 vs 0.818).
- On the CA seed, our hierarchy is at least as good as the walk in every size third.

**My reading.** The bottleneck is the CA seed's lack of an absolute "not the object" reference: it is one word, min–max scaled, so weak background attention is stretched up when the object is small. NERVE's CLIP seed is a softmax against background words. This motivated run 18.

---

## 6. Background-aware cross-attention seed (`runs/18_bg_key_seed`): FAIL

**Commits:** pre-registration fe85e8ef3; Amendment 1 92b15b699 (owner's request, before any scoring; the first smoke run was cancelled during capture); result 1a36f8eb1.

**Idea.** Put the class word and fixed, class-agnostic background words in one cross-attention softmax inside SD, and take the class word's share: NERVE's seed principle without CLIP.

**Design.**
- One live generation per image; seeds differ only in the attribution keys.
- Each word is encoded alone by the frozen text encoder; multi-piece words contribute one key per piece, and the class map sums the class pieces' columns.
- Everything is min–max normalised and goes through the frozen propagation.

| Seed | Keys competing with the class word | Layer weights |
|---|---|---|
| W | frozen attribution (BRACE) | guided |
| S1 | start token | guided |
| B8 | start token + 8 generic words (wall, floor, ground, sky, grass, tree, building, road) | guided |
| R8 | 8 generic words, no start token | guided |
| BN | start token + NERVE's 25 background words | guided |
| RN | NERVE's 25 words, no start token | guided |
| W0, S1u, B8u, R8u, BNu, RNu | same keys | uniform (Amendment 1, descriptive only) |

**Registered selection and test.**
1. Pick the best of S1/B8/R8/BN/RN on VOC-Sim out-of-fold, full pipeline.
2. Test it against W on COCO-Cap once.
3. It passes if the COCO-Cap interval is above zero and VOC-Sim Δ is at least −0.005.

**Fidelity.**
- VOC-Sim: 248/490 bit-identical regenerations; W full pipeline at 0.55 = 0.830.
- COCO-Cap: 391/707 bit-identical; W = 0.697.

### 6.1 VOC-Sim, VOC-selected threshold

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.705 | 0.695 | 0.696 | 0.689 | 0.698 | 0.685 | 0.714 | 0.698 | 0.711 | 0.693 | 0.711 | 0.690 |
| fine-only | 0.810 | 0.798 | 0.793 | 0.809 | 0.799 | 0.805 | 0.805 | 0.771 | 0.798 | 0.810 | 0.803 | 0.803 |
| capped hierarchy | 0.816 | 0.805 | 0.791 | 0.814 | 0.790 | 0.803 | 0.812 | 0.799 | 0.798 | 0.815 | 0.801 | 0.806 |
| fine-only + DenseCRF | 0.823 | 0.809 | 0.804 | 0.820 | 0.812 | 0.818 | 0.817 | 0.781 | 0.812 | 0.821 | 0.818 | 0.817 |
| hierarchy + DenseCRF | **0.830** | 0.819 | 0.806 | 0.824 | 0.803 | 0.815 | 0.825 | 0.811 | 0.812 | 0.824 | 0.816 | 0.818 |

### 6.2 VOC-Sim, fixed 0.55

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.611 | 0.564 | 0.541 | 0.686 | 0.518 | 0.647 | 0.588 | 0.512 | 0.536 | 0.690 | 0.514 | 0.649 |
| fine-only | 0.791 | 0.775 | 0.759 | 0.808 | 0.752 | 0.792 | 0.780 | 0.752 | 0.760 | 0.810 | 0.758 | 0.791 |
| capped hierarchy | 0.817 | 0.806 | 0.792 | 0.807 | 0.791 | 0.801 | 0.813 | 0.798 | 0.798 | 0.806 | 0.800 | 0.803 |
| fine-only + DenseCRF | 0.796 | 0.779 | 0.762 | 0.818 | 0.753 | 0.800 | 0.784 | 0.754 | 0.763 | 0.819 | 0.760 | 0.799 |
| hierarchy + DenseCRF | **0.830** | 0.819 | 0.805 | 0.822 | 0.802 | 0.815 | 0.826 | 0.811 | 0.811 | 0.821 | 0.812 | 0.817 |

### 6.3 VOC-Sim, differences [95% interval]

| Seed | full vs W, VOC-sel. | full vs W, 0.55 | hierarchy − fine-only (+CRF) | hierarchy − seed (raw) | guided − uniform weights |
|---|---|---|---|---|---|
| W | — | — | +0.008 [+0.001, +0.014] | +0.111 [+0.100, +0.123] | +0.005 [+0.000, +0.011] |
| S1 | −0.011 [−0.019, −0.003] | −0.011 [−0.019, −0.003] | +0.010 [+0.002, +0.018] | +0.110 [+0.098, +0.123] | +0.008 [−0.001, +0.018] |
| B8 | −0.024 [−0.031, −0.016] | −0.026 [−0.033, −0.018] | +0.002 [−0.005, +0.010] | +0.094 [+0.083, +0.108] | −0.006 [−0.012, +0.001] |
| R8 | −0.006 [−0.020, +0.007] | −0.008 [−0.024, +0.006] | +0.005 [−0.002, +0.011] | +0.125 [+0.114, +0.138] | +0.001 [−0.003, +0.004] |
| BN | −0.027 [−0.035, −0.018] | −0.028 [−0.036, −0.019] | −0.008 [−0.016, −0.000] | +0.092 [+0.080, +0.107] | −0.013 [−0.019, −0.005] |
| RN | −0.015 [−0.027, −0.003] | −0.015 [−0.027, −0.003] | −0.003 [−0.011, +0.005] | +0.118 [+0.108, +0.131] | −0.003 [−0.008, +0.002] |
| W0 | −0.005 [−0.011, −0.000] | −0.004 [−0.010, +0.001] | +0.008 [+0.001, +0.014] | +0.099 [+0.089, +0.110] | — |
| S1u | −0.019 [−0.030, −0.008] | −0.019 [−0.029, −0.008] | +0.030 [+0.019, +0.041] | +0.101 [+0.089, +0.115] | — |
| B8u | −0.018 [−0.025, −0.012] | −0.020 [−0.027, −0.013] | −0.000 [−0.007, +0.006] | +0.087 [+0.078, +0.098] | — |
| R8u | −0.007 [−0.020, +0.006] | −0.009 [−0.024, +0.004] | +0.002 [−0.005, +0.010] | +0.122 [+0.111, +0.135] | — |
| BNu | −0.014 [−0.021, −0.007] | −0.018 [−0.026, −0.010] | −0.002 [−0.009, +0.005] | +0.090 [+0.080, +0.102] | — |
| RNu | −0.012 [−0.024, +0.000] | −0.013 [−0.025, −0.000] | +0.002 [−0.005, +0.009] | +0.116 [+0.104, +0.129] | — |

### 6.4 COCO-Cap, VOC-selected threshold

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.584 | 0.575 | 0.583 | 0.579 | 0.592 | 0.595 | 0.562 | 0.551 | 0.588 | 0.588 | 0.602 | 0.599 |
| fine-only | 0.660 | 0.653 | 0.660 | 0.651 | 0.676 | 0.665 | 0.631 | 0.614 | 0.648 | 0.662 | 0.657 | 0.665 |
| capped hierarchy | 0.682 | 0.679 | 0.683 | 0.667 | 0.693 | 0.681 | 0.660 | 0.659 | 0.665 | 0.674 | 0.683 | 0.677 |
| fine-only + DenseCRF | 0.662 | 0.654 | 0.664 | 0.653 | 0.679 | 0.679 | 0.644 | 0.625 | 0.652 | 0.665 | 0.673 | 0.680 |
| hierarchy + DenseCRF | **0.697** | 0.693 | 0.686 | 0.678 | 0.697 | 0.694 | 0.663 | 0.654 | 0.669 | 0.678 | 0.688 | 0.690 |

### 6.5 COCO-Cap, fixed 0.55

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.500 | 0.499 | 0.491 | 0.584 | 0.480 | 0.586 | 0.444 | 0.447 | 0.477 | 0.593 | 0.474 | 0.592 |
| fine-only | 0.677 | 0.669 | 0.675 | 0.660 | 0.681 | 0.688 | 0.648 | 0.641 | 0.669 | 0.662 | 0.682 | 0.690 |
| capped hierarchy | 0.682 | 0.679 | 0.683 | 0.642 | 0.693 | 0.671 | 0.660 | 0.650 | 0.677 | 0.645 | 0.692 | 0.668 |
| fine-only + DenseCRF | 0.684 | 0.675 | 0.681 | 0.671 | 0.686 | 0.697 | 0.654 | 0.646 | 0.674 | 0.672 | 0.687 | 0.699 |
| hierarchy + DenseCRF | **0.697** | 0.693 | 0.697 | 0.655 | 0.706 | 0.686 | 0.673 | 0.662 | 0.689 | 0.658 | 0.705 | 0.683 |

### 6.6 COCO-Cap, differences [95% interval]

| Seed | full vs W, VOC-sel. | full vs W, 0.55 | hierarchy − fine-only (+CRF) | hierarchy − seed (raw) | guided − uniform weights |
|---|---|---|---|---|---|
| W | — | — | +0.034 [+0.023, +0.045] | +0.098 [+0.085, +0.114] | +0.034 [+0.023, +0.045] |
| S1 | −0.003 [−0.009, +0.003] | −0.003 [−0.009, +0.003] | +0.039 [+0.028, +0.051] | +0.104 [+0.090, +0.119] | +0.040 [+0.028, +0.052] |
| B8 | −0.011 [−0.017, −0.004] | +0.000 [−0.006, +0.006] | +0.022 [+0.012, +0.032] | +0.100 [+0.086, +0.116] | +0.017 [+0.007, +0.026] |
| R8 | −0.019 [−0.036, −0.004] | −0.041 [−0.058, −0.025] | +0.024 [+0.015, +0.034] | +0.087 [+0.073, +0.104] | −0.001 [−0.009, +0.008] |
| BN | +0.000 [−0.006, +0.008] | +0.010 [+0.003, +0.017] | +0.018 [+0.008, +0.027] | +0.100 [+0.087, +0.115] | +0.009 [+0.000, +0.018] |
| RN | −0.003 [−0.016, +0.011] | −0.011 [−0.024, +0.004] | +0.015 [+0.004, +0.025] | +0.085 [+0.072, +0.100] | +0.004 [−0.005, +0.013] |
| W0 | −0.034 [−0.045, −0.023] | −0.024 [−0.035, −0.013] | +0.019 [+0.007, +0.030] | +0.098 [+0.083, +0.114] | — |
| S1u | −0.043 [−0.056, −0.030] | −0.035 [−0.048, −0.022] | +0.028 [+0.016, +0.040] | +0.108 [+0.093, +0.122] | — |
| B8u | −0.027 [−0.036, −0.018] | −0.007 [−0.016, +0.001] | +0.017 [+0.006, +0.028] | +0.078 [+0.064, +0.091] | — |
| R8u | −0.018 [−0.034, −0.003] | −0.038 [−0.055, −0.021] | +0.013 [+0.004, +0.023] | +0.086 [+0.071, +0.104] | — |
| BNu | −0.009 [−0.017, +0.001] | +0.009 [−0.000, +0.018] | +0.015 [+0.004, +0.026] | +0.081 [+0.067, +0.096] | — |
| RNu | −0.006 [−0.019, +0.006] | −0.014 [−0.027, −0.000] | +0.011 [−0.001, +0.021] | +0.079 [+0.064, +0.094] | — |

### 6.7 Registered selection and test

VOC-Sim out-of-fold, full pipeline:

| W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8302 | 0.8195 | 0.8062 | 0.8242 | 0.8034 | 0.8155 | 0.8249 | 0.8113 | 0.8117 | 0.8236 | 0.8160 | 0.8184 |

- **Selected:** R8 (class + 8 generic words). VOC-Sim Δ vs W: −0.006 [−0.020, +0.007].
- **COCO-Cap test:** R8 − W = −0.019 [−0.036, −0.004], at thresholds 0.600 / 0.550.
- **Verdict: FAIL.**

### 6.8 COCO-Cap failure modes, full pipeline at VOC-selected threshold (descriptive)

Size columns are per-image IoU by ground-truth-area third.

| Seed | thr | mIoU | small | middle | large | below 0.3 IoU | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 0.550 | 0.697 | 0.585 | 0.764 | 0.814 | 11.2% | 0.756 | 0.905 |
| S1 | 0.550 | 0.693 | 0.579 | 0.760 | 0.816 | 11.2% | 0.751 | 0.909 |
| B8 | 0.525 | 0.686 | 0.589 | 0.756 | 0.803 | 10.7% | 0.742 | 0.910 |
| R8 | 0.600 | 0.678 | 0.591 | 0.754 | 0.795 | 10.6% | 0.743 | 0.895 |
| BN | 0.525 | 0.697 | 0.599 | 0.758 | 0.809 | 10.3% | 0.755 | 0.909 |
| RN | 0.575 | 0.694 | 0.599 | 0.758 | 0.810 | 10.6% | 0.755 | 0.903 |
| W0 | 0.525 | 0.663 | 0.550 | 0.736 | 0.800 | 13.6% | 0.717 | 0.915 |
| S1u | 0.525 | 0.654 | 0.532 | 0.730 | 0.797 | 15.6% | 0.707 | 0.917 |
| B8u | 0.500 | 0.669 | 0.573 | 0.741 | 0.800 | 10.7% | 0.719 | 0.922 |
| R8u | 0.600 | 0.678 | 0.596 | 0.758 | 0.796 | 9.2% | 0.745 | 0.895 |
| BNu | 0.500 | 0.688 | 0.606 | 0.752 | 0.809 | 8.9% | 0.740 | 0.920 |
| RNu | 0.575 | 0.690 | 0.605 | 0.759 | 0.811 | 9.9% | 0.753 | 0.899 |

### 6.9 Per class, full pipeline, VOC-selected threshold

**VOC-Sim**

| class | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.88 | 0.88 | 0.88 | 0.90 | 0.88 | 0.90 | 0.89 | 0.88 | 0.88 | 0.90 | 0.89 | 0.90 |
| bicycle | 0.80 | 0.80 | 0.79 | 0.80 | 0.79 | 0.80 | 0.79 | 0.80 | 0.79 | 0.80 | 0.78 | 0.80 |
| bird | 0.92 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.88 | 0.87 | 0.93 | 0.92 | 0.93 | 0.93 |
| boat | 0.86 | 0.85 | 0.85 | 0.84 | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 | 0.83 | 0.84 | 0.84 |
| bottle | 0.88 | 0.87 | 0.87 | 0.89 | 0.85 | 0.87 | 0.90 | 0.91 | 0.90 | 0.88 | 0.88 | 0.87 |
| bus | 0.92 | 0.92 | 0.92 | 0.91 | 0.92 | 0.92 | 0.92 | 0.91 | 0.91 | 0.91 | 0.91 | 0.92 |
| car | 0.85 | 0.87 | 0.86 | 0.81 | 0.85 | 0.83 | 0.85 | 0.78 | 0.85 | 0.81 | 0.85 | 0.83 |
| cat | 0.85 | 0.84 | 0.84 | 0.88 | 0.84 | 0.88 | 0.86 | 0.86 | 0.86 | 0.88 | 0.86 | 0.88 |
| chair | 0.79 | 0.80 | 0.79 | 0.81 | 0.78 | 0.79 | 0.79 | 0.78 | 0.77 | 0.81 | 0.76 | 0.80 |
| cow | 0.90 | 0.88 | 0.89 | 0.96 | 0.89 | 0.96 | 0.90 | 0.89 | 0.88 | 0.96 | 0.91 | 0.96 |
| dining-table | 0.42 | 0.43 | 0.42 | 0.40 | 0.41 | 0.40 | 0.41 | 0.41 | 0.40 | 0.42 | 0.39 | 0.41 |
| dog | 0.93 | 0.92 | 0.92 | 0.93 | 0.92 | 0.93 | 0.93 | 0.91 | 0.91 | 0.92 | 0.92 | 0.92 |
| horse | 0.90 | 0.90 | 0.90 | 0.93 | 0.90 | 0.92 | 0.90 | 0.91 | 0.90 | 0.93 | 0.90 | 0.92 |
| motorbike | 0.87 | 0.87 | 0.87 | 0.85 | 0.87 | 0.86 | 0.87 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| person | 0.81 | 0.62 | 0.49 | 0.60 | 0.39 | 0.39 | 0.82 | 0.69 | 0.65 | 0.59 | 0.62 | 0.47 |
| potted-plant | 0.72 | 0.69 | 0.64 | 0.75 | 0.67 | 0.77 | 0.76 | 0.77 | 0.73 | 0.78 | 0.74 | 0.78 |
| sheep | 0.75 | 0.75 | 0.76 | 0.87 | 0.77 | 0.87 | 0.74 | 0.74 | 0.75 | 0.84 | 0.77 | 0.86 |
| sofa | 0.96 | 0.96 | 0.95 | 0.94 | 0.95 | 0.93 | 0.96 | 0.91 | 0.96 | 0.94 | 0.95 | 0.93 |
| train | 0.74 | 0.73 | 0.72 | 0.82 | 0.79 | 0.85 | 0.66 | 0.65 | 0.65 | 0.81 | 0.76 | 0.86 |
| tv | 0.85 | 0.89 | 0.83 | 0.68 | 0.81 | 0.66 | 0.83 | 0.84 | 0.81 | 0.66 | 0.79 | 0.63 |

**COCO-Cap**

| class | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.80 | 0.78 | 0.77 | 0.82 | 0.80 | 0.81 | 0.72 | 0.71 | 0.72 | 0.82 | 0.76 | 0.82 |
| bicycle | 0.63 | 0.62 | 0.59 | 0.57 | 0.60 | 0.59 | 0.61 | 0.61 | 0.59 | 0.58 | 0.59 | 0.60 |
| bird | 0.78 | 0.78 | 0.78 | 0.79 | 0.78 | 0.79 | 0.76 | 0.76 | 0.75 | 0.79 | 0.76 | 0.79 |
| boat | 0.59 | 0.58 | 0.58 | 0.53 | 0.58 | 0.54 | 0.53 | 0.52 | 0.53 | 0.54 | 0.53 | 0.54 |
| bottle | 0.50 | 0.51 | 0.50 | 0.52 | 0.52 | 0.55 | 0.46 | 0.46 | 0.47 | 0.50 | 0.48 | 0.51 |
| bus | 0.92 | 0.92 | 0.91 | 0.91 | 0.92 | 0.92 | 0.91 | 0.90 | 0.91 | 0.91 | 0.91 | 0.91 |
| car | 0.77 | 0.76 | 0.76 | 0.73 | 0.77 | 0.72 | 0.62 | 0.58 | 0.74 | 0.72 | 0.74 | 0.72 |
| cat | 0.87 | 0.87 | 0.87 | 0.86 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.86 | 0.87 | 0.87 |
| chair | 0.47 | 0.47 | 0.46 | 0.48 | 0.47 | 0.51 | 0.46 | 0.40 | 0.46 | 0.49 | 0.48 | 0.52 |
| cow | 0.87 | 0.88 | 0.89 | 0.89 | 0.89 | 0.91 | 0.87 | 0.87 | 0.88 | 0.88 | 0.89 | 0.91 |
| dining table | 0.45 | 0.50 | 0.46 | 0.47 | 0.46 | 0.51 | 0.46 | 0.46 | 0.49 | 0.48 | 0.52 | 0.51 |
| dog | 0.75 | 0.75 | 0.73 | 0.74 | 0.74 | 0.76 | 0.75 | 0.75 | 0.74 | 0.74 | 0.75 | 0.76 |
| horse | 0.72 | 0.72 | 0.67 | 0.72 | 0.71 | 0.74 | 0.71 | 0.71 | 0.67 | 0.72 | 0.70 | 0.73 |
| motorbike | 0.67 | 0.67 | 0.65 | 0.63 | 0.65 | 0.63 | 0.64 | 0.65 | 0.63 | 0.62 | 0.63 | 0.62 |
| person | 0.28 | 0.31 | 0.32 | 0.40 | 0.30 | 0.34 | 0.30 | 0.30 | 0.39 | 0.47 | 0.41 | 0.40 |
| potted plant | 0.74 | 0.70 | 0.78 | 0.74 | 0.79 | 0.79 | 0.68 | 0.68 | 0.70 | 0.70 | 0.72 | 0.72 |
| sheep | 0.87 | 0.87 | 0.88 | 0.86 | 0.89 | 0.88 | 0.87 | 0.87 | 0.87 | 0.85 | 0.88 | 0.88 |
| sofa | 0.70 | 0.68 | 0.67 | 0.71 | 0.69 | 0.75 | 0.64 | 0.61 | 0.65 | 0.72 | 0.67 | 0.77 |
| train | 0.75 | 0.71 | 0.69 | 0.73 | 0.77 | 0.79 | 0.65 | 0.58 | 0.58 | 0.72 | 0.76 | 0.79 |
| tv | 0.79 | 0.78 | 0.75 | 0.44 | 0.77 | 0.50 | 0.76 | 0.79 | 0.73 | 0.44 | 0.73 | 0.43 |

### 6.10 The pipeline on the plain seed (no layer weighting), from the same run (owner's question)

Frozen class-word attribution with uniform layer weights (W0), step by step. BRACE is the last row. Step Δs are plain differences.

| Step | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|
| CA seed only | 0.714 | 0.562 | 0.588 | 0.444 |
| fine-only propagation | 0.805 | 0.631 | 0.780 | 0.648 |
| capped hierarchy | 0.812 | 0.660 | 0.813 | 0.660 |
| + DenseCRF | 0.825 | 0.663 | 0.826 | 0.673 |
| BRACE (guided weights) | 0.830 | 0.697 | 0.830 | 0.697 |

### 6.11 Reading

- No background-word seed beats BRACE on either benchmark; the registered pick loses on COCO-Cap.
- The background words do help where intended:
  - small objects +0.014 to +0.021;
  - flooded images 11.2% → 8.9% (BNu);
  - COCO-Cap person 0.28 → up to 0.47.
- They lose as much on medium and large objects, and they break two classes:
  - person on VOC-Sim, 0.81 → 0.39 with NERVE's words;
  - tv on COCO-Cap, 0.79 → 0.44 whenever the start token is removed.
- SD's own background-word keys do not give the absolute scale that CLIP's softmax gives.
- The hierarchy adds about +0.08 to +0.13 over the raw seed for every seed.
- The hierarchy's gain over fine-only is small on VOC-Sim (−0.008 to +0.030) and +0.011 to +0.039 on COCO-Cap. That COCO-Cap number again depends on each map using its own VOC-selected threshold (Section 3).

---

## 7. Side N's independent sanity reruns (context only; Narmeen's agent, not audited by me)

Threshold 0.5, no DenseCRF, each method from its own released code:

| Method | VOC-Sim | COCO-Cap |
|---|---:|---:|
| GoCA SD 1.5 | 0.671 | 0.452 |
| GoCA SDXL | 0.788 | 0.607 |
| DiffSegmenter | 0.713 | 0.653 |
| NERVE, variant A (class word + VOC background words) | 0.830 | 0.729 |
| NERVE, variant V | 0.824 | 0.743 |
| NERVE, variant W | 0.810 | 0.729 |

- NERVE uses CLIP ViT-L/14.
- SPARK: no-go. GoCA SD 2.1: blocked (official weights removed).

---

## 8. Summary of verdicts

| Run | Kind | Verdict |
|---|---|---|
| 14A green-screen tokens | registered screen | STOP |
| 15 token ladder | registered | with the token, hierarchy / weights / cap all fail on COCO-Cap; class word reproduces 0.830 / 0.696 |
| 16 +0.034 trace | post hoc | +0.034 = two maps at different thresholds; +0.012 at the same 0.55; +0.018 at each map's best |
| 17 seed × propagation swap | registered | mixed: NERVE's COCO-Cap lead is the CLIP seed with its walk together (interaction +0.052) |
| swap breakdown | exploratory | the gap is in small objects and flooded maps; the CA seed is the bottleneck |
| 18 background-aware seed | registered | FAIL (−0.019 on COCO-Cap) |

## 9. Questions for the reviewer

1. Is the reading of runs/16 fair: is comparing two maps at their own VOC-selected thresholds a legitimate "transfer" claim, or should the paper report the same-threshold number (+0.012, not significant) as primary?
2. Given runs/15 and 18, is it defensible to keep claiming the hierarchy and the layer weighting as contributions, and how should the threshold dependence be stated?
3. In runs/17, is the "mixed" verdict the right reading, or does the interaction mean something more specific about why NERVE's walk needs a CLIP-like seed?
4. Runs/18 closes Stable-Diffusion-only seed competition with background words. Is there a seed-calibration idea we have missed for the flooded small-object failure, other than using CLIP? Already closed: per-image thresholds (Otsu and five other rules), generic-word subtraction, synonyms, caption-based attribution, embedding-space purification, green-screen tokens, background-key softmax.
5. Anything in the conventions, the bootstrap, or the pre-registration handling that looks wrong to you.
