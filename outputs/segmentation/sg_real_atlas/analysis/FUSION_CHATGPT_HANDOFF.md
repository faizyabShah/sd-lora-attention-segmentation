# Handoff: late and early fusion of the cross-attention seed and the CLIP seed in BRACE (generated sets, 2026-11-10)

## Context
- **BRACE** = cross-attention (A3) seed → capped 16→32→64 self-attention hierarchy (V3) → DenseCRF R2, frozen at τ 0.55.
- **CLIP-seeded BRACE** = the same propagation, but seeded with the SCLIP ViT-B/16 VOC21 target-class probability (min-max).
  - The seed setup was fixed on real photos before any generated-set run.
  - It is accepted as the best paper variant: VOC-Sim 0.845, COCO-Cap 0.717, vs BRACE 0.830 / 0.696.
- **Question:** does combining the two seeds beat CLIP-seeded BRACE?
- Two fusion tests, each pre-registered and committed before scoring:
  - **Late:** fuse the two propagated maps.
  - **Early:** fuse the two seeds, then propagate.
- **Shared rule for both tests:**
  1. Choose the candidate on VOC-Sim at τ 0.55 + R2.
  2. Test it on COCO-Cap against CLIP-seeded BRACE.
  3. The image-bootstrap 95% CI must be > 0.
- Every comparison is paired within one regeneration per image.

## Summary (τ 0.55 + R2, class-macro mIoU)

| variant | VOC-Sim | COCO-Cap | COCO-Cap Δ vs CLIP-seeded (95% CI) |
|---|---:|---:|---|
| BRACE (cross-attention seed) | 0.830 | 0.696 | −0.020 (−0.045…+0.004) |
| CLIP-seeded BRACE | 0.845 | 0.717 | — |
| late: mean of maps (**registered pick**) | 0.850 | 0.728 | +0.012 (−0.009…+0.034) |
| late: geometric mean of maps | 0.846 | 0.734 | +0.017 (+0.002…+0.033) |
| late: min of maps | 0.829 | 0.726 | +0.009 (−0.002…+0.019) |
| early: mean seed (**registered pick**) | 0.847 | 0.725 | +0.008 (−0.013…+0.031) |
| early: geometric-mean seed | 0.844 | **0.741** | +0.024 (+0.010…+0.040) |
| early: product seed (CLIP-gated) | 0.836 | 0.739 | +0.022 (+0.010…+0.034) |
| early: min seed | 0.835 | 0.735 | +0.018 (+0.010…+0.026) |

## Verdict
- **Both registered tests fail.** The mean fusion was chosen on VOC-Sim each time, and it does not beat CLIP-seeded BRACE on COCO-Cap. CLIP-seeded BRACE stays the claimable best variant.
- **A consistent post-hoc pattern:**
  - VOC-Sim is saturated: there is one object per prompt, and all variants are within about ±0.01.
  - On COCO-Cap (several objects per image), agreement-style fusions win, with CIs > 0: the geometric mean, the product and the min.
  - The best is the early geometric-mean seed: 0.741, which is +0.045 over BRACE and +0.024 over CLIP-seeded BRACE.
  - Person goes from 0.28 (BRACE) to 0.66 (CLIP-seeded) to 0.68 (early geometric mean).
- **Interpretation:** the two seeds are complementary. Cross-attention is better on aeroplane, tv, potted plant and train; CLIP is better on person, chair and dining table. Agreement fusion helps when other objects compete.
- **The selection rule picked the wrong family** because VOC-Sim cannot separate the candidates.
- **The geometric-mean fusion is not claimable.** It would need a fresh annotated multi-object set for confirmation.
- **Suggested paper use:** report CLIP-seeded BRACE as the headline. The fusion rows can go in as an exploratory ablation, clearly labelled.

## Caveats
- **Regeneration is not bit-exact** for about half the images, because of GPU nondeterminism under shared load (mean |Δpixel| about 0.6–0.7/255). BRACE re-scored on each capture reproduces 0.830 / 0.696. All rows of a table come from the same capture.
- **CLIP is SCLIP ViT-B/16 only.**
- **The protocol is class-given and binary,** as in the paper.

## Questions
1. Is the post-hoc pattern worth reporting as an exploratory ablation, or does it invite "garden of forking paths" criticism?
2. Is there a defensible confirmation set for the geometric-mean fusion without new annotation?
3. Framing: does "seed-agnostic propagation, two instantiations (self-contained vs CLIP-seeded)" hold up against CLIPer/NERVE?

---

# A. Late fusion — pre-registration


Written 2026-11-09, before any fused map has been scored. Code: `scripts/sg_gen_capab.py fuse-score / fuse-report`.

**Inputs.** Both maps come from the stored cap-ablation capture (`/home/shafa/sg_comp/gen_capab`), one capture per image:
- **a:** the capped-hierarchy map from the cross-attention seed (BRACE).
- **c:** the same map from the SCLIP ViT-B/16 VOC21 target-probability seed (CLIP-seeded BRACE).

Each map is min-max normalised at 64×64.

**Candidates.** All are parameter-free. Each fused map is min-max normalised again, upsampled, thresholded and passed through the frozen DenseCRF R2, exactly like BRACE.

| name | definition |
|---|---|
| geo | sqrt(a · c) |
| mean | (a + c) / 2 |
| min | min(a, c) |

**Settings.** Frozen: τ = 0.55 and the R2 CRF. Nothing is tuned.

**Rule.**
1. On VOC-Sim, choose the candidate with the highest class-macro mIoU at τ 0.55 + R2.
2. On COCO-Cap, compare it with CLIP-seeded BRACE (c alone, τ 0.55 + R2).
3. The image-bootstrap 95% CI (1000 resamples) must be > 0 for "fusion helps".

**Secondary** (reported, not pass/fail):
- every candidate vs BRACE and vs CLIP-seeded BRACE on both sets;
- out-of-fold τ results;
- precision / recall;
- per class.

**If the rule fails,** CLIP-seeded BRACE stays the best variant, and late fusion is not used.

---

# B. Late fusion — full report


Pre-registered in `gen_fusion_prereg.md`, committed before scoring. a, c = the capped-hierarchy maps of the two seeds from one capture per image; fused map min-max normalised, τ 0.55, then the frozen DenseCRF R2. Δ with image-bootstrap 95% CI.

## VOC-Sim (490 images)

| map | mIoU @0.55 + R2 | prec / rec | Δ vs a (BRACE) | Δ vs c (CLIP-seeded) | OOF + R2 | raw @0.55 |
|---|---:|---|---|---|---:|---:|
| geo: sqrt(a·c) | 0.846 | 0.897 / 0.930 | +0.016 (+0.007…+0.025) | +0.001 (-0.006…+0.009) | 0.846 | 0.833 |
| mean: (a+c)/2 | 0.850 | 0.896 / 0.934 | +0.020 (+0.010…+0.028) | +0.005 (-0.003…+0.013) | 0.848 | 0.836 |
| min(a, c) | 0.829 | 0.909 / 0.891 | -0.001 (-0.009…+0.007) | -0.015 (-0.025…-0.006) | 0.834 | 0.818 |
| a = BRACE (cross-attention seed) | 0.830 | 0.890 / 0.910 | — | -0.015 (-0.027…-0.001) | 0.830 | 0.817 |
| c = CLIP-seeded BRACE | 0.845 | 0.893 / 0.933 | +0.015 (+0.001…+0.027) | — | 0.845 | 0.832 |

Per class @0.55 + R2 (a / c / geo / mean / min): aeroplane 0.88 / 0.91 / 0.90 / 0.90 / 0.90, bicycle 0.79 / 0.80 / 0.80 / 0.80 / 0.80, bird 0.92 / 0.93 / 0.93 / 0.93 / 0.93, boat 0.86 / 0.85 / 0.84 / 0.87 / 0.86, bottle 0.88 / 0.85 / 0.87 / 0.87 / 0.85, bus 0.92 / 0.91 / 0.92 / 0.91 / 0.92, car 0.85 / 0.86 / 0.84 / 0.83 / 0.83, cat 0.85 / 0.92 / 0.90 / 0.88 / 0.85, chair 0.79 / 0.79 / 0.79 / 0.79 / 0.80, cow 0.90 / 0.96 / 0.96 / 0.96 / 0.91, dining-table 0.42 / 0.44 / 0.45 / 0.44 / 0.45, dog 0.93 / 0.94 / 0.94 / 0.94 / 0.93, horse 0.90 / 0.93 / 0.93 / 0.93 / 0.91, motorbike 0.87 / 0.86 / 0.87 / 0.87 / 0.87, person 0.81 / 0.85 / 0.84 / 0.85 / 0.82, potted-plant 0.72 / 0.75 / 0.77 / 0.78 / 0.70, sheep 0.75 / 0.97 / 0.89 / 0.92 / 0.79, sofa 0.96 / 0.90 / 0.95 / 0.95 / 0.94, train 0.74 / 0.80 / 0.79 / 0.79 / 0.82, tv 0.85 / 0.68 / 0.75 / 0.78 / 0.70

## COCO-Cap (707 images)

| map | mIoU @0.55 + R2 | prec / rec | Δ vs a (BRACE) | Δ vs c (CLIP-seeded) | OOF + R2 | raw @0.55 |
|---|---:|---|---|---|---:|---:|
| geo: sqrt(a·c) | 0.734 | 0.835 / 0.868 | +0.038 (+0.019…+0.055) | +0.017 (+0.002…+0.033) | 0.737 | 0.721 |
| mean: (a+c)/2 | 0.729 | 0.802 / 0.899 | +0.032 (+0.020…+0.044) | +0.012 (-0.009…+0.034) | 0.734 | 0.715 |
| min(a, c) | 0.726 | 0.868 / 0.827 | +0.029 (+0.007…+0.051) | +0.009 (-0.002…+0.019) | 0.726 | 0.714 |
| a = BRACE (cross-attention seed) | 0.696 | 0.747 / 0.886 | — | -0.020 (-0.045…+0.005) | 0.707 | 0.682 |
| c = CLIP-seeded BRACE | 0.717 | 0.837 / 0.855 | +0.020 (-0.005…+0.045) | — | 0.713 | 0.705 |

Per class @0.55 + R2 (a / c / geo / mean / min): aeroplane 0.80 / 0.71 / 0.80 / 0.80 / 0.84, bicycle 0.63 / 0.73 / 0.70 / 0.69 / 0.73, bird 0.78 / 0.83 / 0.83 / 0.81 / 0.83, boat 0.59 / 0.59 / 0.64 / 0.64 / 0.63, bottle 0.50 / 0.58 / 0.66 / 0.54 / 0.62, bus 0.92 / 0.85 / 0.88 / 0.91 / 0.87, car 0.76 / 0.74 / 0.75 / 0.81 / 0.74, cat 0.87 / 0.86 / 0.86 / 0.87 / 0.86, chair 0.47 / 0.63 / 0.53 / 0.51 / 0.63, cow 0.87 / 0.90 / 0.90 / 0.90 / 0.88, dining table 0.45 / 0.55 / 0.49 / 0.49 / 0.40, dog 0.75 / 0.77 / 0.77 / 0.77 / 0.77, horse 0.72 / 0.76 / 0.76 / 0.77 / 0.78, motorbike 0.67 / 0.69 / 0.68 / 0.70 / 0.69, person 0.28 / 0.66 / 0.58 / 0.52 / 0.49, potted plant 0.74 / 0.63 / 0.76 / 0.74 / 0.62, sheep 0.87 / 0.90 / 0.90 / 0.90 / 0.89, sofa 0.70 / 0.73 / 0.74 / 0.75 / 0.80, train 0.75 / 0.66 / 0.75 / 0.75 / 0.77, tv 0.79 / 0.59 / 0.69 / 0.70 / 0.66

## Registered test

Chosen on VOC-Sim: **mean** (0.850 vs c 0.845). COCO-Cap: mean 0.729 vs CLIP-seeded BRACE 0.717, Δ +0.012 (CI -0.009…+0.034) → **no gain over CLIP-seeded BRACE**.

---

# C. Early fusion — pre-registration


Written 2026-11-09, before any early-fused map has been computed. Code: `scripts/sg_gen_capab.py early-capture / early-score / early-report`.

**Seeds.** Each image is regenerated with the frozen live pipeline. Two seeds, each min-max normalised at 64×64:
- **a:** the cross-attention (A3) seed, as in BRACE.
- **c:** the SCLIP ViT-B/16 VOC21 target-class probability, as in CLIP-seeded BRACE.

**Candidates.** Each fused seed is min-max normalised and then propagated through the image's own self-attention with the frozen BRACE capped hierarchy (V3). Then τ = 0.55 and the frozen DenseCRF R2. All candidates are parameter-free.

| name | fused seed |
|---|---|
| e_geo | sqrt(a · c) |
| e_mean | (a + c) / 2 |
| e_min | min(a, c) |
| e_prod | a · c (CLIP-gated cross-attention) |

**References,** recomputed in the same capture so every comparison is paired:
- BRACE (seed a);
- CLIP-seeded BRACE (seed c);
- late mean fusion of their propagated maps, the variant chosen in `gen_fusion_prereg.md`.

**Rule.**
1. On VOC-Sim, choose the candidate with the highest class-macro mIoU at τ 0.55 + R2.
2. On COCO-Cap, compare it with CLIP-seeded BRACE (seed c, τ 0.55 + R2).
3. The image-bootstrap 95% CI (1000 resamples) must be > 0 for "early fusion helps".

**Secondary** (reported, not pass/fail):
- every candidate vs BRACE, vs CLIP-seeded BRACE and vs the late mean fusion;
- out-of-fold τ;
- raw maps;
- precision / recall;
- per class.

**If the rule fails,** CLIP-seeded BRACE stays the best variant.

---

# D. Early fusion — full report


Pre-registered in `gen_early_fusion_prereg.md`, committed before the capture. One regeneration per image; every seed is propagated through that run's self-attention with the frozen BRACE capped hierarchy, then τ 0.55 and the frozen DenseCRF R2. a = the cross-attention seed exactly as in BRACE; fused seeds use min-max a and c. Δ with image-bootstrap 95% CI.

## VOC-Sim (490 images; 238 regenerated bit-identically)

| seed / map | mIoU @0.55 + R2 | prec / rec | Δ vs BRACE | Δ vs CLIP-seeded | Δ vs late mean | OOF + R2 | raw @0.55 |
|---|---:|---|---|---|---|---:|---:|
| BRACE (cross-attention seed) | 0.830 | 0.890 / 0.910 | — | -0.015 (-0.027…-0.001) | -0.020 (-0.028…-0.010) | 0.830 | 0.817 |
| CLIP-seeded BRACE | 0.845 | 0.893 / 0.933 | +0.015 (+0.001…+0.027) | — | -0.005 (-0.013…+0.003) | 0.845 | 0.832 |
| late mean fusion of a and c maps | 0.850 | 0.896 / 0.934 | +0.020 (+0.010…+0.028) | +0.005 (-0.003…+0.013) | — | 0.848 | 0.836 |
| early: seed sqrt(a·c) | 0.844 | 0.896 / 0.927 | +0.014 (+0.005…+0.023) | -0.001 (-0.009…+0.008) | -0.005 (-0.010…-0.002) | 0.842 | 0.831 |
| early: seed (a+c)/2 | 0.847 | 0.893 / 0.934 | +0.017 (+0.009…+0.026) | +0.003 (-0.006…+0.012) | -0.002 (-0.005…+0.001) | 0.849 | 0.834 |
| early: seed min(a, c) | 0.835 | 0.903 / 0.905 | +0.005 (-0.003…+0.012) | -0.010 (-0.020…-0.000) | -0.015 (-0.022…-0.008) | 0.833 | 0.823 |
| early: seed a·c (CLIP-gated) | 0.836 | 0.919 / 0.893 | +0.006 (-0.003…+0.014) | -0.009 (-0.020…+0.001) | -0.014 (-0.022…-0.006) | 0.841 | 0.826 |

Per class @0.55 + R2 (a / c / l_mean / e_geo / e_mean / e_min / e_prod): aeroplane 0.88 / 0.91 / 0.90 / 0.90 / 0.90 / 0.90 / 0.90, bicycle 0.79 / 0.80 / 0.80 / 0.80 / 0.80 / 0.80 / 0.80, bird 0.92 / 0.93 / 0.93 / 0.93 / 0.93 / 0.93 / 0.93, boat 0.86 / 0.85 / 0.87 / 0.85 / 0.87 / 0.85 / 0.85, bottle 0.88 / 0.85 / 0.87 / 0.87 / 0.88 / 0.86 / 0.82, bus 0.92 / 0.91 / 0.91 / 0.91 / 0.91 / 0.92 / 0.92, car 0.85 / 0.86 / 0.83 / 0.83 / 0.84 / 0.83 / 0.82, cat 0.85 / 0.92 / 0.88 / 0.88 / 0.88 / 0.86 / 0.85, chair 0.79 / 0.79 / 0.79 / 0.79 / 0.79 / 0.80 / 0.80, cow 0.90 / 0.96 / 0.96 / 0.96 / 0.96 / 0.93 / 0.92, dining-table 0.42 / 0.44 / 0.44 / 0.45 / 0.44 / 0.45 / 0.51, dog 0.93 / 0.94 / 0.94 / 0.94 / 0.94 / 0.93 / 0.92, horse 0.90 / 0.93 / 0.93 / 0.94 / 0.93 / 0.92 / 0.92, motorbike 0.87 / 0.86 / 0.87 / 0.87 / 0.86 / 0.87 / 0.87, person 0.81 / 0.85 / 0.85 / 0.86 / 0.85 / 0.86 / 0.83, potted-plant 0.72 / 0.75 / 0.78 / 0.77 / 0.77 / 0.73 / 0.72, sheep 0.75 / 0.97 / 0.92 / 0.87 / 0.87 / 0.80 / 0.79, sofa 0.96 / 0.90 / 0.95 / 0.95 / 0.94 / 0.96 / 0.96, train 0.74 / 0.80 / 0.79 / 0.78 / 0.78 / 0.79 / 0.85, tv 0.85 / 0.68 / 0.78 / 0.74 / 0.80 / 0.70 / 0.72

## COCO-Cap (707 images; 243 regenerated bit-identically)

| seed / map | mIoU @0.55 + R2 | prec / rec | Δ vs BRACE | Δ vs CLIP-seeded | Δ vs late mean | OOF + R2 | raw @0.55 |
|---|---:|---|---|---|---|---:|---:|
| BRACE (cross-attention seed) | 0.696 | 0.747 / 0.885 | — | -0.020 (-0.045…+0.004) | -0.032 (-0.044…-0.020) | 0.707 | 0.682 |
| CLIP-seeded BRACE | 0.717 | 0.837 / 0.855 | +0.020 (-0.004…+0.045) | — | -0.012 (-0.034…+0.009) | 0.713 | 0.705 |
| late mean fusion of a and c maps | 0.728 | 0.802 / 0.899 | +0.032 (+0.020…+0.044) | +0.012 (-0.009…+0.034) | — | 0.734 | 0.715 |
| early: seed sqrt(a·c) | 0.741 | 0.837 / 0.879 | +0.045 (+0.025…+0.063) | +0.024 (+0.010…+0.040) | +0.013 (-0.004…+0.027) | 0.746 | 0.729 |
| early: seed (a+c)/2 | 0.725 | 0.792 / 0.907 | +0.028 (+0.018…+0.040) | +0.008 (-0.013…+0.031) | -0.004 (-0.010…+0.002) | 0.739 | 0.712 |
| early: seed min(a, c) | 0.735 | 0.851 / 0.861 | +0.038 (+0.015…+0.060) | +0.018 (+0.010…+0.026) | +0.006 (-0.014…+0.025) | 0.735 | 0.723 |
| early: seed a·c (CLIP-gated) | 0.739 | 0.878 / 0.839 | +0.043 (+0.018…+0.066) | +0.022 (+0.010…+0.034) | +0.011 (-0.011…+0.030) | 0.742 | 0.729 |

Per class @0.55 + R2 (a / c / l_mean / e_geo / e_mean / e_min / e_prod): aeroplane 0.80 / 0.71 / 0.80 / 0.80 / 0.79 / 0.82 / 0.87, bicycle 0.63 / 0.73 / 0.69 / 0.71 / 0.68 / 0.73 / 0.74, bird 0.78 / 0.83 / 0.81 / 0.83 / 0.79 / 0.83 / 0.82, boat 0.59 / 0.59 / 0.64 / 0.65 / 0.61 / 0.64 / 0.62, bottle 0.50 / 0.57 / 0.54 / 0.67 / 0.51 / 0.59 / 0.64, bus 0.92 / 0.85 / 0.91 / 0.88 / 0.92 / 0.87 / 0.88, car 0.76 / 0.74 / 0.81 / 0.75 / 0.78 / 0.76 / 0.71, cat 0.87 / 0.86 / 0.87 / 0.86 / 0.87 / 0.86 / 0.85, chair 0.47 / 0.63 / 0.52 / 0.54 / 0.50 / 0.62 / 0.65, cow 0.87 / 0.90 / 0.90 / 0.90 / 0.90 / 0.88 / 0.87, dining table 0.45 / 0.55 / 0.49 / 0.57 / 0.55 / 0.55 / 0.50, dog 0.75 / 0.77 / 0.77 / 0.77 / 0.77 / 0.77 / 0.76, horse 0.72 / 0.76 / 0.77 / 0.76 / 0.77 / 0.76 / 0.79, motorbike 0.67 / 0.69 / 0.70 / 0.69 / 0.69 / 0.69 / 0.70, person 0.28 / 0.66 / 0.52 / 0.68 / 0.54 / 0.66 / 0.69, potted plant 0.74 / 0.63 / 0.74 / 0.79 / 0.73 / 0.64 / 0.63, sheep 0.87 / 0.90 / 0.90 / 0.90 / 0.90 / 0.90 / 0.90, sofa 0.70 / 0.73 / 0.75 / 0.72 / 0.72 / 0.75 / 0.80, train 0.75 / 0.66 / 0.75 / 0.72 / 0.71 / 0.74 / 0.74, tv 0.79 / 0.59 / 0.70 / 0.64 / 0.74 / 0.60 / 0.63

## Registered test

Chosen on VOC-Sim: **e_mean** (0.847 vs CLIP-seeded 0.845). COCO-Cap: e_mean 0.725 vs CLIP-seeded BRACE 0.717, Δ +0.008 (CI -0.013…+0.031) → **no gain over CLIP-seeded BRACE**.
