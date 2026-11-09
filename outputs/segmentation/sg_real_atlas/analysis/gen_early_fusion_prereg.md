# Pre-registration: early fusion of the cross-attention seed and the CLIP seed (generated sets)

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
