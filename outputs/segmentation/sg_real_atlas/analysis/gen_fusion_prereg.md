# Pre-registration: late fusion of the cross-attention-seeded and CLIP-seeded BRACE maps (generated sets)

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
