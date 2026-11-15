# Pre-registration: grouped run of ideas 1, 3, 4, 6, 7 on VOC-Sim and COCO-Cap

Written 2026-11-15, before any of these maps has been computed. Code: `scripts/ideas_grouped.py`.

## Why grouped

All five signals come from the same generation. One frozen live capture (for BRACE) and one base trajectory per image are shared, so every candidate is compared with BRACE inside the same process and capture pair.

This run **replaces** the separate idea-1 run (`runs/01_knockout`), which was never executed. Its definitions, candidates and rule (`runs/01_knockout/PREREG.md`) are kept unchanged here as group 1.

**Idea 5 (self-verification) is not included.** The composition criterion first sketched for it is trivially minimised at the extreme thresholds (an all-object or all-background composite is a natural image), so it needs its own design.

## Shared generation

These parts are exactly as in `runs/01_knockout/PREREG.md`:
- frozen SD1.5 settings: same seed and prompt, 30 steps (31 PNDM timesteps), CFG 7.5, fp32 UNet;
- a manual loop equivalent to `pipe()`, with no safety checker;
- the base trajectory with a cross-attention processor that is a no-op;
- the label-free target-token rule: the class word, or on COCO-Cap the fallback token most similar to the A3 seed.

**Recorded on the base trajectory:**
- the latents z_i;
- the scheduler state at steps 5 and 10;
- the unconditional and conditional noise predictions ε_u, ε_c;
- the x₀ prediction x̂₀,i = (z_i − √(1−ᾱ_t) ε_g) / √ᾱ_t, where ε_g = ε_u + 7.5 (ε_c − ε_u);
- the mean conditional token attention over heads, layers and steps at 16×16 and 32×32.

## Post-processing shared by the effect maps (groups 1, 4, 7)

1. Gaussian smoothing on the 64 grid, σ = 1.0.
2. Bilinear upsampling to 512.
3. Robust normalisation: clip((m − p1) / (p99 − p1), 0, 1).

Every candidate is scored raw on the τ grid 0.30–0.90 and with the frozen DenseCRF R2 (τ step 0.05) on the base-loop image.

## Groups and candidates

### Group 1: knockout (idea 1)

8 candidates, as pre-registered in `runs/01_knockout/PREREG.md`:
- `attn_fin_k5`, `attn_fin_k10`, `attn_ins_k5`, `attn_ins_k10`;
- `swap_fin_k5`, `swap_fin_k10`, `swap_ins_k5`, `swap_ins_k10`.

### Group 3: chrono-segmentation (idea 3)

1. **Features.** For each of the 4096 latent pixels, take its x̂₀ trajectory over all 31 steps (31 × 4 = 124 dimensions) and z-score each dimension across pixels.
2. **Clustering.** k-means with K ∈ {4, 8, 16}: k-means++ initialisation (torch generator seed 0), 50 Lloyd iterations, 4 restarts, lowest inertia kept.
3. **Map.** Each pixel gets its cluster's mean of the min–max-normalised A3 seed. The map is then min–max normalised and upsampled bilinearly.

Candidates: `chrono_K4`, `chrono_K8`, `chrono_K16`.

### Group 4: Jacobian propagation (idea 4)

At base steps i ∈ {10, 20}:
1. Draw 4 probes u_r = mm(A3 seed) ⊙ g_r, where g_r ~ N(0, I) of size 4×64×64 (torch generator seed 0).
2. Compute the vector–Jacobian product v_r = (∂ε_c(z_i)/∂z_i)ᵀ u_r. This is the conditional noise prediction only; the identity term of x̂₀ is excluded.
3. The map is the mean over r of the per-pixel channel-L2 norm of v_r, then the shared post-processing.

Candidates: `jac_s10`, `jac_s20`.

### Group 6: optimal-transport assignment (idea 6)

1. **Columns.** At 32×32, use one column per content token of the generation prompt (content positions as in `sg24_ca.content_positions`), with the target tokens merged into a single column. Add one background column: the sum of all other tokens, including SOT, EOT and padding. Renormalise each pixel's row to sum to 1 (Q).
2. **Cost and marginals.**
   - Cost: −log(Q + 1e−8).
   - Pixel marginals: uniform.
   - Target column mass: â = fraction of mm(A3 seed) ≥ 0.5, clipped to [0.02, 0.98].
   - The remaining 1 − â is shared among the other columns in proportion to their mean Q.
3. **Solver.** Log-domain Sinkhorn, ε = 0.05, 200 iterations.
4. **Map.** The per-pixel target probability Π[:, target] / (1/1024).

Candidates:
- `ot`: the map itself, bilinear to 512 and min–max normalised;
- `ot_v3`: the map bilinear to 64, used as the seed of the frozen capped hierarchy (V3) with the live capture's self-attention, then min–max normalised and upsampled.

### Group 7: per-pixel denoising contrast (idea 7)

G_k = Σ over steps i ≥ k of the per-pixel channel-L2 norm of ε_c(z_i) − ε_u(z_i) on the base trajectory, then the shared post-processing.

Candidates: `cfg_k0`, `cfg_k10`.

The class-specific DiffEdit-style contrast (prompt vs prompt with the class word swapped) is group 1's `swap_ins_*`.

## Comparator

BRACE (frozen V3 + R2) from the frozen live capture of the same image in the same process. Fixed τ 0.55, plus OOF τ.

## Rule (per group)

1. On VOC-Sim, choose the group's candidate with the highest **OOF + R2** class-macro mIoU (τ chosen on 4 folds, crc32(image_id) % 5).
2. On COCO-Cap, compare that candidate with BRACE OOF + R2 using the image bootstrap (1000 resamples).
3. **Claim threshold.** Five groups are tested together, so the claim "<group> beats BRACE" requires the **Bonferroni 99% CI** (0.5–99.5 percentile) to be > 0. The 95% CI is also reported. For group 1, the original idea-1 rule (95% CI > 0) is also reported as written.

## Secondary (reported, not pass/fail)

- every candidate vs BRACE on both sets;
- raw (no CRF);
- @0.55 + R2;
- precision / recall;
- per class;
- COCO-Cap token-rule split;
- base-loop vs live pixel difference;
- runtime.

## If a group fails

It is reported as an exploratory signal. Nothing is promoted.
