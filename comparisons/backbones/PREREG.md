# JC #5: fine-only and BRACE on SD 1.5, SD 2.1-base and SDXL, single noising pass (pre-registration)

Side N, 2026-11-19. Written and fixed **before any capture or scoring** of this experiment. This was side F's
item in `docs/JC_REVIEW_WORKPLAN.md` (Section 2, #5); the user (Narmeen) asked side N to run it today so that
nothing is left as "planned" for the meeting. Side F's committed code is imported unchanged; all new code and
outputs are side N's (`comparisons/backbones/`, `~/TRDP/external/n_side/backbones/`).

**Status label: exploratory.** No row here is confirmatory. The frozen method is not changed; it is mapped onto
two new backbones by the rules below, written before any number exists.

## 1. Setting (JC #5: "on the existing images, adding noise and reading attention in one denoising pass")

- **Images:** the stored OVAM benchmark images and annotations of `comparisons/manifests/`
  (490 VOC-Sim, 707 COCO-Cap), the same items, masks and keys as every N-side comparison row.
- **Encode:** RGB / 127.5 − 1 → VAE `latent_dist.mean` × `scaling_factor` (as `scripts/noise_level.py`).
  SD 1.5 and SD 2.1-base at 512 px. SDXL: 512 → 1024 bicubic upscale (as the GoCA SDXL run).
- **Noise:** one draw ε from `torch.Generator("cpu").manual_seed(0)` with the latent's shape;
  z_t = √ᾱ_t z₀ + √(1 − ᾱ_t) ε, with ᾱ from each backbone's own scheduler config; **t = 100** for every backbone
  (the timestep of the GoCA backbone runs, `config_model_*.py`, t=100). Fixed, not tuned.
- **One UNet call**, conditional only (no classifier-free guidance), fp32 UNet for all three backbones
  (SDXL's cached fp16 weights upcast to fp32). SDXL gets its standard added conditions (pooled text
  embedding; time ids (1024, 1024, 0, 0, 1024, 1024)).
- **Prompt and query:** the same prompt and class-word query as generation-time BRACE in runs/20:
  `scripts/jc_newrerun.rows` (corrected prompts). The UNet is conditioned on the row's generation prompt; the
  cross-attention probabilities are read against the keys of the row's `attribution_prompt` (identical to the
  generation prompt on VOC-Sim; on COCO-Cap the prompt that contains the class word), as the frozen recorder
  does (`src/segmentation/sg24_ca.SG24Recorder.record`). Target positions: the class word's token pieces
  (`find_token_subsequence`), mean over pieces.
- **Difference from generation-time BRACE:** one call at t = 100 instead of CA over 31 calls and SA over calls
  10-20 of a 30-step generation. The SD 1.5 row of this experiment is the like-for-like reference; it is never
  mixed with generation-time numbers in one comparison.

## 2. The frozen method in one call, and the per-backbone mapping

The seed and the two maps are re-implemented for a single call (side F's `aggregate` requires exactly 11 SA
calls and 8 heads at 16/32/64, so it cannot be called unchanged). Functions reused unchanged:
`sg24_ca.affinity_similarity` (layer score), `sg25_sa._apply`, `sg25_sa._resize`, `sg_combined_v2.cap`,
`controlled_cv.normalize_batch` (min-max). The re-implementation is checked before scoring: on SD 1.5, feeding
the single call's SA 11 times to `sg_combined_v2.aggregate` must give the same V0 and V3 as the single-call code
(max abs difference < 1e-5 after min-max).

- **CA seed (A3):** per selected CA module, softmax(q kᵀ · scale) over the 77 tokens (conditional query,
  attribution-prompt keys), class-word columns averaged, mean over all heads, bilinear to 64×64; weighted by the
  module's SA-agreement score (affinity of its paired self-attention vs the pseudo-global affinity of the
  reference query, clamped at 0 and normalised; uniform if all are 0); summed; min-max.
- **Fine-only (FINE-P2, "V0"):** u_h = A_h c, V0 = mean_h A_h u_h, with A the fine SA operator (64×64 grid).
- **BRACE ("V3"):** seed area-resized to the coarsest SA grid, propagated coarse → fine through the SA grids,
  each step bilinear-upsampled into the next grid's operator (mean over heads), with the one-sided cap
  (`sg_combined_v2.cap`, scale 1) at the **first (coarsest) transition only**, as frozen.
- **DenseCRF R2** (`src/segmentation/ovam_full.apply_ovam_densecrf`, frozen OVAM parameters) on both maps.

| | SD 1.5 (512 px) | SD 2.1-base (512 px) | SDXL base 1.0 (1024 px) |
|---|---|---|---|
| CA modules | frozen set: sorted attn2 indices [0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15] (8, 16, 64 px grids) | same indices (identical module layout) | **all 70 attn2 modules** (its only grids are 32 and 64; the SD 1.5 exclusion of the 32 px modules has no analogue) |
| Reference query | `up_blocks.2.attentions.1.transformer_blocks.0.attn2` | same | `up_blocks.1.attentions.0.transformer_blocks.0.attn2` (GoCA's SDXL `ref_layer`, up-level1-repeat0) |
| SA grids (coarse → fine) | 16 → 32 → 64: `up_blocks.{1,2,3}.attentions.2.transformer_blocks.0.attn1` | same modules | 32 → 64: `up_blocks.{0,1}.attentions.2.transformer_blocks.0.attn1` |
| Rule | frozen | same | last attention block of the up level at each grid, first transformer block (the rule the SD 1.5 choice follows) |
| Hierarchy | two steps, cap at 16 → 32 | same | **one step** (workplan: "SDXL has two attention scales, so the hierarchy has one step"), cap at 32 → 64 |
| Heads | 8 | 5 / 10 / 20 / 20 | 10 / 20 |

Weights:
- SD 1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5` (the repo's model).
- SDXL: `stabilityai/stable-diffusion-xl-base-1.0` (the snapshot used by the GoCA SDXL run, fp16 variant).
- SD 2.1-base: `sd2-community/stable-diffusion-2-1-base`, a community copy (Stability removed the official repo).
  Its unet / text_encoder / vae safetensors have the same sha256 as `kaseris/stable-diffusion-2-1-base` (a clone
  of the Stability repo), and its `v2-1_512-ema-pruned.ckpt` sha256 (88ecb782…) matches the LFS pointer in
  Stability's original commit. **The user (Narmeen) approved using this copy for BRACE and fine-only on
  2026-11-19; side F has not confirmed it yet.** The workplan asked for F/N agreement on the weights; this row is
  therefore flagged until F confirms.

## 3. Scoring (fixed now)

- Maps min-max (`sg_steps_test.mm`), bilinear to the annotation size (`sg_steps_test.up`), thresholded on the
  27-value grid of `comparisons/official/common.THR_EXT`; counts with `sg_class_comp.counts`; class-macro mIoU
  (`sg_class_comp.miou`).
- Threshold rules: **VOC-sel.** (VOC-Sim 5-fold OOF, folds crc32(key) % 5; full-VOC-Sim-best threshold applied
  unchanged to COCO-Cap) = primary; **fixed 0.55** (BRACE's frozen threshold, both maps). Raw and DenseCRF are
  reported separately and never mixed in one comparison.
- **Paired Δ** (2000-sample image bootstrap, 95% interval, `sg_class_comp.boot`):
  (a) BRACE − fine-only per backbone (primary question: does the hierarchy hold across backbones?);
  (b) BRACE − GoCA on the same backbone (GoCA's soft maps from the N-side GoCA runs, same images, both at
  VOC-sel., raw and DenseCRF). These are same-image pairs from different captures (different code, same
  stored images and masks), labelled so.
- Runtime and peak GPU memory per image, per backbone (A100, node renyi): the shared UNet pass + seed, and the
  V0 and V3 steps separately.
- No result here changes BRACE or promotes anything. A failure (e.g. the hierarchy losing on a backbone) is
  reported as found.

## Amendment A1 (2026-11-19, before any scoring; no number of this experiment had been computed)

DenseCRF R2 costs 4.5 s per call on one CPU core (measured on a 512 × 512 smoke image), so the planned grid
(3 backbones × 1,197 items × 27 thresholds × 2 maps ≈ 194k calls ≈ 240 CPU-hours) cannot finish today under
the cluster's per-user CPU limit. Change, fixed now:
- **Raw maps:** unchanged (full 27-value grid; VOC-sel. primary, fixed 0.55 also reported).
- **DenseCRF R2:** applied **only at the frozen threshold 0.55** (BRACE's frozen rule: 0.55, then DenseCRF), to
  both maps. The DenseCRF table therefore has fixed-0.55 rows only, and the DenseCRF paired Δ (BRACE − fine-only)
  is at fixed 0.55. No VOC-sel. DenseCRF row is computed.
- **BRACE − GoCA:** primary pair is **raw, VOC-sel.** for both. A DenseCRF pair is reported as BRACE+R2 at 0.55
  vs GoCA+R2 at its VOC-sel. threshold (GoCA's counts already exist on the full grid), labelled as a
  mixed-rule pair.
