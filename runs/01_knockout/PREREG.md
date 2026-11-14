# Pre-registration: counterfactual knockout maps (idea 1) on VOC-Sim and COCO-Cap

Written 2026-11-14, before any knockout map has been computed. Code: `scripts/idea1_knockout.py`.

Idea 2 (dose–response, `runs/02_dose_response`) has already been run and failed. Two design choices here were made in response to it, before any knockout result existed:
- the effect maps are smoothed and robustly normalised (below);
- branches start after the layout has formed (k ≥ 5), not at step 0.

## Question

Which pixels of a generated image **depend causally** on the class word? Knock the word out partway through generation and measure what changes. No attention map is used as the segmentation signal.

## Generation

Use the frozen SD1.5 settings: same seed, same prompt, 30 steps, CFG 7.5, fp32 UNet, default PNDM scheduler.

- **Manual denoising loop.** It is numerically equivalent to `pipe()`. The `selftest` subcommand checks this on the A40 before the run.
- **Base trajectory.** Run with a cross-attention processor that is a no-op at w = 1, and keep the latent z_k at every step.
- **Branches.** Each branch starts at step k from a copy of the base latent z_k and of the scheduler state, and runs to step 30.

## Knockouts (applied from step k on)

| mode | intervention |
|---|---|
| `attn` | cross-attention probability to the target token(s) × 0 (P2P-style, after softmax, conditional half only, no renormalisation) |
| `swap` | conditional text embedding replaced by the embedding of the same token sequence with the target token span replaced by the single token "object" |

## Effect measures

| measure | definition |
|---|---|
| `fin` | per-pixel L2 norm over the 4 channels of z_KO,final − z_base,final (64×64) |
| `ins` | instantaneous effect along the **base** trajectory: Σ over steps i ≥ k of the per-pixel L2 norm of ε_cond,KO(z_i) − ε_cond(z_i) (64×64). No branch is needed. |

## Map post-processing (fixed, label-free)

1. Gaussian smoothing on the 64×64 grid, σ = 1.0.
2. Bilinear upsampling to 512.
3. Robust normalisation: x = clip((m − p1) / (p99 − p1), 0, 1), where p1 and p99 are that image's own percentiles.
4. Threshold on the τ grid 0.30–0.90 (raw), and also pass through the frozen DenseCRF R2 (τ step 0.05) on the base-loop image.

## Candidates

2 modes × 2 measures × k ∈ {5, 10}, 8 in total:

| measure | `attn` | `swap` |
|---|---|---|
| `fin` | `attn_fin_k5`, `attn_fin_k10` | `swap_fin_k5`, `swap_fin_k10` |
| `ins` | `attn_ins_k5`, `attn_ins_k10` | `swap_ins_k5`, `swap_ins_k10` |

## Target tokens (label-free; same rule as idea 2)

- **VOC-Sim:** the class-word tokens of "A photograph of a {cls}".
- **COCO-Cap:**
  - the class-name tokens, if they occur in the caption;
  - otherwise, the single content token whose 16×16 cross-attention (base run, conditional, mean over heads, layers and steps) has the highest cosine similarity with BRACE's A3 seed (64→16 area-pooled).

## Comparator

BRACE (frozen V3 + R2), from the frozen live capture of the same image in the same process. Fixed τ 0.55, plus OOF τ.

## Rule

1. On VOC-Sim, choose the candidate with the highest **OOF + R2** class-macro mIoU (τ chosen on 4 folds, crc32(image_id) % 5).
2. On COCO-Cap, compare that candidate with BRACE OOF + R2. The image-bootstrap 95% CI (1000 resamples) must be > 0 for "knockout beats BRACE".

## Secondary (reported, not pass/fail)

- every candidate vs BRACE on both sets;
- raw (no CRF);
- @0.55 + R2;
- precision / recall;
- per class;
- COCO-Cap token-rule split;
- base-loop vs live-capture pixel difference;
- runtime.

## If the rule fails

Knockout is reported as an exploratory signal. Nothing is promoted.
