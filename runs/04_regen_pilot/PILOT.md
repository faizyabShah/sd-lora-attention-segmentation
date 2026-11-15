# Pilot: attention-constrained regeneration as a test of candidate masks (exploratory)

Written 2026-11-15, before any regeneration has been run. Code: `scripts/regen_pilot.py`. This is a diagnostic pilot, not a registered test. Its only purpose is to decide whether the idea deserves a pre-registered screen.

## Question

Take a generated image and its exact generation trajectory. Replay the trajectory with the class token's cross-attention forced to follow a candidate mask M. Is the error between the replayed image and the original image **lowest for correct masks**?

The pilot also checks the alternative: that the error only reflects mask area, intervention strength, or SD's own attention support.

## Images

- **Set.** 20 images from the class-stratified screening subsets (`ideas_grouped.screen_ids`): 10 VOC-Sim and 10 COCO-Cap, listed in `pilot_ids.json`.
- **Selection.** Deterministic, using the stored paper capture:
  - Only objects covering 1–90% of the image count, because the shift, flip and full controls are degenerate otherwise.
  - 5 images per set have the largest gain from an oracle threshold over τ 0.55, with distinct classes. These are cases where a fixed τ is clearly wrong.
  - 5 images per set sit at the 10th, 30th, 50th, 70th and 90th percentile of object area among the remaining images.
- **Use of ground truth.** Selection uses ground truth, which is acceptable for a diagnostic. These 20 images are therefore **excluded from any later confirmation set**.

## Procedure (fixed; not tuned after the run)

### Generation

1. Frozen live capture → BRACE V3 map → DenseCRF R2 masks at τ ∈ {0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75}. These are the candidates a method could choose from.
2. Base trajectory: a manual PNDM loop equal to `pipe()` (30 steps, 31 timesteps, CFG 7.5, same seed and prompt). It keeps the latent and the scheduler state before step **K0 = 8**.
3. Class token: the class word, or on COCO-Cap the label-free fallback token (`idea2_dose.choose_tokens`).

### Intervention

For each candidate, replay steps 8–30 from the stored latent. The edit applies to every cross-attention layer and head, on the conditional branch only, and keeps every probability row normalised to 1:
- **Outside M:** the class-token probability is set to 0, and the other tokens are rescaled to sum to 1.
- **Inside M:** the class-token probability is set to max(p, ρ). Here ρ is the 90th percentile of p over the layer's pixels, computed per head and per call from the row's own unedited probabilities. The other tokens are rescaled to sum to 1 − p′. For multi-piece class words, the pieces keep their original shares.
- **Pooling:** M is pooled to each layer's resolution (8, 16, 32, 64) by area averaging and ≥ 0.5. The same rule is used for every candidate.
- **Unconditional branch:** unchanged.

### Measurements

- **D (primary):** whole-image squared error of the decoded replay against the decoded unconstrained base image, summed over all pixels and channels, with values in [0, 1].
- **Secondary:** final-latent squared error.
- **Intervention magnitude:** the mean KL(p′ ‖ p) of the edit and the mean |Δ class-token mass|, over layers, steps and pixels.
- **Candidate statistics:** IoU with ground truth and area.

### Controls

Ground truth is used only to build these controls and to score them, never by a method:
- `gt`;
- `gt` dilated and eroded by 8 px and by 16 px (8 px is one latent cell);
- `gt` shifted by a quarter of the image (roll);
- `gt` flipped in both axes (same shape and area, elsewhere);
- empty;
- full;
- an unconstrained replay, which must reproduce the base image (D close to 0).

## Pass bars (all four required; fixed now)

1. `gt` has lower D than each of `gt_shift`, `gt_flip`, `gt_dil8` and `gt_ero8` on **at least 15 of 20** images.
2. Over the 7 τ masks, the median within-image Spearman correlation between D and IoU is **≤ −0.5**.
3. The τ with the lowest D beats τ 0.55: mean IoU gain **> 0**, and strictly better on **at least 12 of 20** images.
4. Per image, over all candidates except the replay, regress D on standardised area and KL and take the residual. The median Spearman correlation between that residual and IoU must be **≤ −0.3**.

**If all four pass:** freeze this exact rule (K0, ρ, D) and pre-register a two-stage screen as in `runs/03_grouped/PREREG.md` (amendment 1), excluding these 20 images.

**If any bar fails:** the idea is reported as not viable as specified and closed. There is no search over K0, ρ, distance measures or attention constraints.

The replay D and the per-image scatter of IoU against D (`fig/scatter.png`) are reported in all cases.
