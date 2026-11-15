# Pre-registration: does the SD VAE decoder carry sub-cell boundary ownership beyond its footprint and RGB? (pilot)

Written 2026-11-15, before any decoder perturbation has been run. Code: `scripts/vae_owner.py`.

This is a **diagnostic pilot**, not a method test:
- BRACE stays frozen.
- Ground truth only measures.
- Nothing here is tuned after the run: not ε, the directions, the 0.5 rule, the band, or the cell sets.
- If any gate fails, the direction is closed.

## Hypothesis

BRACE decides ownership on a 64 × 64 grid, 8 px per cell. The SD VAE decoder turns each latent cell into its 8 × 8 patch, but it also mixes cells, including through its mid-block self-attention.

If the decoder groups pixels by object, then perturbing **object** latents and **background** latents near a boundary will affect the boundary pixels **on their own side** more than proximity alone predicts.

The failure mode to rule out is **footprint**: the response simply falls off with distance from the perturbed cells. In that case, ownership is a blurred copy of BRACE's mask.

## Images (fixed now)

- **Source.** The rich capture of the class-stratified screening subsets (`CACHE/ana_dump`).
- **Selection.** 15 VOC-Sim and 15 COCO-Cap images:
  - in crc32(image_id) order;
  - with the target object covering 1–90% of the image;
  - excluding the 20 regeneration-pilot images;
  - distinct classes first, then filling in the same order.
- **Exclusion from confirmation.** These 30 images are excluded from any later confirmation set.

## Latent and gate

- **Latent.** The final latent z is regenerated with the frozen base trajectory: the manual PNDM loop equal to `pipe()`, same seed and prompt.
- **Gate 0.** The decoded D(z) must match the stored image to a mean absolute difference ≤ 1 grey level on every image. Otherwise the run stops.

## BRACE mask, band and cell sets (label-free)

- **M:** BRACE's frozen mask. min–max(V3) from the capture, bilinear to 512, τ 0.55, DenseCRF R2.
- **Band:** pixels within 16 px of M's boundary, on either side.
- **Object core cells F:** cells with min–max V3 ≥ 0.9.
- **Clear background cells B:** cells with min–max V3 ≤ 0.1.
- **Ring:** both sets are restricted to cells within Chebyshev distance 4 of a cell that touches the band. Band-touching cells themselves are excluded, since they are never perturbed.
- **Matching.** The larger set is subsampled to the size of the smaller, with a fixed seed of crc32(image_id). F and B therefore have **equal cell counts**, and their mean distances to the band are reported.
- **Exclusion.** Images with fewer than 4 cells in either set are excluded from the gates and reported.

## Perturbation (fixed; equal energy by construction)

- **Directions.** Each perturbed cell q gets an independent 4-channel direction drawn from N(0, I), normalised to unit length and scaled to ε · r̄.
  - r̄ is the image's mean per-cell latent norm.
  - **ε = 0.05.**
  - Every cell receives the same magnitude, and F and B contain the same number of cells, so the total perturbation energy is equal.
- **Response energy.** For a set S ∈ {F, B} and a direction draw v: ΔD_S = D(z + v_S) − D(z), in RGB scaled to [0, 1].
- **Draws.** 16 draws per set, in two independent seed sets of 8. This gives e_S(p) = the mean over draws of Σ_c ΔD_S(p, c)².
- **Ownership:** o(p) = e_F(p) / (e_F(p) + e_B(p) + 1e−12).
  - Symmetric design means 0.5 is indifference.
  - **Repeatability:** report the per-image Spearman correlation of o between the two seed sets over band pixels.

## Footprint control (the decoder's own average kernel)

- **Kernel K.** Isolated cells are perturbed on a 4 × 4 lattice (spacing 16 cells = 128 px), at 4 lattice offsets, 2 direction draws each. That is 8 decodes per image.
- **Measurement.** For each perturbed cell, ΔD² is recorded in a window from −64 to +71 px around the cell's top-left corner. K is the mean of these windows over cells, draws and all 30 images.
- **Footprint energy:** ê_S(p) = Σ_{q∈S} K(p − 8q).
- **Footprint ownership:** o_fp = ê_F / (ê_F + ê_B + 1e−12).

## Gates (all three required; fixed now)

All gates are evaluated on band pixels, with ground truth for scoring only.

**A. Beyond footprint.**
- Per image: AUROC(o) − AUROC(o_fp) for ground-truth object vs background pixels.
- Pass: mean over images **≥ +0.02** and the image-bootstrap 95% CI lower bound **> 0** (2000 resamples).

**B. Beyond RGB and footprint (information diagnostic).**
- Model: out-of-fold logistic regression, L2, standardised features, folds crc32(image_id) % 5. At most 4000 band pixels per image, sampled with a fixed seed.
- Baseline features:
  - Lab distance of the pixel to the mean colour of F's pixels, and to B's pixels;
  - Sobel gradient magnitude (Lab, σ 1);
  - signed distance to M's boundary;
  - logit(o_fp).
- Test model: the baseline plus logit(o).
- Pass: ΔAUROC **≥ +0.05**, with image-bootstrap 95% CI lower bound **> 0**.
- This is an information diagnostic. It does not show the signal is usable without training, which is why gate C exists.

**C. Usable without supervision.**
- **Corrected mask M′:** M outside the band; inside the band, the pixel is object iff o(p) > 0.5.
- Pass requires all of:
  - IoU(M′) > IoU(M) on **at least 20 of 30** images;
  - mean IoU gain **> 0**;
  - mean IoU gain **greater than** that of the same rule applied with o_fp.
- If fewer than 30 images qualify, the 20/30 bar scales to two-thirds of the included images.

## Reported, not gating

- **Within-cell discrimination.** Cells that the ground-truth boundary crosses, with at least 8 object and 8 background pixels, inside the band.
  - The per-cell AUROC of o, o_fp and the colour contrast (Lab distance to B − distance to F), averaged over cells.
  - The same restricted to cells where M is wrong on at least 8 pixels.
  - This is the most direct test of sub-cell information.
- **Per-set breakdown.** VOC-Sim and COCO-Cap separately.
- **Matched-set distances.** Mean distances of F and B to the band.
- **Kernel shape.** The radial profile of K.

## Outcome

- **All three gates pass:** freeze this exact rule and pre-register a full screen on new images.
- **Any gate fails:** close the direction. There is no search over ε, the cell sets, the band width or the rule.

## Amendment 1 (2026-11-15, after a 2-image code test, before the full run)

- **What the test showed.** F and B had equal counts but different distances to the band: on average F was 2.8 cells away and B was 3.6. The decoder's influence kernel falls by roughly a factor of e every 4 px, so this mismatch biases o towards F irrespective of image content. That makes the fixed 0.5 rule (gate C) meaningless.
- **Change.** F and B are now matched **ring by ring**. At each Chebyshev distance d = 1…4 cells from the band-touching cells, both sets get min(|F_d|, |B_d|) cells, sampled with the same fixed seed. Their distance distributions to the band are therefore identical.
- **Unchanged.** Everything else: ε, draws, ownership, footprint control, gates, bars and images.
- **Disclosure.** The 2-image test numbers (in `result_test.md`) are not evidence and are not used.
