# Pre-registration: dose–response maps (idea 2) on VOC-Sim and COCO-Cap

Written 2026-11-14, before any dose–response map has been computed. Code: `scripts/idea2_dose.py`.

## Question

Can we segment the target by measuring how strongly each pixel of the generated image responds to turning the class word up or down during generation? This uses no attention map as the segmentation signal.

## Signal

For each image, regenerate it with the frozen SD1.5 settings: same seed, same prompt, 30 steps, CFG 7.5, fp32 UNet.

1. **Base generation:** cross-attention probability to the target token(s) multiplied by w = 1 (P2P-style reweighting, applied after softmax, conditional branch only, no renormalisation). This uses the same custom processor as the other branches, so numerics match.
2. **Dose branches:** w ∈ {0.5, 2.0}, applied at every UNet call from denoising step s on.
3. **Response:**
   - d_w = per-pixel L2 norm of (output_w − output_1);
   - R = mean over w of d_w / |ln w|, which equals the mean of the two d_w here.

**Candidates** (2 output spaces × 2 starts):

| name | output space | start s |
|---|---|---|
| lat_s0 | final latent (4×64×64), bilinear up to 512 | 0 |
| lat_s10 | final latent | 10 |
| pix_s0 | decoded RGB (512×512) | 0 |
| pix_s10 | decoded RGB | 10 |

Each R is min-max normalised, thresholded (τ grid 0.30–0.90) and also passed through the frozen DenseCRF R2 (τ step 0.05).

## Target tokens (label-free)

- **VOC-Sim:** the class-word tokens of the generation prompt "A photograph of a {cls}".
- **COCO-Cap:** the image is generated from the caption, so the class word is often absent.
  - If the class-name tokens occur in the caption, use them.
  - Otherwise, use the single content token of the caption whose cross-attention at 16×16 (base run, conditional branch, mean over heads, layers and steps) has the highest cosine similarity with BRACE's A3 seed (64→16 area-pooled).
  - No ground truth is used.

## Comparator

BRACE (frozen V3 + R2) from the frozen live capture of the same image, run in the same process. Fixed τ 0.55, plus OOF τ for a like-for-like comparison.

## Rule

1. On VOC-Sim, choose the candidate with the highest **OOF + R2** class-macro mIoU (τ chosen on 4 folds, crc32(image_id) % 5).
2. On COCO-Cap, compare that candidate with BRACE OOF + R2. The image-bootstrap 95% CI (1000 resamples) must be > 0 for "dose–response beats BRACE".

## Secondary (reported, not pass/fail)

- every candidate vs BRACE on both sets;
- raw (no CRF);
- precision / recall;
- per class;
- the token rule on COCO-Cap: % class word present vs fallback, and mIoU split by that;
- runtime per image.

## If the rule fails

Dose–response is reported as an exploratory signal. Nothing is promoted.
