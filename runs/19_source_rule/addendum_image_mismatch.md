# Addendum (post hoc diagnostic): why the NERVE cell is 0.720 here and 0.737 in runs/17

Not pre-registered. Scripts (copied to `diag/`): `nerve_gap.py`, `nerve_gap2.py`, `nerve_pass_diag.py`,
`nerve_img_diff.py`, `nerve_matched.py`.

## Cause: the input image, not the code

- NERVE's pass is one forward of the image: VAE mean latent, noise added at scheduler index 45 of 50 (t = 101), empty
  prompt, one UNet call (`diffusion_model/stable_diffusion.py`, `one_step`). runs/19 reimplements exactly this.
- On 12 COCO-Cap images (6 worst, 6 typical), NERVE's own code reproduces runs/17 (correlation 1.00) for noise seeds 0–3,
  on both A100 and V100. The runs/19 reimplementation reproduces runs/19 for noise seeds 0–3, in fp32+autocast and in fp16.
  Noise, precision and GPU do not matter.
- The difference is the image. runs/17 (and side N) run NERVE on the **official benchmark PNG**, the image the
  annotation was drawn on. runs/19 runs every cell on **our live regeneration**, by design (one capture).
  Our regenerations (and the stored paper captures) are not the official images on a subset:

| Set | images | mean abs pixel difference < 1 | ≥ 5 | ≥ 20 | ours all black |
|---|---:|---:|---:|---:|---:|
| COCO-Cap | 707 | 528 | 101 | 38 | 2 (224079 chair, 556255 dog) |
| VOC-Sim | 490 | 405 | 55 | 20 | 0 |

  On differing images the CLIP seed (computed on the official PNG) no longer lines up with the attention of the
  regenerated image, and the annotation describes a different image.

- NERVE cell, CLIP × NERVE walk, 0.5 + DenseCRF: on the 528 matched COCO-Cap images runs/19 = runs/17 = 0.736
  (Δ 0.000); on the 101 differing images 0.659 vs 0.760. VOC-Sim: 0.861 = 0.861 matched.

## Consequence for BRACE vs NERVE (post hoc split, label-free: defined by image difference only)

| COCO-Cap, VOC-sel. thr + DenseCRF | all 707 | matched 528 | differs ≥ 5 (101) |
|---|---:|---:|---:|
| BRACE (live regeneration) | 0.696 | 0.737 | 0.566 |
| NERVE (official image, runs/17) | 0.737 | 0.736 | 0.760 |
| NERVE − BRACE | +0.041 [+0.015, +0.067] | −0.001 [−0.028, +0.043] | +0.194 [+0.137, +0.300] |

VOC-Sim: +0.012 [−0.002, +0.024] all; +0.005 [−0.010, +0.020] matched.

On images where our generation reproduces the benchmark image, BRACE and NERVE tie. Most of NERVE's COCO-Cap lead
comes from images on which BRACE is scored against the annotation of a different image.

## Consequence for amendment 1 (COCO-Cap, matched 528)

| Cell | CA | CLIP | hA | hB |
|---|---:|---:|---:|---:|
| seed alone | 0.602 | 0.647 | 0.601 | 0.618 |
| generation SA, BRACE hierarchy | 0.737 | 0.721 | 0.756 | 0.711 |
| NERVE pass SA, NERVE walk | 0.703 | 0.736 | 0.737 | 0.706 |

V (hA − CA): hierarchy +0.018 [+0.007, +0.033], walk +0.034 [+0.013, +0.054]: survives.
K (CLIP − hA): hierarchy −0.035 [−0.054, −0.007]: on matched images CLIP's ranking is worse than ours under the hierarchy.
VOC-Sim matched: hA hierarchy 0.874 vs CA 0.856 (V +0.017 [+0.008, +0.026]).
