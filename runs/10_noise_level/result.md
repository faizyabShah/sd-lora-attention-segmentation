# Exploratory: noise variability of SD1.5 cross-attention at BRACE's bleed, per noise level (`NOTE.md`)

COCO-Cap screening images (150 scored, 0 excluded); 6 noise seeds; one conditional UNet call per level. **Exploratory, in-sample (same images as run 09), 9 levels: not confirmatory.**

Baseline per level = run 09 baseline + this level's mean CA; test = + this level's v_CA. Bar for 'promising': Δ ≥ +0.05, CI > 0.

| t | √ᾱ (image share) | in BRACE window | mean CA obj/bg cell AUROC | v_CA single AUROC (bleed) | baseline | + v_CA Δ (95% CI) | promising |
|---:|---:|:-:|---:|---:|---:|---|:-:|
| 101 | 0.94 |  | 0.931 | 0.501 | 0.755 | +0.0122 [+0.0014, +0.0239] | no |
| 201 | 0.87 |  | 0.932 | 0.507 | 0.761 | +0.0173 [+0.0039, +0.0301] | no |
| 331 | 0.73 | yes | 0.932 | 0.527 | 0.760 | +0.0209 [+0.0037, +0.0377] | no |
| 430 | 0.61 | yes | 0.932 | 0.519 | 0.766 | +0.0118 [-0.0012, +0.0252] | no |
| 529 | 0.49 | yes | 0.935 | 0.543 | 0.771 | +0.0126 [-0.0007, +0.0244] | no |
| 661 | 0.33 | yes | 0.932 | 0.513 | 0.752 | +0.0062 [-0.0024, +0.0147] | no |
| 761 | 0.23 |  | 0.924 | 0.524 | 0.751 | +0.0041 [-0.0051, +0.0137] | no |
| 861 | 0.14 |  | 0.895 | 0.489 | 0.752 | -0.0043 [-0.0101, +0.0008] | no |
| 941 | 0.10 |  | 0.858 | 0.434 | 0.752 | -0.0067 [-0.0116, -0.0027] | no |

Cells 109,686 (31.1% bleed); 10 s/image capture. Run 09 reference (11 BRACE timesteps averaged, COCO-Cap): +0.0252 [+0.0047, +0.0444].

A level marked promising licenses only a new pre-registered single-level test on images not used here.
