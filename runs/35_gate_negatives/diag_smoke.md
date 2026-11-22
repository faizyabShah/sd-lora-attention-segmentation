# Run 35, N4 diagnostics (`PREREG.md` §3)

**Descriptive.** Cluster statistics from profiles fitted on all dev images of the regime; agreement and size from held-out MK4 gates on the run 31 dev dumps.


## 1. Support of the MK4 absent centroids

| Regime | centroids | median pixels | median n_eff | n_eff < 5 | median largest single-image share | PD kept (mean per class×module) |
|---|---:|---:|---:|---:|---:|---:|
| X | 797 | 283 | 31.3 | 0.000 | 0.046 | 3.62 |
| V | 727 | 233 | 16.7 | 0.000 | 0.085 | 3.30 |

Per module (regime X): median n_eff / fraction n_eff < 5

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31.9 / 0.00 | 46.9 / 0.00 | 24.2 / 0.00 | 21.9 / 0.00 | 76.0 / 0.00 | 19.5 / 0.00 | 18.9 / 0.00 | 23.8 / 0.00 | 35.8 / 0.00 | 46.4 / 0.00 | 50.0 / 0.00 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

**PD reading rule (≥ 10% of centroids with n_eff < 5 in regime V or X): not met, PD is descriptive.**


## 2. What the absent clusters contain

Share of the most common member-image label / most common other class by attention, per cluster vs its whole pool.

| Regime | cluster label share | pool label share | lift | cluster attention share | pool attention share | lift |
|---|---:|---:|---:|---:|---:|---:|
| X | 0.192 | 0.105 | +0.087 | 0.566 | 0.510 | +0.055 |
| V | 0.227 | 0.129 | +0.098 | 0.571 | 0.521 | +0.050 |

Per class, regime X, module down_blocks.0.attentions.0.transformer_blocks.0.attn2: each cluster as `label (share) · attention (share) · n_eff`

| Class | clusters |
|---|---|
| aeroplane | person (0.11) · motorbike (0.35) · 35; car (0.11) · motorbike (0.76) · 28; bird (0.17) · motorbike (0.37) · 29; bird (0.12) · motorbike (0.54) · 49 |
| bicycle | bird (0.18) · motorbike (0.84) · 38; car (0.12) · motorbike (0.88) · 26; person (0.18) · motorbike (0.84) · 24; aeroplane (0.14) · motorbike (0.67) · 33 |
| bird | sheep (0.18) · motorbike (0.46) · 31; bicycle (0.17) · motorbike (0.53) · 35; car (0.12) · motorbike (0.69) · 24; aeroplane (0.11) · motorbike (0.33) · 32 |
| boat | bird (0.18) · sofa (0.43) · 30; car (0.11) · motorbike (0.72) · 28; person (0.12) · motorbike (0.44) · 41; bird (0.13) · motorbike (0.58) · 51 |
| bottle | bird (0.16) · sofa (0.76) · 31; car (0.13) · motorbike (0.71) · 24; bicycle (0.14) · motorbike (0.33) · 33; bird (0.16) · sofa (0.72) · 32 |
| bus | sheep (0.18) · sofa (0.52) · 23; car (0.15) · motorbike (0.66) · 20; bird (0.16) · sofa (0.65) · 44; bicycle (0.13) · motorbike (0.41) · 38 |
| car | person (0.12) · motorbike (0.29) · 29; bird (0.17) · motorbike (0.74) · 39; bus (0.12) · motorbike (0.80) · 33; bicycle (0.11) · motorbike (0.67) · 46 |
| cat | sheep (0.19) · bicycle (0.46) · 33; bus (0.12) · motorbike (0.72) · 27; bicycle (0.13) · motorbike (0.36) · 41; bird (0.19) · motorbike (0.39) · 27 |
| chair | bird (0.15) · sofa (0.63) · 45; person (0.19) · sofa (0.70) · 32; cat (0.10) · motorbike (0.37) · 47; bicycle (0.15) · motorbike (0.71) · 24 |
| cow | bicycle (0.14) · motorbike (0.47) · 39; bird (0.23) · motorbike (0.61) · 34; car (0.11) · motorbike (0.76) · 28; person (0.10) · motorbike (0.46) · 41 |
| dining table | bird (0.12) · sofa (0.95) · 38; bicycle (0.16) · motorbike (0.63) · 22; cat (0.15) · sofa (0.43) · 29; bird (0.13) · sofa (0.85) · 43 |
| dog | bicycle (0.14) · motorbike (0.54) · 39; bird (0.22) · motorbike (0.54) · 31; bus (0.12) · motorbike (0.73) · 23; aeroplane (0.12) · motorbike (0.32) · 31 |
| horse | car (0.11) · motorbike (0.72) · 27; person (0.11) · motorbike (0.42) · 44; bird (0.24) · motorbike (0.58) · 32; bicycle (0.15) · motorbike (0.48) · 42 |
| motorbike | aeroplane (0.15) · bicycle (0.63) · 20; sheep (0.17) · bicycle (0.42) · 31; bus (0.10) · bicycle (0.63) · 31; bicycle (0.11) · bicycle (0.65) · 39 |
| person | car (0.12) · motorbike (0.71) · 23; train (0.10) · motorbike (0.34) · 32; bird (0.11) · motorbike (0.49) · 41; bird (0.27) · sofa (0.41) · 20 |
| potted plant | bicycle (0.11) · motorbike (0.42) · 37; bicycle (0.12) · motorbike (0.76) · 30; sheep (0.18) · sofa (0.37) · 25; bird (0.23) · sofa (0.49) · 25 |
| sheep | bird (0.24) · motorbike (0.48) · 31; bus (0.12) · motorbike (0.74) · 28; bicycle (0.13) · motorbike (0.47) · 39; person (0.12) · motorbike (0.41) · 36 |
| sofa | car (0.14) · motorbike (0.77) · 24; bird (0.18) · dining table (0.43) · 43; boat (0.13) · dining table (0.81) · 27; bicycle (0.10) · motorbike (0.41) · 33 |
| train | bird (0.18) · sofa (0.46) · 43; person (0.12) · motorbike (0.38) · 34; bicycle (0.13) · motorbike (0.44) · 43; car (0.14) · motorbike (0.68) · 20 |
| tv | bicycle (0.11) · motorbike (0.75) · 29; bird (0.18) · motorbike (0.55) · 34; bird (0.19) · sofa (0.50) · 23; aeroplane (0.11) · motorbike (0.38) · 36 |

## 3. Module agreement (MK4, held out, 64 × 64 grid)

| Setting | mean pairwise corr of g | object px: some module < 0.5, median ≥ 0.5 | object px: some module < 0.5 | far px: median < 0.5 | far px: some module < 0.5 |
|---|---:|---:|---:|---:|---:|
| V on voc | 0.217 | 0.568 | 0.996 | 0.965 | 1.000 |
| X on coco | 0.169 | 0.286 | 0.998 | 0.985 | 1.000 |
| X on voc | 0.219 | 0.574 | 0.989 | 0.937 | 1.000 |
| V on coco | 0.205 | 0.439 | 0.988 | 0.952 | 1.000 |

Per-module AUROC of MK4 (object vs far background), class-balanced:

| Setting | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V on voc | 0.554 | 0.464 | 0.928 | 0.862 | 0.987 | 0.998 | 0.924 | 0.935 | 0.516 | 0.568 | 0.581 |
| X on coco | 0.519 | 0.531 | 0.635 | 0.752 | 0.952 | 0.974 | 0.893 | 0.906 | 0.539 | 0.556 | 0.566 |
| X on voc | 0.587 | 0.530 | 0.796 | 0.847 | 0.984 | 0.962 | 0.875 | 0.882 | 0.536 | 0.607 | 0.585 |
| V on coco | 0.518 | 0.518 | 0.779 | 0.870 | 0.983 | 0.980 | 0.916 | 0.917 | 0.535 | 0.572 | 0.614 |

## 4. Object size (mass-weighted object retention at γ = 1, image means)

| Setting | gate | small third | middle third | large third | Spearman ρ (area, retention) |
|---|---|---:|---:|---:|---:|
| V on voc | MK4 | 0.504 | 0.550 | 0.535 | +0.303 |
| V on voc | IB | 0.466 | 0.505 | 0.483 | +0.212 |
| V on voc | NC | 0.526 | 0.587 | 0.565 | +0.230 |
| X on coco | MK4 | 0.431 | 0.498 | 0.439 | +0.012 |
| X on coco | IB | 0.389 | 0.442 | 0.397 | -0.007 |
| X on coco | NC | 0.494 | 0.540 | 0.484 | -0.059 |
