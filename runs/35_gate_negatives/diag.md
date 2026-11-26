# Run 35, N4 diagnostics (`PREREG.md` §3)

**Descriptive.** Cluster statistics from profiles fitted on all dev images of the regime; agreement and size from held-out MK4 gates on the run 31 dev dumps.


## 1. Support of the MK4 absent centroids

| Regime | centroids | median pixels | median n_eff | n_eff < 5 | median largest single-image share | PD kept (mean per class×module) |
|---|---:|---:|---:|---:|---:|---:|
| X | 880 | 699 | 223.2 | 0.000 | 0.007 | 4.00 |
| V | 877 | 275 | 90.6 | 0.000 | 0.015 | 3.99 |

Per module (regime X): median n_eff / fraction n_eff < 5

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 248.7 / 0.00 | 363.5 / 0.00 | 182.1 / 0.00 | 170.7 / 0.00 | 145.0 / 0.00 | 146.7 / 0.00 | 131.7 / 0.00 | 161.8 / 0.00 | 326.4 / 0.00 | 379.0 / 0.00 | 388.0 / 0.00 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

**PD reading rule (≥ 10% of centroids with n_eff < 5 in regime V or X): not met, PD is descriptive.**


## 2. What the absent clusters contain

Share of the most common member-image label / most common other class by attention, per cluster vs its whole pool.

| Regime | cluster label share | pool label share | lift | cluster attention share | pool attention share | lift |
|---|---:|---:|---:|---:|---:|---:|
| X | 0.183 | 0.066 | +0.117 | 0.561 | 0.481 | +0.080 |
| V | 0.183 | 0.054 | +0.129 | 0.557 | 0.485 | +0.072 |

Per class, regime X, module down_blocks.0.attentions.0.transformer_blocks.0.attn2: each cluster as `label (share) · attention (share) · n_eff`

| Class | clusters |
|---|---|
| aeroplane | bus (0.12) · motorbike (0.76) · 224; bicycle (0.11) · motorbike (0.51) · 365; cow (0.07) · motorbike (0.42) · 278; dog (0.13) · sofa (0.36) · 204 |
| bicycle | train (0.12) · motorbike (0.72) · 306; person (0.12) · motorbike (0.73) · 202; sheep (0.11) · motorbike (0.81) · 254; bus (0.13) · motorbike (0.88) · 220 |
| bird | sheep (0.12) · motorbike (0.39) · 243; aeroplane (0.10) · motorbike (0.38) · 267; bus (0.14) · motorbike (0.71) · 214; bicycle (0.15) · motorbike (0.54) · 276 |
| boat | dog (0.12) · sofa (0.47) · 226; cow (0.08) · motorbike (0.46) · 312; bus (0.14) · motorbike (0.72) · 224; aeroplane (0.10) · motorbike (0.53) · 384 |
| bottle | aeroplane (0.10) · sofa (0.71) · 319; bus (0.16) · motorbike (0.71) · 178; train (0.12) · motorbike (0.38) · 264; cat (0.10) · sofa (0.75) · 387 |
| bus | car (0.11) · motorbike (0.69) · 156; dog (0.08) · sofa (0.58) · 382; train (0.09) · motorbike (0.44) · 350; person (0.10) · sofa (0.81) · 277 |
| car | bicycle (0.11) · motorbike (0.67) · 359; bus (0.14) · motorbike (0.81) · 245; dog (0.11) · motorbike (0.69) · 279; tv (0.09) · motorbike (0.35) · 221 |
| cat | bicycle (0.11) · motorbike (0.44) · 352; dog (0.15) · sofa (0.46) · 177; bus (0.13) · motorbike (0.72) · 220; sheep (0.14) · bicycle (0.51) · 229 |
| chair | bus (0.14) · motorbike (0.68) · 199; dog (0.08) · sofa (0.64) · 300; person (0.13) · sofa (0.70) · 228; bicycle (0.09) · motorbike (0.39) · 360 |
| cow | bicycle (0.13) · motorbike (0.52) · 328; sheep (0.12) · motorbike (0.50) · 246; bus (0.13) · motorbike (0.75) · 228; aeroplane (0.08) · motorbike (0.43) · 303 |
| dining table | bus (0.14) · motorbike (0.61) · 183; sheep (0.09) · sofa (0.85) · 347; bicycle (0.10) · sofa (0.96) · 267; aeroplane (0.12) · motorbike (0.34) · 245 |
| dog | cat (0.13) · motorbike (0.49) · 232; bicycle (0.12) · motorbike (0.48) · 326; person (0.07) · motorbike (0.41) · 289; bus (0.13) · motorbike (0.74) · 218 |
| horse | cow (0.07) · motorbike (0.45) · 317; sheep (0.13) · motorbike (0.50) · 226; bus (0.13) · motorbike (0.72) · 234; bicycle (0.14) · motorbike (0.49) · 315 |
| motorbike | bicycle (0.09) · bicycle (0.60) · 204; sheep (0.11) · bicycle (0.47) · 249; bus (0.12) · bicycle (0.57) · 254; bicycle (0.12) · bicycle (0.62) · 294 |
| person | bus (0.07) · motorbike (0.42) · 258; cat (0.14) · sofa (0.39) · 188; bus (0.13) · motorbike (0.73) · 214; aeroplane (0.11) · motorbike (0.51) · 321 |
| potted plant | bicycle (0.11) · motorbike (0.48) · 322; bus (0.11) · motorbike (0.76) · 248; cat (0.13) · sofa (0.53) · 205; sheep (0.13) · sofa (0.28) · 181 |
| sheep | aeroplane (0.09) · motorbike (0.41) · 267; dog (0.13) · motorbike (0.45) · 235; bus (0.12) · motorbike (0.73) · 238; bicycle (0.12) · motorbike (0.49) · 327 |
| sofa | boat (0.09) · dining table (0.83) · 227; bus (0.15) · motorbike (0.76) · 215; train (0.12) · motorbike (0.43) · 240; cat (0.10) · dining table (0.45) · 366 |
| train | bus (0.08) · motorbike (0.40) · 236; bus (0.16) · motorbike (0.73) · 170; bicycle (0.11) · motorbike (0.43) · 319; dog (0.09) · sofa (0.47) · 319 |
| tv | dog (0.11) · sofa (0.42) · 246; aeroplane (0.10) · motorbike (0.51) · 320; bus (0.12) · motorbike (0.74) · 242; aeroplane (0.08) · motorbike (0.42) · 260 |

## 3. Module agreement (MK4, held out, 64 × 64 grid)

| Setting | mean pairwise corr of g | object px: some module < 0.5, median ≥ 0.5 | object px: some module < 0.5 | far px: median < 0.5 | far px: some module < 0.5 |
|---|---:|---:|---:|---:|---:|
| V on voc | 0.286 | 0.809 | 0.912 | 0.904 | 1.000 |
| X on coco | 0.254 | 0.720 | 0.942 | 0.930 | 1.000 |
| X on voc | 0.272 | 0.837 | 0.898 | 0.878 | 0.999 |
| V on coco | 0.260 | 0.581 | 0.968 | 0.972 | 1.000 |

Per-module AUROC of MK4 (object vs far background), class-balanced:

| Setting | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V on voc | 0.650 | 0.460 | 0.902 | 0.947 | 0.987 | 0.994 | 0.970 | 0.964 | 0.594 | 0.683 | 0.593 |
| X on coco | 0.625 | 0.482 | 0.825 | 0.914 | 0.973 | 0.982 | 0.951 | 0.964 | 0.563 | 0.587 | 0.520 |
| X on voc | 0.666 | 0.466 | 0.864 | 0.961 | 0.993 | 0.995 | 0.973 | 0.970 | 0.575 | 0.627 | 0.538 |
| V on coco | 0.609 | 0.497 | 0.818 | 0.899 | 0.964 | 0.975 | 0.936 | 0.967 | 0.583 | 0.639 | 0.574 |

## 4. Object size (mass-weighted object retention at γ = 1, image means)

| Setting | gate | small third | middle third | large third | Spearman ρ (area, retention) |
|---|---|---:|---:|---:|---:|
| V on voc | MK4 | 0.664 | 0.659 | 0.662 | -0.035 |
| V on voc | IB | 0.611 | 0.606 | 0.603 | -0.049 |
| V on voc | NC | 0.753 | 0.748 | 0.746 | -0.096 |
| X on coco | MK4 | 0.613 | 0.656 | 0.647 | +0.158 |
| X on coco | IB | 0.565 | 0.601 | 0.589 | +0.105 |
| X on coco | NC | 0.722 | 0.775 | 0.772 | +0.220 |
