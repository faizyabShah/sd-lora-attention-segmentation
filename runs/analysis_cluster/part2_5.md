# Map analysis parts 2–5 (rich capture of a fixed development subset; exploratory diagnostics)

`scripts/ana_capture.py` + `scripts/ana_dump.py`. Ground truth is used only to measure. Tags: `<res>_b<block>_<branch>`, block 0/1/2 = steps 0–9 / 10–19 / 20–30, branch u = unconditional, c = conditional. Best IoU = best threshold of the map (perfect use of its ranking), at the map's own resolution unless noted.

## VOC-Sim (150 images; base-loop vs live mean |Δpixel| 0.00/255)

### 2a. Is the seed or the propagation the bottleneck? (512 px)

| map | best IoU (mean) | IoU @ τ 0.55 + R2 (mean) |
|---|---:|---:|
| A3 seed (raw) | 0.777 | — |
| BRACE V3 (A3 seed) | 0.892 | 0.825 |
| V3 with oracle seed (truth pooled to 64) | 0.912 | 0.887 |

### 2b. Self-attention spectra

Ground-truth energy in the top-k eigenvectors of D^-1/2 W D^-1/2 (fraction of ‖g‖², mean); best single non-trivial eigenvector (|cos|, best IoU, mean index); Fiedler split labelled by the seed (IoU); eigengap estimate vs ground-truth components.

| tag | E1 | E2 | E4 | E8 | E16 | E32 | E64 | best vec |cos| | best vec IoU | best vec idx | Fiedler IoU | eigengap k (mean) | corr(eigengap k, instances) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16_b0_c | 0.39 | 0.70 | 0.79 | 0.84 | 0.88 | 0.91 | 0.93 | 0.56 | 0.797 | 1.2 | 0.595 | 2.3 | +0.04 |
| 16_b0_u | 0.38 | 0.67 | 0.77 | 0.82 | 0.87 | 0.89 | 0.92 | 0.56 | 0.790 | 1.4 | 0.584 | 2.3 | +0.00 |
| 16_b1_c | 0.39 | 0.71 | 0.80 | 0.86 | 0.90 | 0.93 | 0.95 | 0.57 | 0.832 | 1.4 | 0.621 | 2.5 | -0.02 |
| 16_b1_u | 0.40 | 0.71 | 0.80 | 0.86 | 0.90 | 0.93 | 0.95 | 0.57 | 0.835 | 1.4 | 0.625 | 2.5 | -0.02 |
| 16_b2_c | 0.39 | 0.68 | 0.80 | 0.86 | 0.90 | 0.93 | 0.95 | 0.57 | 0.828 | 1.4 | 0.611 | 2.7 | +0.03 |
| 16_b2_u | 0.39 | 0.68 | 0.80 | 0.86 | 0.90 | 0.93 | 0.95 | 0.57 | 0.832 | 1.4 | 0.616 | 2.7 | +0.02 |
| 32_b0_c | 0.38 | 0.65 | 0.81 | 0.86 | 0.89 | 0.92 | 0.94 | 0.57 | 0.792 | 1.7 | 0.575 | 2.6 | -0.03 |
| 32_b0_u | 0.38 | 0.61 | 0.79 | 0.85 | 0.88 | 0.91 | 0.93 | 0.56 | 0.775 | 1.8 | 0.539 | 2.7 | +0.11 |
| 32_b1_c | 0.38 | 0.71 | 0.85 | 0.90 | 0.92 | 0.94 | 0.96 | 0.61 | 0.853 | 1.5 | 0.635 | 2.8 | -0.04 |
| 32_b1_u | 0.38 | 0.71 | 0.85 | 0.90 | 0.92 | 0.94 | 0.96 | 0.61 | 0.852 | 1.6 | 0.640 | 2.8 | -0.04 |
| 32_b2_c | 0.38 | 0.69 | 0.84 | 0.89 | 0.92 | 0.94 | 0.96 | 0.61 | 0.846 | 2.0 | 0.616 | 2.8 | -0.03 |
| 32_b2_u | 0.38 | 0.69 | 0.85 | 0.89 | 0.92 | 0.94 | 0.96 | 0.61 | 0.846 | 1.6 | 0.617 | 2.8 | -0.03 |
| 64_b0_c | 0.37 | 0.60 | 0.81 | 0.85 | 0.88 | 0.91 | 0.93 | 0.58 | 0.795 | 2.0 | 0.542 | 2.7 | -0.03 |
| 64_b0_u | 0.37 | 0.56 | 0.79 | 0.84 | 0.87 | 0.90 | 0.92 | 0.56 | 0.771 | 2.1 | 0.502 | 2.6 | -0.02 |
| 64_b1_c | 0.37 | 0.69 | 0.86 | 0.90 | 0.93 | 0.94 | 0.96 | 0.61 | 0.844 | 1.8 | 0.624 | 3.2 | -0.05 |
| 64_b1_u | 0.37 | 0.69 | 0.86 | 0.90 | 0.93 | 0.94 | 0.96 | 0.61 | 0.841 | 1.9 | 0.621 | 3.1 | -0.03 |
| 64_b2_c | 0.37 | 0.68 | 0.83 | 0.89 | 0.92 | 0.94 | 0.95 | 0.59 | 0.817 | 1.7 | 0.620 | 3.4 | -0.05 |
| 64_b2_u | 0.37 | 0.68 | 0.83 | 0.89 | 0.92 | 0.94 | 0.95 | 0.59 | 0.818 | 1.7 | 0.620 | 3.4 | -0.02 |

Spectral clustering on the top-8 non-trivial eigenvectors, clusters labelled by the A3 seed (best IoU of the cluster-mean seed map / fixed rule: clusters with mean seed ≥ half the max / oracle: clusters with > 50% object):

| tag | K=2 | K=4 | K=8 |
|---|---|---|---|
| 16_b0_c | 0.567 / 0.473 / 0.325 | 0.675 / 0.590 / 0.553 | 0.751 / 0.656 / 0.714 |
| 16_b1_c | 0.564 / 0.475 / 0.329 | 0.681 / 0.602 / 0.570 | 0.770 / 0.676 / 0.736 |
| 16_b2_c | 0.560 / 0.477 / 0.329 | 0.692 / 0.603 / 0.573 | 0.770 / 0.679 / 0.740 |
| 32_b0_c | 0.567 / 0.457 / 0.310 | 0.677 / 0.581 / 0.538 | 0.780 / 0.691 / 0.750 |
| 32_b1_c | 0.584 / 0.473 / 0.330 | 0.728 / 0.624 / 0.616 | 0.821 / 0.733 / 0.802 |
| 32_b2_c | 0.581 / 0.473 / 0.348 | 0.715 / 0.609 / 0.578 | 0.825 / 0.734 / 0.808 |
| 64_b0_c | 0.597 / 0.477 / 0.363 | 0.736 / 0.655 / 0.649 | 0.790 / 0.722 / 0.768 |
| 64_b0_u | 0.589 / 0.467 / 0.348 | 0.714 / 0.628 / 0.608 | 0.779 / 0.714 / 0.760 |
| 64_b1_c | 0.617 / 0.512 / 0.401 | 0.785 / 0.701 / 0.708 | 0.842 / 0.764 / 0.819 |
| 64_b1_u | 0.614 / 0.510 / 0.396 | 0.790 / 0.699 / 0.711 | 0.843 / 0.768 / 0.823 |
| 64_b2_c | 0.621 / 0.520 / 0.406 | 0.778 / 0.694 / 0.705 | 0.845 / 0.771 / 0.823 |
| 64_b2_u | 0.619 / 0.520 / 0.403 | 0.780 / 0.694 / 0.710 | 0.848 / 0.772 / 0.824 |

Seed diffusion A^d s at 64 (head-mean SA of the block), best IoU at 512 (mean), vs BRACE V3 0.892:

| tag | d=1 | d=2 | d=4 | d=8 | d=16 |
|---|---:|---:|---:|---:|---:|
| 64_b0_c | 0.821 (small 0.756 / large 0.885) | 0.814 (small 0.743 / large 0.886) | 0.790 (small 0.706 / large 0.873) | 0.738 (small 0.633 / large 0.843) | 0.651 (small 0.523 / large 0.778) |
| 64_b0_u | 0.805 (small 0.737 / large 0.874) | 0.795 (small 0.719 / large 0.871) | 0.759 (small 0.668 / large 0.849) | 0.702 (small 0.592 / large 0.812) | 0.611 (small 0.470 / large 0.752) |
| 64_b1_c | 0.878 (small 0.836 / large 0.920) | 0.888 (small 0.847 / large 0.930) | 0.883 (small 0.835 / large 0.931) | 0.861 (small 0.804 / large 0.918) | 0.833 (small 0.761 / large 0.904) |
| 64_b1_u | 0.877 (small 0.835 / large 0.919) | 0.888 (small 0.846 / large 0.929) | 0.882 (small 0.833 / large 0.930) | 0.861 (small 0.804 / large 0.918) | 0.832 (small 0.760 / large 0.905) |
| 64_b2_c | 0.869 (small 0.826 / large 0.913) | 0.884 (small 0.842 / large 0.926) | 0.880 (small 0.832 / large 0.928) | 0.860 (small 0.802 / large 0.918) | 0.825 (small 0.751 / large 0.899) |
| 64_b2_u | 0.869 (small 0.826 / large 0.913) | 0.884 (small 0.841 / large 0.926) | 0.880 (small 0.832 / large 0.928) | 0.860 (small 0.802 / large 0.919) | 0.825 (small 0.750 / large 0.899) |

Per-head mass retention (mean over object queries of the attention mass that stays on the object; same for background), per head 0–7:

| tag | object stay | background stay |
|---|---|---|
| 16_b0_c | 0.49 0.68 0.37 0.63 0.61 0.60 0.44 0.62 | 0.53 0.76 0.64 0.69 0.77 0.72 0.61 0.71 |
| 16_b0_u | 0.45 0.60 0.37 0.57 0.51 0.55 0.42 0.56 | 0.56 0.73 0.64 0.68 0.75 0.70 0.63 0.71 |
| 16_b1_c | 0.56 0.83 0.47 0.77 0.78 0.77 0.55 0.79 | 0.56 0.80 0.62 0.77 0.83 0.79 0.62 0.80 |
| 16_b1_u | 0.57 0.83 0.48 0.78 0.78 0.77 0.56 0.79 | 0.56 0.80 0.61 0.77 0.83 0.79 0.62 0.80 |
| 16_b2_c | 0.58 0.84 0.52 0.80 0.82 0.80 0.60 0.81 | 0.62 0.84 0.66 0.83 0.85 0.84 0.68 0.85 |
| 16_b2_u | 0.59 0.84 0.52 0.80 0.82 0.80 0.60 0.81 | 0.62 0.83 0.65 0.83 0.86 0.84 0.68 0.84 |
| 32_b0_c | 0.48 0.67 0.61 0.53 0.54 0.65 0.58 0.69 | 0.62 0.80 0.69 0.70 0.70 0.78 0.77 0.82 |
| 32_b0_u | 0.46 0.62 0.57 0.50 0.51 0.62 0.54 0.64 | 0.63 0.78 0.69 0.70 0.69 0.77 0.75 0.80 |
| 32_b1_c | 0.62 0.87 0.76 0.71 0.72 0.83 0.80 0.89 | 0.67 0.91 0.77 0.76 0.75 0.90 0.86 0.92 |
| 32_b1_u | 0.62 0.87 0.76 0.71 0.72 0.83 0.80 0.89 | 0.67 0.91 0.77 0.76 0.75 0.90 0.87 0.92 |
| 32_b2_c | 0.67 0.89 0.78 0.75 0.76 0.86 0.83 0.90 | 0.74 0.93 0.83 0.82 0.81 0.93 0.89 0.94 |
| 32_b2_u | 0.67 0.89 0.78 0.75 0.76 0.86 0.83 0.90 | 0.75 0.93 0.83 0.82 0.81 0.93 0.89 0.94 |
| 64_b0_c | 0.64 0.60 0.56 0.51 0.49 0.61 0.66 0.57 | 0.78 0.77 0.76 0.74 0.70 0.77 0.81 0.73 |
| 64_b0_u | 0.61 0.57 0.54 0.49 0.47 0.58 0.62 0.55 | 0.76 0.76 0.75 0.72 0.70 0.75 0.79 0.72 |
| 64_b1_c | 0.90 0.83 0.81 0.72 0.68 0.87 0.91 0.80 | 0.93 0.89 0.90 0.85 0.80 0.91 0.93 0.85 |
| 64_b1_u | 0.90 0.83 0.81 0.72 0.68 0.87 0.91 0.80 | 0.93 0.89 0.90 0.85 0.80 0.91 0.93 0.85 |
| 64_b2_c | 0.91 0.86 0.86 0.78 0.73 0.90 0.93 0.82 | 0.94 0.90 0.91 0.87 0.84 0.94 0.95 0.89 |
| 64_b2_u | 0.91 0.86 0.86 0.78 0.73 0.90 0.93 0.82 | 0.94 0.90 0.91 0.87 0.84 0.94 0.95 0.89 |

### 3. Cross-attention

Mean over all layers, heads and blocks (pooled to 16): best IoU 0.821, AUROC 0.957. PC1 of the centred map stack: best IoU 0.753, AUROC 0.922. Explained variance of the first 5 PCs: 0.62, 0.12, 0.06, 0.04, 0.03; rank-1 share of the uncentred stack 0.66.

Top 15 and bottom 5 single maps (layer, block, head; best IoU at own resolution, mean):

| layer | block | head | best IoU |
|---|---|---|---:|
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 2 | 0.715 |
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b1 | 2 | 0.691 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 2 | 0.687 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b1 | 2 | 0.676 |
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 5 | 0.665 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b1 | 4 | 0.656 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b2 | 4 | 0.636 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 0 | 0.635 |
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b1 | 5 | 0.634 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 5 | 0.628 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b2 | 2 | 0.626 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b1 | 0 | 0.621 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 3 | 0.619 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b1 | 5 | 0.619 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b0 | 4 | 0.618 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 7 | 0.378 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b2 | 5 | 0.378 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b0 | 5 | 0.375 |
| down_blocks.0.attentions.1.transformer_blocks.0.attn2 | b0 | mean | 0.374 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b1 | 5 | 0.374 |

Special tokens, AUROC for the object (head- and layer-mean; < 0.5 = background-like):

| token | 16 | 32 |
|---|---:|---:|
| sot | 0.236 | 0.180 |
| eot | 0.618 | 0.785 |
| pad | 0.642 | 0.800 |

### 4. Decoder features

Per-image: best of the top-3 PCs (best IoU); within-image ridge probe on a checkerboard split (best IoU on held-out pixels, an upper bound); k-means K=8 oracle cluster selection (IoU).

| features | top-3 PC | within-image probe | k-means K=8 oracle |
|---|---:|---:|---:|
| up1/s15 | 0.692 | 0.667 | 0.731 |
| up1/s25 | 0.682 | 0.674 | 0.740 |
| up1/s5 | 0.681 | 0.674 | 0.654 |
| up2/s15 | 0.654 | 0.944 | 0.723 |
| up2/s25 | 0.641 | 0.948 | 0.698 |
| up2/s5 | 0.662 | 0.896 | 0.681 |
| x0 trajectory (31×4) | — | — | 0.643 |

### 5. Dynamics

x0 settling step (first step after which the x0 prediction stays within 10% of its initial distance to the final): inside the object 25.8, outside 27.0.

## COCO-Cap (150 images; base-loop vs live mean |Δpixel| 0.00/255)

### 2a. Is the seed or the propagation the bottleneck? (512 px)

| map | best IoU (mean) | IoU @ τ 0.55 + R2 (mean) |
|---|---:|---:|
| A3 seed (raw) | 0.673 | — |
| BRACE V3 (A3 seed) | 0.798 | 0.710 |
| V3 with oracle seed (truth pooled to 64) | 0.869 | 0.836 |

### 2b. Self-attention spectra

Ground-truth energy in the top-k eigenvectors of D^-1/2 W D^-1/2 (fraction of ‖g‖², mean); best single non-trivial eigenvector (|cos|, best IoU, mean index); Fiedler split labelled by the seed (IoU); eigengap estimate vs ground-truth components.

| tag | E1 | E2 | E4 | E8 | E16 | E32 | E64 | best vec |cos| | best vec IoU | best vec idx | Fiedler IoU | eigengap k (mean) | corr(eigengap k, instances) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16_b0_c | 0.25 | 0.50 | 0.66 | 0.75 | 0.81 | 0.86 | 0.89 | 0.56 | 0.750 | 2.2 | 0.381 | 2.6 | -0.01 |
| 16_b0_u | 0.24 | 0.45 | 0.61 | 0.72 | 0.79 | 0.83 | 0.87 | 0.54 | 0.733 | 2.6 | 0.360 | 2.6 | -0.15 |
| 16_b1_c | 0.25 | 0.51 | 0.69 | 0.79 | 0.85 | 0.89 | 0.93 | 0.58 | 0.799 | 2.1 | 0.398 | 3.0 | -0.08 |
| 16_b1_u | 0.25 | 0.51 | 0.69 | 0.79 | 0.85 | 0.89 | 0.93 | 0.58 | 0.801 | 2.2 | 0.399 | 2.8 | -0.08 |
| 16_b2_c | 0.25 | 0.47 | 0.68 | 0.78 | 0.84 | 0.89 | 0.94 | 0.57 | 0.786 | 2.7 | 0.375 | 3.0 | -0.15 |
| 16_b2_u | 0.25 | 0.47 | 0.68 | 0.78 | 0.85 | 0.89 | 0.94 | 0.57 | 0.793 | 2.5 | 0.377 | 3.0 | -0.14 |
| 32_b0_c | 0.24 | 0.41 | 0.65 | 0.76 | 0.82 | 0.86 | 0.90 | 0.54 | 0.710 | 2.8 | 0.345 | 2.5 | -0.06 |
| 32_b0_u | 0.24 | 0.39 | 0.61 | 0.73 | 0.80 | 0.85 | 0.89 | 0.52 | 0.684 | 3.2 | 0.324 | 2.6 | -0.02 |
| 32_b1_c | 0.25 | 0.46 | 0.72 | 0.82 | 0.87 | 0.90 | 0.93 | 0.60 | 0.791 | 2.7 | 0.381 | 2.9 | +0.02 |
| 32_b1_u | 0.25 | 0.46 | 0.72 | 0.82 | 0.87 | 0.90 | 0.93 | 0.60 | 0.791 | 2.7 | 0.379 | 3.0 | +0.01 |
| 32_b2_c | 0.24 | 0.43 | 0.70 | 0.81 | 0.87 | 0.90 | 0.93 | 0.59 | 0.770 | 2.8 | 0.356 | 3.2 | +0.00 |
| 32_b2_u | 0.24 | 0.43 | 0.70 | 0.81 | 0.87 | 0.90 | 0.93 | 0.59 | 0.779 | 2.9 | 0.358 | 3.3 | +0.02 |
| 64_b0_c | 0.24 | 0.39 | 0.64 | 0.76 | 0.81 | 0.85 | 0.88 | 0.56 | 0.701 | 3.9 | 0.339 | 2.5 | -0.03 |
| 64_b0_u | 0.24 | 0.38 | 0.61 | 0.73 | 0.79 | 0.84 | 0.87 | 0.53 | 0.673 | 4.1 | 0.323 | 2.6 | -0.02 |
| 64_b1_c | 0.24 | 0.42 | 0.69 | 0.82 | 0.87 | 0.90 | 0.92 | 0.59 | 0.755 | 3.8 | 0.355 | 3.5 | +0.22 |
| 64_b1_u | 0.24 | 0.42 | 0.69 | 0.81 | 0.87 | 0.90 | 0.92 | 0.59 | 0.755 | 3.2 | 0.354 | 3.4 | +0.11 |
| 64_b2_c | 0.24 | 0.41 | 0.65 | 0.79 | 0.86 | 0.90 | 0.92 | 0.56 | 0.728 | 3.6 | 0.350 | 3.5 | +0.08 |
| 64_b2_u | 0.24 | 0.41 | 0.65 | 0.79 | 0.86 | 0.90 | 0.92 | 0.56 | 0.727 | 3.6 | 0.350 | 3.5 | +0.07 |

Spectral clustering on the top-8 non-trivial eigenvectors, clusters labelled by the A3 seed (best IoU of the cluster-mean seed map / fixed rule: clusters with mean seed ≥ half the max / oracle: clusters with > 50% object):

| tag | K=2 | K=4 | K=8 |
|---|---|---|---|
| 16_b0_c | 0.379 / 0.284 / 0.140 | 0.479 / 0.385 / 0.285 | 0.638 / 0.500 / 0.556 |
| 16_b1_c | 0.384 / 0.286 / 0.148 | 0.515 / 0.402 / 0.348 | 0.665 / 0.528 / 0.576 |
| 16_b2_c | 0.409 / 0.312 / 0.191 | 0.525 / 0.411 / 0.355 | 0.663 / 0.541 / 0.582 |
| 32_b0_c | 0.384 / 0.282 / 0.150 | 0.500 / 0.395 / 0.324 | 0.651 / 0.535 / 0.597 |
| 32_b1_c | 0.389 / 0.289 / 0.157 | 0.524 / 0.413 / 0.345 | 0.702 / 0.562 / 0.646 |
| 32_b2_c | 0.387 / 0.293 / 0.159 | 0.538 / 0.416 / 0.346 | 0.707 / 0.570 / 0.651 |
| 64_b0_c | 0.394 / 0.288 / 0.145 | 0.516 / 0.409 / 0.365 | 0.657 / 0.544 / 0.616 |
| 64_b0_u | 0.384 / 0.286 / 0.129 | 0.510 / 0.403 / 0.358 | 0.637 / 0.535 / 0.595 |
| 64_b1_c | 0.409 / 0.299 / 0.164 | 0.564 / 0.449 / 0.396 | 0.729 / 0.593 / 0.690 |
| 64_b1_u | 0.412 / 0.298 / 0.159 | 0.569 / 0.448 / 0.399 | 0.725 / 0.594 / 0.677 |
| 64_b2_c | 0.409 / 0.308 / 0.165 | 0.555 / 0.442 / 0.405 | 0.694 / 0.572 / 0.650 |
| 64_b2_u | 0.409 / 0.308 / 0.166 | 0.550 / 0.435 / 0.390 | 0.696 / 0.577 / 0.656 |

Seed diffusion A^d s at 64 (head-mean SA of the block), best IoU at 512 (mean), vs BRACE V3 0.798:

| tag | d=1 | d=2 | d=4 | d=8 | d=16 |
|---|---:|---:|---:|---:|---:|
| 64_b0_c | 0.715 (small 0.603 / large 0.826) | 0.682 (small 0.550 / large 0.813) | 0.607 (small 0.449 / large 0.766) | 0.521 (small 0.338 / large 0.704) | 0.472 (small 0.289 / large 0.655) |
| 64_b0_u | 0.694 (small 0.579 / large 0.808) | 0.660 (small 0.528 / large 0.792) | 0.593 (small 0.444 / large 0.741) | 0.507 (small 0.324 / large 0.690) | 0.434 (small 0.231 / large 0.636) |
| 64_b1_c | 0.785 (small 0.700 / large 0.871) | 0.790 (small 0.697 / large 0.884) | 0.765 (small 0.653 / large 0.878) | 0.705 (small 0.559 / large 0.850) | 0.623 (small 0.449 / large 0.798) |
| 64_b1_u | 0.782 (small 0.696 / large 0.869) | 0.787 (small 0.691 / large 0.882) | 0.762 (small 0.649 / large 0.876) | 0.702 (small 0.556 / large 0.848) | 0.622 (small 0.445 / large 0.798) |
| 64_b2_c | 0.781 (small 0.701 / large 0.861) | 0.791 (small 0.705 / large 0.876) | 0.773 (small 0.670 / large 0.875) | 0.720 (small 0.593 / large 0.848) | 0.639 (small 0.476 / large 0.803) |
| 64_b2_u | 0.781 (small 0.700 / large 0.861) | 0.791 (small 0.705 / large 0.876) | 0.773 (small 0.671 / large 0.875) | 0.720 (small 0.593 / large 0.848) | 0.641 (small 0.478 / large 0.803) |

Per-head mass retention (mean over object queries of the attention mass that stays on the object; same for background), per head 0–7:

| tag | object stay | background stay |
|---|---|---|
| 16_b0_c | 0.32 0.57 0.25 0.49 0.49 0.46 0.29 0.49 | 0.74 0.88 0.79 0.84 0.88 0.86 0.79 0.85 |
| 16_b0_u | 0.29 0.45 0.24 0.42 0.37 0.39 0.27 0.41 | 0.76 0.86 0.80 0.82 0.87 0.83 0.80 0.84 |
| 16_b1_c | 0.39 0.77 0.32 0.70 0.71 0.65 0.40 0.69 | 0.76 0.92 0.79 0.89 0.92 0.90 0.79 0.90 |
| 16_b1_u | 0.39 0.77 0.32 0.69 0.69 0.65 0.39 0.68 | 0.76 0.91 0.79 0.89 0.92 0.90 0.80 0.90 |
| 16_b2_c | 0.42 0.79 0.37 0.73 0.76 0.70 0.44 0.73 | 0.80 0.93 0.81 0.92 0.93 0.92 0.82 0.92 |
| 16_b2_u | 0.42 0.79 0.37 0.74 0.75 0.70 0.45 0.73 | 0.80 0.93 0.81 0.92 0.93 0.92 0.83 0.92 |
| 32_b0_c | 0.33 0.54 0.45 0.38 0.40 0.51 0.44 0.57 | 0.79 0.89 0.83 0.84 0.83 0.87 0.87 0.90 |
| 32_b0_u | 0.31 0.48 0.41 0.35 0.37 0.47 0.40 0.50 | 0.79 0.87 0.83 0.84 0.83 0.86 0.86 0.89 |
| 32_b1_c | 0.46 0.80 0.63 0.57 0.59 0.73 0.70 0.83 | 0.82 0.95 0.88 0.87 0.87 0.94 0.93 0.97 |
| 32_b1_u | 0.46 0.79 0.63 0.56 0.59 0.72 0.69 0.82 | 0.82 0.95 0.88 0.87 0.87 0.94 0.92 0.96 |
| 32_b2_c | 0.52 0.84 0.67 0.63 0.65 0.78 0.75 0.86 | 0.86 0.96 0.90 0.90 0.90 0.96 0.94 0.97 |
| 32_b2_u | 0.52 0.84 0.67 0.63 0.65 0.78 0.75 0.86 | 0.86 0.96 0.91 0.90 0.90 0.96 0.94 0.97 |
| 64_b0_c | 0.51 0.47 0.42 0.37 0.34 0.48 0.54 0.43 | 0.88 0.87 0.87 0.85 0.83 0.87 0.89 0.85 |
| 64_b0_u | 0.48 0.44 0.40 0.35 0.33 0.44 0.50 0.40 | 0.87 0.86 0.86 0.84 0.83 0.86 0.88 0.84 |
| 64_b1_c | 0.83 0.75 0.70 0.59 0.53 0.78 0.85 0.68 | 0.96 0.94 0.94 0.91 0.89 0.95 0.97 0.92 |
| 64_b1_u | 0.83 0.75 0.70 0.59 0.53 0.78 0.85 0.67 | 0.96 0.94 0.94 0.91 0.89 0.95 0.97 0.92 |
| 64_b2_c | 0.86 0.78 0.77 0.66 0.59 0.84 0.89 0.71 | 0.97 0.95 0.95 0.92 0.91 0.96 0.98 0.93 |
| 64_b2_u | 0.86 0.78 0.77 0.66 0.59 0.84 0.89 0.71 | 0.97 0.95 0.95 0.92 0.91 0.96 0.98 0.93 |

### 3. Cross-attention

Mean over all layers, heads and blocks (pooled to 16): best IoU 0.737, AUROC 0.907. PC1 of the centred map stack: best IoU 0.661, AUROC 0.864. Explained variance of the first 5 PCs: 0.62, 0.12, 0.06, 0.04, 0.03; rank-1 share of the uncentred stack 0.66.

Top 15 and bottom 5 single maps (layer, block, head; best IoU at own resolution, mean):

| layer | block | head | best IoU |
|---|---|---|---:|
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b1 | 4 | 0.562 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b1 | 6 | 0.555 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b1 | 0 | 0.553 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b2 | 4 | 0.551 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 2 | 0.540 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 6 | 0.538 |
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 5 | 0.536 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 0 | 0.536 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b1 | 2 | 0.534 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b0 | 6 | 0.530 |
| down_blocks.1.attentions.0.transformer_blocks.0.attn2 | b2 | 2 | 0.522 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 3 | 0.516 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b1 | 3 | 0.507 |
| up_blocks.1.attentions.0.transformer_blocks.0.attn2 | b0 | 0 | 0.502 |
| down_blocks.2.attentions.1.transformer_blocks.0.attn2 | b2 | 2 | 0.499 |
| down_blocks.2.attentions.0.transformer_blocks.0.attn2 | b2 | 7 | 0.248 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b2 | 5 | 0.247 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b0 | 5 | 0.246 |
| down_blocks.1.attentions.1.transformer_blocks.0.attn2 | b1 | 5 | 0.245 |
| down_blocks.0.attentions.1.transformer_blocks.0.attn2 | b0 | mean | 0.242 |

Special tokens, AUROC for the object (head- and layer-mean; < 0.5 = background-like):

| token | 16 | 32 |
|---|---:|---:|
| sot | 0.226 | 0.226 |
| eot | 0.569 | 0.700 |
| pad | 0.596 | 0.709 |

Competition (COCO-Cap, images with other content tokens: 150): inside the object, the target's share of content-token attention 0.43; strongest other token's mean / target mean 4.30; correlation of target share with BRACE best IoU +0.33.

Token rule split: class_word 95, fallback 55

### 4. Decoder features

Per-image: best of the top-3 PCs (best IoU); within-image ridge probe on a checkerboard split (best IoU on held-out pixels, an upper bound); k-means K=8 oracle cluster selection (IoU).

| features | top-3 PC | within-image probe | k-means K=8 oracle |
|---|---:|---:|---:|
| up1/s15 | 0.552 | nan | 0.581 |
| up1/s25 | 0.526 | nan | 0.581 |
| up1/s5 | 0.556 | nan | 0.507 |
| up2/s15 | 0.468 | 0.903 | 0.525 |
| up2/s25 | 0.439 | 0.906 | 0.486 |
| up2/s5 | 0.519 | 0.839 | 0.509 |
| x0 trajectory (31×4) | — | — | 0.476 |

### 5. Dynamics

x0 settling step (first step after which the x0 prediction stays within 10% of its initial distance to the final): inside the object 25.4, outside 26.4.

## Cross-image linear probe (features standardised per image)

Ridge on 512 sampled pixels per image. Train: first half of the VOC-Sim subset; test: second half and COCO-Cap. Reported: pixel AUROC on the test images' samples. Upper bound of a single class-agnostic linear 'object of the prompt' direction.

| features | VOC-Sim test AUROC | COCO-Cap AUROC |
|---|---:|---:|
| up1/s15 | 0.964 | 0.891 |
| up1/s25 | 0.958 | 0.883 |
| up1/s5 | 0.931 | 0.851 |
| up2/s15 | 0.946 | 0.875 |
| up2/s25 | 0.942 | 0.872 |
| up2/s5 | 0.929 | 0.838 |
