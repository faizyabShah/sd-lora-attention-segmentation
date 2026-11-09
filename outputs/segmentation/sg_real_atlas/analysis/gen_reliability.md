# Reliability of the frozen BRACE mask on generated images (pre-registered in `gen_reliab_prereg.md`)

## VOC-Sim (490 images; frozen BRACE mask class-macro IoU 0.830; 30 masks with IoU < 0.5)

Image level. ρ = Spearman with the mask's true IoU; ρ_wc = mean within-class ρ; ρ_wc|size = within-class partial ρ controlling log predicted area; AUROC_wc = detecting IoU < 0.5 with scores ranked within class; retained class-macro mIoU when keeping the top q of each class (random keep = all-data mIoU in expectation).

| signal | ρ | ρ_wc | ρ_wc\|size | AUROC_wc | keep 90% | keep 70% | keep 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| s_area | +0.087 | +0.237 | +0.017 | 0.598 | 0.833 | 0.839 | 0.848 |
| s_seedpeak | +0.018 | -0.124 | +0.111 | 0.465 | 0.825 | 0.823 | 0.829 |
| s_clip_iou | +0.443 | +0.418 | +0.359 | 0.702 | 0.848 | 0.866 | 0.880 |
| s_clip_soft | +0.349 | +0.268 | +0.359 | 0.663 | 0.847 | 0.864 | 0.871 |
| s_conf | +0.684 | +0.549 | +0.512 | 0.707 | 0.845 | 0.867 | 0.881 |
| s_cf_iou | +0.403 | +0.261 | +0.291 | 0.609 | 0.840 | 0.850 | 0.858 |
| s_c16_iou | +0.526 | +0.414 | +0.422 | 0.671 | 0.842 | 0.861 | 0.876 |
| s_crf_iou | +0.200 | +0.232 | +0.135 | 0.590 | 0.836 | 0.843 | 0.846 |
| s_call_iou | +0.456 | +0.257 | +0.225 | 0.574 | 0.837 | 0.846 | 0.860 |
| oracle | +1.000 | +1.000 | +1.000 | 0.761 | 0.859 | 0.895 | 0.918 |

All images: 0.830.

Pixel level: which pixels of the BRACE mask are wrong? AUROC of each uncertainty map for error pixels, mean over images (40k pixels each); *strat* = the same within boundary-distance bins (<2, 2–5, 5–10, 10–20, >20 px), i.e. what a map adds beyond 'errors sit at the boundary'.

| uncertainty | AUROC | AUROC within distance bins |
|---|---:|---:|
| u_boundary | 0.878 | 0.590 |
| u_conf | 0.887 | 0.652 |
| u_cf | 0.696 | 0.559 |
| u_c16 | 0.672 | 0.575 |
| u_clip | 0.827 | 0.631 |
| u_seed | 0.815 | 0.634 |
| u_call | 0.699 | 0.549 |

## COCO-Cap (707 images; frozen BRACE mask class-macro IoU 0.696; 142 masks with IoU < 0.5)

Image level. ρ = Spearman with the mask's true IoU; ρ_wc = mean within-class ρ; ρ_wc|size = within-class partial ρ controlling log predicted area; AUROC_wc = detecting IoU < 0.5 with scores ranked within class; retained class-macro mIoU when keeping the top q of each class (random keep = all-data mIoU in expectation).

| signal | ρ | ρ_wc | ρ_wc\|size | AUROC_wc | keep 90% | keep 70% | keep 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| s_area | -0.055 | +0.123 | -0.016 | 0.495 | 0.699 | 0.701 | 0.711 |
| s_seedpeak | +0.287 | +0.003 | +0.100 | 0.552 | 0.694 | 0.697 | 0.700 |
| s_clip_iou | +0.451 | +0.433 | +0.354 | 0.733 | 0.717 | 0.751 | 0.783 |
| s_clip_soft | +0.445 | +0.351 | +0.381 | 0.734 | 0.714 | 0.758 | 0.779 |
| s_conf | +0.502 | +0.441 | +0.435 | 0.581 | 0.709 | 0.734 | 0.750 |
| s_cf_iou | +0.455 | +0.344 | +0.357 | 0.638 | 0.714 | 0.730 | 0.757 |
| s_c16_iou | +0.490 | +0.402 | +0.404 | 0.657 | 0.717 | 0.745 | 0.759 |
| s_crf_iou | +0.066 | +0.180 | +0.162 | 0.483 | 0.700 | 0.703 | 0.709 |
| s_call_iou | +0.487 | +0.392 | +0.380 | 0.649 | 0.713 | 0.733 | 0.759 |
| oracle | +1.000 | +1.000 | +1.000 | 0.892 | 0.743 | 0.815 | 0.871 |

All images: 0.696.

Pixel level: which pixels of the BRACE mask are wrong? AUROC of each uncertainty map for error pixels, mean over images (40k pixels each); *strat* = the same within boundary-distance bins (<2, 2–5, 5–10, 10–20, >20 px), i.e. what a map adds beyond 'errors sit at the boundary'.

| uncertainty | AUROC | AUROC within distance bins |
|---|---:|---:|
| u_boundary | 0.856 | 0.589 |
| u_conf | 0.860 | 0.637 |
| u_cf | 0.720 | 0.601 |
| u_c16 | 0.662 | 0.589 |
| u_clip | 0.862 | 0.687 |
| u_seed | 0.859 | 0.684 |
| u_call | 0.754 | 0.583 |

## Registered transfer test

BRACE-internal signal chosen on VOC-Sim by within-class size-controlled ρ: **s_conf** (VOC-Sim +0.512). COCO-Cap: s_conf +0.435 vs CLIP-agreement (s_clip_soft, prior-art style) +0.381.
Δ (BRACE-internal − CLIP agreement) = +0.054, image-bootstrap 95% CI -0.056…+0.158 → **no difference**.

