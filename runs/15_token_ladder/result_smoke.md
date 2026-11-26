# Component ladder: class word vs OVAM's optimized token (`PREREG.md`)

`scripts/jc_ladder.py`. One live generation per image with the frozen runtime; both query columns, all rows, from that generation. **W** = class word (frozen attribution). **T** = OVAM's released VOC-Sim token (few-shot: needs annotated images; the same VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *@0.55* = fixed threshold (primary). *VOC-sel.* = threshold selected on VOC-Sim: 5-fold out-of-fold on VOC-Sim, the VOC-Sim-best threshold applied unchanged on COCO-Cap. Δ with image-bootstrap 95% interval (2000). Class-word results are in-sample for the method design (these sets were used to develop it).

## VOC-Sim (3 images)

Regeneration: 1/3 images bit-identical to the stored image; class-word maps vs the stored capture: median max |Δ| 9.2e-03, 95th percentile 1.2e-02.

| Row | Step | W @0.55 | Δ W vs ref | T @0.55 | Δ T vs ref | T − W @0.55 | W VOC-sel. | T VOC-sel. | folds W | folds T |
|---|---|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | CA seed, uniform layer weights | 0.625 | — | 0.674 | — | +0.048 [+0.024, +0.106] | 0.686 | 0.750 | — | — |
| 2 | CA seed, SA-guided layer weights | 0.690 | +0.065 [+0.019, +0.098] | 0.682 | +0.009 [-0.026, +0.039] | -0.008 [-0.035, +0.061] | 0.624 | 0.733 | 0/5 | 0/5 |
| 3 | + fine-only SA propagation (two passes) | 0.887 | +0.197 [+0.146, +0.266] | 0.848 | +0.165 [+0.153, +0.196] | -0.039 [-0.074, -0.007] | 0.890 | 0.877 | 3/5 | 3/5 |
| 4 | + 16→32→64 hierarchy, no cap | 0.901 | +0.014 [-0.048, +0.061] | 0.903 | +0.055 [-0.026, +0.131] | +0.002 [-0.004, +0.013] | 0.881 | 0.868 | 1/5 | 1/5 |
| 5 | + one-sided cap at 16→32 | 0.892 | -0.009 [-0.016, -0.005] | 0.902 | -0.001 [-0.007, +0.003] | +0.010 [+0.003, +0.027] | 0.886 | 0.884 | 1/5 | 3/5 |
| 6 | + DenseCRF (full pipeline) | 0.943 | +0.051 [+0.022, +0.126] | 0.942 | +0.040 [+0.019, +0.098] | -0.001 [-0.003, +0.000] | 0.938 | 0.942 | 3/5 | 3/5 |
| 3c | fine-only + DenseCRF | 0.911 | — | 0.873 | — | -0.038 [-0.071, -0.007] | 0.948 | 0.946 | — | — |
| 6u | full pipeline, uniform layer weights | 0.944 | — | 0.942 | — | -0.002 [-0.003, +0.000] | 0.943 | 0.948 | — | — |

Reference of each Δ: the row above (rows 2–6). Row 6 − row 5 is the DenseCRF effect (raw to DenseCRF); all other Δ compare like with like.

## COCO-Cap (3 images)

Regeneration: 1/3 images bit-identical to the stored image; class-word maps vs the stored capture: median max |Δ| 1.7e-03, 95th percentile 1.4e-02.

| Row | Step | W @0.55 | Δ W vs ref | T @0.55 | Δ T vs ref | T − W @0.55 | W VOC-sel. | T VOC-sel. |
|---|---|---:|---|---:|---|---|---:|---:|
| 1 | CA seed, uniform layer weights | 0.603 | — | 0.589 | — | -0.014 [-0.045, +0.043] | 0.662 | 0.646 |
| 2 | CA seed, SA-guided layer weights | 0.628 | +0.026 [-0.088, +0.090] | 0.597 | +0.009 [-0.054, +0.065] | -0.031 [-0.079, +0.017] | 0.612 | 0.600 |
| 3 | + fine-only SA propagation (two passes) | 0.820 | +0.191 [+0.125, +0.307] | 0.841 | +0.243 [+0.185, +0.311] | +0.021 [-0.004, +0.044] | 0.777 | 0.780 |
| 4 | + 16→32→64 hierarchy, no cap | 0.780 | -0.039 [-0.102, -0.013] | 0.793 | -0.048 [-0.109, -0.009] | +0.012 [-0.004, +0.025] | 0.810 | 0.805 |
| 5 | + one-sided cap at 16→32 | 0.759 | -0.021 [-0.033, -0.001] | 0.780 | -0.012 [-0.018, -0.003] | +0.021 [-0.006, +0.041] | 0.820 | 0.804 |
| 6 | + DenseCRF (full pipeline) | 0.881 | +0.122 [+0.095, +0.316] | 0.912 | +0.132 [+0.099, +0.322] | +0.031 [-0.000, +0.057] | 0.873 | 0.900 |
| 3c | fine-only + DenseCRF | 0.907 | — | 0.921 | — | +0.014 [+0.000, +0.023] | 0.866 | 0.898 |
| 6u | full pipeline, uniform layer weights | 0.881 | — | 0.908 | — | +0.027 [+0.000, +0.049] | 0.874 | 0.892 |

Reference of each Δ: the row above (rows 2–6). Row 6 − row 5 is the DenseCRF effect (raw to DenseCRF); all other Δ compare like with like.

## Registered comparisons (@0.55)

| ID | Comparison | Query | VOC-Sim Δ | COCO-Cap Δ | Verdict |
|---|---|---|---|---|---|
| H | capped hierarchy vs fine-only (both + DenseCRF) | optimized token (registered) | +0.069 [+0.005, +0.122] | -0.009 [-0.029, +0.020] | **does not hold** |
| H | capped hierarchy vs fine-only (both + DenseCRF) | class word (rerun, reference) | +0.032 [-0.002, +0.054] | -0.026 [-0.063, +0.024] | does not hold |
| L | SA-guided vs uniform layer weights (full pipeline) | optimized token (registered) | -0.000 [-0.001, +0.000] | +0.004 [-0.000, +0.008] | **does not hold** |
| L | SA-guided vs uniform layer weights (full pipeline) | class word (rerun, reference) | -0.001 [-0.001, -0.000] | -0.000 [-0.000, +0.000] | does not hold |
| C | cap vs no cap (raw) | optimized token (registered) | -0.001 [-0.007, +0.003] | -0.012 [-0.018, -0.003] | **does not hold** |
| C | cap vs no cap (raw) | class word (rerun, reference) | -0.009 [-0.016, -0.005] | -0.021 [-0.033, -0.001] | does not hold |
| Q | optimized token vs class word, full pipeline | descriptive | -0.001 [-0.003, +0.000] | +0.031 [-0.000, +0.057] | — |

Rule (H, L, C separately): holds if the COCO-Cap Δ is positive with its interval above zero and the VOC-Sim Δ is not below −0.005.

Per class, VOC-Sim, full pipeline @0.55 (W / T): aeroplane 0.94 / 0.94

Per class, COCO-Cap, full pipeline @0.55 (W / T): aeroplane 0.88 / 0.91

