# Component ladder: class word vs OVAM's optimized token (`PREREG.md`)

`scripts/jc_ladder.py`. One live generation per image with the frozen runtime; both query columns, all rows, from that generation. **W** = class word (frozen attribution). **T** = OVAM's released VOC-Sim token (few-shot: needs annotated images; the same VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *@0.55* = fixed threshold (primary). *VOC-sel.* = threshold selected on VOC-Sim: 5-fold out-of-fold on VOC-Sim, the VOC-Sim-best threshold applied unchanged on COCO-Cap. Δ with image-bootstrap 95% interval (2000). Class-word results are in-sample for the method design (these sets were used to develop it).

## VOC-Sim (490 images)

Regeneration: 238/490 images bit-identical to the stored image; class-word maps vs the stored capture: median max |Δ| 8.3e-04, 95th percentile 5.5e-02.

| Row | Step | W @0.55 | Δ W vs ref | T @0.55 | Δ T vs ref | T − W @0.55 | W VOC-sel. | T VOC-sel. | folds W | folds T |
|---|---|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | CA seed, uniform layer weights | 0.588 | — | 0.766 | — | +0.178 [+0.166, +0.189] | 0.713 | 0.796 | — | — |
| 2 | CA seed, SA-guided layer weights | 0.611 | +0.023 [+0.016, +0.030] | 0.754 | -0.012 [-0.016, -0.009] | +0.143 [+0.132, +0.153] | 0.705 | 0.775 | 0/5 | 0/5 |
| 3 | + fine-only SA propagation (two passes) | 0.791 | +0.180 [+0.168, +0.193] | 0.845 | +0.091 [+0.082, +0.100] | +0.054 [+0.042, +0.064] | 0.810 | 0.858 | 5/5 | 5/5 |
| 4 | + 16→32→64 hierarchy, no cap | 0.807 | +0.016 [+0.008, +0.024] | 0.835 | -0.010 [-0.016, -0.002] | +0.028 [+0.019, +0.037] | 0.807 | 0.835 | 3/5 | 0/5 |
| 5 | + one-sided cap at 16→32 | 0.817 | +0.010 [+0.004, +0.016] | 0.847 | +0.012 [+0.007, +0.016] | +0.030 [+0.021, +0.037] | 0.816 | 0.848 | 4/5 | 5/5 |
| 6 | + DenseCRF (full pipeline) | 0.830 | +0.013 [+0.010, +0.016] | 0.861 | +0.014 [+0.011, +0.016] | +0.031 [+0.022, +0.039] | 0.830 | 0.861 | 5/5 | 5/5 |
| 3c | fine-only + DenseCRF | 0.796 | — | 0.851 | — | +0.055 [+0.042, +0.067] | 0.824 | 0.870 | — | — |
| 6u | full pipeline, uniform layer weights | 0.826 | — | 0.857 | — | +0.031 [+0.022, +0.041] | 0.826 | 0.857 | — | — |

Reference of each Δ: the row above (rows 2–6). Row 6 − row 5 is the DenseCRF effect (raw to DenseCRF); all other Δ compare like with like.

## COCO-Cap (707 images)

Regeneration: 243/707 images bit-identical to the stored image; class-word maps vs the stored capture: median max |Δ| 2.9e-03, 95th percentile 6.1e-02.

| Row | Step | W @0.55 | Δ W vs ref | T @0.55 | Δ T vs ref | T − W @0.55 | W VOC-sel. | T VOC-sel. |
|---|---|---:|---|---:|---|---|---:|---:|
| 1 | CA seed, uniform layer weights | 0.444 | — | 0.625 | — | +0.181 [+0.164, +0.197] | 0.563 | 0.638 |
| 2 | CA seed, SA-guided layer weights | 0.501 | +0.056 [+0.046, +0.065] | 0.617 | -0.008 [-0.014, -0.002] | +0.116 [+0.100, +0.132] | 0.584 | 0.624 |
| 3 | + fine-only SA propagation (two passes) | 0.677 | +0.176 [+0.160, +0.193] | 0.689 | +0.072 [+0.060, +0.086] | +0.012 [-0.002, +0.029] | 0.660 | 0.673 |
| 4 | + 16→32→64 hierarchy, no cap | 0.687 | +0.010 [-0.001, +0.023] | 0.669 | -0.020 [-0.029, -0.012] | -0.018 [-0.033, -0.002] | 0.678 | 0.669 |
| 5 | + one-sided cap at 16→32 | 0.682 | -0.005 [-0.011, -0.001] | 0.667 | -0.002 [-0.006, +0.003] | -0.014 [-0.028, +0.000] | 0.682 | 0.674 |
| 6 | + DenseCRF (full pipeline) | 0.696 | +0.014 [+0.012, +0.017] | 0.681 | +0.013 [+0.012, +0.016] | -0.015 [-0.030, +0.000] | 0.696 | 0.681 |
| 3c | fine-only + DenseCRF | 0.684 | — | 0.698 | — | +0.014 [-0.002, +0.031] | 0.674 | 0.678 |
| 6u | full pipeline, uniform layer weights | 0.672 | — | 0.677 | — | +0.004 [-0.011, +0.021] | 0.672 | 0.677 |

Reference of each Δ: the row above (rows 2–6). Row 6 − row 5 is the DenseCRF effect (raw to DenseCRF); all other Δ compare like with like.

## Registered comparisons (@0.55)

| ID | Comparison | Query | VOC-Sim Δ | COCO-Cap Δ | Verdict |
|---|---|---|---|---|---|
| H | capped hierarchy vs fine-only (both + DenseCRF) | optimized token (registered) | +0.010 [+0.002, +0.017] | -0.017 [-0.028, -0.007] | **does not hold** |
| H | capped hierarchy vs fine-only (both + DenseCRF) | class word (rerun, reference) | +0.034 [+0.024, +0.044] | +0.012 [-0.002, +0.026] | does not hold |
| L | SA-guided vs uniform layer weights (full pipeline) | optimized token (registered) | +0.003 [+0.000, +0.008] | +0.004 [-0.002, +0.011] | **does not hold** |
| L | SA-guided vs uniform layer weights (full pipeline) | class word (rerun, reference) | +0.004 [-0.001, +0.010] | +0.024 [+0.013, +0.036] | holds |
| C | cap vs no cap (raw) | optimized token (registered) | +0.012 [+0.007, +0.016] | -0.002 [-0.006, +0.003] | **does not hold** |
| C | cap vs no cap (raw) | class word (rerun, reference) | +0.010 [+0.004, +0.016] | -0.005 [-0.011, -0.001] | does not hold |
| Q | optimized token vs class word, full pipeline | descriptive | +0.031 [+0.022, +0.039] | -0.015 [-0.030, +0.000] | — |

Rule (H, L, C separately): holds if the COCO-Cap Δ is positive with its interval above zero and the VOC-Sim Δ is not below −0.005.

Per class, VOC-Sim, full pipeline @0.55 (W / T): aeroplane 0.88 / 0.90, bicycle 0.79 / 0.80, bird 0.92 / 0.93, boat 0.86 / 0.85, bottle 0.88 / 0.90, bus 0.92 / 0.92, car 0.85 / 0.86, cat 0.85 / 0.88, chair 0.79 / 0.80, cow 0.90 / 0.95, dining-table 0.42 / 0.43, dog 0.93 / 0.93, horse 0.90 / 0.92, motorbike 0.87 / 0.88, person 0.81 / 0.86, potted-plant 0.72 / 0.78, sheep 0.75 / 0.87, sofa 0.96 / 0.96, train 0.74 / 0.88, tv 0.85 / 0.92

Per class, COCO-Cap, full pipeline @0.55 (W / T): aeroplane 0.80 / 0.87, bicycle 0.63 / 0.60, bird 0.78 / 0.77, boat 0.59 / 0.57, bottle 0.50 / 0.46, bus 0.92 / 0.92, car 0.76 / 0.66, cat 0.87 / 0.84, chair 0.47 / 0.46, cow 0.87 / 0.87, dining table 0.45 / 0.51, dog 0.75 / 0.71, horse 0.72 / 0.66, motorbike 0.67 / 0.64, person 0.28 / 0.62, potted plant 0.74 / 0.71, sheep 0.87 / 0.82, sofa 0.70 / 0.34, train 0.75 / 0.85, tv 0.79 / 0.73

