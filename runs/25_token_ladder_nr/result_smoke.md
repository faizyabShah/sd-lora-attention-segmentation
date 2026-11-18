# Stage ladder: class word vs OVAM's optimized token, corrected data (`PREREG.md`)

`scripts/jc_ladder_nr.py`. One live generation per image (frozen runtime, runs/20 corrected prompts, safety checker off); both query columns and all stages from that generation. **Class word** = frozen attribution (BRACE). **Optimized token** = OVAM's released VOC-Sim token (few-shot: needs annotated images; the same VOC-Sim tokens on COCO-Cap). Class-macro mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold applied unchanged on COCO-Cap. Δ with paired image-bootstrap 95% interval (50). Class-word results are in-sample for the method design. Supersedes runs/15 (old prompts).

## Data checks

- VOC-Sim: 1 images; image identical to the runs/20 capture on 1/1; class-word BRACE map vs runs/20: median max |Δ| 1.2e-03.
- COCO-Cap: 1 images; image identical to the runs/20 capture on 0/1; class-word BRACE map vs runs/20: median max |Δ| 6.3e-03.

## Stage table, with DenseCRF (primary), SA-guided layer weights

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.273 | 0.316 | 0.460 | 0.416 | 0.434 | 0.422 | 0.621 | 0.640 |
| CA + fine SA | 0.273 | 0.286 | 0.405 | 0.384 | 0.750 | 0.711 | 0.964 | 0.910 |
| CA + hierarchy (BRACE) | 0.196 | 0.195 | 0.335 | 0.324 | 0.829 | 0.831 | 0.959 | 0.956 |

## Stage table, raw (secondary), SA-guided layer weights

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.276 | 0.314 | 0.459 | 0.415 | 0.460 | 0.432 | 0.601 | 0.627 |
| CA + fine SA | 0.272 | 0.284 | 0.402 | 0.379 | 0.727 | 0.699 | 0.905 | 0.875 |
| CA + hierarchy (BRACE) | 0.197 | 0.195 | 0.327 | 0.317 | 0.804 | 0.806 | 0.905 | 0.901 |

## Uniform layer weights, with DenseCRF (secondary)

| Stage | VOC-Sim word, VOC-sel. | VOC-Sim token, VOC-sel. | VOC-Sim word, 0.55 | VOC-Sim token, 0.55 | COCO-Cap word, VOC-sel. | COCO-Cap token, VOC-sel. | COCO-Cap word, 0.55 | COCO-Cap token, 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CA | 0.275 | 0.309 | 0.370 | 0.433 | 0.667 | 0.459 | 0.499 | 0.624 |
| CA + fine SA | 0.246 | 0.287 | 0.396 | 0.388 | 0.771 | 0.696 | 0.960 | 0.916 |
| CA + hierarchy (BRACE) | 0.188 | 0.198 | 0.297 | 0.301 | 0.810 | 0.816 | 0.960 | 0.957 |

## Registered comparisons (with DenseCRF)

| ID | Comparison | Query | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 | Verdict (0.55) |
|---|---|---|---|---|---|---|---|
| F | CA + fine − CA | class word | +0.001 [+0.001, +0.001] | +0.315 [+0.315, +0.315] | -0.055 [-0.055, -0.055] | +0.343 [+0.343, +0.343] | does not hold |
| F | CA + fine − CA | optimized token | -0.030 [-0.030, -0.030] | +0.289 [+0.289, +0.289] | -0.032 [-0.032, -0.032] | +0.271 [+0.271, +0.271] | does not hold |
| H | CA + hierarchy − CA + fine | class word | -0.077 [-0.077, -0.077] | +0.079 [+0.079, +0.079] | -0.070 [-0.070, -0.070] | -0.006 [-0.006, -0.006] | does not hold |
| H | CA + hierarchy − CA + fine | optimized token | -0.091 [-0.091, -0.091] | +0.120 [+0.120, +0.120] | -0.060 [-0.060, -0.060] | +0.046 [+0.046, +0.046] | does not hold |
| — | CA + hierarchy − CA | class word | -0.077 [-0.077, -0.077] | +0.394 [+0.394, +0.394] | -0.125 [-0.125, -0.125] | +0.337 [+0.337, +0.337] | — |
| — | CA + hierarchy − CA | optimized token | -0.121 [-0.121, -0.121] | +0.409 [+0.409, +0.409] | -0.092 [-0.092, -0.092] | +0.317 [+0.317, +0.317] | — |
| Q | token − word: CA | — | +0.043 [+0.043, +0.043] | -0.013 [-0.013, -0.013] | -0.044 [-0.044, -0.044] | +0.018 [+0.018, +0.018] | descriptive |
| Q | token − word: CA + fine SA | — | +0.013 [+0.013, +0.013] | -0.039 [-0.039, -0.039] | -0.021 [-0.021, -0.021] | -0.054 [-0.054, -0.054] | descriptive |
| Q | token − word: CA + hierarchy (BRACE) | — | -0.001 [-0.001, -0.001] | +0.002 [+0.002, +0.002] | -0.011 [-0.011, -0.011] | -0.002 [-0.002, -0.002] | descriptive |

Rule (F, H, per query): holds if the COCO-Cap Δ at 0.55 is positive with interval above zero and the VOC-Sim Δ at 0.55 is not below −0.005.

Per class, VOC-Sim, CA + hierarchy + DenseCRF @0.55 (word / token): dining-table 0.34 / 0.32

Per class, COCO-Cap, CA + hierarchy + DenseCRF @0.55 (word / token): cat 0.96 / 0.96

