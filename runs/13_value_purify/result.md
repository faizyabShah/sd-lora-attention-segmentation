# (V) SA value-space compatibility and (P) semantic subspace purification (diagnostics; `PREREG.md`)

Replayed stored trajectories of the 150 + 150 screening images with the exact A3 seed weighting; all arms from one replay. Development images (disclosed). GT scores only.

## Verdicts (decisive COCO-Cap): V value compatibility **CLOSE**, P purification **CLOSE**

### (V) Edges: crossing (object↔background) vs within-object, among each cell's 16 strongest fine-SA links

| set | images | edges (crossing) | baseline AUROC | + value cosine Δ (95% CI) |
|---|---:|---|---:|---|
| VOC-Sim | 150 | 298,873 (4.7%) | 0.882 | -0.0003 [-0.0012, +0.0007] |
| COCO-Cap | 150 | 289,879 (8.1%) | 0.882 | +0.0003 [-0.0016, +0.0024] |

Single-feature AUROC for crossing (higher = larger on crossing edges):

| feature | VOC-Sim | COCO-Cap |
|---|---:|---:|
| dlab | 0.667 | 0.671 |
| contour | 0.692 | 0.678 |
| dist | 0.692 | 0.683 |
| aff | 0.393 | 0.368 |
| featcos | 0.238 | 0.256 |
| conf_min | 0.186 | 0.170 |
| conf_max | 0.267 | 0.244 |
| valcos | 0.308 | 0.306 |

### (P) Purified class context (τ 0.55 + R2, class-macro mIoU; Δ vs exact-A3 replay `base`, image bootstrap)

| arm | role | VOC-Sim mIoU | Δ (95% CI) | COCO-Cap mIoU | Δ (95% CI) |
|---|---|---:|---|---:|---|
| live | fidelity (stored live seed) | 0.8028 | +0.0172 [+0.0083, +0.0265] | 0.6612 | +0.0164 [-0.0102, +0.0427] |
| base | exact A3 replay | 0.7856 | — | 0.6448 | — |
| pur1 | **primary** (α 1) | 0.7728 | -0.0128 [-0.0227, -0.0031] | 0.6302 | -0.0146 [-0.0344, +0.0029] |
| pur0.5 | secondary (α 0.5) | 0.7834 | -0.0022 [-0.0057, +0.0015] | 0.6365 | -0.0083 [-0.0201, +0.0004] |
| rand | control: random 8-d subspace | 0.7849 | -0.0008 [-0.0021, +0.0003] | 0.6445 | -0.0003 [-0.0038, +0.0031] |
| meandir | control: mean nuisance direction only | 0.7775 | -0.0082 [-0.0158, -0.0004] | 0.6257 | -0.0190 [-0.0406, +0.0047] |
| pur1 − rand | gate condition | | -0.0121 [-0.0222, -0.0020] | | -0.0143 [-0.0349, +0.0039] |

VOC-Sim replay fidelity: r(replay A3 seed, live seed) = 0.953; r(V3 from live seed, stored V3) = 0.981; r(V3 from replay seed, stored V3) = 0.974; 12 s/image.

COCO-Cap replay fidelity: r(replay A3 seed, live seed) = 0.900; r(V3 from live seed, stored V3) = 0.978; r(V3 from replay seed, stored V3) = 0.922; 8 s/image.
