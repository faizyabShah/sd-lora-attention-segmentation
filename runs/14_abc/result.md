# (A) green-screen tokens, (B) depth–time signature, (C) caption-competitive attribution (`PREREG.md`)

Replayed stored trajectories of the 150 + 150 screening images, exact A3 seed weighting, all arms from one replay. Development images (disclosed). τ 0.55 + DenseCRF R2, class-macro mIoU, image-bootstrap 95% CI.

## Verdicts: A tokens **STOP**, B signature **CLOSE**, C caption attribution **STOP**

| arm | role | VOC-Sim mIoU | Δ vs base (95% CI) | COCO-Cap mIoU | Δ vs base (95% CI) |
|---|---|---:|---|---:|---|
| live | fidelity (stored live seed) | 0.8028 | +0.0172 [+0.0083, +0.0265] | 0.6612 | +0.0165 [-0.0102, +0.0427] |
| base | frozen attribution, exact A3 replay | 0.7856 | — | 0.6447 | — |
| clsonly | class name alone | 0.7891 | +0.0035 [-0.0051, +0.0118] | 0.6447 | +0.0000 [+0.0000, +0.0000] |
| tmpl | C control: "A photograph of a {class}" | 0.7856 | +0.0000 [+0.0000, +0.0000] | 0.6412 | -0.0035 [-0.0198, +0.0131] |
| incap | **C primary**: class word inside the caption | 0.7856 | +0.0000 [+0.0000, +0.0000] | 0.6471 | +0.0023 [-0.0231, +0.0313] |
| gs | **A primary**: green-screen token | 0.7780 | -0.0076 [-0.0276, +0.0109] | 0.6633 | +0.0185 [-0.0147, +0.0547] |
| gs_lin | A secondary: closed-form step | 0.7907 | +0.0050 [-0.0033, +0.0129] | 0.6459 | +0.0012 [-0.0143, +0.0192] |
| gs − clsonly | token effect alone | | -0.0111 [-0.0285, +0.0066] | | +0.0185 [-0.0147, +0.0547] |
| incap − tmpl | competition beyond a sentence | | +0.0000 [+0.0000, +0.0000] | | +0.0058 [-0.0198, +0.0320] |

### (C) COCO-Cap split

| subset | images | base | incap | Δ (95% CI) |
|---|---:|---:|---:|---|
| word in caption | 95 | 0.6694 | 0.7008 | +0.0314 [-0.0021, +0.0623] |
| word appended | 55 | 0.6144 | 0.5848 | -0.0296 [-0.0659, -0.0136] |

Fallbacks to the class name alone (class word truncated): 0.

### (B) Bleed vs correct on Q cells: baseline of run 09 vs + `sig` + `fl`

| set | cells (bleed) | baseline AUROC | Δ (95% CI) | single: conf / sig / fl |
|---|---|---:|---|---|
| VOC-Sim | 145,030 (15.2%) | 0.777 | -0.0002 [-0.0106, +0.0093] | 0.229 / 0.356 / 0.624 |
| COCO-Cap | 109,686 (31.1%) | 0.751 | +0.0002 [-0.0105, +0.0105] | 0.315 / 0.344 / 0.610 |

### (A) Exemplars and tokens

| class | generated | kept | token | loss first → last |
|---|---:|---:|---|---|
| aeroplane | 18 | 10 | learned | 0.940 → 0.302 |
| bicycle | 40 | 8 | learned | 1.003 → 0.346 |
| bird | 20 | 10 | learned | 0.890 → 0.455 |
| boat | 40 | 9 | learned | 0.901 → 0.634 |
| bottle | 13 | 10 | learned | 0.943 → 0.415 |
| bus | 40 | 5 | learned | 0.864 → 0.427 |
| car | 39 | 10 | learned | 0.946 → 0.402 |
| cat | 14 | 10 | learned | 0.857 → 0.321 |
| chair | 17 | 10 | learned | 1.018 → 0.523 |
| cow | 40 | 9 | learned | 0.845 → 0.441 |
| dining table | 26 | 10 | learned | 1.165 → 0.566 |
| dog | 14 | 10 | learned | 0.920 → 0.363 |
| horse | 20 | 10 | learned | 0.898 → 0.525 |
| motorbike | 27 | 10 | learned | 0.944 → 0.383 |
| person | 23 | 10 | learned | 1.059 → 0.518 |
| potted plant | 15 | 10 | learned | 0.921 → 0.480 |
| sheep | 29 | 10 | learned | 0.920 → 0.448 |
| sofa | 21 | 10 | learned | 1.081 → 0.507 |
| train | 40 | 2 | original (too few exemplars) | — |
| tv | 19 | 10 | learned | 0.924 → 0.469 |

VOC-Sim: 150 images, 142 with a learned token; r(replay seed, live seed) = 0.953; r(V3 from replay seed, stored V3) = 0.974; 12 s/image.

COCO-Cap: 150 images, 142 with a learned token; r(replay seed, live seed) = 0.899; r(V3 from replay seed, stored V3) = 0.922; 12 s/image.

Screens and diagnostics only. A promising arm licenses a pre-registered confirmation on fresh images with a live capture, never a BRACE change.
