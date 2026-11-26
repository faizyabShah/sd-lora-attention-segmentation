# (A) green-screen tokens, (B) depth–time signature, (C) caption-competitive attribution (`PREREG.md`)

**Note (added by hand):** this is the token-free replay (`bc`). It is the registered result for (B) and (C) only. No token existed, so the `gs` rows equal `clsonly` and the "A tokens" verdict below is not a result; (A) is scored in `result.md`.

Replayed stored trajectories of the 150 + 150 screening images, exact A3 seed weighting, all arms from one replay. Development images (disclosed). τ 0.55 + DenseCRF R2, class-macro mIoU, image-bootstrap 95% CI.

## Verdicts: A tokens **STOP**, B signature **CLOSE**, C caption attribution **STOP**

| arm | role | VOC-Sim mIoU | Δ vs base (95% CI) | COCO-Cap mIoU | Δ vs base (95% CI) |
|---|---|---:|---|---:|---|
| live | fidelity (stored live seed) | 0.8028 | +0.0172 [+0.0083, +0.0265] | 0.6612 | +0.0164 [-0.0102, +0.0427] |
| base | frozen attribution, exact A3 replay | 0.7856 | — | 0.6448 | — |
| clsonly | class name alone | 0.7891 | +0.0035 [-0.0051, +0.0118] | 0.6448 | +0.0000 [+0.0000, +0.0000] |
| tmpl | C control: "A photograph of a {class}" | 0.7856 | +0.0000 [+0.0000, +0.0000] | 0.6413 | -0.0035 [-0.0198, +0.0131] |
| incap | **C primary**: class word inside the caption | 0.7856 | +0.0000 [+0.0000, +0.0000] | 0.6471 | +0.0023 [-0.0231, +0.0313] |
| gs | **A primary**: green-screen token | 0.7891 | +0.0035 [-0.0051, +0.0118] | 0.6448 | +0.0000 [+0.0000, +0.0000] |
| gs_lin | A secondary: closed-form step | 0.7891 | +0.0035 [-0.0051, +0.0118] | 0.6448 | +0.0000 [+0.0000, +0.0000] |
| gs − clsonly | token effect alone | | +0.0000 [+0.0000, +0.0000] | | +0.0000 [+0.0000, +0.0000] |
| incap − tmpl | competition beyond a sentence | | +0.0000 [+0.0000, +0.0000] | | +0.0058 [-0.0198, +0.0320] |

### (C) COCO-Cap split

| subset | images | base | incap | Δ (95% CI) |
|---|---:|---:|---:|---|
| word in caption | 95 | 0.6694 | 0.7008 | +0.0314 [-0.0022, +0.0623] |
| word appended | 55 | 0.6145 | 0.5848 | -0.0296 [-0.0659, -0.0137] |

Fallbacks to the class name alone (class word truncated): 0.

### (B) Bleed vs correct on Q cells: baseline of run 09 vs + `sig` + `fl`

| set | cells (bleed) | baseline AUROC | Δ (95% CI) | single: conf / sig / fl |
|---|---|---:|---|---|
| VOC-Sim | 145,030 (15.2%) | 0.777 | -0.0002 [-0.0106, +0.0093] | 0.229 / 0.356 / 0.624 |
| COCO-Cap | 109,686 (31.1%) | 0.751 | +0.0002 [-0.0105, +0.0105] | 0.315 / 0.344 / 0.610 |

VOC-Sim: 150 images, 0 with a learned token; r(replay seed, live seed) = 0.953; r(V3 from replay seed, stored V3) = 0.974; 12 s/image.

COCO-Cap: 150 images, 0 with a learned token; r(replay seed, live seed) = 0.900; r(V3 from replay seed, stored V3) = 0.922; 8 s/image.

Screens and diagnostics only. A promising arm licenses a pre-registered confirmation on fresh images with a live capture, never a BRACE change.
