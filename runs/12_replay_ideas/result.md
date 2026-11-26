# Training-free SA head weighting (SCW) and closed-form context refinement (ROC): screen (`PREREG.md`)

Replayed stored trajectories of the 150 + 150 screening images (development data; disclosed). All arms from the same replay. Fixed τ 0.55 + DenseCRF R2. Class-macro mIoU; image-bootstrap 95% CI of Δ vs replay baseline (2000).

## Screen verdicts (primary): SCW **STOP**, ROC β 0.1 **STOP**

Bar: COCO-Cap Δ ≥ +0.010 with CI lower > 0, and VOC-Sim Δ ≥ −0.005 (vs replay baseline).

| arm | role | VOC-Sim mIoU | Δ (95% CI) | COCO-Cap mIoU | Δ (95% CI) |
|---|---|---:|---|---:|---|
| live | fidelity: stored live A3 seed + replayed SA | 0.8028 | +0.0137 [+0.0024, +0.0266] | 0.6612 | +0.0586 [+0.0230, +0.0903] |
| base | replay baseline (A3 modules, uniform layer weights) | 0.7891 | — | 0.6026 | — |
| ent | reference: iSeg-style entropy weighting | 0.7874 | -0.0017 [-0.0098, +0.0069] | 0.6192 | +0.0165 [-0.0016, +0.0452] |
| scw | **primary A** | 0.7828 | -0.0063 [-0.0138, +0.0019] | 0.6127 | +0.0100 [-0.0081, +0.0318] |
| roc0.05 | secondary | 0.7850 | -0.0041 [-0.0075, -0.0009] | 0.6157 | +0.0131 [+0.0013, +0.0349] |
| roc0.1 | **primary B** | 0.7809 | -0.0083 [-0.0144, -0.0024] | 0.6207 | +0.0181 [+0.0016, +0.0452] |
| roc0.2 | secondary | 0.7714 | -0.0177 [-0.0290, -0.0069] | 0.6280 | +0.0253 [+0.0050, +0.0559] |
| scw+roc0.1 | secondary combo | 0.7734 | -0.0158 [-0.0277, -0.0037] | 0.6379 | +0.0353 [+0.0114, +0.0777] |

VOC-Sim: 150 images; replay fidelity r(live-seed V3 replay, stored live V3) = 0.981; r(replay seed, live A3 seed) = 0.937; 8 s/image.

COCO-Cap: 150 images; replay fidelity r(live-seed V3 replay, stored live V3) = 0.978; r(replay seed, live A3 seed) = 0.730; 4 s/image.

Screen only (development images). A promising arm licenses a pre-registered confirmation on fresh images, not a BRACE change.
