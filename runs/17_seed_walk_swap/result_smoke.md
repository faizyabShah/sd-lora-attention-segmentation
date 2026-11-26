# Seed × propagation swap between BRACE and NERVE (`PREREG.md`)

`scripts/jc_swap.py`, `scripts/jc_swap_nerve.py`. Seeds: **CA** = BRACE's cross-attention seed (live generation); **CLIP** = NERVE's CLIP ViT-L/14 seed for {class word} + NERVE's background words (stored image). Propagations: BRACE fine-only and capped hierarchy (generation self-attention, frozen code); **NERVE walk** = NERVE's released refinement, unchanged (stored image, one noised pass, closed-form walk). CA × hierarchy is BRACE; CLIP × walk is NERVE (side N's variant A). Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold applied unchanged on COCO-Cap). Primary number: VOC-sel. with DenseCRF. Δ with image-bootstrap 95% interval (2000). CLIP-seed cells use a second model (CLIP) and are not SD-only.

## VOC-Sim (4 images)

Regeneration: 4/4 images bit-identical to the stored image; BRACE maps vs the stored capture: median max |Δ| 0.0e+00.

**VOC-sel. threshold, with DenseCRF (primary)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.652 | 0.474 | -0.178 [-0.413, +0.058] |
| BRACE fine-only | 0.799 | 0.927 | +0.128 [-0.018, +0.274] |
| BRACE capped hierarchy | 0.820 | 0.918 | +0.098 [+0.000, +0.197] |
| NERVE walk | 0.762 | 0.934 | +0.171 [-0.027, +0.369] |

**VOC-sel. threshold, raw**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.531 | 0.477 | -0.054 [-0.225, +0.116] |
| BRACE fine-only | 0.733 | 0.686 | -0.048 [-0.358, +0.262] |
| BRACE capped hierarchy | 0.732 | 0.861 | +0.129 [+0.017, +0.241] |
| NERVE walk | 0.712 | 0.687 | -0.025 [-0.385, +0.334] |

**Fixed threshold 0.5, raw (side N's sanity convention)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) |
|---|---:|---:|
| none (seed alone) | 0.634 | 0.577 |
| BRACE fine-only | 0.759 | 0.736 |
| BRACE capped hierarchy | 0.814 | 0.862 |
| NERVE walk | 0.725 | 0.750 |

## COCO-Cap (4 images)

Regeneration: 0/4 images bit-identical to the stored image; BRACE maps vs the stored capture: median max |Δ| 8.4e-03.

**VOC-sel. threshold, with DenseCRF (primary)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.571 | 0.700 | +0.129 [-0.186, +0.445] |
| BRACE fine-only | 0.557 | 0.766 | +0.209 [+0.068, +0.350] |
| BRACE capped hierarchy | 0.606 | 0.799 | +0.192 [+0.052, +0.333] |
| NERVE walk | 0.591 | 0.787 | +0.196 [+0.044, +0.349] |

**VOC-sel. threshold, raw**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.526 | 0.604 | +0.078 [-0.135, +0.290] |
| BRACE fine-only | 0.551 | 0.703 | +0.151 [+0.016, +0.286] |
| BRACE capped hierarchy | 0.651 | 0.825 | +0.173 [+0.022, +0.324] |
| NERVE walk | 0.547 | 0.709 | +0.162 [+0.007, +0.316] |

**Fixed threshold 0.5, raw (side N's sanity convention)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) |
|---|---:|---:|
| none (seed alone) | 0.557 | 0.634 |
| BRACE fine-only | 0.673 | 0.795 |
| BRACE capped hierarchy | 0.625 | 0.767 |
| NERVE walk | 0.668 | 0.826 |

## Thresholds selected on full VOC-Sim

| Cell | raw | with DenseCRF |
|---|---:|---:|
| cross-attention seed (BRACE) × none (seed alone) | 0.425 | 0.375 |
| cross-attention seed (BRACE) × BRACE fine-only | 0.350 | 0.300 |
| cross-attention seed (BRACE) × BRACE capped hierarchy | 0.525 | 0.400 |
| cross-attention seed (BRACE) × NERVE walk | 0.300 | 0.300 |
| CLIP seed (NERVE) × none (seed alone) | 0.375 | 0.700 |
| CLIP seed (NERVE) × BRACE fine-only | 0.375 | 0.375 |
| CLIP seed (NERVE) × BRACE capped hierarchy | 0.575 | 0.475 |
| CLIP seed (NERVE) × NERVE walk | 0.300 | 0.300 |

## Registered contrasts (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images only | COCO-Cap, raw |
|---|---|---|---|---|---|
| P-CLIP | walk − hierarchy, CLIP seed | +0.015 [-0.018, +0.048] | -0.012 [-0.033, +0.010] | — | -0.116 [-0.162, -0.071] |
| P-CA | walk − hierarchy, CA seed | -0.058 [-0.194, +0.078] | -0.016 [-0.086, +0.054] | — | -0.105 [-0.161, -0.048] |
| S-H | CLIP − CA seed, hierarchy | +0.098 [+0.000, +0.197] | +0.192 [+0.052, +0.333] | — | +0.173 [+0.022, +0.324] |
| S-W | CLIP − CA seed, walk | +0.171 [-0.027, +0.369] | +0.196 [+0.044, +0.349] | — | +0.162 [+0.007, +0.316] |
| I | interaction (P-CLIP − P-CA) | +0.073 [-0.092, +0.238] | +0.004 [-0.045, +0.053] | — | -0.012 [-0.114, +0.091] |

**Reading rule (COCO-Cap).** Seed explains NERVE's lead (S-H and S-W both above zero): **yes**. Walk explains NERVE's lead (P-CLIP and P-CA both above zero): **no**.

## Fidelity checks

- BRACE cell (CA × hierarchy, 0.55, DenseCRF): 0.853 VOC-Sim, 0.708 COCO-Cap (paper: 0.830 / 0.696).
- NERVE cell (CLIP × walk, 0.5 raw, side N's convention), VOC-Sim: 0.7497 here vs 0.7492 from side N's committed counts; per-image IoU: identical counts on 0/4 images, max |ΔIoU| 0.0018, mean |ΔIoU| 0.00051.
- NERVE cell (CLIP × walk, 0.5 raw, side N's convention), COCO-Cap: 0.8262 here vs 0.8295 from side N's committed counts; per-image IoU: identical counts on 0/4 images, max |ΔIoU| 0.0137, mean |ΔIoU| 0.00354.

Per class, VOC-Sim, VOC-sel. + DenseCRF (CA×hierarchy / CA×walk / CLIP×hierarchy / CLIP×walk): aeroplane 0.95 / 0.97 / 0.95 / 0.96, bird 0.92 / 0.91 / 0.92 / 0.92, bottle 0.82 / 0.96 / 0.95 / 0.92, cat 0.59 / 0.21 / 0.85 / 0.94

Per class, COCO-Cap, VOC-sel. + DenseCRF (CA×hierarchy / CA×walk / CLIP×hierarchy / CLIP×walk): aeroplane 0.86 / 0.88 / 0.88 / 0.85, bicycle 0.13 / 0.03 / 0.59 / 0.56, boat 0.78 / 0.72 / 0.87 / 0.83, car 0.65 / 0.74 / 0.86 / 0.90

