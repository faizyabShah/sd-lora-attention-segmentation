# Seed × propagation swap between BRACE and NERVE (`PREREG.md`)

`scripts/jc_swap.py`, `scripts/jc_swap_nerve.py`. Seeds: **CA** = BRACE's cross-attention seed (live generation); **CLIP** = NERVE's CLIP ViT-L/14 seed for {class word} + NERVE's background words (stored image). Propagations: BRACE fine-only and capped hierarchy (generation self-attention, frozen code); **NERVE walk** = NERVE's released refinement, unchanged (stored image, one noised pass, closed-form walk). CA × hierarchy is BRACE; CLIP × walk is NERVE (side N's variant A). Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold applied unchanged on COCO-Cap). Primary number: VOC-sel. with DenseCRF. Δ with image-bootstrap 95% interval (2000). CLIP-seed cells use a second model (CLIP) and are not SD-only.

## VOC-Sim (490 images)

Regeneration: 238/490 images bit-identical to the stored image; BRACE maps vs the stored capture: median max |Δ| 8.3e-04.

**VOC-sel. threshold, with DenseCRF (primary)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.735 | 0.759 | +0.024 [+0.012, +0.037] |
| BRACE fine-only | 0.823 | 0.830 | +0.007 [-0.008, +0.022] |
| BRACE capped hierarchy | 0.830 | 0.836 | +0.006 [-0.007, +0.018] |
| NERVE walk | 0.828 | 0.842 | +0.014 [+0.001, +0.029] |

**VOC-sel. threshold, raw**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.705 | 0.727 | +0.022 [+0.012, +0.033] |
| BRACE fine-only | 0.810 | 0.814 | +0.004 [-0.009, +0.018] |
| BRACE capped hierarchy | 0.816 | 0.825 | +0.010 [-0.002, +0.022] |
| NERVE walk | 0.819 | 0.832 | +0.013 [+0.000, +0.026] |

**Fixed threshold 0.5, raw (side N's sanity convention)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) |
|---|---:|---:|
| none (seed alone) | 0.661 | 0.714 |
| BRACE fine-only | 0.807 | 0.810 |
| BRACE capped hierarchy | 0.810 | 0.804 |
| NERVE walk | 0.791 | 0.830 |

## COCO-Cap (707 images)

Regeneration: 243/707 images bit-identical to the stored image; BRACE maps vs the stored capture: median max |Δ| 2.9e-03.

**VOC-sel. threshold, with DenseCRF (primary)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.619 | 0.693 | +0.074 [+0.050, +0.098] |
| BRACE fine-only | 0.662 | 0.704 | +0.042 [+0.020, +0.065] |
| BRACE capped hierarchy | 0.696 | 0.704 | +0.008 [-0.012, +0.029] |
| NERVE walk | 0.678 | 0.737 | +0.060 [+0.035, +0.083] |

**VOC-sel. threshold, raw**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) | CLIP − CA |
|---|---:|---:|---|
| none (seed alone) | 0.584 | 0.650 | +0.067 [+0.047, +0.087] |
| BRACE fine-only | 0.660 | 0.707 | +0.047 [+0.025, +0.070] |
| BRACE capped hierarchy | 0.682 | 0.699 | +0.017 [-0.003, +0.039] |
| NERVE walk | 0.668 | 0.744 | +0.076 [+0.052, +0.101] |

**Fixed threshold 0.5, raw (side N's sanity convention)**

| Propagation | cross-attention seed (BRACE) | CLIP seed (NERVE) |
|---|---:|---:|
| none (seed alone) | 0.546 | 0.633 |
| BRACE fine-only | 0.676 | 0.683 |
| BRACE capped hierarchy | 0.655 | 0.657 |
| NERVE walk | 0.682 | 0.730 |

## Thresholds selected on full VOC-Sim

| Cell | raw | with DenseCRF |
|---|---:|---:|
| cross-attention seed (BRACE) × none (seed alone) | 0.400 | 0.400 |
| cross-attention seed (BRACE) × BRACE fine-only | 0.450 | 0.425 |
| cross-attention seed (BRACE) × BRACE capped hierarchy | 0.550 | 0.550 |
| cross-attention seed (BRACE) × NERVE walk | 0.400 | 0.400 |
| CLIP seed (NERVE) × none (seed alone) | 0.625 | 0.650 |
| CLIP seed (NERVE) × BRACE fine-only | 0.575 | 0.525 |
| CLIP seed (NERVE) × BRACE capped hierarchy | 0.625 | 0.600 |
| CLIP seed (NERVE) × NERVE walk | 0.550 | 0.500 |

## Registered contrasts (VOC-sel. threshold, with DenseCRF)

| ID | Contrast | VOC-Sim | COCO-Cap | COCO-Cap, bit-identical images only | COCO-Cap, raw |
|---|---|---|---|---|---|
| P-CLIP | walk − hierarchy, CLIP seed | +0.006 [-0.002, +0.015] | +0.033 [+0.019, +0.049] | +0.030 [+0.006, +0.058] | +0.045 [+0.030, +0.061] |
| P-CA | walk − hierarchy, CA seed | -0.002 [-0.013, +0.008] | -0.019 [-0.031, -0.005] | -0.013 [-0.035, +0.008] | -0.014 [-0.026, -0.002] |
| S-H | CLIP − CA seed, hierarchy | +0.006 [-0.007, +0.018] | +0.008 [-0.012, +0.029] | +0.005 [-0.024, +0.038] | +0.017 [-0.003, +0.039] |
| S-W | CLIP − CA seed, walk | +0.014 [+0.001, +0.029] | +0.060 [+0.035, +0.083] | +0.048 [+0.012, +0.088] | +0.076 [+0.052, +0.101] |
| I | interaction (P-CLIP − P-CA) | +0.008 [-0.004, +0.021] | +0.052 [+0.034, +0.070] | +0.043 [+0.015, +0.072] | +0.059 [+0.041, +0.077] |

**Reading rule (COCO-Cap).** Seed explains NERVE's lead (S-H and S-W both above zero): **no**. Walk explains NERVE's lead (P-CLIP and P-CA both above zero): **no**. Result: mixed.

## Fidelity checks

- BRACE cell (CA × hierarchy, 0.55, DenseCRF): 0.830 VOC-Sim, 0.696 COCO-Cap (paper: 0.830 / 0.696).
- NERVE cell (CLIP × walk, 0.5 raw, side N's convention), VOC-Sim: 0.8302 here vs 0.8302 from side N's committed counts; per-image IoU: identical counts on 40/490 images, max |ΔIoU| 0.0016, mean |ΔIoU| 0.00007.
- NERVE cell (CLIP × walk, 0.5 raw, side N's convention), COCO-Cap: 0.7295 here vs 0.7295 from side N's committed counts; per-image IoU: identical counts on 94/707 images, max |ΔIoU| 0.0021, mean |ΔIoU| 0.00010.

Per class, VOC-Sim, VOC-sel. + DenseCRF (CA×hierarchy / CA×walk / CLIP×hierarchy / CLIP×walk): aeroplane 0.88 / 0.88 / 0.90 / 0.86, bicycle 0.79 / 0.75 / 0.80 / 0.78, bird 0.92 / 0.93 / 0.92 / 0.92, boat 0.86 / 0.83 / 0.85 / 0.86, bottle 0.88 / 0.87 / 0.83 / 0.79, bus 0.92 / 0.92 / 0.89 / 0.89, car 0.85 / 0.84 / 0.83 / 0.82, cat 0.85 / 0.79 / 0.91 / 0.92, chair 0.79 / 0.77 / 0.77 / 0.84, cow 0.90 / 0.93 / 0.96 / 0.96, dining-table 0.42 / 0.46 / 0.40 / 0.43, dog 0.93 / 0.93 / 0.95 / 0.95, horse 0.90 / 0.91 / 0.92 / 0.93, motorbike 0.87 / 0.86 / 0.86 / 0.86, person 0.81 / 0.75 / 0.78 / 0.76, potted-plant 0.72 / 0.75 / 0.79 / 0.85, sheep 0.75 / 0.90 / 0.90 / 0.94, sofa 0.96 / 0.96 / 0.91 / 0.91, train 0.74 / 0.69 / 0.84 / 0.88, tv 0.85 / 0.83 / 0.71 / 0.67

Per class, COCO-Cap, VOC-sel. + DenseCRF (CA×hierarchy / CA×walk / CLIP×hierarchy / CLIP×walk): aeroplane 0.80 / 0.68 / 0.67 / 0.64, bicycle 0.63 / 0.54 / 0.70 / 0.72, bird 0.78 / 0.80 / 0.78 / 0.83, boat 0.59 / 0.59 / 0.64 / 0.69, bottle 0.50 / 0.54 / 0.45 / 0.60, bus 0.92 / 0.94 / 0.87 / 0.84, car 0.76 / 0.68 / 0.78 / 0.75, cat 0.87 / 0.87 / 0.88 / 0.92, chair 0.47 / 0.46 / 0.59 / 0.68, cow 0.87 / 0.88 / 0.89 / 0.91, dining table 0.45 / 0.51 / 0.56 / 0.60, dog 0.75 / 0.77 / 0.79 / 0.91, horse 0.72 / 0.70 / 0.74 / 0.73, motorbike 0.67 / 0.67 / 0.71 / 0.73, person 0.28 / 0.26 / 0.64 / 0.71, potted plant 0.74 / 0.68 / 0.79 / 0.85, sheep 0.87 / 0.90 / 0.89 / 0.92, sofa 0.70 / 0.67 / 0.72 / 0.74, train 0.75 / 0.67 / 0.73 / 0.72, tv 0.79 / 0.74 / 0.26 / 0.25

