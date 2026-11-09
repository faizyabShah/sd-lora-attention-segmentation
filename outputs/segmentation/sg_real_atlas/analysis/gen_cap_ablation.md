# Cap ablation of the 16→32→64 hierarchy, A3 seed vs SCLIP seed, generated sets (exploratory, 2026-11-09)

`scripts/sg_gen_capab.py`. Each image is regenerated with the frozen live pipeline; both seeds are propagated through the image's own self-attention with the frozen variant code (`src/segmentation/sg_combined_v2.py`). Seeds: **A3** = BRACE's cross-attention seed; **SCLIP** = SCLIP ViT-B/16 VOC21 target-class probability (min-max). Class-macro mIoU; *OOF* = τ chosen on 4 folds (crc32(image_id) % 5) and applied to the 5th; *@0.55* = BRACE's fixed τ. R2 = BRACE's DenseCRF. Δ = row − V3 (same seed, same post-processing), image-bootstrap 95% CI. Exploratory: V3 was selected earlier on the A3 seed, so the A3 rows are not a fresh test of the cap; the SCLIP rows are.

Variants: **V0** fine-only 64 graph (no hierarchy); **V1** hierarchy, no cap; **V2core** two-sided KL projection, both levels; **V2** two-sided KL projection, both levels + completion; **V3** one-sided cap at 16→32 (BRACE); **V4** one-sided cap at 16→32 and 32→64.

## VOC-Sim (490 images)

Regeneration: 238/490 images bit-identical to the stored image; mean |Δpixel| over the rest 0.61/255 (GPU kernel nondeterminism under shared-GPU load; all variants and both seeds of an image come from the same capture). BRACE V3+R2 @0.55 from this capture: 0.830.

**with R2 (DenseCRF)**

| variant | A3 OOF | A3 @0.55 | Δ vs V3 (A3, OOF) | SCLIP OOF | SCLIP @0.55 | Δ vs V3 (SCLIP, OOF) |
|---|---:|---:|---|---:|---:|---|
| V0 fine-only 64 graph (no hierarchy) | 0.824 | 0.796 | -0.006 (-0.012…+0.000) | 0.830 | 0.802 | -0.014 (-0.022…-0.007) |
| V1 hierarchy, no cap | 0.819 | 0.817 | -0.011 (-0.016…-0.007) | 0.838 | 0.838 | -0.007 (-0.011…-0.003) |
| V2core two-sided KL projection, both levels | 0.785 | 0.762 | -0.045 (-0.055…-0.036) | 0.785 | 0.743 | -0.059 (-0.070…-0.048) |
| V2 two-sided KL projection, both levels + completion | 0.822 | 0.817 | -0.008 (-0.012…-0.004) | 0.837 | 0.835 | -0.007 (-0.012…-0.003) |
| **V3** one-sided cap at 16→32 (BRACE) | 0.830 | 0.830 | — | 0.845 | 0.845 | — |
| V4 one-sided cap at 16→32 and 32→64 | 0.826 | 0.818 | -0.004 (-0.009…+0.001) | 0.844 | 0.833 | -0.001 (-0.005…+0.002) |

**raw maps, no CRF**

| variant | A3 OOF | A3 @0.55 | Δ vs V3 (A3, OOF) | SCLIP OOF | SCLIP @0.55 | Δ vs V3 (SCLIP, OOF) |
|---|---:|---:|---|---:|---:|---|
| V0 fine-only 64 graph (no hierarchy) | 0.810 | 0.791 | -0.006 (-0.012…+0.000) | 0.813 | 0.799 | -0.022 (-0.030…-0.015) |
| V1 hierarchy, no cap | 0.807 | 0.807 | -0.008 (-0.012…-0.004) | 0.827 | 0.827 | -0.008 (-0.011…-0.004) |
| V2core two-sided KL projection, both levels | 0.759 | 0.737 | -0.057 (-0.065…-0.049) | 0.758 | 0.725 | -0.077 (-0.087…-0.067) |
| V2 two-sided KL projection, both levels + completion | 0.804 | 0.802 | -0.011 (-0.016…-0.007) | 0.819 | 0.820 | -0.016 (-0.021…-0.012) |
| **V3** one-sided cap at 16→32 (BRACE) | 0.816 | 0.817 | — | 0.835 | 0.832 | — |
| V4 one-sided cap at 16→32 and 32→64 | 0.815 | 0.799 | -0.000 (-0.004…+0.003) | 0.832 | 0.813 | -0.003 (-0.006…-0.000) |

Per class, SCLIP seed, +R2 OOF (V1 / V2core / V3 / V4):

aeroplane 0.90 / 0.80 / 0.91 / 0.90, bicycle 0.79 / 0.70 / 0.80 / 0.80, bird 0.93 / 0.88 / 0.93 / 0.92, boat 0.84 / 0.64 / 0.85 / 0.84, bottle 0.85 / 0.63 / 0.85 / 0.82, bus 0.91 / 0.89 / 0.91 / 0.91, car 0.85 / 0.82 / 0.86 / 0.86, cat 0.92 / 0.86 / 0.92 / 0.91, chair 0.80 / 0.65 / 0.79 / 0.79, cow 0.96 / 0.92 / 0.96 / 0.96, dining-table 0.42 / 0.49 / 0.44 / 0.47, dog 0.93 / 0.87 / 0.94 / 0.94, horse 0.93 / 0.88 / 0.93 / 0.93, motorbike 0.86 / 0.82 / 0.86 / 0.87, person 0.83 / 0.82 / 0.85 / 0.84, potted-plant 0.72 / 0.73 / 0.75 / 0.74, sheep 0.93 / 0.92 / 0.97 / 0.97, sofa 0.89 / 0.90 / 0.90 / 0.92, train 0.82 / 0.80 / 0.80 / 0.82, tv 0.68 / 0.69 / 0.68 / 0.66

## COCO-Cap (707 images)

Regeneration: 243/707 images bit-identical to the stored image; mean |Δpixel| over the rest 0.71/255 (GPU kernel nondeterminism under shared-GPU load; all variants and both seeds of an image come from the same capture). BRACE V3+R2 @0.55 from this capture: 0.696.

**with R2 (DenseCRF)**

| variant | A3 OOF | A3 @0.55 | Δ vs V3 (A3, OOF) | SCLIP OOF | SCLIP @0.55 | Δ vs V3 (SCLIP, OOF) |
|---|---:|---:|---|---:|---:|---|
| V0 fine-only 64 graph (no hierarchy) | 0.682 | 0.684 | -0.025 (-0.038…-0.012) | 0.697 | 0.656 | -0.016 (-0.026…-0.006) |
| V1 hierarchy, no cap | 0.706 | 0.700 | -0.000 (-0.006…+0.005) | 0.703 | 0.705 | -0.010 (-0.015…-0.004) |
| V2core two-sided KL projection, both levels | 0.684 | 0.661 | -0.022 (-0.035…-0.010) | 0.601 | 0.566 | -0.112 (-0.129…-0.088) |
| V2 two-sided KL projection, both levels + completion | 0.712 | 0.715 | +0.005 (-0.005…+0.014) | 0.677 | 0.677 | -0.036 (-0.051…-0.020) |
| **V3** one-sided cap at 16→32 (BRACE) | 0.707 | 0.696 | — | 0.713 | 0.717 | — |
| V4 one-sided cap at 16→32 and 32→64 | 0.701 | 0.662 | -0.006 (-0.012…+0.000) | 0.718 | 0.712 | +0.005 (-0.000…+0.010) |

**raw maps, no CRF**

| variant | A3 OOF | A3 @0.55 | Δ vs V3 (A3, OOF) | SCLIP OOF | SCLIP @0.55 | Δ vs V3 (SCLIP, OOF) |
|---|---:|---:|---|---:|---:|---|
| V0 fine-only 64 graph (no hierarchy) | 0.680 | 0.677 | -0.016 (-0.026…-0.007) | 0.682 | 0.654 | -0.024 (-0.034…-0.015) |
| V1 hierarchy, no cap | 0.696 | 0.687 | -0.001 (-0.005…+0.004) | 0.697 | 0.695 | -0.008 (-0.013…-0.004) |
| V2core two-sided KL projection, both levels | 0.651 | 0.636 | -0.045 (-0.056…-0.035) | 0.584 | 0.541 | -0.122 (-0.138…-0.102) |
| V2 two-sided KL projection, both levels + completion | 0.699 | 0.698 | +0.003 (-0.004…+0.010) | 0.663 | 0.663 | -0.042 (-0.056…-0.027) |
| **V3** one-sided cap at 16→32 (BRACE) | 0.697 | 0.682 | — | 0.706 | 0.705 | — |
| V4 one-sided cap at 16→32 and 32→64 | 0.692 | 0.644 | -0.004 (-0.009…+0.001) | 0.705 | 0.694 | -0.000 (-0.005…+0.003) |

Per class, SCLIP seed, +R2 OOF (V1 / V2core / V3 / V4):

aeroplane 0.74 / 0.68 / 0.71 / 0.71, bicycle 0.70 / 0.70 / 0.73 / 0.74, bird 0.81 / 0.44 / 0.83 / 0.83, boat 0.51 / 0.30 / 0.54 / 0.61, bottle 0.56 / 0.26 / 0.57 / 0.58, bus 0.84 / 0.82 / 0.85 / 0.85, car 0.72 / 0.66 / 0.73 / 0.74, cat 0.83 / 0.74 / 0.86 / 0.85, chair 0.63 / 0.43 / 0.63 / 0.62, cow 0.89 / 0.82 / 0.89 / 0.89, dining table 0.49 / 0.50 / 0.55 / 0.56, dog 0.76 / 0.63 / 0.77 / 0.77, horse 0.77 / 0.72 / 0.76 / 0.76, motorbike 0.68 / 0.63 / 0.69 / 0.70, person 0.63 / 0.50 / 0.66 / 0.65, potted plant 0.61 / 0.53 / 0.63 / 0.63, sheep 0.89 / 0.82 / 0.90 / 0.90, sofa 0.74 / 0.72 / 0.73 / 0.73, train 0.64 / 0.66 / 0.66 / 0.66, tv 0.62 / 0.48 / 0.59 / 0.59

