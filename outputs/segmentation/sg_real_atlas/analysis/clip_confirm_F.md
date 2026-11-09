# Confirmation on fresh real VOC photos (F): CLIP seed + SD propagation

Pre-registered in `clip_confirm_prereg.md` (committed before the run). F: 294 pairs, 200 photos, never used before. All thresholds fixed on D; nothing fitted on F except the secondary OOF column.

| pipeline | τ (from D) | mIoU on F | prec / rec / FP-on-other |
|---|---:|---:|---|
| clip C16 | 0.575 | 0.733 | 0.828 / 0.862 / 13% |
| sclip R2 | — | 0.663 | 0.823 / 0.796 / 11% |
| sclip standalone | — | 0.627 | 0.772 / 0.810 / 9% |
| comp C16 | 0.50 | 0.591 | 0.600 / 0.862 / 40% |
| base V3+R2 | 0.70 | 0.534 | 0.553 / 0.801 / 38% |

**Primary (registered):** clip C16 − SCLIP + R2 = +0.069 (photo-bootstrap 95% CI +0.037…+0.108) → **CONFIRMED**.

## Secondary

| comparison | Δ | 95% CI |
|---|---:|---|
| clip C16 − sclip standalone | +0.106 | +0.075…+0.141 |
| clip C16 − comp C16 | +0.142 | +0.085…+0.172 |
| clip C16 − base V3+R2 | +0.199 | +0.132…+0.236 |

OOF within F (τ chosen on 4 F folds): clip C16 0.733, clip V3+R2 0.712, clip P1+R2 0.731, comp C16 0.599, base V3+R2 0.530, sclip V3+R2 0.663

Seed-level AUROC (tgt vs other / tgt vs bg): base 0.728 / 0.930, comp 0.736 / 0.915, clip 0.838 / 0.897

## By object size (fixed-τ IoU, class-macro within bin)

| bin | n | clip C16 | sclip R2 | comp C16 | base V3+R2 |
|---|---:|---:|---:|---:|---:|
| <1% | 25 | 0.152 | 0.144 | 0.133 | 0.035 |
| 1–5% | 58 | 0.330 | 0.269 | 0.275 | 0.261 |
| 5–20% | 121 | 0.685 | 0.580 | 0.625 | 0.568 |
| >20% | 90 | 0.857 | 0.749 | 0.752 | 0.724 |

## Per class (clip C16 / SCLIP + R2 / comp C16)

aeroplane 0.51 / 0.59 / 0.65, bicycle 0.58 / 0.55 / 0.44, bird 0.81 / 0.71 / 0.74, boat 0.70 / 0.38 / 0.36, bottle 0.66 / 0.76 / 0.31, bus 0.78 / 0.77 / 0.84, car 0.69 / 0.68 / 0.57, cat 0.93 / 0.81 / 0.90, chair 0.59 / 0.20 / 0.22, cow 0.91 / 0.90 / 0.76, dining-table 0.70 / 0.56 / 0.44, dog 0.90 / 0.90 / 0.89, horse 0.81 / 0.75 / 0.68, motorbike 0.79 / 0.68 / 0.71, person 0.58 / 0.70 / 0.44, potted-plant 0.63 / 0.73 / 0.32, sheep 0.90 / 0.82 / 0.88, sofa 0.77 / 0.57 / 0.47, train 0.74 / 0.64 / 0.81, tv 0.68 / 0.56 / 0.40

