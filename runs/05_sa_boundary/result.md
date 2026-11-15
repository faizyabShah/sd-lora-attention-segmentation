# Does SD self-attention carry boundary information beyond RGB? (diagnostic; pre-registered in `PREREG.md`)

Stored rich capture of the 150/150 screening images; native grids; ground truth only measures. Primary setting: 32 px, block 1 (steps 10–19). Decisive: Q3(a) on COCO-Cap.

## VOC-Sim (150 images)

| res | block | Q1 AUROC SA / colour / contour (95% CI for SA) | Q2 AUROC RGB → RGB+SA, Δ (95% CI) | Q3(a) core: RGB → RGB+SA, Δ (95% CI) [bleed/correct cells] | Q3 single-feature AUROC, core ref: dist / colour / contour / −SA | Q3(b) gt ref, single −SA |
|---:|---:|---|---|---|---|---|
| 16 | 0 | 0.618 (0.589…0.644) / 0.817 / 0.769 | 0.825 → 0.852, +0.028 (+0.017…+0.038) | 0.712 → 0.742, +0.030 (-0.010…+0.063) [1135/4947 in 65 img] | 0.733 / 0.617 / 0.698 / 0.701 | 0.726 |
| 16 | 1 | 0.690 (0.664…0.714) / 0.817 / 0.769 | 0.825 → 0.863, +0.038 (+0.027…+0.051) | 0.712 → 0.761, +0.048 (-0.002…+0.090) [1135/4947 in 65 img] | 0.733 / 0.617 / 0.698 / 0.751 | 0.835 |
| 16 | 2 | 0.750 (0.729…0.768) / 0.817 / 0.769 | 0.825 → 0.872, +0.047 (+0.036…+0.058) | 0.712 → 0.764, +0.052 (+0.004…+0.090) [1135/4947 in 65 img] | 0.733 / 0.617 / 0.698 / 0.763 | 0.872 |
| 32 | 0 | 0.680 (0.656…0.702) / 0.854 / 0.815 | 0.865 → 0.879, +0.013 (+0.006…+0.021) | 0.726 → 0.751, +0.026 (-0.006…+0.056) [5113/35887 in 109 img] | 0.722 / 0.585 / 0.729 / 0.728 | 0.826 |
| 32 **(primary)** | 1 | 0.789 (0.771…0.806) / 0.854 / 0.815 | 0.867 → 0.890, +0.023 (+0.015…+0.032) | 0.726 → 0.773, +0.047 (+0.014…+0.082) [5113/35887 in 109 img] | 0.722 / 0.585 / 0.729 / 0.759 | 0.896 |
| 32 | 2 | 0.833 (0.817…0.847) / 0.854 / 0.815 | 0.866 → 0.897, +0.031 (+0.023…+0.039) | 0.726 → 0.768, +0.042 (+0.011…+0.074) [5113/35887 in 109 img] | 0.722 / 0.585 / 0.729 / 0.753 | 0.917 |

## COCO-Cap (150 images)

| res | block | Q1 AUROC SA / colour / contour (95% CI for SA) | Q2 AUROC RGB → RGB+SA, Δ (95% CI) | Q3(a) core: RGB → RGB+SA, Δ (95% CI) [bleed/correct cells] | Q3 single-feature AUROC, core ref: dist / colour / contour / −SA | Q3(b) gt ref, single −SA |
|---:|---:|---|---|---|---|---|
| 16 | 0 | 0.547 (0.515…0.580) / 0.813 / 0.756 | 0.815 → 0.824, +0.009 (+0.002…+0.015) | 0.601 → 0.619, +0.018 (-0.027…+0.061) [2179/3582 in 87 img] | 0.663 / 0.540 / 0.545 / nan | nan |
| 16 | 1 | 0.643 (0.616…0.671) / 0.813 / 0.756 | 0.815 → 0.836, +0.021 (+0.012…+0.030) | 0.601 → 0.648, +0.047 (-0.008…+0.100) [2179/3582 in 87 img] | 0.663 / 0.540 / 0.545 / nan | nan |
| 16 | 2 | 0.701 (0.679…0.724) / 0.813 / 0.756 | 0.815 → 0.841, +0.025 (+0.015…+0.034) | 0.601 → 0.651, +0.049 (-0.005…+0.100) [2179/3582 in 87 img] | 0.663 / 0.540 / 0.545 / nan | nan |
| 32 | 0 | 0.613 (0.584…0.641) / 0.844 / 0.805 | 0.858 → 0.862, +0.005 (+0.000…+0.009) | 0.643 → 0.661, +0.018 (-0.019…+0.056) [9450/20266 in 107 img] | 0.680 / 0.553 / 0.574 / 0.663 | 0.840 |
| 32 **(primary)** | 1 | 0.739 (0.716…0.761) / 0.844 / 0.805 | 0.858 → 0.873, +0.015 (+0.008…+0.022) | 0.643 → 0.682, +0.040 (-0.010…+0.089) [9450/20266 in 107 img] | 0.680 / 0.553 / 0.574 / 0.685 | 0.897 |
| 32 | 2 | 0.789 (0.771…0.806) / 0.844 / 0.805 | 0.857 → 0.878, +0.021 (+0.013…+0.028) | 0.643 → 0.681, +0.038 (-0.010…+0.087) [9450/20266 in 107 img] | 0.680 / 0.553 / 0.574 / 0.683 | 0.915 |

## Decision (fixed in PREREG.md)

Q3(a) on COCO-Cap, 32 px, block 1: ΔAUROC +0.040 (95% CI -0.010…+0.089); rule: Δ ≥ +0.05 and CI lower bound > 0 → **CLOSE: no boundary mechanism is built**

Q1: per-image AUROC of −log affinity (or colour distance / contour) for boundary-crossing vs within edges, averaged over distance bands 1–3. Q2/Q3: pooled out-of-fold AUROC of a logistic model (5 folds by image), image-bootstrap CI of the difference. Q3 RGB baseline = distance to the nearest reference cell, colour distance to the reference mean, intervening contour to the nearest reference cell; + SA = log mean affinity to the reference set.
