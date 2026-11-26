# Does BRACE's cap help most where competing objects are present? (mechanism diagnostic; `PREREG.md`)

Stored paper captures, one capture per image. V0 + R2 at τ₀ = 0.45 (full-VOC rule; the same rule gives V3 0.55); V3 + R2 at 0.55. Gate: max count difference / pixels 0.0e+00 (VOC), 0.0e+00 (COCO). Raw secondary: V0 τ 0.450, V3 τ 0.550.

## Verdict (COCO-Cap, region am, R2): **Not supported** (H1 fail, H2 fail)

## H1: Δ IoU(V3+R2 − V0+R2) ~ z(C) + log area + class

| set | score | β per SD | image-strat CI | class-cluster CI | no-area β (CI) | raw-mask β (CI) | within-class ρ (classes) |
|---|---|---:|---|---|---|---|---|
| VOC-Sim | C, η 0.5 | -0.0016 | [-0.0055, +0.0025] | [-0.0060, +0.1737] | -0.0018 [-0.0057, +0.0022] | -0.0023 [-0.0066, +0.0014] | +0.026 (8) |
| VOC-Sim | C, η 0.3 | -0.0021 | [-0.0064, +0.0032] | [-0.0124, +0.0892] | -0.0023 [-0.0067, +0.0030] | -0.0026 [-0.0069, +0.0026] | +0.019 (13) |
| VOC-Sim | C, η am | +0.0006 | [-0.0089, +0.0113] | [-0.0107, +0.0238] | +0.0004 [-0.0092, +0.0110] | -0.0014 [-0.0105, +0.0084] | -0.100 (20) |
| VOC-Sim | K (count) | +0.0042 | [-0.0044, +0.0133] | [-0.0050, +0.0144] | | | |
| COCO-Cap | C, η 0.5 | -0.0029 | [-0.0107, +0.0046] | [-0.0136, +0.0044] | -0.0026 [-0.0104, +0.0052] | -0.0027 [-0.0101, +0.0045] | -0.002 (19) |
| COCO-Cap | C, η 0.3 | -0.0032 | [-0.0120, +0.0053] | [-0.0163, +0.0048] | -0.0030 [-0.0120, +0.0055] | -0.0035 [-0.0128, +0.0046] | +0.016 (19) |
| COCO-Cap | C, η am | -0.0034 | [-0.0131, +0.0059] | [-0.0169, +0.0074] | -0.0030 [-0.0129, +0.0066] | -0.0041 [-0.0137, +0.0046] | +0.061 (20) |
| COCO-Cap | K (count) | +0.0018 | [-0.0087, +0.0123] | [-0.0055, +0.0093] | | | |

**Score distributions:**

- VOC-Sim, C at η 0.5: mean 0.002; 10/50/90% 0.000, 0.000, 0.000; zero on 92% of images.
- VOC-Sim, C at η 0.3: mean 0.007; 10/50/90% 0.000, 0.000, 0.004; zero on 86% of images.
- VOC-Sim, C at η am: mean 0.019; 10/50/90% 0.000, 0.003, 0.056; zero on 14% of images.
- VOC-Sim, K: mean 0.37; zero on 70% of images.
- COCO-Cap, C at η 0.5: mean 0.024; 10/50/90% 0.000, 0.000, 0.086; zero on 75% of images.
- COCO-Cap, C at η 0.3: mean 0.039; 10/50/90% 0.000, 0.000, 0.152; zero on 63% of images.
- COCO-Cap, C at η am: mean 0.092; 10/50/90% 0.000, 0.017, 0.296; zero on 10% of images.
- COCO-Cap, K: mean 0.83; zero on 47% of images.

## H2: is a V0 false-positive pixel more likely to be removed by V3 when it lies on a competitor?

Mantel–Haenszel log odds ratio, strata = image × distance band from the ground-truth object.

| set | η | log OR (MH) | CI | unstratified | per band (0–8, 8–16, 16–32, 32–64, >64 px) | FP₀ on R |
|---|---|---:|---|---:|---|---:|
| VOC-Sim | 0.5 | -2.4869 | [-4.7618, -1.5942] | -3.8379 | -1.4375, -2.0986, -2.8437, -1.9563, -3.4389 | 2.7% |
| VOC-Sim | 0.3 | -1.2313 | [-2.3072, -0.2863] | -2.8535 | -0.3125, -0.2674, -0.6954, -2.1295, -1.9483 | 9.0% |
| VOC-Sim | am | +0.0718 | [-1.1175, +1.0924] | -1.4386 | +0.2633, +0.5108, +0.7495, +0.0155, -0.8728 | 10.8% |
| COCO-Cap | 0.5 | +0.9561 | [+0.0800, +1.6176] | +0.3470 | +0.5847, +0.7225, +0.8466, +0.9881, +1.1539 | 3.4% |
| COCO-Cap | 0.3 | +0.5207 | [-0.2627, +1.2024] | +0.1632 | +0.5346, +0.3689, +0.2983, +0.4911, +0.6325 | 6.8% |
| COCO-Cap | am | +0.2708 | [-0.2810, +0.7738] | +0.0324 | +0.4875, +0.2760, +0.1831, +0.4370, +0.1638 | 18.1% |

## H3: what does V3 change relative to V0 (both + R2)?

| set | mIoU V0 → V3 | only FP change | only TP change | pooled P, R: V0 → V3 | class-macro P, R: V0 → V3 | ΔFP / GT px | ΔTP / GT px | images up / down / same |
|---|---|---:|---:|---|---|---:|---:|---|
| VOC-Sim | 0.8240 → 0.8302 | 0.8233 | 0.8309 | 0.895, 0.901 → 0.890, 0.910 | 0.895, 0.918 → 0.895, 0.926 | +0.007 | +0.009 | 48% / 41% / 11% |
| COCO-Cap | 0.6734 → 0.6963 | 0.6907 | 0.6788 | 0.724, 0.876 → 0.747, 0.885 | 0.736, 0.898 → 0.756, 0.905 | -0.034 | +0.009 | 53% / 35% / 12% |

**False-positive pixels on and off the competitor region (V0 → V3):**

- VOC-Sim, η 0.5: on R 138,127 → 159,623; off R 4,893,166 → 5,192,738.
- VOC-Sim, η 0.3: on R 451,356 → 517,153; off R 4,579,937 → 4,835,208.
- VOC-Sim, η am: on R 543,747 → 642,067; off R 4,487,546 → 4,710,294.
- COCO-Cap, η 0.5: on R 489,445 → 392,358; off R 13,811,587 → 12,462,583.
- COCO-Cap, η 0.3: on R 971,439 → 831,111; off R 13,329,593 → 12,023,830.
- COCO-Cap, η am: on R 2,586,388 → 2,325,620; off R 11,714,644 → 10,529,321.

VOC-Sim does not decide (little variation in competition by design). Exploratory mechanism analysis; COCO-Cap is used post hoc to explain an existing gain, not to select anything.
