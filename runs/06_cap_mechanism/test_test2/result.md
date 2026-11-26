# Does BRACE's cap help most where competing objects are present? (mechanism diagnostic; `PREREG.md`)

Stored paper captures, one capture per image. V0 + R2 at τ₀ = 0.55 (full-VOC rule; the same rule gives V3 0.60); V3 + R2 at 0.55. Gate: max count difference / pixels 0.0e+00 (VOC), 0.0e+00 (COCO). Raw secondary: V0 τ 0.550, V3 τ 0.625.

## Verdict (COCO-Cap, region am, R2): **Not supported** (H1 fail, H2 fail)

## H1: Δ IoU(V3+R2 − V0+R2) ~ z(C) + log area + class

| set | score | β per SD | image-strat CI | class-cluster CI | no-area β (CI) | raw-mask β (CI) | within-class ρ (classes) |
|---|---|---:|---|---|---|---|---|
| VOC-Sim | C, η 0.5 | nan | [nan, nan] | [nan, nan] | nan [nan, nan] | nan [nan, nan] | +nan (0) |
| VOC-Sim | C, η 0.3 | nan | [nan, nan] | [nan, nan] | nan [nan, nan] | nan [nan, nan] | +nan (0) |
| VOC-Sim | C, η am | -0.0217 | [-0.3321, +0.9126] | [-0.3282, -0.0217] | -0.0144 [-0.3289, +0.1785] | -0.0110 [-0.1934, +0.2545] | +nan (0) |
| VOC-Sim | K (count) | -0.0301 | [-0.2373, +0.3074] | [-0.2345, -0.0064] | | | |
| COCO-Cap | C, η 0.5 | nan | [nan, nan] | [nan, nan] | nan [nan, nan] | nan [nan, nan] | +nan (0) |
| COCO-Cap | C, η 0.3 | -0.0040 | [-0.0064, -0.0040] | [-0.0129, -0.0004] | -0.0066 [-0.0075, -0.0057] | -0.0114 [-0.0114, -0.0026] | +nan (0) |
| COCO-Cap | C, η am | -0.0100 | [-0.0100, -0.0052] | [-0.0142, +0.0289] | -0.1761 [-0.1761, -0.0054] | +0.0117 [-0.0052, +0.0117] | +nan (0) |
| COCO-Cap | K (count) | -0.0040 | [-0.0064, -0.0040] | [-0.0129, -0.0004] | | | |

**Score distributions:**

- VOC-Sim, C at η 0.5: mean 0.000; 10/50/90% 0.000, 0.000, 0.000; zero on 100% of images.
- VOC-Sim, C at η 0.3: mean 0.000; 10/50/90% 0.000, 0.000, 0.000; zero on 100% of images.
- VOC-Sim, C at η am: mean 0.014; 10/50/90% 0.002, 0.007, 0.033; zero on 0% of images.
- VOC-Sim, K: mean 0.50; zero on 67% of images.
- COCO-Cap, C at η 0.5: mean 0.000; 10/50/90% 0.000, 0.000, 0.000; zero on 100% of images.
- COCO-Cap, C at η 0.3: mean 0.000; 10/50/90% 0.000, 0.000, 0.000; zero on 83% of images.
- COCO-Cap, C at η am: mean 0.007; 10/50/90% 0.000, 0.000, 0.021; zero on 50% of images.
- COCO-Cap, K: mean 0.17; zero on 83% of images.

## H2: is a V0 false-positive pixel more likely to be removed by V3 when it lies on a competitor?

Mantel–Haenszel log odds ratio, strata = image × distance band from the ground-truth object.

| set | η | log OR (MH) | CI | unstratified | per band (0–8, 8–16, 16–32, 32–64, >64 px) | FP₀ on R |
|---|---|---:|---|---:|---|---:|
| VOC-Sim | 0.5 | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.0% |
| VOC-Sim | 0.3 | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.0% |
| VOC-Sim | am | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.6% |
| COCO-Cap | 0.5 | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.0% |
| COCO-Cap | 0.3 | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.0% |
| COCO-Cap | am | nan | [nan, nan] | nan | nan, nan, nan, nan, nan | 0.0% |

## H3: what does V3 change relative to V0 (both + R2)?

| set | mIoU V0 → V3 | only FP change | only TP change | pooled P, R: V0 → V3 | class-macro P, R: V0 → V3 | ΔFP / GT px | ΔTP / GT px | images up / down / same |
|---|---|---:|---:|---|---|---:|---:|---|
| VOC-Sim | 0.8012 → 0.8064 | 0.7518 | 0.8614 | 0.924, 0.851 → 0.852, 0.917 | 0.920, 0.850 → 0.858, 0.916 | +0.089 | +0.065 | 67% / 33% / 0% |
| COCO-Cap | 0.9387 → 0.9459 | 0.9287 | 0.9561 | 0.984, 0.950 → 0.974, 0.968 | 0.984, 0.953 → 0.974, 0.971 | +0.011 | +0.018 | 67% / 33% / 0% |

**False-positive pixels on and off the competitor region (V0 → V3):**

- VOC-Sim, η 0.5: on R 0 → 0; off R 24,946 → 56,760.
- VOC-Sim, η 0.3: on R 0 → 0; off R 24,946 → 56,760.
- VOC-Sim, η am: on R 145 → 895; off R 24,801 → 55,865.
- COCO-Cap, η 0.5: on R 0 → 0; off R 6,081 → 10,334.
- COCO-Cap, η 0.3: on R 0 → 0; off R 6,081 → 10,334.
- COCO-Cap, η am: on R 0 → 0; off R 6,081 → 10,334.

VOC-Sim does not decide (little variation in competition by design). Exploratory mechanism analysis; COCO-Cap is used post hoc to explain an existing gain, not to select anything.
