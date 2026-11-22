# Run 34, stage 2: full pipeline (`PREREG.md`)

Test halves; thresholds chosen on dev. **Exploratory.** One generation-time capture for every arm. Regime V = VOC-Sim-only profiles (COCO-Cap is transfer); regime X = VOC-Sim + COCO-Cap dev profiles (label-free domain adaptation).

Test images: VOC-Sim 0, COCO-Cap 1. Stage 1 verdicts: MP2 fail, MP4 fail, FD4 fail, FP2 fail, FP4 fail. A contrast is read only for gates that passed stage 1.


## Regime X

| Arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|
| A | nan | 0.977 | nan | 0.966 |
| X:NC | nan | 0.976 | nan | 0.907 |
| X:MK4 | nan | 0.971 | nan | 0.822 |
| X:MP2 | nan | 0.973 | nan | 0.838 |
| X:MP4 | nan | 0.973 | nan | 0.836 |
| X:FD4 | nan | 0.975 | nan | 0.827 |
| X:FP2 | nan | 0.974 | nan | 0.840 |
| X:FP4 | nan | 0.973 | nan | 0.838 |

Hierarchy + DenseCRF, dev-selected thresholds (0.55 in brackets column).

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| MP2 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| MP4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FD4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FP2 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FP4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| MK4 − NC | — | n/a | n/a | n/a | n/a | — |
| MK4 − A | — | n/a | n/a | n/a | n/a | — |
| NC − A | — | n/a | n/a | n/a | n/a | — |

Per class, COCO-Cap test, dev-selected (A / MK4 / MP2 / MP4 / FD4 / FP2 / FP4): cow 0.98 / 0.97 / 0.97 / 0.97 / 0.97 / 0.97 / 0.97


## Regime V

| Arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|
| A | nan | 0.976 | nan | 0.966 |
| V:NC | nan | 0.976 | nan | 0.935 |
| V:MK4 | nan | 0.972 | nan | 0.939 |
| V:MP2 | nan | 0.971 | nan | 0.943 |
| V:MP4 | nan | 0.971 | nan | 0.942 |
| V:FD4 | nan | 0.971 | nan | 0.940 |
| V:FP2 | nan | 0.971 | nan | 0.944 |
| V:FP4 | nan | 0.971 | nan | 0.943 |

Hierarchy + DenseCRF, dev-selected thresholds (0.55 in brackets column).

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| MP2 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| MP4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FD4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FP2 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| FP4 − MK4 | fail | n/a | n/a | n/a | n/a | descriptive (failed stage 1) |
| MK4 − NC | — | n/a | n/a | n/a | n/a | — |
| MK4 − A | — | n/a | n/a | n/a | n/a | — |
| NC − A | — | n/a | n/a | n/a | n/a | — |

Per class, COCO-Cap test, dev-selected (A / MK4 / MP2 / MP4 / FD4 / FP2 / FP4): cow 0.98 / 0.97 / 0.97 / 0.97 / 0.97 / 0.97 / 0.97

