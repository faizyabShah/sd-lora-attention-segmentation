# Run 31, Part D: multi-negative gate evaluation (`PREREG.md`)

Test halves only; thresholds and per-class strengths chosen on dev. **Exploratory**: both benchmarks were seen in runs 28–30. Regime V = VOC-Sim-only development → transfer; regime X = mixed VOC-Sim + COCO-Cap development.

Test images: VOC-Sim 0, COCO-Cap 2.


## Regime X (mixed development)

| Seed arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | VOC-Sim, test-best | COCO-Cap, test-best |
|---|---:|---:|---:|---:|---:|---:|
| A | nan | 0.700 | nan | 0.699 | nan | 0.846 |
| X:NC | nan | 0.685 | nan | 0.685 | nan | 0.834 |
| X:NO | nan | 0.684 | nan | 0.684 | nan | 0.833 |
| X:NH | nan | 0.690 | nan | 0.688 | nan | 0.838 |
| X:MK2 | nan | 0.684 | nan | 0.685 | nan | 0.834 |
| X:MK4 | nan | 0.685 | nan | 0.685 | nan | 0.834 |
| X:MK8 | nan | 0.685 | nan | 0.686 | nan | 0.835 |
| X:MG4 | nan | 0.685 | nan | 0.685 | nan | 0.834 |
| X:MS4 | nan | 0.685 | nan | 0.685 | nan | 0.834 |
| CAg | nan | 0.685 | — | — | — | — |

Hierarchy + DenseCRF, dev-selected thresholds.

| ID | Comparison | VOC-Sim test | COCO-Cap test |
|---|---|---|---|
| T1 | MK4 − NC | n/a | -0.000 [-0.000, +0.000] |
| T2 | MK4 − A | n/a | -0.016 [-0.032, +0.000] |
| T3 | CAg − A | n/a | -0.016 [-0.031, +0.000] |
| T4 | MK4 − MG4 | n/a | +0.000 [+0.000, +0.000] |
| — | MS4 − MK4 | n/a | +0.000 [+0.000, +0.000] |
| — | MK2 − NC | n/a | -0.000 [-0.001, +0.000] |
| — | MK8 − NC | n/a | +0.000 [+0.000, +0.001] |
| — | NO − NC | n/a | -0.001 [-0.002, +0.000] |
| — | NH − NC | n/a | +0.006 [+0.000, +0.012] |
| — | NC − A | n/a | -0.016 [-0.031, +0.000] |
| — | MG4 − A | n/a | -0.016 [-0.032, +0.000] |

Reading (rule fixed in PREREG): T1 not promising; T2 not promising; T3 not promising; spurious-mode claim (T4 above zero on COCO-Cap) not supported.

Seed and fine-only (raw, dev-selected thresholds), arm − A:

| Arm | seed VOC-Sim | seed COCO-Cap | fine-only VOC-Sim | fine-only COCO-Cap |
|---|---|---|---|---|
| X:NC | n/a | +0.058 [+0.000, +0.116] | n/a | +0.035 [+0.000, +0.069] |
| X:NO | n/a | +0.047 [-0.018, +0.111] | n/a | +0.038 [+0.000, +0.075] |
| X:NH | n/a | +0.036 [+0.000, +0.072] | n/a | +0.023 [+0.000, +0.046] |
| X:MK2 | n/a | +0.059 [+0.000, +0.117] | n/a | +0.038 [+0.000, +0.075] |
| X:MK4 | n/a | +0.057 [+0.000, +0.114] | n/a | +0.035 [+0.000, +0.069] |
| X:MK8 | n/a | +0.055 [+0.000, +0.110] | n/a | +0.036 [+0.000, +0.071] |
| X:MG4 | n/a | +0.039 [-0.018, +0.095] | n/a | +0.036 [+0.000, +0.071] |
| X:MS4 | n/a | +0.057 [+0.000, +0.114] | n/a | +0.035 [+0.000, +0.070] |

CAg choices (non-A): aeroplane NH, cow NC

Per class, COCO-Cap test (A / NC / MK4 / MG4 / CAg): cow 0.78 / 0.74 / 0.74 / 0.74 / 0.74, person 0.62 / 0.62 / 0.62 / 0.62 / 0.62


## Regime V (VOC-Sim-only development)

| Seed arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | VOC-Sim, test-best | COCO-Cap, test-best |
|---|---:|---:|---:|---:|---:|---:|
| A | nan | 0.705 | nan | 0.699 | nan | 0.846 |
| V:NC | nan | 0.696 | nan | 0.689 | nan | 0.839 |
| V:NO | nan | 0.690 | nan | 0.685 | nan | 0.834 |
| V:NH | nan | 0.700 | nan | 0.694 | nan | 0.842 |
| V:MK2 | nan | 0.694 | nan | 0.688 | nan | 0.838 |
| V:MK4 | nan | 0.695 | nan | 0.688 | nan | 0.839 |
| V:MK8 | nan | 0.695 | nan | 0.688 | nan | 0.839 |
| V:MG4 | nan | 0.691 | nan | 0.686 | nan | 0.835 |
| V:MS4 | nan | 0.695 | nan | 0.689 | nan | 0.839 |
| CAg | nan | 0.705 | — | — | — | — |

Hierarchy + DenseCRF, dev-selected thresholds.

| ID | Comparison | VOC-Sim test | COCO-Cap test |
|---|---|---|---|
| T1 | MK4 − NC | n/a | -0.001 [-0.001, +0.000] |
| T2 | MK4 − A | n/a | -0.010 [-0.019, +0.000] |
| T3 | CAg − A | n/a | +0.000 [+0.000, +0.000] |
| T4 | MK4 − MG4 | n/a | +0.004 [+0.000, +0.008] |
| — | MS4 − MK4 | n/a | +0.000 [+0.000, +0.000] |
| — | MK2 − NC | n/a | -0.002 [-0.003, +0.000] |
| — | MK8 − NC | n/a | -0.001 [-0.001, +0.000] |
| — | NO − NC | n/a | -0.006 [-0.012, +0.000] |
| — | NH − NC | n/a | +0.004 [+0.000, +0.008] |
| — | NC − A | n/a | -0.009 [-0.018, +0.000] |
| — | MG4 − A | n/a | -0.013 [-0.027, +0.000] |

Reading (rule fixed in PREREG): T1 not promising; T2 not promising; T3 not promising; spurious-mode claim (T4 above zero on COCO-Cap) not supported.

Seed and fine-only (raw, dev-selected thresholds), arm − A:

| Arm | seed VOC-Sim | seed COCO-Cap | fine-only VOC-Sim | fine-only COCO-Cap |
|---|---|---|---|---|
| V:NC | n/a | +0.051 [+0.000, +0.101] | n/a | -0.006 [-0.036, +0.023] |
| V:NO | n/a | +0.074 [+0.000, +0.147] | n/a | +0.009 [-0.036, +0.054] |
| V:NH | n/a | +0.028 [+0.000, +0.056] | n/a | -0.015 [-0.036, +0.005] |
| V:MK2 | n/a | +0.051 [+0.000, +0.102] | n/a | -0.002 [-0.036, +0.032] |
| V:MK4 | n/a | +0.051 [+0.000, +0.101] | n/a | -0.004 [-0.036, +0.027] |
| V:MK8 | n/a | +0.051 [+0.000, +0.101] | n/a | -0.004 [-0.036, +0.028] |
| V:MG4 | n/a | +0.064 [+0.000, +0.128] | n/a | +0.004 [-0.036, +0.043] |
| V:MS4 | n/a | +0.051 [+0.000, +0.102] | n/a | -0.005 [-0.036, +0.025] |

CAg choices (non-A): 

Per class, COCO-Cap test (A / NC / MK4 / MG4 / CAg): cow 0.79 / 0.77 / 0.77 / 0.76 / 0.79, person 0.62 / 0.62 / 0.62 / 0.62 / 0.62

