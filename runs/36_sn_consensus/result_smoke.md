# Run 36, stage 2: self-negative gate and module-consensus gates, full pipeline (`PREREG.md`)

Test halves; thresholds chosen on dev. **Exploratory.** One generation-time capture for every arm. Regime V = VOC-Sim-only profiles (COCO-Cap is transfer); regime X = VOC-Sim + COCO-Cap dev profiles (label-free domain adaptation). Every gate except MK4 and SN1 runs at its dev-matched strength γ* (`stage1.md`).

Test images: VOC-Sim 2, COCO-Cap 2. Stage 1 verdicts (vs own base): MK4-AV fail, MK4-LR pass, MK4-RW pass, SN-AV fail, SN-LR pass, SN-RW pass. SN passed run 35 stage 1.


## Regime X

| Arm | γ | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|---:|
| A | — | 0.807 | 0.866 | 0.808 | 0.858 |
| X:MK4 | 1.00 | 0.812 | 0.868 | 0.814 | 0.860 |
| X:SN | 0.52 | 0.808 | 0.868 | 0.809 | 0.861 |
| X:SN1 | 1.00 | 0.811 | 0.868 | 0.813 | 0.859 |
| X:MK4-AV | 0.99 | 0.810 | 0.867 | 0.810 | 0.858 |
| X:MK4-LR | 0.71 | 0.809 | 0.870 | 0.810 | 0.861 |
| X:MK4-RW | 0.76 | 0.810 | 0.869 | 0.811 | 0.861 |
| X:SN-AV | 0.55 | 0.806 | 0.868 | 0.807 | 0.859 |
| X:SN-LR | 0.51 | 0.806 | 0.868 | 0.807 | 0.858 |
| X:SN-RW | 0.53 | 0.806 | 0.868 | 0.807 | 0.858 |

Hierarchy + DenseCRF. Paired image bootstrap; "improves" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005.

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| SN − MK4 | pass (run 35) | -0.004 [-0.011, +0.003] | -0.000 [-0.003, +0.002] | -0.004 [-0.011, +0.002] | +0.001 [-0.003, +0.005] | does not improve on MK4 |
| MK4-AV − MK4 | fail | -0.002 [-0.005, +0.001] | -0.001 [-0.002, -0.001] | -0.003 [-0.006, -0.000] | -0.002 [-0.004, -0.001] | descriptive (failed stage 1) |
| MK4-LR − MK4 | pass | -0.003 [-0.007, +0.001] | +0.001 [+0.001, +0.001] | -0.004 [-0.008, +0.000] | +0.001 [+0.000, +0.002] | does not improve on MK4 |
| MK4-RW − MK4 | pass | -0.002 [-0.005, +0.001] | +0.001 [+0.001, +0.001] | -0.003 [-0.006, +0.000] | +0.001 [+0.000, +0.001] | does not improve on MK4 |
| SN-AV − SN | fail | -0.002 [-0.003, -0.001] | -0.001 [-0.003, +0.002] | -0.002 [-0.003, -0.002] | -0.002 [-0.006, +0.002] | descriptive (failed stage 1) |
| SN-LR − SN | pass | -0.002 [-0.003, -0.001] | -0.001 [-0.004, +0.002] | -0.002 [-0.002, -0.002] | -0.003 [-0.008, +0.002] | does not improve on SN |
| SN-RW − SN | pass | -0.002 [-0.002, -0.001] | -0.000 [-0.003, +0.002] | -0.002 [-0.002, -0.002] | -0.003 [-0.007, +0.002] | does not improve on SN |
| SN1 − MK4 | — | -0.000 [-0.001, +0.000] | -0.001 [-0.001, -0.000] | -0.001 [-0.003, +0.001] | -0.001 [-0.001, +0.000] | descriptive |
| SN − A | — | +0.001 [-0.003, +0.006] | +0.002 [+0.001, +0.004] | +0.002 [-0.003, +0.006] | +0.003 [+0.002, +0.003] | descriptive |
| MK4 − A | — | +0.005 [-0.006, +0.017] | +0.003 [-0.001, +0.006] | +0.006 [-0.005, +0.017] | +0.002 [-0.003, +0.006] | descriptive |
| MK4-LR − MK4-AV | — | -0.001 [-0.002, -0.001] | +0.002 [+0.002, +0.003] | -0.000 [-0.002, +0.001] | +0.004 [+0.001, +0.006] | descriptive |
| MK4-RW − MK4-AV | — | -0.000 [-0.001, +0.000] | +0.002 [+0.001, +0.003] | +0.000 [-0.000, +0.001] | +0.003 [+0.001, +0.005] | descriptive |
| SN-LR − SN-AV | — | -0.000 [-0.000, +0.000] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | -0.001 [-0.002, +0.000] | descriptive |
| SN-RW − SN-AV | — | +0.000 [-0.000, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.001] | -0.000 [-0.001, +0.000] | descriptive |

SN − MK4 before the hierarchy (descriptive; dev-selected thresholds on each map):

| Map | VOC-Sim test | COCO-Cap test |
|---|---|---|
| seed | +0.011 [+0.003, +0.018] | +0.004 [-0.011, +0.018] |
| V0 | -0.004 [-0.009, +0.000] | -0.001 [-0.006, +0.003] |
| V3 | -0.007 [-0.014, +0.000] | +0.001 [-0.002, +0.004] |
| V3+R2 | -0.004 [-0.011, +0.003] | -0.000 [-0.003, +0.002] |

Per class, COCO-Cap test, dev-selected (A / MK4 / SN / SN-LR / SN-RW / MK4-LR / MK4-RW): car 0.79 / 0.79 / 0.79 / 0.79 / 0.79 / 0.79 / 0.79, horse 0.94 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95


## Regime V

| Arm | γ | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|---:|
| A | — | 0.804 | 0.875 | 0.808 | 0.858 |
| V:MK4 | 1.00 | 0.809 | 0.870 | 0.812 | 0.844 |
| V:SN | 0.48 | 0.806 | 0.873 | 0.808 | 0.856 |
| V:SN1 | 1.00 | 0.808 | 0.871 | 0.812 | 0.848 |
| V:MK4-AV | 0.98 | 0.807 | 0.870 | 0.810 | 0.846 |
| V:MK4-LR | 0.72 | 0.808 | 0.873 | 0.811 | 0.853 |
| V:MK4-RW | 0.79 | 0.809 | 0.873 | 0.811 | 0.851 |
| V:SN-AV | 0.51 | 0.805 | 0.873 | 0.806 | 0.853 |
| V:SN-LR | 0.46 | 0.805 | 0.873 | 0.807 | 0.853 |
| V:SN-RW | 0.48 | 0.805 | 0.873 | 0.807 | 0.854 |

Hierarchy + DenseCRF. Paired image bootstrap; "improves" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005.

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| SN − MK4 | pass (run 35) | -0.003 [-0.010, +0.003] | +0.004 [+0.000, +0.008] | -0.004 [-0.010, +0.002] | +0.012 [-0.000, +0.023] | does not improve on MK4 |
| MK4-AV − MK4 | fail | -0.002 [-0.005, +0.002] | +0.001 [+0.000, +0.001] | -0.002 [-0.004, -0.000] | +0.001 [+0.001, +0.002] | descriptive (failed stage 1) |
| MK4-LR − MK4 | pass | -0.001 [-0.003, +0.002] | +0.004 [+0.003, +0.004] | -0.001 [-0.003, +0.000] | +0.009 [+0.002, +0.016] | **improves on MK4** |
| MK4-RW − MK4 | pass | -0.000 [-0.002, +0.001] | +0.003 [+0.003, +0.003] | -0.001 [-0.002, +0.000] | +0.006 [+0.002, +0.011] | **improves on MK4** |
| SN-AV − SN | fail | -0.001 [-0.002, -0.000] | -0.001 [-0.003, +0.001] | -0.002 [-0.003, -0.001] | -0.002 [-0.006, +0.001] | descriptive (failed stage 1) |
| SN-LR − SN | pass | -0.001 [-0.001, -0.001] | -0.001 [-0.004, +0.002] | -0.001 [-0.001, -0.001] | -0.003 [-0.007, +0.001] | does not improve on SN |
| SN-RW − SN | pass | -0.001 [-0.001, -0.000] | -0.001 [-0.003, +0.002] | -0.001 [-0.001, -0.001] | -0.002 [-0.006, +0.001] | does not improve on SN |
| SN1 − MK4 | — | -0.001 [-0.002, +0.001] | +0.002 [-0.000, +0.004] | -0.001 [-0.002, +0.001] | +0.004 [-0.000, +0.008] | descriptive |
| SN − A | — | +0.002 [-0.001, +0.004] | -0.001 [-0.004, +0.002] | +0.000 [-0.004, +0.005] | -0.003 [-0.007, +0.002] | descriptive |
| MK4 − A | — | +0.005 [-0.005, +0.014] | -0.005 [-0.012, +0.002] | +0.004 [-0.006, +0.014] | -0.014 [-0.030, +0.002] | descriptive |
| MK4-LR − MK4-AV | — | +0.001 [-0.000, +0.002] | +0.003 [+0.002, +0.004] | +0.001 [+0.001, +0.002] | +0.007 [+0.001, +0.014] | descriptive |
| MK4-RW − MK4-AV | — | +0.001 [-0.000, +0.003] | +0.002 [+0.002, +0.003] | +0.002 [+0.001, +0.003] | +0.005 [+0.001, +0.009] | descriptive |
| SN-LR − SN-AV | — | +0.000 [-0.000, +0.001] | -0.000 [-0.001, +0.000] | +0.001 [-0.001, +0.002] | -0.000 [-0.001, +0.000] | descriptive |
| SN-RW − SN-AV | — | +0.000 [-0.000, +0.001] | +0.000 [-0.000, +0.000] | +0.001 [-0.000, +0.001] | +0.000 [-0.000, +0.000] | descriptive |

SN − MK4 before the hierarchy (descriptive; dev-selected thresholds on each map):

| Map | VOC-Sim test | COCO-Cap test |
|---|---|---|
| seed | -0.002 [-0.017, +0.012] | +0.003 [-0.011, +0.016] |
| V0 | -0.006 [-0.012, -0.000] | -0.001 [-0.003, +0.002] |
| V3 | -0.008 [-0.016, -0.001] | +0.007 [-0.000, +0.013] |
| V3+R2 | -0.003 [-0.010, +0.003] | +0.004 [+0.000, +0.008] |

Per class, COCO-Cap test, dev-selected (A / MK4 / SN / SN-LR / SN-RW / MK4-LR / MK4-RW): car 0.80 / 0.79 / 0.79 / 0.79 / 0.79 / 0.79 / 0.79, horse 0.95 / 0.95 / 0.95 / 0.96 / 0.96 / 0.96 / 0.96


## Combined reading (`PREREG.md` §4)

- Regime X: SN − MK4 does not improve; SN consensus improving on SN: none; MK4 consensus improving on MK4: none.
- Regime V: SN − MK4 does not improve; SN consensus improving on SN: none; MK4 consensus improving on MK4: MK4-LR, MK4-RW.

BRACE stays frozen; any change to the paper's method is a separate owner decision.

