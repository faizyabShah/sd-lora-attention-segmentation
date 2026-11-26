# VAE-decoder ownership pilot (pre-registered in `PREREG.md`)

2 images scored, 2 included (≥ 4 matched cells per set). ε 0.05, 16 draws per set. Gate 0 (decode reproduces the stored image): max mean |Δ| 0.006 grey levels.

## Verdict: **CLOSE**

| gate | statistic | bar | result |
|---|---|---|---|
| A. beyond footprint | mean per-image AUROC(o) − AUROC(o_fp), band pixels | ≥ +0.02, CI > 0 | +0.1159 [+0.0363, +0.1956] → pass |
| B. beyond RGB + footprint | OOF ΔAUROC (0.814 → 0.807) | ≥ +0.05, CI > 0 | -0.0063 [-0.0080, -0.0046] → fail |
| C. label-free correction (o > 0.5 in band) | wins vs R2; mean IoU gain; vs footprint rule | ≥ 2/2; > 0; > fp | 0/2; -0.1381; fp -0.1765 → fail |

## Per set

| set | images | AUROC o | AUROC o_fp | IoU R2 | IoU o-rule | IoU fp-rule | repeatability ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOC-Sim | 2 | 0.572 | 0.456 | 0.8469 | 0.7088 | 0.6704 | 0.906 |

## Within-cell discrimination (cells crossed by the ground-truth boundary, centre in band; reported, not gating)

| subset | cells | AUROC o | AUROC o_fp | AUROC colour |
|---|---:|---:|---:|---:|
| all | 268 | 0.567 | 0.650 | 0.359 |
| BRACE wrong on ≥ 8 px | 249 | 0.548 | 0.639 | 0.322 |

## Matched cell sets and kernel

- Mean matched cells per set: 100.5; mean distance to band (cells): F 2.80, B 3.62.
- Excluded images (< 4 cells): none.
- Kernel radial profile (mean ΔD², 4-px rings from the cell centre): 4.93e-04, 2.12e-04, 6.39e-05, 2.59e-05, 8.31e-06, 3.03e-06, 1.22e-06, 5.44e-07, 2.37e-07, 9.99e-08.
