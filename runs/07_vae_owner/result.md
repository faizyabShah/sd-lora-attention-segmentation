# VAE-decoder ownership pilot (pre-registered in `PREREG.md`)

30 images scored, 22 included (≥ 4 matched cells per set). ε 0.05, 16 draws per set. Gate 0 (decode reproduces the stored image): max mean |Δ| 0.237 grey levels.

## Verdict: **CLOSE**

| gate | statistic | bar | result |
|---|---|---|---|
| A. beyond footprint | mean per-image AUROC(o) − AUROC(o_fp), band pixels | ≥ +0.02, CI > 0 | +0.1054 [+0.0455, +0.1586] → pass |
| B. beyond RGB + footprint | OOF ΔAUROC (0.881 → 0.884) | ≥ +0.05, CI > 0 | +0.0035 [-0.0117, +0.0158] → fail |
| C. label-free correction (o > 0.5 in band) | wins vs R2; mean IoU gain; vs footprint rule | ≥ 15/22; > 0; > fp | 2/22; -0.1092; fp -0.1516 → fail |

## Per set

| set | images | AUROC o | AUROC o_fp | IoU R2 | IoU o-rule | IoU fp-rule | repeatability ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOC-Sim | 12 | 0.754 | 0.630 | 0.8433 | 0.7368 | 0.6887 | 0.936 |
| COCO-Cap | 10 | 0.779 | 0.697 | 0.8059 | 0.6935 | 0.6579 | 0.959 |

## Within-cell discrimination (cells crossed by the ground-truth boundary, centre in band; reported, not gating)

| subset | cells | AUROC o | AUROC o_fp | AUROC colour |
|---|---:|---:|---:|---:|
| all | 2598 | 0.720 | 0.708 | 0.721 |
| BRACE wrong on ≥ 8 px | 1783 | 0.703 | 0.715 | 0.654 |

## Matched cell sets and kernel

- Mean matched cells per set: 112.2; mean distance to band (cells): F 2.62, B 2.68.
- Excluded images (< 4 cells): ovam-voc-sim-15, ovam-voc-sim-413, ovam-voc-sim-156, ovam-coco-819494-bicycle-1112191427, ovam-coco-273132-motorbike-2763104245, ovam-coco-472976-person-4017470167, ovam-coco-148907-horse-1389681562, ovam-coco-770260-chair-1724683861.
- Kernel radial profile (mean ΔD², 4-px rings from the cell centre): 3.39e-04, 1.69e-04, 6.68e-05, 2.47e-05, 8.61e-06, 3.51e-06, 1.28e-06, 5.30e-07, 2.09e-07, 9.00e-08.
