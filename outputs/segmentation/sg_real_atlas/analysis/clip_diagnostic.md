# Dense CLIP (SCLIP ViT-B/16) vs SD attention on real VOC D (exploratory)

307 pairs, 200 photos. Pre-registered definitions and rules in `scripts/sg_clip_diag.py`. The SCLIP port reproduces the repo on VOC val (VOC20 81.49 vs 81.54, VOC21 59.64 vs 59.63). CLIP map = SCLIP's VOC21 class probability at 64×64. Class-given per-pair binary masks, class-macro IoU, OOF τ over 5 photo-grouped folds, photo-bootstrap 95% CIs. Max replay |Δ| 1.3e-03.

## Q1 — seed-level separation (before any SD propagation)

| seed | AUROC tgt vs other object | AUROC tgt vs background | AP |
|---|---:|---:|---:|
| base | 0.724 | 0.924 | 0.745 |
| comp | 0.754 | 0.911 | 0.744 |
| clip | 0.807 | 0.874 | 0.768 |
| clipcomp | 0.832 | 0.935 | 0.795 |
| fuse | 0.829 | 0.955 | 0.820 |

clip − A3 seed, tgt vs other: +0.083 (CI +0.049…+0.120)

clip − A3 seed, tgt vs background: -0.051 (CI -0.069…-0.030)

clip − A3 seed, AP: +0.023 (CI -0.003…+0.048)

By object size (seed AUROC tgt vs other / tgt vs bg):

| bin | n | A3 | CLIP |
|---|---:|---|---|
| <1% | 41 | 0.663 / 0.918 | 0.556 / 0.639 |
| 1–5% | 60 | 0.670 / 0.891 | 0.709 / 0.787 |
| 5–20% | 104 | 0.771 / 0.942 | 0.889 / 0.944 |
| >20% | 102 | 0.751 / 0.929 | 0.939 / 0.948 |

## Segmentation (OOF mIoU)

| variant | seed | C16 | V3+R2 | P1+R2 | best-map prec / rec / FP-on-other |
|---|---:|---:|---:|---:|---:|
| base | 0.488 | 0.553 | 0.538 | 0.547 | 0.562 / 0.786 / 50% (C16) |
| comp | 0.487 | 0.596 | 0.601 | 0.574 | 0.567 / 0.829 / 49% (V3+R2) |
| clip | 0.629 | 0.735 | 0.720 | 0.730 | 0.818 / 0.839 / 20% (C16) |
| clipcomp | 0.558 | 0.609 | 0.578 | 0.607 | 0.611 / 0.828 / 42% (C16) |
| fuse | 0.598 | 0.630 | 0.587 | 0.644 | 0.643 / 0.840 / 53% (P1+R2) |
| sclip | 0.639 | 0.639 | 0.673 | 0.673 | 0.825 / 0.787 / 17% (V3+R2) |

## Q2 — registered hybrid comparison

| comparison | Δ | 95% CI | pass |
|---|---:|---|---|
| best hybrid (fuse C16) − comp C16 (registered) | +0.034 | +0.007…+0.059 | yes |
| best hybrid (fuse C16) − best CLIP-only (clip C16) | -0.105 | -0.132…-0.061 | no |
| best CLIP-only − comp C16 | +0.139 | +0.091…+0.169 | — |
| clip C16 − clip seed (does SD SA help CLIP?) | +0.106 | +0.074…+0.134 | — |
| SCLIP standalone − comp C16 | +0.044 | -0.007…+0.080 | — |

**Hybrid claim:** FAIL — no hybrid is claimed.

## Per-class OOF IoU (comp C16 / SCLIP standalone / best hybrid)

aeroplane 0.75 / 0.54 / 0.78, bicycle 0.29 / 0.34 / 0.38, bird 0.74 / 0.75 / 0.52, boat 0.68 / 0.40 / 0.69, bottle 0.18 / 0.61 / 0.22, bus 0.67 / 0.81 / 0.75, car 0.41 / 0.77 / 0.56, cat 0.89 / 0.79 / 0.90, chair 0.26 / 0.41 / 0.30, cow 0.81 / 0.77 / 0.85, dining-table 0.65 / 0.66 / 0.70, dog 0.75 / 0.80 / 0.77, horse 0.81 / 0.59 / 0.81, motorbike 0.78 / 0.77 / 0.80, person 0.35 / 0.68 / 0.38, potted-plant 0.15 / 0.64 / 0.23, sheep 0.79 / 0.78 / 0.80, sofa 0.48 / 0.43 / 0.61, train 0.84 / 0.64 / 0.86, tv 0.63 / 0.60 / 0.69

## By object size (OOF IoU, class-macro within bin)

| bin | n | base V3+R2 | comp C16 | sclip seed | clip C16 | fuse C16 |
|---|---:|---:|---:|---:|---:|---:|
| <1% | 41 | 0.108 | 0.144 | 0.139 | 0.158 | 0.140 |
| 1–5% | 60 | 0.179 | 0.320 | 0.327 | 0.312 | 0.323 |
| 5–20% | 104 | 0.550 | 0.625 | 0.576 | 0.701 | 0.695 |
| >20% | 102 | 0.781 | 0.764 | 0.684 | 0.823 | 0.815 |

## Post-hoc check (not registered; computed after reading the table above)

The best CLIP-only pipeline is itself CLIP semantics + SD self-attention propagation. Against SCLIP with the same kind of refinement (DenseCRF), photo-bootstrap 95% CI:

| comparison | Δ | 95% CI |
|---|---:|---|
| clip C16 − SCLIP standalone + R2 | +0.063 | +0.028…+0.094 |
| clip C16 − SCLIP standalone | +0.096 | +0.066…+0.125 |
| clip P1+R2 − SCLIP standalone + R2 | +0.058 | +0.032…+0.082 |

Precision / recall: clip C16 0.818 / 0.839; SCLIP + R2 0.825 / 0.787. SD propagation gains recall at roughly constant precision. SCLIP's optional PAMR refinement was not tested.
