# Operator batch A: label-free threshold and region operators on the stored BRACE maps (exploratory)

`scripts/ana_ops_a.py`. Stored paper capture; ground truth only scores. A2's fitted constant c = 1.064 comes from VOC-Sim only and is applied unchanged to COCO-Cap. Nothing here is a registered test.

## VOC-Sim (490 images; BRACE mask components: mean 39.6, median 25)

| group | map | score | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |
|---|---|---:|---|---|---:|---:|
| — | BRACE (V3 + R2, OOF) | 0.830 | — | 0.890 / 0.910 | 0.816 | 0.830 |
| A1 stability τ | stab_crf (mask) | 0.820 | -0.011 (-0.022…+0.003) | 0.855 / 0.940 | — | — |
| A1 stability τ | stab_raw (mask) | 0.809 | -0.021 (-0.035…-0.006) | 0.837 / 0.953 | — | — |
| A2 area-matched τ | area_c1 (mask) | 0.741 | -0.089 (-0.101…-0.078) | 0.912 / 0.765 | — | — |
| A2 area-matched τ | area_cfit (mask) | 0.756 | -0.075 (-0.086…-0.064) | 0.898 / 0.796 | — | — |
| A3 components | cc_mass05 (mask) | 0.830 | -0.000 (-0.001…+0.000) | 0.891 / 0.909 | — | — |
| A3 components | cc_top1 (mask) | 0.829 | -0.001 (-0.003…+0.000) | 0.892 / 0.908 | — | — |
| A3 components | cc_seedpk (mask) | 0.830 | -0.000 (-0.001…+0.001) | 0.892 / 0.908 | — | — |
| A4 geodesic | geo (OOF + R2) | 0.755 | -0.075 (-0.082…-0.068) | 0.909 / 0.797 | 0.746 | 0.582 |
| A5 matting | matte05 (mask) | 0.799 | -0.032 (-0.043…-0.021) | 0.836 / 0.934 | — | — |
| A5 matting | matte05+R2 (mask) | 0.801 | -0.029 (-0.041…-0.018) | 0.841 / 0.929 | — | — |

Δ for OOF maps is against BRACE OOF + R2; Δ for single label-free masks is against BRACE @ 0.55 + R2.

Chosen τ (mean ± sd): stab_crf 0.486 ± 0.110, stab_raw 0.444 ± 0.106, area_c1 0.631 ± 0.174, area_cfit 0.585 ± 0.179

## COCO-Cap (707 images; BRACE mask components: mean 30.7, median 21)

| group | map | score | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |
|---|---|---:|---|---|---:|---:|
| — | BRACE (V3 + R2, OOF) | 0.707 | — | 0.775 / 0.860 | 0.697 | 0.696 |
| A1 stability τ | stab_crf (mask) | 0.660 | -0.036 (-0.047…-0.023) | 0.698 / 0.898 | — | — |
| A1 stability τ | stab_raw (mask) | 0.618 | -0.078 (-0.090…-0.061) | 0.648 / 0.905 | — | — |
| A2 area-matched τ | area_c1 (mask) | 0.622 | -0.074 (-0.092…-0.056) | 0.831 / 0.703 | — | — |
| A2 area-matched τ | area_cfit (mask) | 0.633 | -0.063 (-0.082…-0.045) | 0.819 / 0.729 | — | — |
| A3 components | cc_mass05 (mask) | 0.696 | -0.000 (-0.001…+0.000) | 0.749 / 0.882 | — | — |
| A3 components | cc_top1 (mask) | 0.682 | -0.014 (-0.021…-0.008) | 0.752 / 0.855 | — | — |
| A3 components | cc_seedpk (mask) | 0.686 | -0.010 (-0.018…-0.004) | 0.754 / 0.857 | — | — |
| A4 geodesic | geo (OOF + R2) | 0.646 | -0.061 (-0.068…-0.052) | 0.762 / 0.787 | 0.635 | 0.543 |
| A5 matting | matte05 (mask) | 0.620 | -0.076 (-0.088…-0.063) | 0.658 / 0.889 | — | — |
| A5 matting | matte05+R2 (mask) | 0.623 | -0.073 (-0.085…-0.059) | 0.663 / 0.885 | — | — |

Δ for OOF maps is against BRACE OOF + R2; Δ for single label-free masks is against BRACE @ 0.55 + R2.

Chosen τ (mean ± sd): stab_crf 0.504 ± 0.111, stab_raw 0.471 ± 0.113, area_c1 0.652 ± 0.185, area_cfit 0.625 ± 0.190

