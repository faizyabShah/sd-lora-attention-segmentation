# Operator batches B and C on the class-stratified 150/150 screening images (exploratory)

`scripts/ana_ops_bc.py`. One frozen live capture per image; every map is paired with BRACE (V3) from the same capture. Ground truth only scores. Small subsets: treat differences below ~0.01 as noise. Nothing here is a registered test.

## VOC-Sim (150 images; sinks per image mean 0.0; 46 s/image)

| group | map | score | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |
|---|---|---:|---|---|---:|---:|
| — | BRACE (V3 + R2, OOF) | 0.814 | — | 0.877 / 0.905 | 0.806 | 0.819 |
| 6 PageRank | ppr01 (OOF + R2) | 0.797 | -0.018 (-0.029…-0.005) | 0.868 / 0.901 | 0.787 | 0.761 |
| 6 PageRank | ppr02 (OOF + R2) | 0.800 | -0.015 (-0.028…-0.000) | 0.853 / 0.920 | 0.782 | 0.742 |
| 6 PageRank | ppr04 (OOF + R2) | 0.790 | -0.025 (-0.043…-0.008) | 0.851 / 0.909 | 0.765 | 0.704 |
| 6 PageRank | ppr02_v3 (OOF + R2) | 0.812 | -0.003 (-0.016…+0.010) | 0.870 / 0.903 | 0.802 | 0.812 |
| 7 fg/bg competition | comp_border (OOF + R2) | 0.720 | -0.094 (-0.122…-0.071) | 0.854 / 0.813 | 0.693 | 0.693 |
| 7 fg/bg competition | comp_inv (OOF + R2) | 0.706 | -0.109 (-0.140…-0.074) | 0.860 / 0.787 | 0.701 | 0.714 |
| 7 fg/bg competition | rw (OOF + R2) | 0.484 | -0.330 (-0.410…-0.270) | 0.784 / 0.568 | 0.493 | 0.200 |
| 8 graph cut | gc_l05 (mask) | 0.768 | -0.051 (-0.067…-0.036) | 0.810 / 0.934 | — | — |
| 8 graph cut | gc_l2 (mask) | 0.706 | -0.113 (-0.151…-0.088) | 0.770 / 0.909 | — | — |
| 8 graph cut | gc_l05+R2 (mask) | 0.801 | -0.018 (-0.033…-0.002) | 0.841 / 0.930 | — | — |
| 8 graph cut | gc_l2+R2 (mask) | 0.728 | -0.091 (-0.131…-0.062) | 0.793 / 0.905 | — | — |
| 9 low-pass | lp8 (OOF + R2) | 0.813 | -0.001 (-0.022…+0.018) | 0.874 / 0.909 | 0.794 | 0.642 |
| 9 low-pass | lp16 (OOF + R2) | 0.785 | -0.029 (-0.052…-0.006) | 0.858 / 0.892 | 0.762 | 0.637 |
| 9 low-pass | lp32 (OOF + R2) | 0.765 | -0.049 (-0.074…-0.026) | 0.840 / 0.881 | 0.737 | 0.609 |
| 9 low-pass | lp16_v3 (OOF + R2) | 0.818 | +0.004 (-0.007…+0.014) | 0.884 / 0.897 | 0.807 | 0.818 |
| 10 sinks / temperature | p1 (OOF + R2) | 0.804 | -0.010 (-0.023…+0.005) | 0.863 / 0.915 | 0.788 | 0.750 |
| 10 sinks / temperature | desink_p1 (OOF + R2) | 0.804 | -0.010 (-0.023…+0.005) | 0.863 / 0.915 | 0.788 | 0.750 |
| 10 sinks / temperature | temp05_p1 (OOF + R2) | 0.785 | -0.030 (-0.048…-0.011) | 0.858 / 0.891 | 0.762 | 0.697 |
| 10 sinks / temperature | temp2_p1 (OOF + R2) | 0.805 | -0.009 (-0.020…+0.001) | 0.874 / 0.901 | 0.797 | 0.775 |
| 10 sinks / temperature | temp05_v3 (OOF + R2) | 0.803 | -0.012 (-0.021…-0.002) | 0.865 / 0.911 | 0.778 | 0.802 |
| 10 sinks / temperature | temp2_v3 (OOF + R2) | 0.792 | -0.022 (-0.045…-0.002) | 0.854 / 0.902 | 0.787 | 0.797 |
| 11 SA-kernel CRF | sacrf_w1 (OOF + R2) | 0.819 | +0.005 (-0.002…+0.010) | 0.878 / 0.907 | 0.808 | 0.819 |
| 11 SA-kernel CRF | sacrf_w3 (OOF + R2) | 0.799 | -0.015 (-0.027…+0.001) | 0.858 / 0.910 | 0.796 | 0.800 |
| 12 rollout | rollout (OOF + R2) | 0.783 | -0.031 (-0.045…-0.017) | 0.868 / 0.875 | 0.777 | 0.776 |
| 13 re-noising ensemble | rn200_v3 (OOF + R2) | 0.818 | +0.004 (-0.007…+0.014) | 0.889 / 0.897 | 0.804 | 0.818 |
| 13 re-noising ensemble | rn400_v3 (OOF + R2) | 0.818 | +0.004 (-0.004…+0.011) | 0.882 / 0.903 | 0.803 | 0.818 |
| 13 re-noising ensemble | rn600_v3 (OOF + R2) | 0.802 | -0.012 (-0.024…-0.002) | 0.874 / 0.895 | 0.785 | 0.802 |
| 13 re-noising ensemble | ens_mean (OOF + R2) | 0.818 | +0.004 (-0.004…+0.011) | 0.883 / 0.900 | 0.805 | 0.818 |
| 13 re-noising ensemble | ens_conf (OOF + R2) | 0.814 | -0.000 (-0.012…+0.008) | 0.892 / 0.888 | 0.803 | 0.814 |

Δ for OOF maps is against BRACE OOF + R2; Δ for single label-free masks is against BRACE @ 0.55 + R2.

## COCO-Cap (150 images; sinks per image mean 0.0; 46 s/image)

| group | map | score | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |
|---|---|---:|---|---|---:|---:|
| — | BRACE (V3 + R2, OOF) | 0.688 | — | 0.766 / 0.800 | 0.681 | 0.685 |
| 6 PageRank | ppr01 (OOF + R2) | 0.657 | -0.031 (-0.052…-0.009) | 0.763 / 0.772 | 0.646 | 0.649 |
| 6 PageRank | ppr02 (OOF + R2) | 0.651 | -0.037 (-0.057…-0.015) | 0.755 / 0.792 | 0.636 | 0.630 |
| 6 PageRank | ppr04 (OOF + R2) | 0.634 | -0.054 (-0.077…-0.027) | 0.756 / 0.779 | 0.612 | 0.592 |
| 6 PageRank | ppr02_v3 (OOF + R2) | 0.662 | -0.026 (-0.047…-0.007) | 0.743 / 0.799 | 0.654 | 0.630 |
| 7 fg/bg competition | comp_border (OOF + R2) | 0.542 | -0.147 (-0.189…-0.103) | 0.694 / 0.654 | 0.527 | 0.432 |
| 7 fg/bg competition | comp_inv (OOF + R2) | 0.555 | -0.133 (-0.176…-0.090) | 0.722 / 0.635 | 0.549 | 0.568 |
| 7 fg/bg competition | rw (OOF + R2) | 0.320 | -0.368 (-0.448…-0.305) | 0.671 / 0.474 | 0.340 | 0.111 |
| 8 graph cut | gc_l05 (mask) | 0.609 | -0.076 (-0.092…-0.059) | 0.664 / 0.852 | — | — |
| 8 graph cut | gc_l2 (mask) | 0.555 | -0.130 (-0.184…-0.105) | 0.654 / 0.795 | — | — |
| 8 graph cut | gc_l05+R2 (mask) | 0.642 | -0.043 (-0.056…-0.025) | 0.693 / 0.849 | — | — |
| 8 graph cut | gc_l2+R2 (mask) | 0.574 | -0.111 (-0.166…-0.084) | 0.673 / 0.794 | — | — |
| 9 low-pass | lp8 (OOF + R2) | 0.638 | -0.050 (-0.079…-0.016) | 0.732 / 0.782 | 0.633 | 0.592 |
| 9 low-pass | lp16 (OOF + R2) | 0.650 | -0.039 (-0.072…-0.001) | 0.743 / 0.790 | 0.637 | 0.578 |
| 9 low-pass | lp32 (OOF + R2) | 0.632 | -0.056 (-0.083…-0.019) | 0.724 / 0.788 | 0.605 | 0.531 |
| 9 low-pass | lp16_v3 (OOF + R2) | 0.673 | -0.015 (-0.034…+0.000) | 0.755 / 0.789 | 0.671 | 0.661 |
| 10 sinks / temperature | p1 (OOF + R2) | 0.662 | -0.026 (-0.050…+0.005) | 0.754 / 0.786 | 0.652 | 0.641 |
| 10 sinks / temperature | desink_p1 (OOF + R2) | 0.662 | -0.026 (-0.050…+0.005) | 0.754 / 0.786 | 0.652 | 0.641 |
| 10 sinks / temperature | temp05_p1 (OOF + R2) | 0.648 | -0.040 (-0.067…-0.007) | 0.765 / 0.776 | 0.623 | 0.605 |
| 10 sinks / temperature | temp2_p1 (OOF + R2) | 0.643 | -0.045 (-0.066…-0.025) | 0.740 / 0.755 | 0.644 | 0.646 |
| 10 sinks / temperature | temp05_v3 (OOF + R2) | 0.692 | +0.004 (-0.017…+0.031) | 0.777 / 0.790 | 0.672 | 0.697 |
| 10 sinks / temperature | temp2_v3 (OOF + R2) | 0.602 | -0.087 (-0.125…-0.055) | 0.676 / 0.794 | 0.598 | 0.571 |
| 11 SA-kernel CRF | sacrf_w1 (OOF + R2) | 0.691 | +0.003 (-0.006…+0.013) | 0.770 / 0.810 | 0.686 | 0.673 |
| 11 SA-kernel CRF | sacrf_w3 (OOF + R2) | 0.649 | -0.039 (-0.057…-0.020) | 0.714 / 0.817 | 0.647 | 0.651 |
| 12 rollout | rollout (OOF + R2) | 0.600 | -0.088 (-0.120…-0.058) | 0.733 / 0.695 | 0.599 | 0.583 |
| 13 re-noising ensemble | rn200_v3 (OOF + R2) | 0.691 | +0.003 (-0.007…+0.014) | 0.770 / 0.799 | 0.675 | 0.681 |
| 13 re-noising ensemble | rn400_v3 (OOF + R2) | 0.695 | +0.007 (-0.005…+0.019) | 0.767 / 0.809 | 0.682 | 0.683 |
| 13 re-noising ensemble | rn600_v3 (OOF + R2) | 0.667 | -0.021 (-0.040…-0.005) | 0.767 / 0.766 | 0.657 | 0.647 |
| 13 re-noising ensemble | ens_mean (OOF + R2) | 0.694 | +0.005 (-0.006…+0.017) | 0.766 / 0.812 | 0.685 | 0.677 |
| 13 re-noising ensemble | ens_conf (OOF + R2) | 0.692 | +0.004 (-0.007…+0.015) | 0.776 / 0.797 | 0.684 | 0.683 |

Δ for OOF maps is against BRACE OOF + R2; Δ for single label-free masks is against BRACE @ 0.55 + R2.

