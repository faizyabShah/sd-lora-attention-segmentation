# Pilot: OVAM optimized token + final SA method

49 annotated VOC-Sim images (aeroplane 24, bicycle 25), offline from disk captures. These are **all** VOC-Sim images of these two classes, so the released-OVAM per-class numbers below are on exactly the same images.

## Pooled class IoU

| Configuration | aeroplane | bicycle | mIoU (2 cls) | mean image IoU |
|---|---:|---:|---:|---:|
| OVAM released (unoptimized) | 0.717 | 0.606 | 0.662 | 0.680 |
| OVAM released (token-optimized) | 0.736 | 0.815 | 0.775 | 0.798 |
| Faiz final method (live run, same images) | 0.838 | 0.760 | 0.799 | — |
| `opt_all16/ca_only_ovammax_t0.40` | 0.485 | 0.372 | 0.429 | 0.430 |
| `opt_all16/ca_only_pct_t0.45` | 0.765 | 0.770 | 0.768 | 0.756 |
| `opt_all16/sa_iter1` | 0.856 | 0.786 | 0.821 | 0.813 |
| `opt_all16/sa_iter2` | 0.868 | 0.787 | 0.827 | 0.820 |
| `opt_all16/sa_iter3` | 0.871 | 0.782 | 0.826 | 0.820 |
| `opt_all16/sa_iter4` | 0.871 | 0.776 | 0.823 | 0.817 |
| `opt_all16/sa_iter4+crf` | 0.896 | 0.791 | 0.843 | 0.844 |
| `opt_all16/sa_iter4_nocleanup` | 0.870 | 0.775 | 0.822 | 0.816 |
| `opt_all16/sa_iter4_t0.35` | 0.834 | 0.708 | 0.771 | 0.762 |
| `opt_all16/sa_iter4_t0.40` | 0.853 | 0.745 | 0.799 | 0.792 |
| `opt_all16/sa_iter4_t0.50` | 0.878 | 0.794 | 0.836 | 0.832 |
| `opt_all16/sa_iter4_t0.55` | 0.866 | 0.797 | 0.831 | 0.831 |
| `opt_ovam15/ca_only_ovammax_t0.40` | 0.514 | 0.388 | 0.451 | 0.455 |
| `opt_ovam15/ca_only_ovammax_t0.40+crf` | 0.549 | 0.380 | 0.465 | 0.480 |
| `opt_ovam15/ca_only_pct_t0.45` | 0.779 | 0.785 | 0.782 | 0.772 |
| `opt_ovam15/sa_iter1` | 0.860 | 0.791 | 0.825 | 0.817 |
| `opt_ovam15/sa_iter2` | 0.868 | 0.788 | 0.828 | 0.821 |
| `opt_ovam15/sa_iter3` | 0.869 | 0.782 | 0.826 | 0.820 |
| `opt_ovam15/sa_iter4` | 0.870 | 0.773 | 0.822 | 0.817 |
| `opt_ovam15/sa_iter4_nocleanup` | 0.868 | 0.773 | 0.821 | 0.815 |
| `opt_ovam15/sa_iter4_t0.35` | 0.833 | 0.708 | 0.771 | 0.761 |
| `opt_ovam15/sa_iter4_t0.40` | 0.853 | 0.743 | 0.798 | 0.791 |
| `opt_ovam15/sa_iter4_t0.50` | 0.876 | 0.793 | 0.835 | 0.832 |
| `opt_ovam15/sa_iter4_t0.55` | 0.862 | 0.797 | 0.829 | 0.830 |
| `opt_top4/ca_only_ovammax_t0.40` | 0.651 | 0.596 | 0.623 | 0.619 |
| `opt_top4/ca_only_pct_t0.45` | 0.768 | 0.777 | 0.772 | 0.763 |
| `opt_top4/sa_iter1` | 0.852 | 0.787 | 0.819 | 0.811 |
| `opt_top4/sa_iter1+crf` | 0.898 | 0.798 | 0.848 | 0.847 |
| `opt_top4/sa_iter2` | 0.866 | 0.786 | 0.826 | 0.819 |
| `opt_top4/sa_iter3` | 0.870 | 0.783 | 0.827 | 0.819 |
| `opt_top4/sa_iter4` | 0.871 | 0.776 | 0.824 | 0.817 |
| `opt_top4/sa_iter4+crf` | 0.900 | 0.790 | 0.845 | 0.844 |
| `opt_top4/sa_iter4_nocleanup` | 0.871 | 0.775 | 0.823 | 0.816 |
| `opt_top4/sa_iter4_nocleanup+crf` | 0.900 | 0.789 | 0.845 | 0.843 |
| `opt_top4/sa_iter4_t0.35` | 0.831 | 0.706 | 0.769 | 0.759 |
| `opt_top4/sa_iter4_t0.40` | 0.853 | 0.744 | 0.799 | 0.790 |
| `opt_top4/sa_iter4_t0.50` | 0.881 | 0.795 | 0.838 | 0.833 |
| `opt_top4/sa_iter4_t0.55` | 0.878 | 0.799 | 0.839 | 0.837 |
| `plain_ovam15/ca_only_ovammax_t0.40` | 0.497 | 0.528 | 0.513 | 0.524 |
| `plain_ovam15/ca_only_ovammax_t0.40+crf` | 0.532 | 0.525 | 0.529 | 0.565 |
| `plain_ovam15/ca_only_pct_t0.45` | 0.695 | 0.691 | 0.693 | 0.688 |
| `plain_ovam15/sa_iter1` | 0.806 | 0.763 | 0.785 | 0.782 |
| `plain_ovam15/sa_iter2` | 0.827 | 0.769 | 0.798 | 0.795 |
| `plain_ovam15/sa_iter3` | 0.836 | 0.764 | 0.800 | 0.796 |
| `plain_ovam15/sa_iter4` | 0.839 | 0.758 | 0.798 | 0.795 |
| `plain_ovam15/sa_iter4_nocleanup` | 0.838 | 0.758 | 0.798 | 0.794 |
| `plain_ovam15/sa_iter4_t0.35` | 0.783 | 0.716 | 0.750 | 0.747 |
| `plain_ovam15/sa_iter4_t0.40` | 0.816 | 0.745 | 0.781 | 0.776 |
| `plain_ovam15/sa_iter4_t0.50` | 0.858 | 0.756 | 0.807 | 0.805 |
| `plain_ovam15/sa_iter4_t0.55` | 0.864 | 0.732 | 0.798 | 0.800 |
| `plain_top4/ca_only_ovammax_t0.40` | 0.615 | 0.611 | 0.613 | 0.610 |
| `plain_top4/ca_only_pct_t0.45` | 0.687 | 0.683 | 0.685 | 0.681 |
| `plain_top4/sa_iter1` | 0.801 | 0.757 | 0.779 | 0.777 |
| `plain_top4/sa_iter2` | 0.823 | 0.766 | 0.794 | 0.793 |
| `plain_top4/sa_iter3` | 0.833 | 0.764 | 0.799 | 0.797 |
| `plain_top4/sa_iter4` | 0.838 | 0.760 | 0.799 | 0.798 |
| `plain_top4/sa_iter4+crf` | 0.876 | 0.776 | 0.826 | 0.832 |
| `plain_top4/sa_iter4_nocleanup` | 0.836 | 0.760 | 0.798 | 0.797 |
| `plain_top4/sa_iter4_nocleanup+crf` | 0.875 | 0.777 | 0.826 | 0.831 |
| `plain_top4/sa_iter4_t0.35` | 0.781 | 0.713 | 0.747 | 0.745 |
| `plain_top4/sa_iter4_t0.40` | 0.808 | 0.745 | 0.777 | 0.775 |
| `plain_top4/sa_iter4_t0.50` | 0.859 | 0.760 | 0.809 | 0.810 |
| `plain_top4/sa_iter4_t0.55` | 0.876 | 0.746 | 0.811 | 0.812 |

## Paired per-image comparisons (mean image-IoU difference, bootstrap 95% CI, 20,000 resamples)

| A | B | mean(A−B) | 95% CI | A wins / B wins |
|---|---|---:|---|---|
| `opt_top4/sa_iter4` | `plain_top4/sa_iter4` | +0.0192 | [+0.0087, +0.0307] | 39 / 10 |
| `opt_top4/sa_iter4` | `OVAM released (token-optimized)` | +0.0189 | [-0.0162, +0.0575] | 20 / 29 |
| `plain_top4/sa_iter4` | `OVAM released (token-optimized)` | -0.0003 | [-0.0337, +0.0353] | 18 / 31 |
| `opt_top4/sa_iter4` | `OVAM released (unoptimized)` | +0.1369 | [+0.0987, +0.1768] | 41 / 8 |
| `opt_top4/sa_iter4+crf` | `opt_top4/sa_iter4` | +0.0273 | [+0.0122, +0.0423] | 37 / 12 |
| `plain_top4/sa_iter4+crf` | `plain_top4/sa_iter4` | +0.0341 | [+0.0182, +0.0500] | 40 / 9 |
| `opt_top4/sa_iter4+crf` | `OVAM released (token-optimized)` | +0.0462 | [+0.0095, +0.0876] | 30 / 19 |
| `opt_ovam15/ca_only_ovammax_t0.40+crf` | `OVAM released (token-optimized)` | -0.3177 | [-0.3632, -0.2708] | 1 / 48 |

Released-OVAM images are the authors' renders; ours are regenerated captures (small pixel differences), scored against the same released annotations. Two classes only: treat as a pilot.
