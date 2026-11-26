# Background-aware cross-attention seed (`PREREG.md`)

`scripts/jc_bgkey.py`. One live generation per image with the frozen runtime; seeds differ only in the attribution keys. Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold out-of-fold on VOC-Sim; full-VOC-Sim-best threshold applied unchanged on COCO-Cap). Δ vs W with image-bootstrap 95% interval (2000). Stable Diffusion only.

| Seed | Keys |
|---|---|
| W | frozen class-word attribution (BRACE) |
| S1 | ⟨SoT⟩ + class word |
| B8 | ⟨SoT⟩ + class + 8 generic words |
| R8 | class + 8 generic words |
| BN | ⟨SoT⟩ + class + NERVE's 25 words |
| RN | class + NERVE's 25 words |
| W0 | frozen class-word attribution, uniform layer weights |
| S1u | ⟨SoT⟩ + class word, uniform layer weights |
| B8u | ⟨SoT⟩ + class + 8 generic words, uniform layer weights |
| R8u | class + 8 generic words, uniform layer weights |
| BNu | ⟨SoT⟩ + class + NERVE's 25 words, uniform layer weights |
| RNu | class + NERVE's 25 words, uniform layer weights |

## VOC-Sim (10 images)

Regeneration: 5/10 images bit-identical to the stored image; W maps vs the stored capture: median max |Δ| 1.9e-03. W full pipeline at 0.55: 0.877.

**VOC-sel. threshold**

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.612 | 0.645 | 0.646 | 0.721 | 0.642 | 0.697 | 0.690 | 0.684 | 0.704 | 0.726 | 0.698 | 0.714 |
| fine-only | 0.825 | 0.849 | 0.850 | 0.859 | 0.832 | 0.863 | 0.819 | 0.836 | 0.836 | 0.867 | 0.817 | 0.872 |
| capped hierarchy | 0.833 | 0.829 | 0.836 | 0.877 | 0.836 | 0.875 | 0.835 | 0.835 | 0.829 | 0.884 | 0.820 | 0.876 |
| fine-only + DenseCRF | 0.857 | 0.880 | 0.871 | 0.863 | 0.882 | 0.851 | 0.833 | 0.824 | 0.861 | 0.870 | 0.859 | 0.883 |
| capped hierarchy + DenseCRF | 0.862 | 0.865 | 0.865 | 0.886 | 0.865 | 0.887 | 0.848 | 0.852 | 0.861 | 0.887 | 0.857 | 0.892 |

**fixed threshold 0.55**

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.609 | 0.591 | 0.576 | 0.733 | 0.564 | 0.700 | 0.588 | 0.499 | 0.559 | 0.738 | 0.546 | 0.711 |
| fine-only | 0.812 | 0.800 | 0.768 | 0.866 | 0.763 | 0.857 | 0.808 | 0.776 | 0.789 | 0.876 | 0.788 | 0.866 |
| capped hierarchy | 0.862 | 0.860 | 0.848 | 0.871 | 0.848 | 0.868 | 0.861 | 0.860 | 0.859 | 0.871 | 0.857 | 0.871 |
| fine-only + DenseCRF | 0.823 | 0.808 | 0.775 | 0.877 | 0.771 | 0.868 | 0.822 | 0.784 | 0.798 | 0.886 | 0.795 | 0.876 |
| capped hierarchy + DenseCRF | 0.877 | 0.880 | 0.868 | 0.893 | 0.869 | 0.890 | 0.873 | 0.871 | 0.878 | 0.892 | 0.876 | 0.890 |

**Δ vs W, full pipeline (capped hierarchy + DenseCRF)**

| Seed | VOC-sel. | fixed 0.55 | hierarchy − fine-only on this seed (both + DenseCRF, VOC-sel.) | hierarchy − seed alone (raw, VOC-sel.) | SA-guided − uniform layer weights (full pipeline, VOC-sel.) |
|---|---|---|---|---|---|
| W | — | — | +0.006 | +0.221 | +0.014 |
| S1 | +0.003 | +0.004 | -0.015 | +0.184 | +0.014 |
| B8 | +0.003 | -0.009 | -0.006 | +0.190 | +0.004 |
| R8 | +0.024 | +0.016 | +0.023 | +0.156 | -0.000 |
| BN | +0.003 | -0.008 | -0.016 | +0.194 | +0.008 |
| RN | +0.024 | +0.014 | +0.035 | +0.178 | -0.005 |
| W0 | -0.014 | -0.004 | +0.015 | +0.145 | — |
| S1u | -0.011 | -0.006 | +0.028 | +0.150 | — |
| B8u | -0.001 | +0.002 | +0.000 | +0.125 | — |
| R8u | +0.024 | +0.015 | +0.016 | +0.158 | — |
| BNu | -0.005 | -0.001 | -0.001 | +0.121 | — |
| RNu | +0.030 | +0.013 | +0.009 | +0.162 | — |

## COCO-Cap (10 images)

Regeneration: 0/10 images bit-identical to the stored image; W maps vs the stored capture: median max |Δ| 4.9e-03. W full pipeline at 0.55: 0.701.

**VOC-sel. threshold**

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.557 | 0.568 | 0.584 | 0.612 | 0.575 | 0.570 | 0.575 | 0.574 | 0.591 | 0.601 | 0.589 | 0.570 |
| fine-only | 0.639 | 0.625 | 0.649 | 0.692 | 0.632 | 0.721 | 0.637 | 0.610 | 0.653 | 0.697 | 0.643 | 0.708 |
| capped hierarchy | 0.672 | 0.668 | 0.680 | 0.718 | 0.658 | 0.721 | 0.681 | 0.681 | 0.675 | 0.720 | 0.641 | 0.722 |
| fine-only + DenseCRF | 0.665 | 0.642 | 0.669 | 0.718 | 0.649 | 0.752 | 0.661 | 0.635 | 0.680 | 0.712 | 0.670 | 0.721 |
| capped hierarchy + DenseCRF | 0.698 | 0.698 | 0.713 | 0.741 | 0.673 | 0.755 | 0.709 | 0.705 | 0.720 | 0.744 | 0.666 | 0.756 |

**fixed threshold 0.55**

| Cell | W | S1 | B8 | R8 | BN | RN | W0 | S1u | B8u | R8u | BNu | RNu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed alone | 0.479 | 0.474 | 0.463 | 0.612 | 0.451 | 0.573 | 0.415 | 0.399 | 0.442 | 0.609 | 0.433 | 0.579 |
| fine-only | 0.619 | 0.618 | 0.627 | 0.702 | 0.628 | 0.724 | 0.600 | 0.577 | 0.618 | 0.703 | 0.620 | 0.725 |
| capped hierarchy | 0.672 | 0.670 | 0.682 | 0.706 | 0.645 | 0.722 | 0.681 | 0.686 | 0.670 | 0.710 | 0.644 | 0.724 |
| fine-only + DenseCRF | 0.626 | 0.623 | 0.629 | 0.722 | 0.632 | 0.747 | 0.603 | 0.583 | 0.620 | 0.723 | 0.625 | 0.751 |
| capped hierarchy + DenseCRF | 0.701 | 0.698 | 0.711 | 0.740 | 0.671 | 0.755 | 0.711 | 0.715 | 0.698 | 0.743 | 0.672 | 0.757 |

**Δ vs W, full pipeline (capped hierarchy + DenseCRF)**

| Seed | VOC-sel. | fixed 0.55 | hierarchy − fine-only on this seed (both + DenseCRF, VOC-sel.) | hierarchy − seed alone (raw, VOC-sel.) | SA-guided − uniform layer weights (full pipeline, VOC-sel.) |
|---|---|---|---|---|---|
| W | — | — | +0.032 | +0.115 | -0.011 |
| S1 | +0.001 | -0.003 | +0.056 | +0.100 | -0.007 |
| B8 | +0.016 | +0.009 | +0.044 | +0.096 | -0.006 |
| R8 | +0.043 | +0.038 | +0.023 | +0.107 | -0.004 |
| BN | -0.024 | -0.030 | +0.024 | +0.084 | +0.008 |
| RN | +0.057 | +0.053 | +0.003 | +0.151 | -0.001 |
| W0 | +0.011 | +0.009 | +0.048 | +0.106 | — |
| S1u | +0.007 | +0.014 | +0.070 | +0.108 | — |
| B8u | +0.022 | -0.003 | +0.039 | +0.085 | — |
| R8u | +0.047 | +0.041 | +0.032 | +0.119 | — |
| BNu | -0.032 | -0.030 | -0.004 | +0.052 | — |
| RNu | +0.058 | +0.056 | +0.035 | +0.152 | — |

## Registered selection and test

VOC-Sim out-of-fold mIoU, full pipeline: W 0.8622, S1 0.8654, B8 0.8653, R8 0.8864, BN 0.8651, RN 0.8867, W0 0.8480, S1u 0.8516, B8u 0.8608, R8u 0.8867, BNu 0.8575, RNu 0.8918.

- **Selected on VOC-Sim:** RN (class + NERVE's 25 words); VOC-Sim Δ vs W +0.024.
- **COCO-Cap test:** RN − W = +0.057 (thresholds 0.575 / 0.475).
- **Verdict: FAIL** (rule: COCO-Cap interval above zero and VOC-Sim Δ ≥ −0.005).

## Registered secondary readouts, COCO-Cap, full pipeline at the VOC-sel. threshold (descriptive)

| Seed | thr | mIoU | per-image IoU: small third | middle | large | images below 0.3 IoU | macro precision | macro recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 0.475 | 0.698 | 0.550 | 0.828 | 0.710 | 0.100 | 0.725 | 0.951 |
| S1 | 0.550 | 0.698 | 0.600 | 0.823 | 0.678 | 0.100 | 0.757 | 0.906 |
| B8 | 0.525 | 0.713 | 0.585 | 0.825 | 0.726 | 0.100 | 0.759 | 0.927 |
| R8 | 0.575 | 0.741 | 0.572 | 0.799 | 0.823 | 0.100 | 0.766 | 0.942 |
| BN | 0.525 | 0.673 | 0.588 | 0.824 | 0.625 | 0.200 | 0.776 | 0.871 |
| RN | 0.575 | 0.755 | 0.593 | 0.824 | 0.824 | 0.100 | 0.791 | 0.938 |
| W0 | 0.475 | 0.709 | 0.550 | 0.828 | 0.738 | 0.100 | 0.725 | 0.966 |
| S1u | 0.475 | 0.705 | 0.551 | 0.826 | 0.730 | 0.100 | 0.724 | 0.963 |
| B8u | 0.475 | 0.720 | 0.552 | 0.829 | 0.763 | 0.100 | 0.730 | 0.975 |
| R8u | 0.575 | 0.744 | 0.572 | 0.802 | 0.830 | 0.100 | 0.770 | 0.943 |
| BNu | 0.475 | 0.666 | 0.554 | 0.828 | 0.628 | 0.200 | 0.722 | 0.903 |
| RNu | 0.575 | 0.756 | 0.590 | 0.828 | 0.826 | 0.100 | 0.790 | 0.942 |

Per class, VOC-Sim, full pipeline, VOC-sel. (W / S1 / B8 / R8 / BN / RN / W0 / S1u / B8u / R8u / BNu / RNu): aeroplane 0.95 / 0.95 / 0.95 / 0.94 / 0.95 / 0.94 / 0.95 / 0.95 / 0.95 / 0.94 / 0.95 / 0.94, bird 0.74 / 0.75 / 0.76 / 0.82 / 0.76 / 0.83 / 0.73 / 0.73 / 0.75 / 0.81 / 0.74 / 0.84, boat 0.87 / 0.85 / 0.86 / 0.91 / 0.88 / 0.92 / 0.88 / 0.87 / 0.86 / 0.92 / 0.87 / 0.93, bus 0.95 / 0.95 / 0.95 / 0.92 / 0.95 / 0.95 / 0.93 / 0.94 / 0.95 / 0.92 / 0.95 / 0.95, cat 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98 / 0.98, cow 0.80 / 0.81 / 0.85 / 0.96 / 0.84 / 0.95 / 0.66 / 0.67 / 0.78 / 0.96 / 0.76 / 0.95, dog 0.91 / 0.91 / 0.91 / 0.91 / 0.91 / 0.91 / 0.90 / 0.91 / 0.91 / 0.91 / 0.91 / 0.91, motorbike 0.89 / 0.90 / 0.88 / 0.87 / 0.88 / 0.85 / 0.89 / 0.90 / 0.88 / 0.89 / 0.87 / 0.88, potted-plant 0.55 / 0.55 / 0.51 / 0.56 / 0.52 / 0.54 / 0.56 / 0.56 / 0.55 / 0.55 / 0.54 / 0.54, sofa 1.00 / 1.00 / 0.99 / 0.99 / 1.00 / 0.99 / 1.00 / 1.00 / 1.00 / 0.99 / 1.00 / 1.00

Per class, COCO-Cap, full pipeline, VOC-sel. (W / S1 / B8 / R8 / BN / RN / W0 / S1u / B8u / R8u / BNu / RNu): aeroplane 0.87 / 0.89 / 0.88 / 0.89 / 0.88 / 0.89 / 0.87 / 0.87 / 0.88 / 0.89 / 0.88 / 0.89, bicycle 0.58 / 0.59 / 0.58 / 0.50 / 0.57 / 0.58 / 0.59 / 0.58 / 0.59 / 0.51 / 0.58 / 0.59, bird 0.95 / 0.94 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.95, bus 0.89 / 0.90 / 0.90 / 0.90 / 0.90 / 0.90 / 0.89 / 0.89 / 0.89 / 0.90 / 0.89 / 0.90, cat 0.57 / 0.70 / 0.66 / 0.60 / 0.66 / 0.67 / 0.57 / 0.57 / 0.57 / 0.60 / 0.57 / 0.66, cow 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.94 / 0.95 / 0.95 / 0.95 / 0.94 / 0.95 / 0.94, dog 0.93 / 0.77 / 0.82 / 0.91 / 0.80 / 0.87 / 0.93 / 0.91 / 0.91 / 0.90 / 0.90 / 0.86, motorbike 0.21 / 0.22 / 0.22 / 0.23 / 0.22 / 0.23 / 0.21 / 0.21 / 0.21 / 0.23 / 0.21 / 0.22, person 0.61 / 0.57 / 0.68 / 0.85 / 0.22 / 0.86 / 0.73 / 0.73 / 0.84 / 0.85 / 0.25 / 0.88, sofa 0.41 / 0.47 / 0.50 / 0.64 / 0.58 / 0.67 / 0.40 / 0.39 / 0.42 / 0.67 / 0.47 / 0.67

