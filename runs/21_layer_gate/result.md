# Per-location CA layer reliability gate and signed temporal CA contrast (`PREREG.md`)

`scripts/jc_layergate.py`. One live capture per image (runs/20 corrected prompts, safety checker off). Class-macro mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold applied unchanged on COCO-Cap. Δ with image-bootstrap 95% interval (2000). Exploratory.

## Data checks

- VOC-Sim: 490 images. Per-layer reconstruction vs runtime seed: max |Δ| 0.0e+00. Image vs runs/20 capture: identical on 164/490, median max |Δ| 11. BRACE hierarchy map vs runs/20: median max |Δ| 2.4e-03.
- COCO-Cap: 707 images. Per-layer reconstruction vs runtime seed: max |Δ| 0.0e+00. Image vs runs/20 capture: identical on 177/707, median max |Δ| 22. BRACE hierarchy map vs runs/20: median max |Δ| 3.7e-03.

## Diagnostics (pixel level; ΔAUC = augmented − base, VOC-Sim 5-fold OOF, COCO-Cap fit on VOC-Sim)

| set | target | pixels | error rate | AUC base | AUC +reliability (Exp. 1) | ΔAUC Exp. 1 | AUC +early/late (Exp. 2) | ΔAUC Exp. 2 | AUC D=E−L | AUC −D |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| VOC-Sim | FP vs TP | 193724 | 0.119 | 0.8217 | 0.8214 | -0.0002 [-0.0027, +0.0022] | 0.8293 | +0.0076 [+0.0014, +0.0139] | 0.5240 | 0.4760 |
| VOC-Sim | FN vs TN | 308036 | 0.051 | 0.8955 | 0.8991 | +0.0036 [+0.0004, +0.0071] | 0.9069 | +0.0114 [-0.0035, +0.0255] | 0.5125 | 0.4875 |
| COCO-Cap | FP vs TP | 204194 | 0.262 | 0.7765 | 0.7630 | -0.0135 [-0.0176, -0.0097] | 0.7922 | +0.0157 [+0.0068, +0.0244] | 0.5047 | 0.4953 |
| COCO-Cap | FN vs TN | 519774 | 0.033 | 0.7515 | 0.7545 | +0.0030 [-0.0014, +0.0073] | 0.7538 | +0.0023 [-0.0233, +0.0265] | 0.5270 | 0.4730 |

**Diagnostic pass (VOC-Sim ΔAUC ≥ 0.01, lower bound > 0, either target):** Exp. 1 **fail**, Exp. 2 **fail**.

Agreeing-wrong check (≥ 6/11 layers fire at ≥ 0.5): VOC-Sim FP 0.007 vs TP 0.035, mean R̄ FP 0.355 vs TP 0.344; COCO-Cap FP 0.006 vs TP 0.031, mean R̄ FP 0.351 vs TP 0.310.

## Components, VOC-Sim (490 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.736 | 0.824 | 0.831 | 0.830 |
| UN: uniform layer weights | 0.748 | 0.820 | 0.828 | 0.827 |
| Gpx: gate, pixels | 0.793 | 0.822 | 0.825 | 0.839 |
| Gpxx: gate, pixels, shuffled | 0.739 | 0.824 | 0.833 | 0.828 |
| Gs16: gate, SA k-means k=16 | 0.765 | 0.827 | 0.829 | 0.841 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.692 | 0.807 | 0.822 | 0.808 |
| Gs64: gate, SA k-means k=64 | 0.772 | 0.827 | 0.829 | 0.841 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.708 | 0.821 | 0.826 | 0.816 |
| Gs256: gate, SA k-means k=256 | 0.786 | 0.828 | 0.830 | 0.844 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.722 | 0.825 | 0.832 | 0.826 |
| Gg16: gate, grid 4x4 | 0.737 | 0.829 | 0.832 | 0.836 |
| Gg16x: gate, grid 4x4, shuffled | 0.704 | 0.815 | 0.826 | 0.823 |
| Gg64: gate, grid 8x8 | 0.760 | 0.824 | 0.829 | 0.837 |
| Gg64x: gate, grid 8x8, shuffled | 0.709 | 0.816 | 0.832 | 0.823 |
| Gg256: gate, grid 16x16 | 0.779 | 0.824 | 0.828 | 0.838 |
| Gg256x: gate, grid 16x16, shuffled | 0.724 | 0.825 | 0.834 | 0.830 |
| TE: early calls 0-9 | 0.644 | 0.808 | 0.822 | 0.798 |
| TM: middle calls 10-20 | 0.750 | 0.818 | 0.825 | 0.833 |
| TL: late calls 21-30 | 0.747 | 0.814 | 0.815 | 0.821 |
| TD+: seed + early - late | 0.569 | 0.767 | 0.800 | 0.739 |
| TD-: seed + late - early | 0.641 | 0.782 | 0.795 | 0.787 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.628 | 0.796 | 0.832 | 0.759 |
| UN: uniform layer weights | 0.603 | 0.786 | 0.829 | 0.751 |
| Gpx: gate, pixels | 0.580 | 0.801 | 0.826 | 0.784 |
| Gpxx: gate, pixels, shuffled | 0.240 | 0.796 | 0.833 | 0.757 |
| Gs16: gate, SA k-means k=16 | 0.604 | 0.790 | 0.831 | 0.763 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.522 | 0.766 | 0.823 | 0.727 |
| Gs64: gate, SA k-means k=64 | 0.580 | 0.794 | 0.831 | 0.770 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.457 | 0.786 | 0.829 | 0.733 |
| Gs256: gate, SA k-means k=256 | 0.564 | 0.796 | 0.830 | 0.775 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.378 | 0.782 | 0.834 | 0.735 |
| Gg16: gate, grid 4x4 | 0.609 | 0.800 | 0.832 | 0.771 |
| Gg16x: gate, grid 4x4, shuffled | 0.552 | 0.787 | 0.827 | 0.745 |
| Gg64: gate, grid 8x8 | 0.605 | 0.803 | 0.829 | 0.780 |
| Gg64x: gate, grid 8x8, shuffled | 0.479 | 0.788 | 0.832 | 0.744 |
| Gg256: gate, grid 16x16 | 0.602 | 0.802 | 0.828 | 0.783 |
| Gg256x: gate, grid 16x16, shuffled | 0.407 | 0.794 | 0.835 | 0.751 |
| TE: early calls 0-9 | 0.608 | 0.778 | 0.822 | 0.743 |
| TM: middle calls 10-20 | 0.520 | 0.784 | 0.825 | 0.735 |
| TL: late calls 21-30 | 0.387 | 0.750 | 0.813 | 0.685 |
| TD+: seed + early - late | 0.518 | 0.738 | 0.801 | 0.699 |
| TD-: seed + late - early | 0.243 | 0.694 | 0.790 | 0.601 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.706 | 0.811 | 0.818 | 0.819 |
| UN: uniform layer weights | 0.715 | 0.808 | 0.815 | 0.813 |
| Gpx: gate, pixels | 0.755 | 0.813 | 0.813 | 0.824 |
| Gpxx: gate, pixels, shuffled | 0.692 | 0.810 | 0.818 | 0.818 |
| Gs16: gate, SA k-means k=16 | 0.733 | 0.814 | 0.817 | 0.825 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.664 | 0.796 | 0.810 | 0.797 |
| Gs64: gate, SA k-means k=64 | 0.739 | 0.815 | 0.816 | 0.828 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.671 | 0.806 | 0.818 | 0.804 |
| Gs256: gate, SA k-means k=256 | 0.752 | 0.814 | 0.815 | 0.831 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.678 | 0.812 | 0.820 | 0.811 |
| Gg16: gate, grid 4x4 | 0.704 | 0.814 | 0.817 | 0.824 |
| Gg16x: gate, grid 4x4, shuffled | 0.673 | 0.802 | 0.812 | 0.810 |
| Gg64: gate, grid 8x8 | 0.722 | 0.814 | 0.814 | 0.823 |
| Gg64x: gate, grid 8x8, shuffled | 0.672 | 0.805 | 0.819 | 0.810 |
| Gg256: gate, grid 16x16 | 0.741 | 0.814 | 0.813 | 0.825 |
| Gg256x: gate, grid 16x16, shuffled | 0.679 | 0.815 | 0.820 | 0.819 |
| TE: early calls 0-9 | 0.622 | 0.797 | 0.808 | 0.788 |
| TM: middle calls 10-20 | 0.717 | 0.807 | 0.812 | 0.820 |
| TL: late calls 21-30 | 0.709 | 0.797 | 0.800 | 0.804 |
| TD+: seed + early - late | 0.554 | 0.755 | 0.787 | 0.730 |
| TD-: seed + late - early | 0.612 | 0.770 | 0.780 | 0.774 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.613 | 0.792 | 0.819 | 0.758 |
| UN: uniform layer weights | 0.591 | 0.782 | 0.815 | 0.750 |
| Gpx: gate, pixels | 0.581 | 0.797 | 0.813 | 0.783 |
| Gpxx: gate, pixels, shuffled | 0.301 | 0.792 | 0.819 | 0.756 |
| Gs16: gate, SA k-means k=16 | 0.594 | 0.786 | 0.817 | 0.762 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.516 | 0.763 | 0.810 | 0.726 |
| Gs64: gate, SA k-means k=64 | 0.577 | 0.791 | 0.817 | 0.770 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.462 | 0.781 | 0.816 | 0.733 |
| Gs256: gate, SA k-means k=256 | 0.569 | 0.792 | 0.817 | 0.775 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.402 | 0.778 | 0.820 | 0.735 |
| Gg16: gate, grid 4x4 | 0.595 | 0.796 | 0.818 | 0.769 |
| Gg16x: gate, grid 4x4, shuffled | 0.541 | 0.782 | 0.814 | 0.744 |
| Gg64: gate, grid 8x8 | 0.592 | 0.798 | 0.815 | 0.777 |
| Gg64x: gate, grid 8x8, shuffled | 0.477 | 0.783 | 0.818 | 0.743 |
| Gg256: gate, grid 16x16 | 0.592 | 0.798 | 0.814 | 0.781 |
| Gg256x: gate, grid 16x16, shuffled | 0.423 | 0.789 | 0.821 | 0.750 |
| TE: early calls 0-9 | 0.589 | 0.771 | 0.808 | 0.739 |
| TM: middle calls 10-20 | 0.524 | 0.782 | 0.812 | 0.737 |
| TL: late calls 21-30 | 0.412 | 0.750 | 0.800 | 0.688 |
| TD+: seed + early - late | 0.504 | 0.732 | 0.788 | 0.694 |
| TD-: seed + late - early | 0.286 | 0.697 | 0.779 | 0.610 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR | 0.736 | 0.827 | 0.832 | 0.832 |
| UN | 0.749 | 0.822 | 0.829 | 0.827 |
| Gpx | 0.793 | 0.826 | 0.827 | 0.839 |
| Gpxx | 0.739 | 0.826 | 0.833 | 0.831 |
| Gs16 | 0.765 | 0.829 | 0.831 | 0.841 |
| Gs16x | 0.692 | 0.808 | 0.823 | 0.808 |
| Gs64 | 0.773 | 0.830 | 0.831 | 0.841 |
| Gs64x | 0.708 | 0.822 | 0.829 | 0.818 |
| Gs256 | 0.786 | 0.829 | 0.830 | 0.844 |
| Gs256x | 0.722 | 0.825 | 0.834 | 0.826 |
| Gg16 | 0.737 | 0.829 | 0.832 | 0.836 |
| Gg16x | 0.704 | 0.818 | 0.827 | 0.823 |
| Gg64 | 0.760 | 0.825 | 0.829 | 0.837 |
| Gg64x | 0.709 | 0.819 | 0.832 | 0.824 |
| Gg256 | 0.779 | 0.826 | 0.828 | 0.838 |
| Gg256x | 0.726 | 0.828 | 0.835 | 0.832 |
| TE | 0.644 | 0.811 | 0.822 | 0.799 |
| TM | 0.750 | 0.823 | 0.827 | 0.833 |
| TL | 0.747 | 0.814 | 0.816 | 0.821 |
| TD+ | 0.569 | 0.769 | 0.801 | 0.741 |
| TD- | 0.641 | 0.785 | 0.796 | 0.787 |

## Components, COCO-Cap (707 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.629 | 0.673 | 0.708 | 0.675 |
| UN: uniform layer weights | 0.616 | 0.657 | 0.674 | 0.645 |
| Gpx: gate, pixels | 0.637 | 0.643 | 0.673 | 0.641 |
| Gpxx: gate, pixels, shuffled | 0.624 | 0.669 | 0.705 | 0.683 |
| Gs16: gate, SA k-means k=16 | 0.619 | 0.636 | 0.676 | 0.637 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.572 | 0.642 | 0.659 | 0.630 |
| Gs64: gate, SA k-means k=64 | 0.623 | 0.652 | 0.674 | 0.638 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.575 | 0.656 | 0.702 | 0.652 |
| Gs256: gate, SA k-means k=256 | 0.635 | 0.638 | 0.670 | 0.638 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.595 | 0.663 | 0.697 | 0.665 |
| Gg16: gate, grid 4x4 | 0.599 | 0.643 | 0.676 | 0.642 |
| Gg16x: gate, grid 4x4, shuffled | 0.576 | 0.663 | 0.699 | 0.663 |
| Gg64: gate, grid 8x8 | 0.615 | 0.652 | 0.668 | 0.642 |
| Gg64x: gate, grid 8x8, shuffled | 0.588 | 0.669 | 0.697 | 0.664 |
| Gg256: gate, grid 16x16 | 0.623 | 0.643 | 0.669 | 0.643 |
| Gg256x: gate, grid 16x16, shuffled | 0.603 | 0.683 | 0.700 | 0.673 |
| TE: early calls 0-9 | 0.530 | 0.622 | 0.652 | 0.623 |
| TM: middle calls 10-20 | 0.635 | 0.693 | 0.708 | 0.683 |
| TL: late calls 21-30 | 0.611 | 0.651 | 0.680 | 0.637 |
| TD+: seed + early - late | 0.481 | 0.593 | 0.627 | 0.585 |
| TD-: seed + late - early | 0.518 | 0.602 | 0.648 | 0.599 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.527 | 0.698 | 0.708 | 0.678 |
| UN: uniform layer weights | 0.458 | 0.669 | 0.683 | 0.652 |
| Gpx: gate, pixels | 0.447 | 0.673 | 0.664 | 0.677 |
| Gpxx: gate, pixels, shuffled | 0.216 | 0.697 | 0.705 | 0.677 |
| Gs16: gate, SA k-means k=16 | 0.486 | 0.676 | 0.676 | 0.670 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.437 | 0.652 | 0.668 | 0.628 |
| Gs64: gate, SA k-means k=64 | 0.466 | 0.674 | 0.674 | 0.671 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.381 | 0.677 | 0.694 | 0.646 |
| Gs256: gate, SA k-means k=256 | 0.448 | 0.673 | 0.670 | 0.673 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.328 | 0.682 | 0.697 | 0.658 |
| Gg16: gate, grid 4x4 | 0.483 | 0.677 | 0.676 | 0.677 |
| Gg16x: gate, grid 4x4, shuffled | 0.442 | 0.689 | 0.699 | 0.661 |
| Gg64: gate, grid 8x8 | 0.465 | 0.673 | 0.668 | 0.678 |
| Gg64x: gate, grid 8x8, shuffled | 0.410 | 0.687 | 0.697 | 0.670 |
| Gg256: gate, grid 16x16 | 0.457 | 0.673 | 0.669 | 0.678 |
| Gg256x: gate, grid 16x16, shuffled | 0.350 | 0.694 | 0.700 | 0.672 |
| TE: early calls 0-9 | 0.502 | 0.648 | 0.652 | 0.629 |
| TM: middle calls 10-20 | 0.418 | 0.692 | 0.715 | 0.661 |
| TL: late calls 21-30 | 0.304 | 0.654 | 0.689 | 0.616 |
| TD+: seed + early - late | 0.428 | 0.614 | 0.627 | 0.594 |
| TD-: seed + late - early | 0.168 | 0.584 | 0.655 | 0.511 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.591 | 0.670 | 0.692 | 0.675 |
| UN: uniform layer weights | 0.572 | 0.643 | 0.670 | 0.644 |
| Gpx: gate, pixels | 0.594 | 0.643 | 0.660 | 0.630 |
| Gpxx: gate, pixels, shuffled | 0.564 | 0.668 | 0.690 | 0.671 |
| Gs16: gate, SA k-means k=16 | 0.582 | 0.633 | 0.671 | 0.623 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.535 | 0.627 | 0.663 | 0.618 |
| Gs64: gate, SA k-means k=64 | 0.586 | 0.637 | 0.668 | 0.640 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.533 | 0.652 | 0.688 | 0.651 |
| Gs256: gate, SA k-means k=256 | 0.596 | 0.637 | 0.665 | 0.639 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.543 | 0.659 | 0.682 | 0.662 |
| Gg16: gate, grid 4x4 | 0.563 | 0.629 | 0.670 | 0.630 |
| Gg16x: gate, grid 4x4, shuffled | 0.539 | 0.660 | 0.693 | 0.651 |
| Gg64: gate, grid 8x8 | 0.574 | 0.639 | 0.663 | 0.631 |
| Gg64x: gate, grid 8x8, shuffled | 0.543 | 0.667 | 0.693 | 0.664 |
| Gg256: gate, grid 16x16 | 0.582 | 0.639 | 0.661 | 0.632 |
| Gg256x: gate, grid 16x16, shuffled | 0.546 | 0.668 | 0.685 | 0.673 |
| TE: early calls 0-9 | 0.506 | 0.610 | 0.649 | 0.614 |
| TM: middle calls 10-20 | 0.592 | 0.689 | 0.702 | 0.682 |
| TL: late calls 21-30 | 0.562 | 0.633 | 0.666 | 0.633 |
| TD+: seed + early - late | 0.460 | 0.582 | 0.615 | 0.584 |
| TD-: seed + late - early | 0.483 | 0.594 | 0.639 | 0.586 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.510 | 0.690 | 0.692 | 0.676 |
| UN: uniform layer weights | 0.455 | 0.662 | 0.670 | 0.651 |
| Gpx: gate, pixels | 0.462 | 0.665 | 0.650 | 0.675 |
| Gpxx: gate, pixels, shuffled | 0.271 | 0.689 | 0.690 | 0.674 |
| Gs16: gate, SA k-means k=16 | 0.478 | 0.666 | 0.661 | 0.665 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.429 | 0.644 | 0.654 | 0.625 |
| Gs64: gate, SA k-means k=64 | 0.465 | 0.666 | 0.659 | 0.667 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.385 | 0.669 | 0.678 | 0.643 |
| Gs256: gate, SA k-means k=256 | 0.457 | 0.666 | 0.656 | 0.670 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.350 | 0.674 | 0.682 | 0.654 |
| Gg16: gate, grid 4x4 | 0.471 | 0.667 | 0.660 | 0.672 |
| Gg16x: gate, grid 4x4, shuffled | 0.432 | 0.680 | 0.684 | 0.658 |
| Gg64: gate, grid 8x8 | 0.461 | 0.664 | 0.653 | 0.673 |
| Gg64x: gate, grid 8x8, shuffled | 0.407 | 0.680 | 0.682 | 0.667 |
| Gg256: gate, grid 16x16 | 0.460 | 0.664 | 0.654 | 0.674 |
| Gg256x: gate, grid 16x16, shuffled | 0.365 | 0.686 | 0.685 | 0.669 |
| TE: early calls 0-9 | 0.481 | 0.639 | 0.639 | 0.625 |
| TM: middle calls 10-20 | 0.423 | 0.687 | 0.702 | 0.661 |
| TL: late calls 21-30 | 0.331 | 0.649 | 0.676 | 0.617 |
| TD+: seed + early - late | 0.413 | 0.605 | 0.615 | 0.588 |
| TD-: seed + late - early | 0.215 | 0.582 | 0.644 | 0.517 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR | 0.629 | 0.702 | 0.719 | 0.698 |
| UN | 0.616 | 0.671 | 0.697 | 0.670 |
| Gpx | 0.638 | 0.673 | 0.681 | 0.678 |
| Gpxx | 0.628 | 0.703 | 0.718 | 0.696 |
| Gs16 | 0.622 | 0.676 | 0.694 | 0.679 |
| Gs16x | 0.574 | 0.653 | 0.685 | 0.644 |
| Gs64 | 0.623 | 0.675 | 0.691 | 0.678 |
| Gs64x | 0.575 | 0.683 | 0.708 | 0.675 |
| Gs256 | 0.635 | 0.673 | 0.686 | 0.677 |
| Gs256x | 0.597 | 0.686 | 0.710 | 0.680 |
| Gg16 | 0.599 | 0.678 | 0.695 | 0.682 |
| Gg16x | 0.576 | 0.694 | 0.711 | 0.685 |
| Gg64 | 0.615 | 0.673 | 0.687 | 0.680 |
| Gg64x | 0.588 | 0.697 | 0.715 | 0.688 |
| Gg256 | 0.625 | 0.673 | 0.684 | 0.679 |
| Gg256x | 0.603 | 0.700 | 0.713 | 0.702 |
| TE | 0.530 | 0.648 | 0.672 | 0.640 |
| TM | 0.635 | 0.708 | 0.727 | 0.706 |
| TL | 0.611 | 0.665 | 0.700 | 0.658 |
| TD+ | 0.481 | 0.615 | 0.645 | 0.600 |
| TD- | 0.518 | 0.621 | 0.655 | 0.610 |

## Selection and contrasts (hierarchy + DenseCRF)

- SA regions: VOC-Sim OOF Gs16 0.829, Gs64 0.829, Gs256 0.830 → **Gs256**
- grid: VOC-Sim OOF Gg16 0.832, Gg64 0.829, Gg256 0.828 → **Gg16**

| Contrast | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---|---|---|---|
| BR − UN (GOCA weighting) | +0.004 [-0.002, +0.009] | +0.034 [+0.023, +0.046] | +0.003 [-0.002, +0.009] | +0.025 [+0.014, +0.036] |
| Gpx − BR | -0.006 [-0.012, -0.000] | -0.035 [-0.047, -0.024] | -0.006 [-0.013, -0.000] | -0.045 [-0.056, -0.033] |
| Gs256 − BR | -0.001 [-0.007, +0.005] | -0.038 [-0.051, -0.025] | -0.002 [-0.008, +0.003] | -0.038 [-0.051, -0.025] |
| Gg16 − BR | +0.001 [-0.005, +0.006] | -0.033 [-0.043, -0.022] | -0.000 [-0.006, +0.004] | -0.033 [-0.043, -0.022] |
| TD+ − BR | -0.032 [-0.045, -0.018] | -0.081 [-0.096, -0.065] | -0.031 [-0.045, -0.017] | -0.081 [-0.096, -0.065] |
| TD- − BR | -0.036 [-0.050, -0.022] | -0.060 [-0.079, -0.040] | -0.042 [-0.056, -0.028] | -0.053 [-0.074, -0.032] |
| Gpx − Gpxx (shuffled) | -0.008 [-0.014, -0.002] | -0.032 [-0.044, -0.021] | -0.007 [-0.014, -0.001] | -0.042 [-0.054, -0.030] |
| Gs256 − Gs256x (shuffled) | -0.002 [-0.009, +0.004] | -0.027 [-0.041, -0.012] | -0.003 [-0.010, +0.003] | -0.027 [-0.041, -0.012] |
| Gg16 − Gg16x (shuffled) | +0.006 [-0.002, +0.014] | -0.024 [-0.037, -0.011] | +0.005 [-0.003, +0.012] | -0.024 [-0.037, -0.011] |
| Gs256 − Gg16 (regions vs grid) | -0.002 [-0.005, +0.001] | -0.005 [-0.014, +0.003] | -0.002 [-0.005, +0.001] | -0.005 [-0.014, +0.003] |

The same contrasts at the other propagations (VOC-sel. + DenseCRF; VOC-Sim / COCO-Cap):

| Contrast | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---|---|---|---|
| BR − UN (GOCA weighting) | -0.011 / +0.013 | +0.004 / +0.016 | +0.004 / +0.034 | +0.003 / +0.031 |
| Gpx − BR | +0.057 / +0.008 | -0.002 / -0.029 | -0.006 / -0.035 | +0.009 / -0.035 |
| Gs256 − BR | +0.049 / +0.006 | +0.003 / -0.035 | -0.001 / -0.038 | +0.014 / -0.037 |
| Gg16 − BR | +0.000 / -0.030 | +0.005 / -0.029 | +0.001 / -0.033 | +0.006 / -0.034 |
| TD+ − BR | -0.168 / -0.148 | -0.057 / -0.080 | -0.032 / -0.081 | -0.091 / -0.091 |
| TD- − BR | -0.096 / -0.111 | -0.042 / -0.070 | -0.036 / -0.060 | -0.043 / -0.077 |
| Gpx − Gpxx (shuffled) | +0.054 / +0.013 | -0.002 / -0.026 | -0.008 / -0.032 | +0.010 / -0.042 |
| Gs256 − Gs256x (shuffled) | +0.064 / +0.041 | +0.002 / -0.025 | -0.002 / -0.027 | +0.018 / -0.026 |
| Gg16 − Gg16x (shuffled) | +0.033 / +0.023 | +0.014 / -0.019 | +0.006 / -0.024 | +0.013 / -0.022 |
| Gs256 − Gg16 (regions vs grid) | +0.049 / +0.036 | -0.001 / -0.006 | -0.002 / -0.005 | +0.008 / -0.003 |

**Reading** (helps = COCO Δ vs BR interval > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0; gates also need Δ vs shuffled interval > 0 on COCO):

- Gpx (gate, pixels): COCO -0.035 (lower -0.047), VOC -0.006, COCO best-thr. -0.037 → **does not help**; diagnostic failed
- Gs256 (gate, SA k-means k=256): COCO -0.038 (lower -0.051), VOC -0.001, COCO best-thr. -0.033 → **does not help**; diagnostic failed
- Gg16 (gate, grid 4x4): COCO -0.033 (lower -0.043), VOC +0.001, COCO best-thr. -0.024 → **does not help**; diagnostic failed
- TD+ (seed + early - late): COCO -0.081 (lower -0.096), VOC -0.032, COCO best-thr. -0.073 → **does not help**; diagnostic failed
- TD- (seed + late - early): COCO -0.060 (lower -0.079), VOC -0.036, COCO best-thr. -0.064 → **does not help**; diagnostic failed

**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**

| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BR | BRACE hierarchy | 0.708 | 0.601 | 0.779 | 0.821 | 0.105 | 0.0674 | 0.763 | 0.915 |
| BR | NERVE walk | 0.675 | 0.570 | 0.741 | 0.798 | 0.123 | 0.0884 | 0.722 | 0.922 |
| UN | BRACE hierarchy | 0.674 | 0.567 | 0.750 | 0.808 | 0.133 | 0.0897 | 0.722 | 0.926 |
| UN | NERVE walk | 0.645 | 0.560 | 0.716 | 0.777 | 0.137 | 0.1048 | 0.695 | 0.917 |
| Gpx | BRACE hierarchy | 0.673 | 0.575 | 0.769 | 0.799 | 0.117 | 0.0763 | 0.730 | 0.911 |
| Gpx | NERVE walk | 0.641 | 0.526 | 0.737 | 0.773 | 0.139 | 0.0936 | 0.690 | 0.911 |
| Gs256 | BRACE hierarchy | 0.670 | 0.569 | 0.769 | 0.795 | 0.117 | 0.0787 | 0.726 | 0.910 |
| Gs256 | NERVE walk | 0.638 | 0.520 | 0.732 | 0.772 | 0.139 | 0.0963 | 0.687 | 0.910 |
| Gg16 | BRACE hierarchy | 0.676 | 0.570 | 0.772 | 0.799 | 0.116 | 0.0778 | 0.729 | 0.916 |
| Gg16 | NERVE walk | 0.642 | 0.510 | 0.730 | 0.781 | 0.140 | 0.0936 | 0.690 | 0.911 |
| TD+ | BRACE hierarchy | 0.627 | 0.522 | 0.734 | 0.766 | 0.165 | 0.0976 | 0.686 | 0.887 |
| TD+ | NERVE walk | 0.585 | 0.473 | 0.664 | 0.723 | 0.182 | 0.1010 | 0.655 | 0.850 |
| TD- | BRACE hierarchy | 0.648 | 0.598 | 0.733 | 0.756 | 0.144 | 0.0937 | 0.713 | 0.878 |
| TD- | NERVE walk | 0.599 | 0.575 | 0.681 | 0.707 | 0.171 | 0.1188 | 0.672 | 0.855 |

## Per class, VOC-sel. + CRF, hierarchy (BR / Gpx / Gs256)

VOC-Sim: aeroplane 0.88 / 0.88 / 0.88, bicycle 0.79 / 0.80 / 0.80, bird 0.91 / 0.88 / 0.89, boat 0.85 / 0.83 / 0.83, bottle 0.87 / 0.89 / 0.90, bus 0.92 / 0.91 / 0.91, car 0.85 / 0.84 / 0.84, cat 0.85 / 0.86 / 0.87, chair 0.79 / 0.81 / 0.80, cow 0.90 / 0.88 / 0.89, dining-table 0.43 / 0.43 / 0.43, dog 0.93 / 0.93 / 0.94, horse 0.90 / 0.89 / 0.89, motorbike 0.87 / 0.87 / 0.87, person 0.81 / 0.81 / 0.80, potted-plant 0.75 / 0.77 / 0.77, sheep 0.76 / 0.73 / 0.75, sofa 0.96 / 0.96 / 0.96, train 0.74 / 0.73 / 0.75, tv 0.85 / 0.84 / 0.84

COCO-Cap: aeroplane 0.80 / 0.79 / 0.79, bicycle 0.63 / 0.57 / 0.58, bird 0.78 / 0.72 / 0.72, boat 0.59 / 0.54 / 0.53, bottle 0.57 / 0.55 / 0.59, bus 0.92 / 0.91 / 0.91, car 0.76 / 0.72 / 0.72, cat 0.92 / 0.90 / 0.89, chair 0.47 / 0.46 / 0.46, cow 0.87 / 0.86 / 0.86, dining table 0.47 / 0.44 / 0.46, dog 0.81 / 0.73 / 0.73, horse 0.72 / 0.69 / 0.66, motorbike 0.71 / 0.68 / 0.68, person 0.28 / 0.33 / 0.31, potted plant 0.74 / 0.68 / 0.69, sheep 0.87 / 0.86 / 0.86, sofa 0.70 / 0.63 / 0.63, train 0.75 / 0.67 / 0.67, tv 0.79 / 0.73 / 0.67

