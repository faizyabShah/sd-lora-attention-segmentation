# Cross-validated selection

490 images, 20 classes, 78 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| Joint CA-SA | `CA=C/R|SA=C/L` | `CA=C/R|SA=C/L` (5/5) | +0.0000 | `CA=C/R|SA=C/L` | reference won the fold vote | 0.7887 / 0.7887 | +0.0000 | +0.0000 [+0.0000, +0.0000] | no |

## Joint CA-SA

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `CA=C/R|SA=C/L` | no eligible candidate | 0.7892 / 0.7892 | 0.7892 / 0.7892 |
| 1 | `CA=C/R|SA=C/L` | no eligible candidate | 0.7825 / 0.7825 | 0.8116 / 0.8116 |
| 2 | `CA=C/R|SA=C/L` | no eligible candidate | 0.7899 / 0.7899 | 0.7869 / 0.7869 |
| 3 | `CA=C/R|SA=C/L` | no eligible candidate | 0.7908 / 0.7908 | 0.7825 / 0.7825 |
| 4 | `CA=C/R|SA=C/L` | no eligible candidate | 0.7910 / 0.7910 | 0.7757 / 0.7757 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `CA=C/R+H|SA=C/L` | 0.7690 | 0.7904 | 0.7539 | 0.9559 |
| `CA=C/R+H|SA=C/L+H` | 0.7636 | 0.7888 | 0.7563 | 0.9585 |
| `CA=C/R|SA=C/L` | 0.7691 | 0.7887 | 0.7532 | 0.9560 |
| `CA=C/R|SA=C/L+H` | 0.7637 | 0.7871 | 0.7554 | 0.9587 |
| `CA=C/R+H|SA=C/T+L+H` | 0.7661 | 0.7831 | 0.7323 | 0.9514 |
| `CA=C/R+H|SA=C/T+L` | 0.7701 | 0.7826 | 0.7241 | 0.9475 |
| `CA=C/R|SA=C/T+L+H` | 0.7658 | 0.7825 | 0.7330 | 0.9514 |
| `CA=C/R|SA=C/T+L` | 0.7704 | 0.7820 | 0.7260 | 0.9474 |
| `CA=C/L+H|SA=C/L` | 0.7605 | 0.7780 | 0.7493 | 0.9592 |
| `CA=C/L|SA=C/L` | 0.7617 | 0.7773 | 0.7481 | 0.9594 |

## Full CA × SA matrix (490-image in-sample diagnostic)

Rows contain every CA-C cell; columns contain no-SA and every unique SA-C cell. Selection itself is cross-validated above.

| CA configuration | SA none | SA C/reference | SA C/T | SA C/R | SA C/L | SA C/H | SA C/T+R | SA C/T+L | SA C/T+H | SA C/R+H | SA C/L+H | SA C/T+R+H | SA C/T+L+H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C/reference` | 0.6661 | 0.7444 | 0.7573 | 0.7521 | 0.7705 | 0.7495 | 0.7557 | 0.7616 | 0.7582 | 0.7496 | 0.7688 | 0.7480 | 0.7621 |
| `C/R` | 0.6957 | 0.7556 | 0.7710 | 0.7676 | 0.7887 | 0.7613 | 0.7749 | 0.7820 | 0.7732 | 0.7656 | 0.7871 | 0.7692 | 0.7825 |
| `C/L` | 0.6922 | 0.7473 | 0.7622 | 0.7571 | 0.7773 | 0.7529 | 0.7650 | 0.7714 | 0.7651 | 0.7552 | 0.7760 | 0.7603 | 0.7720 |
| `C/H` | 0.6765 | 0.7465 | 0.7606 | 0.7551 | 0.7738 | 0.7519 | 0.7612 | 0.7670 | 0.7624 | 0.7528 | 0.7722 | 0.7549 | 0.7674 |
| `C/R+H` | 0.6961 | 0.7568 | 0.7723 | 0.7691 | 0.7904 | 0.7623 | 0.7765 | 0.7826 | 0.7748 | 0.7671 | 0.7888 | 0.7711 | 0.7831 |
| `C/L+H` | 0.6925 | 0.7475 | 0.7632 | 0.7576 | 0.7780 | 0.7533 | 0.7670 | 0.7730 | 0.7666 | 0.7558 | 0.7765 | 0.7631 | 0.7733 |

## Best partner for each CA configuration (in-sample diagnostic)

| CA | Best SA | mIoU@0.45 | Gain over that CA without SA |
|---|---|---:|---:|
| `C/reference` | `C/L` | 0.7705 | +0.1044 |
| `C/R` | `C/L` | 0.7887 | +0.0929 |
| `C/L` | `C/L` | 0.7773 | +0.0851 |
| `C/H` | `C/L` | 0.7738 | +0.0974 |
| `C/R+H` | `C/L` | 0.7904 | +0.0943 |
| `C/L+H` | `C/L` | 0.7780 | +0.0855 |

## Per-class IoU: closest in-sample alternative (diagnostic only)

`CA=C/R+H|SA=C/L` versus `CA=C/R|SA=C/L`. Δ mIoU@0.45 +0.0017 (marginal); paired mean image-IoU +0.0003 [-0.0016, +0.0024].

| Class | Sequential reference | Comparison | Δ |
|---|---:|---:|---:|
| aeroplane | 0.8450 | 0.8478 | +0.0028 |
| bicycle | 0.7471 | 0.7540 | +0.0068 |
| bird | 0.8531 | 0.8562 | +0.0030 |
| boat | 0.8318 | 0.8275 | -0.0043 |
| bottle | 0.8665 | 0.8662 | -0.0002 |
| bus | 0.9184 | 0.9191 | +0.0007 |
| car | 0.8420 | 0.8465 | +0.0045 |
| cat | 0.7816 | 0.7846 | +0.0030 |
| chair | 0.6503 | 0.6456 | -0.0048 |
| cow | 0.8184 | 0.8247 | +0.0063 |
| dining-table | 0.4677 | 0.4583 | -0.0094 |
| dog | 0.8960 | 0.8961 | +0.0001 |
| horse | 0.8696 | 0.8706 | +0.0010 |
| motorbike | 0.8527 | 0.8545 | +0.0018 |
| person | 0.6664 | 0.6466 | -0.0198 |
| potted-plant | 0.7091 | 0.7054 | -0.0037 |
| sheep | 0.7660 | 0.8242 | +0.0583 |
| sofa | 0.9447 | 0.9442 | -0.0005 |
| train | 0.6697 | 0.6509 | -0.0188 |
| tv | 0.7768 | 0.7846 | +0.0078 |
