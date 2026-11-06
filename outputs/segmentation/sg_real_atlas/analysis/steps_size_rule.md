# Size-dependent propagation depth: real D vs VOC-Sim (exploratory)

P_t = mean over the 11 window calls and 8 heads of A_h^t c on the frozen 64 px BRACE module (P_2 = V0). a_hat = share of the seed with minmax ≥ 0.5 (label-free). Depth rule: one depth per a_hat bin (< 0.05, 0.05–0.2, 0.2–0.42, ≥ 0.42), chosen with τ on training folds. Class-macro mIoU; OOF = τ (and rule) picked on 4 grouped folds, scored on the 5th; Δ CIs by image bootstrap.

## real D, renoise/n0 (307 pairs)

P_2 vs stored V0 (minmax) max abs diff: 1.4e-06. ρ(a_hat, true area) = 0.32; pairs per a_hat bin: 4, 79, 159, 65

| map | OOF mIoU raw | OOF mIoU +R2 |
|---|---:|---:|
| P0 | 0.440 |  |
| P1 | 0.507 | 0.516 |
| P2 | 0.497 | 0.495 |
| P3 | 0.481 | 0.479 |
| P4 | 0.478 | 0.476 |
| P5 | 0.464 | 0.466 |
| P6 | 0.460 | 0.457 |
| V0 | 0.497 | 0.495 |
| V3 | 0.487 | 0.489 |
| C16 | 0.519 |  |
| **size rule** | **0.511** | **0.516** |

Rules picked per fold (depth per bin, τ): raw [1, 1, 1, 6] @0.650; [1, 1, 1, 6] @0.650; [1, 1, 1, 5] @0.650; [2, 1, 2, 6] @0.675; [1, 1, 1, 6] @0.625 — +R2 [1, 1, 1, 6] @0.65; [1, 1, 1, 6] @0.65; [1, 1, 1, 4] @0.65; [1, 1, 1, 6] @0.65; [1, 1, 1, 6] @0.60

- rule − V0: +0.014 (95% CI -0.003 … +0.027)
- rule+R2 − V0+R2: +0.020 (95% CI +0.002 … +0.033)
- rule+R2 − V3+R2: +0.027 (95% CI -0.012 … +0.053)
- rule − C16: -0.008 (95% CI -0.027 … +0.007)

Headroom: mean per-pair IoU at τ 0.55 — P2 0.469; best depth per pair (oracle) 0.509.

Best depth by true-area bin (raw, mean per-pair IoU at each depth's best global τ):
- area 0.00–0.05 (n=101): P0 0.173 P1 0.199 P2 0.189 P3 0.174 P4 0.161 P5 0.146 P6 0.137
- area 0.05–0.20 (n=104): P0 0.478 P1 0.572 P2 0.567 P3 0.556 P4 0.543 P5 0.535 P6 0.530
- area 0.20–0.42 (n=75): P0 0.609 P1 0.723 P2 0.740 P3 0.743 P4 0.738 P5 0.735 P6 0.733
- area 0.42–1.01 (n=27): P0 0.754 P1 0.796 P2 0.804 P3 0.804 P4 0.801 P5 0.799 P6 0.796

## real D, ddim (307 pairs)

P_2 vs stored V0 (minmax) max abs diff: 2.3e-06. ρ(a_hat, true area) = 0.32; pairs per a_hat bin: 10, 100, 151, 46

| map | OOF mIoU raw | OOF mIoU +R2 |
|---|---:|---:|
| P0 | 0.488 |  |
| P1 | 0.542 | 0.547 |
| P2 | 0.536 | 0.531 |
| P3 | 0.521 | 0.521 |
| P4 | 0.499 | 0.495 |
| P5 | 0.502 | 0.503 |
| P6 | 0.487 | 0.480 |
| V0 | 0.536 | 0.531 |
| V3 | 0.536 | 0.538 |
| C16 | 0.553 |  |
| **size rule** | **0.543** | **0.552** |

Rules picked per fold (depth per bin, τ): raw [0, 2, 2, 6] @0.625; [0, 1, 2, 6] @0.625; [0, 2, 1, 6] @0.600; [0, 2, 1, 6] @0.600; [0, 1, 1, 6] @0.575 — +R2 [1, 1, 1, 6] @0.60; [1, 1, 1, 6] @0.60; [1, 1, 1, 6] @0.60; [1, 1, 1, 6] @0.60; [1, 1, 1, 6] @0.60

- rule − V0: +0.007 (95% CI -0.014 … +0.024)
- rule+R2 − V0+R2: +0.021 (95% CI -0.002 … +0.034)
- rule+R2 − V3+R2: +0.014 (95% CI -0.012 … +0.032)
- rule − C16: -0.011 (95% CI -0.032 … +0.010)

Headroom: mean per-pair IoU at τ 0.55 — P2 0.507; best depth per pair (oracle) 0.557.

Best depth by true-area bin (raw, mean per-pair IoU at each depth's best global τ):
- area 0.00–0.05 (n=101): P0 0.225 P1 0.245 P2 0.223 P3 0.211 P4 0.197 P5 0.184 P6 0.175
- area 0.05–0.20 (n=104): P0 0.541 P1 0.636 P2 0.630 P3 0.622 P4 0.611 P5 0.603 P6 0.593
- area 0.20–0.42 (n=75): P0 0.643 P1 0.747 P2 0.764 P3 0.759 P4 0.755 P5 0.752 P6 0.749
- area 0.42–1.01 (n=27): P0 0.753 P1 0.834 P2 0.846 P3 0.847 P4 0.832 P5 0.831 P6 0.830

## VOC-Sim, generate (490 pairs)

P_2 vs stored V0 (minmax) max abs diff: 0.0e+00. ρ(a_hat, true area) = 0.73; pairs per a_hat bin: 0, 113, 283, 94

| map | OOF mIoU raw | OOF mIoU +R2 |
|---|---:|---:|
| P0 | 0.705 |  |
| P1 | 0.798 | 0.810 |
| P2 | 0.810 | 0.824 |
| P3 | 0.811 | 0.824 |
| P4 | 0.810 | 0.814 |
| P5 | 0.808 | 0.811 |
| P6 | 0.805 | 0.809 |
| V0 | 0.810 | 0.824 |
| V3 | 0.816 | 0.830 |
| C16 | 0.802 |  |
| **size rule** | **0.810** | **0.823** |

Rules picked per fold (depth per bin, τ): raw [0, 2, 4, 2] @0.475; [0, 2, 3, 2] @0.450; [0, 2, 3, 3] @0.475; [0, 3, 4, 3] @0.500; [0, 2, 4, 2] @0.475 — +R2 [1, 1, 3, 2] @0.45; [1, 1, 3, 2] @0.45; [1, 2, 2, 3] @0.45; [1, 1, 3, 3] @0.45; [1, 1, 3, 2] @0.45

- rule − V0: -0.000 (95% CI -0.004 … +0.003)
- rule+R2 − V0+R2: -0.001 (95% CI -0.004 … +0.001)
- rule+R2 − V3+R2: -0.007 (95% CI -0.015 … -0.000)
- rule − C16: +0.008 (95% CI -0.000 … +0.016)

Headroom: mean per-pair IoU at τ 0.55 — P2 0.800; best depth per pair (oracle) 0.833.

Best depth by true-area bin (raw, mean per-pair IoU at each depth's best global τ):
- area 0.05–0.20 (n=131): P0 0.639 P1 0.774 P2 0.785 P3 0.784 P4 0.782 P5 0.778 P6 0.774
- area 0.20–0.42 (n=191): P0 0.732 P1 0.850 P2 0.861 P3 0.858 P4 0.856 P5 0.854 P6 0.852
- area 0.42–1.01 (n=164): P0 0.812 P1 0.866 P2 0.869 P3 0.868 P4 0.863 P5 0.859 P6 0.856

## Transfer: rule fitted on all of one dataset, applied frozen to the other

| fitted on → applied to | rule (raw) | mIoU raw | P2 raw at same τ | rule (+R2) | mIoU +R2 | P2+R2 at same τ |
|---|---|---:|---:|---|---:|---:|
| real D, ddim → VOC-Sim, generate | [0, 1, 1, 6] @0.600 | 0.740 | 0.757 | [1, 1, 1, 6] @0.60 | 0.741 | 0.758 |
| real D, renoise/n0 → VOC-Sim, generate | [1, 1, 1, 6] @0.650 | 0.691 | 0.705 | [1, 1, 1, 6] @0.65 | 0.688 | 0.701 |
| VOC-Sim, generate → real D, ddim | [0, 2, 3, 2] @0.475 | 0.477 | 0.470 | [1, 1, 3, 3] @0.45 | 0.478 | 0.457 |
| VOC-Sim, generate → real D, renoise/n0 | [0, 2, 3, 2] @0.475 | 0.417 | 0.423 | [1, 1, 3, 3] @0.45 | 0.420 | 0.412 |
