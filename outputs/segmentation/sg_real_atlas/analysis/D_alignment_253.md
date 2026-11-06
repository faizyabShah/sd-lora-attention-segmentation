# SG-real-atlas alignment diagnostics on D (253 pairs, 168 photos)

Exploratory. The real mask is only used to measure the stored attention. AUROC: 0.5 = chance, 1 = the map ranks every object pixel above every non-object pixel; it ignores the threshold. The SA window is calls 10–20 (t ≈ 661→331). Object-size thirds cut at 0.055 / 0.199 of the image.

## 1. Cross-attention atlas (class token)

### renoise_n0

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.879

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.558 | 0.547 | 0.547 | 0.444 | 0.657 (h0) | 0.477 (h7) |
| d0.1 | 64 | 0.495 | 0.496 | 0.496 | 0.490 | 0.617 (h6) | 0.423 (h5) |
| d1.0 | 32 | 0.602 | 0.525 | 0.524 | 0.377 | 0.750 (h5) | 0.433 (h7) |
| d1.1 | 32 | 0.558 | 0.500 | 0.505 | 0.385 | 0.732 (h0) | 0.360 (h5) |
| d2.0 | 16 | 0.649 | 0.589 | 0.582 | 0.341 | 0.770 (h2) | 0.386 (h7) |
| d2.1 | 16 | 0.628 | 0.582 | 0.582 | 0.448 | 0.805 (h4) | 0.453 (h5) |
| mid.0 | 8 | 0.705 | 0.682 | 0.682 | 0.524 | 0.836 (h5) | 0.623 (h6) |
| u1.0 | 16 | 0.651 | 0.660 | 0.652 | 0.610 | 0.777 (h0) | 0.506 (h4) |
| u1.1 | 16 | 0.613 | 0.625 | 0.612 | 0.627 | 0.712 (h4) | 0.511 (h2) |
| u1.2 | 16 | 0.605 | 0.590 | 0.588 | 0.529 | 0.683 (h6) | 0.484 (h4) |
| u2.0 | 32 | 0.544 | 0.499 | 0.499 | 0.500 | 0.692 (h1) | 0.461 (h6) |
| u2.1 | 32 | 0.669 | 0.581 | 0.583 | 0.421 | 0.740 (h7) | 0.554 (h5) |
| u2.2 | 32 | 0.644 | 0.589 | 0.590 | 0.487 | 0.731 (h5) | 0.503 (h6) |
| u3.0 | 64 | 0.497 | 0.524 | 0.517 | 0.555 | 0.630 (h4) | 0.383 (h6) |
| u3.1 | 64 | 0.556 | 0.518 | 0.517 | 0.489 | 0.669 (h2) | 0.443 (h6) |
| u3.2 | 64 | 0.550 | 0.520 | 0.520 | 0.481 | 0.591 (h1) | 0.460 (h3) |

By call (prob, mean over modules and heads): c0 0.545, c3 0.552, c6 0.566, c9 0.578, c12 0.587, c15 0.595, c18 0.601, c21 0.606, c24 0.609, c27 0.610, c30 0.617

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.841, K=5: 0.841, K=20: 0.842, K=100: 0.811 — versus the A3 seed 0.879.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.98. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.41.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.268; small 0.087, medium 0.245, large 0.460.

Token AUROC (head-mean probability, window, all modules): SOT 0.331, a 0.523, photograph 0.512, of 0.504, a2 0.527, cls 0.702, EOT 0.634, pad1 0.634, pad2 0.634, pad3 0.634, pad4 0.634

By size third (prob, all modules): small 0.594 (seed 0.833), medium 0.601 (seed 0.900), large 0.586 (seed 0.903)

### ddim

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.906

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.563 | 0.549 | 0.549 | 0.428 | 0.689 (h0) | 0.419 (h7) |
| d0.1 | 64 | 0.480 | 0.484 | 0.484 | 0.483 | 0.602 (h6) | 0.402 (h5) |
| d1.0 | 32 | 0.616 | 0.533 | 0.532 | 0.366 | 0.765 (h5) | 0.446 (h7) |
| d1.1 | 32 | 0.557 | 0.497 | 0.502 | 0.379 | 0.740 (h0) | 0.359 (h5) |
| d2.0 | 16 | 0.650 | 0.586 | 0.581 | 0.345 | 0.769 (h2) | 0.367 (h7) |
| d2.1 | 16 | 0.636 | 0.588 | 0.588 | 0.449 | 0.835 (h4) | 0.444 (h5) |
| mid.0 | 8 | 0.720 | 0.697 | 0.696 | 0.528 | 0.852 (h5) | 0.629 (h6) |
| u1.0 | 16 | 0.666 | 0.676 | 0.666 | 0.632 | 0.793 (h0) | 0.512 (h4) |
| u1.1 | 16 | 0.625 | 0.635 | 0.620 | 0.642 | 0.736 (h4) | 0.517 (h5) |
| u1.2 | 16 | 0.611 | 0.595 | 0.591 | 0.541 | 0.696 (h6) | 0.485 (h4) |
| u2.0 | 32 | 0.540 | 0.492 | 0.490 | 0.510 | 0.715 (h1) | 0.445 (h6) |
| u2.1 | 32 | 0.685 | 0.590 | 0.592 | 0.415 | 0.756 (h7) | 0.574 (h5) |
| u2.2 | 32 | 0.665 | 0.601 | 0.603 | 0.475 | 0.751 (h5) | 0.522 (h6) |
| u3.0 | 64 | 0.487 | 0.516 | 0.507 | 0.560 | 0.604 (h4) | 0.369 (h6) |
| u3.1 | 64 | 0.551 | 0.512 | 0.510 | 0.491 | 0.671 (h2) | 0.452 (h6) |
| u3.2 | 64 | 0.541 | 0.524 | 0.524 | 0.492 | 0.584 (h1) | 0.449 (h3) |

By call (prob, mean over modules and heads): c0 0.565, c3 0.573, c6 0.584, c9 0.591, c12 0.595, c15 0.598, c18 0.602, c21 0.604, c24 0.606, c27 0.608, c30 0.617

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.849, K=5: 0.849, K=20: 0.849, K=100: 0.820 — versus the A3 seed 0.906.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.98. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.43.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.274; small 0.096, medium 0.254, large 0.461.

Token AUROC (head-mean probability, window, all modules): SOT 0.324, a 0.519, photograph 0.503, of 0.497, a2 0.524, cls 0.711, EOT 0.642, pad1 0.641, pad2 0.641, pad3 0.641, pad4 0.641

By size third (prob, all modules): small 0.604 (seed 0.865), medium 0.605 (seed 0.931), large 0.586 (seed 0.919)

## 2. Self-attention propagation atlas (real mask, keep / leak)

keep = share of a mask pixel's attention that lands inside the mask; leak = the same for a non-mask pixel. Chance for both is the mask area a. Reported: keep lift (keep − a)/(1 − a) (1 = perfect), leak ratio leak / a (0 = no leak, 1 = chance), for the exact, eroded (interior) and dilated (+2 px at 64) masks.

### renoise_n0

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.229 | 0.802 | 0.213 | 0.806 | 0.335 |
| d0.1 | 64 | 0.359 | 0.697 | 0.317 | 0.691 | 0.146 |
| d1.0 | 32 | 0.328 | 0.737 | 0.283 | 0.686 | 0.395 |
| d1.1 | 32 | 0.356 | 0.736 | 0.303 | 0.706 | 0.244 |
| d2.0 | 16 | 0.257 | 0.839 | 0.215 | 0.777 | 0.392 |
| d2.1 | 16 | 0.238 | 0.853 | 0.189 | 0.791 | 0.517 |
| mid.0 | 8 | 0.178 | 0.956 | 0.140 | 0.901 | 0.458 |
| u1.0 | 16 | 0.298 | 0.816 | 0.251 | 0.733 | 0.490 |
| u1.1 | 16 | 0.421 | 0.652 | 0.350 | 0.546 | 0.204 |
| u1.2 | 16 | 0.271 | 0.845 | 0.211 | 0.757 | 0.473 |
| u2.0 | 32 | 0.514 | 0.405 | 0.477 | 0.344 | 0.188 |
| u2.1 | 32 | 0.534 | 0.410 | 0.461 | 0.380 | 0.250 |
| u2.2 | 32 | 0.384 | 0.684 | 0.316 | 0.646 | 0.438 |
| u3.0 | 64 | 0.616 | 0.314 | 0.566 | 0.335 | 0.087 |
| u3.1 | 64 | 0.607 | 0.327 | 0.519 | 0.364 | 0.256 |
| u3.2 | 64 | 0.441 | 0.530 | 0.369 | 0.543 | 0.359 |

Keep lift by call (exact, mean over modules and heads): c0 0.092, c3 0.123, c6 0.185, c9 0.254, c12 0.324, c15 0.384, c18 0.432, c21 0.466, c24 0.490, c27 0.506, c30 0.513

- small objects: keep lift 0.191, median leak ratio 0.889
- medium objects: keep lift 0.405, median leak ratio 0.658
- large objects: keep lift 0.535, median leak ratio 0.452

### ddim

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.243 | 0.789 | 0.220 | 0.788 | 0.350 |
| d0.1 | 64 | 0.375 | 0.665 | 0.329 | 0.658 | 0.156 |
| d1.0 | 32 | 0.334 | 0.736 | 0.285 | 0.681 | 0.402 |
| d1.1 | 32 | 0.366 | 0.720 | 0.307 | 0.689 | 0.250 |
| d2.0 | 16 | 0.264 | 0.832 | 0.219 | 0.768 | 0.402 |
| d2.1 | 16 | 0.251 | 0.845 | 0.196 | 0.776 | 0.524 |
| mid.0 | 8 | 0.187 | 0.954 | 0.147 | 0.892 | 0.475 |
| u1.0 | 16 | 0.318 | 0.807 | 0.266 | 0.711 | 0.522 |
| u1.1 | 16 | 0.446 | 0.634 | 0.367 | 0.521 | 0.217 |
| u1.2 | 16 | 0.293 | 0.846 | 0.224 | 0.734 | 0.504 |
| u2.0 | 32 | 0.539 | 0.377 | 0.493 | 0.324 | 0.194 |
| u2.1 | 32 | 0.563 | 0.379 | 0.477 | 0.363 | 0.259 |
| u2.2 | 32 | 0.405 | 0.658 | 0.326 | 0.621 | 0.458 |
| u3.0 | 64 | 0.645 | 0.281 | 0.581 | 0.318 | 0.087 |
| u3.1 | 64 | 0.640 | 0.288 | 0.530 | 0.352 | 0.268 |
| u3.2 | 64 | 0.462 | 0.503 | 0.375 | 0.531 | 0.383 |

Keep lift by call (exact, mean over modules and heads): c0 0.132, c3 0.169, c6 0.229, c9 0.292, c12 0.351, c15 0.402, c18 0.441, c21 0.469, c24 0.490, c27 0.504, c30 0.512

- small objects: keep lift 0.208, median leak ratio 0.875
- medium objects: keep lift 0.431, median leak ratio 0.630
- large objects: keep lift 0.548, median leak ratio 0.440

## 3. How many propagation steps? (y_t = A^t seed, head-mean A, averaged over the 11 calls)

### renoise_n0

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.877 | 0.887 | 0.883 | 0.879 | 0.875 | 0.871 | 0.866 | 0.861 | 0.856 |
| 16 | AP | 0.623 | 0.670 | 0.666 | 0.660 | 0.652 | 0.645 | 0.638 | 0.632 | 0.625 |
| 32 | AUROC | 0.879 | 0.894 | 0.893 | 0.890 | 0.883 | 0.875 | 0.868 | 0.862 | 0.856 |
| 32 | AP | 0.628 | 0.687 | 0.682 | 0.679 | 0.672 | 0.662 | 0.652 | 0.643 | 0.632 |
| 64 | AUROC | 0.879 | 0.897 | 0.895 | 0.890 | 0.883 | 0.875 | 0.868 | 0.861 | 0.854 |
| 64 | AP | 0.628 | 0.683 | 0.673 | 0.659 | 0.646 | 0.629 | 0.615 | 0.605 | 0.596 |

C16 path A64·up(A16^t c16): AUROC t=1 0.894, t=2 0.889, t=3 0.885, t=4 0.880; AP t=1 0.690, t=2 0.690, t=3 0.685, t=4 0.678

- small: AP side 16 0.337 0.355 0.339 0.315 0.280 | side 64 0.349 0.363 0.325 0.272 0.187 (t = 0,1,2,4,8)
- medium: AP side 16 0.680 0.745 0.739 0.716 0.678 | side 64 0.684 0.770 0.767 0.729 0.661 (t = 0,1,2,4,8)
- large: AP side 16 0.848 0.905 0.916 0.921 0.912 | side 64 0.847 0.910 0.922 0.933 0.934 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.811 h1 0.893 h2 0.727 h3 0.869 h4 0.886 h5 0.869 h6 0.795 h7 0.887
- per-head AUROC at t=2, side 64: h0 0.894 h1 0.894 h2 0.889 h3 0.883 h4 0.875 h5 0.893 h6 0.895 h7 0.889

### ddim

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.904 | 0.911 | 0.906 | 0.900 | 0.894 | 0.890 | 0.886 | 0.882 | 0.877 |
| 16 | AP | 0.673 | 0.716 | 0.702 | 0.690 | 0.681 | 0.675 | 0.668 | 0.659 | 0.653 |
| 32 | AUROC | 0.905 | 0.918 | 0.913 | 0.904 | 0.895 | 0.888 | 0.880 | 0.872 | 0.866 |
| 32 | AP | 0.678 | 0.739 | 0.726 | 0.704 | 0.689 | 0.677 | 0.662 | 0.650 | 0.637 |
| 64 | AUROC | 0.906 | 0.920 | 0.913 | 0.903 | 0.894 | 0.886 | 0.878 | 0.869 | 0.862 |
| 64 | AP | 0.678 | 0.739 | 0.718 | 0.695 | 0.674 | 0.657 | 0.641 | 0.624 | 0.610 |

C16 path A64·up(A16^t c16): AUROC t=1 0.914, t=2 0.908, t=3 0.902, t=4 0.898; AP t=1 0.735, t=2 0.725, t=3 0.715, t=4 0.711

- small: AP side 16 0.391 0.418 0.383 0.339 0.297 | side 64 0.403 0.445 0.383 0.301 0.205 (t = 0,1,2,4,8)
- medium: AP side 16 0.749 0.810 0.794 0.774 0.730 | side 64 0.754 0.840 0.830 0.773 0.684 (t = 0,1,2,4,8)
- large: AP side 16 0.874 0.915 0.924 0.925 0.925 | side 64 0.873 0.927 0.936 0.944 0.936 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.835 h1 0.912 h2 0.737 h3 0.893 h4 0.908 h5 0.897 h6 0.811 h7 0.907
- per-head AUROC at t=2, side 64: h0 0.915 h1 0.911 h2 0.903 h3 0.895 h4 0.885 h5 0.913 h6 0.918 h7 0.905

## 4. Spectral structure of the SA graphs (first 50 photos, renoise/n0)

76 pairs. E_k = share of the mask's energy in the k smoothest non-trivial eigenvectors (rolled mask = same shape, wrong place = chance).

| side | graph | E_k mask (k=1,2,4,8,16,32) | E_k rolled | E_k seed | best eigvec AUROC (oracle, j≤16) | seed-picked eigvec AUROC | seed low-pass AUROC (k=4,8,16,32,64) | seed AUROC |
|---|---|---|---|---|---:|---:|---|---:|
| 16 | mean | 0.28 0.39 0.49 0.56 0.63 0.70 | 0.11 0.14 0.18 0.24 0.31 0.39 | 0.39 0.56 0.69 0.78 0.84 0.89 | 0.952 | 0.834 | 0.87 0.87 0.87 0.88 0.88 | 0.886 |
| 16 | head | 0.20 0.31 0.40 0.48 0.55 0.61 | 0.08 0.12 0.16 0.21 0.28 0.36 | 0.28 0.46 0.58 0.67 0.73 0.78 | 0.918 | 0.793 | 0.83 0.85 0.86 0.86 0.87 | 0.886 |
| 32 | mean | 0.22 0.35 0.48 0.57 0.63 0.69 | 0.08 0.13 0.17 0.22 0.28 0.36 | 0.28 0.50 0.65 0.74 0.81 0.86 | 0.943 | 0.810 | 0.86 0.87 0.88 0.87 0.87 | 0.876 |
| 32 | head | 0.19 0.32 0.44 0.52 0.59 0.65 | 0.07 0.12 0.16 0.20 0.26 0.33 | 0.26 0.45 0.60 0.69 0.76 0.82 | 0.928 | 0.801 | 0.84 0.86 0.87 0.87 0.87 | 0.876 |
| 64 | mean | 0.19 0.31 0.43 0.52 0.57 0.62 | 0.07 0.11 0.15 0.19 0.23 0.29 | 0.25 0.40 0.55 0.64 0.70 0.76 | 0.920 | 0.798 | 0.83 0.87 0.87 0.87 0.87 | 0.881 |

Seed low-pass AUROC on the head-mean graph by object size (k = 4 / 8 / 16, side 32 and 64) vs seed:
- small (n=28): side 32: 0.757 0.790 0.807 (seed 0.826); side 64: 0.744 0.794 0.805 (seed 0.831)
- medium (n=22): side 32: 0.869 0.886 0.889 (seed 0.888); side 64: 0.832 0.869 0.877 (seed 0.893)
- large (n=26): side 32: 0.949 0.946 0.937 (seed 0.921); side 64: 0.935 0.941 0.937 (seed 0.925)

Pairs where a 16 px graph had no valid in-mask cell (AUROC undefined, dropped from those means): 2

## 5. Where the errors are

Pixel shares at the OOF thresholds of analysis 1. FP split by what the wrong pixel actually is; FN by whether it lies within 8 px (≈ one latent cell) of the object boundary.

| variant | precision | recall | FP: other object | FP: background ≤8 px | FP: background far | FN: boundary band | FN: interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| n0_V0R2 | 0.523 | 0.775 | 47% | 3% | 50% | 39% | 61% |
| ddim_C16 | 0.568 | 0.806 | 52% | 4% | 44% | 40% | 60% |
| ddim_V0 | 0.534 | 0.811 | 45% | 4% | 51% | 39% | 61% |

Instance coverage (share of each instance's pixels predicted), by instance size as a share of the image:

| variant | <1% | 1–5% | 5–20% | >20% | instances missed (<10% covered) |
|---|---:|---:|---:|---:|---:|
| n0_V0R2 | 0.52 (n=126) | 0.69 (n=128) | 0.77 (n=118) | 0.79 (n=67) | 18% of 439 |
| ddim_C16 | 0.52 (n=126) | 0.77 (n=128) | 0.83 (n=118) | 0.82 (n=67) | 16% of 439 |
| ddim_V0 | 0.59 (n=126) | 0.80 (n=128) | 0.83 (n=118) | 0.82 (n=67) | 13% of 439 |

Most FP pixels on another object (n0_V0R2), target → other class: 
chair→diningtable 12%, person→car 6%, person→train 6%, person→diningtable 6%, bottle→person 5%, pottedplant→diningtable 4%, chair→sofa 3%, person→bus 3%, bicycle→person 3%, bottle→sheep 3%, chair→person 3%, diningtable→person 2%

