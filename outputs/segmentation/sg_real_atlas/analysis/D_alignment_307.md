# SG-real-atlas alignment diagnostics on D (307 pairs, 200 images)

Exploratory. The real mask is only used to measure the stored attention. AUROC: 0.5 = chance, 1 = the map ranks every object pixel above every non-object pixel; it ignores the threshold. The SA window is calls 10–20 (t ≈ 661→331). Object-size thirds cut at 0.051 / 0.199 of the image.

## 1. Cross-attention atlas (class token)

### renoise_n0

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.875

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.557 | 0.545 | 0.545 | 0.446 | 0.652 (h0) | 0.474 (h7) |
| d0.1 | 64 | 0.495 | 0.497 | 0.496 | 0.491 | 0.613 (h6) | 0.423 (h5) |
| d1.0 | 32 | 0.599 | 0.523 | 0.522 | 0.379 | 0.738 (h5) | 0.441 (h7) |
| d1.1 | 32 | 0.555 | 0.499 | 0.504 | 0.388 | 0.725 (h0) | 0.366 (h5) |
| d2.0 | 16 | 0.647 | 0.587 | 0.580 | 0.343 | 0.764 (h2) | 0.393 (h7) |
| d2.1 | 16 | 0.630 | 0.582 | 0.582 | 0.449 | 0.809 (h4) | 0.457 (h5) |
| mid.0 | 8 | 0.706 | 0.680 | 0.680 | 0.524 | 0.835 (h5) | 0.624 (h6) |
| u1.0 | 16 | 0.654 | 0.659 | 0.650 | 0.608 | 0.782 (h0) | 0.517 (h4) |
| u1.1 | 16 | 0.615 | 0.624 | 0.611 | 0.621 | 0.716 (h4) | 0.510 (h5) |
| u1.2 | 16 | 0.606 | 0.590 | 0.588 | 0.525 | 0.684 (h6) | 0.482 (h4) |
| u2.0 | 32 | 0.544 | 0.498 | 0.498 | 0.496 | 0.692 (h1) | 0.458 (h6) |
| u2.1 | 32 | 0.665 | 0.578 | 0.580 | 0.420 | 0.733 (h7) | 0.555 (h5) |
| u2.2 | 32 | 0.642 | 0.587 | 0.589 | 0.483 | 0.727 (h5) | 0.503 (h6) |
| u3.0 | 64 | 0.495 | 0.523 | 0.516 | 0.555 | 0.629 (h4) | 0.380 (h5) |
| u3.1 | 64 | 0.552 | 0.520 | 0.518 | 0.493 | 0.659 (h2) | 0.438 (h6) |
| u3.2 | 64 | 0.550 | 0.519 | 0.519 | 0.479 | 0.595 (h1) | 0.455 (h3) |

By call (prob, mean over modules and heads): c0 0.545, c3 0.551, c6 0.565, c9 0.576, c12 0.586, c15 0.594, c18 0.600, c21 0.604, c24 0.607, c27 0.609, c30 0.616

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.838, K=5: 0.837, K=20: 0.841, K=100: 0.810 — versus the A3 seed 0.875.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.98. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.42.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.267; small 0.079, medium 0.244, large 0.463.

Token AUROC (head-mean probability, window, all modules): SOT 0.333, a 0.523, photograph 0.514, of 0.505, a2 0.529, cls 0.700, EOT 0.633, pad1 0.633, pad2 0.633, pad3 0.632, pad4 0.632

By size third (prob, all modules): small 0.592 (seed 0.827), medium 0.602 (seed 0.902), large 0.585 (seed 0.894)

### ddim

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.901

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.564 | 0.548 | 0.548 | 0.428 | 0.690 (h0) | 0.417 (h7) |
| d0.1 | 64 | 0.479 | 0.484 | 0.484 | 0.485 | 0.597 (h6) | 0.402 (h5) |
| d1.0 | 32 | 0.616 | 0.532 | 0.531 | 0.367 | 0.756 (h5) | 0.459 (h7) |
| d1.1 | 32 | 0.555 | 0.498 | 0.502 | 0.381 | 0.739 (h0) | 0.361 (h5) |
| d2.0 | 16 | 0.651 | 0.587 | 0.581 | 0.345 | 0.766 (h2) | 0.380 (h7) |
| d2.1 | 16 | 0.639 | 0.587 | 0.588 | 0.450 | 0.838 (h4) | 0.452 (h5) |
| mid.0 | 8 | 0.718 | 0.692 | 0.691 | 0.528 | 0.849 (h5) | 0.623 (h6) |
| u1.0 | 16 | 0.669 | 0.674 | 0.664 | 0.627 | 0.795 (h0) | 0.527 (h4) |
| u1.1 | 16 | 0.627 | 0.633 | 0.619 | 0.634 | 0.739 (h4) | 0.513 (h5) |
| u1.2 | 16 | 0.612 | 0.592 | 0.589 | 0.536 | 0.698 (h6) | 0.482 (h4) |
| u2.0 | 32 | 0.540 | 0.490 | 0.489 | 0.502 | 0.713 (h1) | 0.444 (h6) |
| u2.1 | 32 | 0.681 | 0.587 | 0.589 | 0.413 | 0.749 (h7) | 0.570 (h5) |
| u2.2 | 32 | 0.662 | 0.599 | 0.601 | 0.471 | 0.745 (h5) | 0.521 (h6) |
| u3.0 | 64 | 0.487 | 0.518 | 0.509 | 0.560 | 0.609 (h4) | 0.369 (h6) |
| u3.1 | 64 | 0.547 | 0.513 | 0.511 | 0.496 | 0.661 (h2) | 0.442 (h6) |
| u3.2 | 64 | 0.543 | 0.522 | 0.523 | 0.489 | 0.589 (h1) | 0.447 (h3) |

By call (prob, mean over modules and heads): c0 0.563, c3 0.571, c6 0.583, c9 0.589, c12 0.594, c15 0.598, c18 0.601, c21 0.604, c24 0.605, c27 0.607, c30 0.616

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.849, K=5: 0.851, K=20: 0.849, K=100: 0.818 — versus the A3 seed 0.901.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.98. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.44.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.273; small 0.088, medium 0.253, large 0.464.

Token AUROC (head-mean probability, window, all modules): SOT 0.326, a 0.520, photograph 0.506, of 0.497, a2 0.526, cls 0.710, EOT 0.641, pad1 0.641, pad2 0.641, pad3 0.641, pad4 0.641

By size third (prob, all modules): small 0.602 (seed 0.859), medium 0.606 (seed 0.930), large 0.586 (seed 0.912)

## 2. Self-attention propagation atlas (real mask, keep / leak)

keep = share of a mask pixel's attention that lands inside the mask; leak = the same for a non-mask pixel. Chance for both is the mask area a. Reported: keep lift (keep − a)/(1 − a) (1 = perfect), leak ratio leak / a (0 = no leak, 1 = chance), for the exact, eroded (interior) and dilated (+2 px at 64) masks.

### renoise_n0

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.223 | 0.806 | 0.209 | 0.809 | 0.328 |
| d0.1 | 64 | 0.350 | 0.705 | 0.312 | 0.694 | 0.144 |
| d1.0 | 32 | 0.323 | 0.744 | 0.282 | 0.690 | 0.389 |
| d1.1 | 32 | 0.351 | 0.739 | 0.303 | 0.705 | 0.243 |
| d2.0 | 16 | 0.253 | 0.844 | 0.215 | 0.783 | 0.389 |
| d2.1 | 16 | 0.234 | 0.857 | 0.189 | 0.796 | 0.509 |
| mid.0 | 8 | 0.175 | 0.955 | 0.138 | 0.897 | 0.454 |
| u1.0 | 16 | 0.291 | 0.818 | 0.249 | 0.736 | 0.482 |
| u1.1 | 16 | 0.411 | 0.656 | 0.347 | 0.548 | 0.199 |
| u1.2 | 16 | 0.265 | 0.843 | 0.209 | 0.757 | 0.465 |
| u2.0 | 32 | 0.504 | 0.413 | 0.473 | 0.347 | 0.184 |
| u2.1 | 32 | 0.522 | 0.421 | 0.455 | 0.385 | 0.246 |
| u2.2 | 32 | 0.376 | 0.687 | 0.313 | 0.648 | 0.433 |
| u3.0 | 64 | 0.605 | 0.327 | 0.559 | 0.338 | 0.086 |
| u3.1 | 64 | 0.596 | 0.338 | 0.513 | 0.370 | 0.254 |
| u3.2 | 64 | 0.433 | 0.536 | 0.366 | 0.546 | 0.356 |

Keep lift by call (exact, mean over modules and heads): c0 0.093, c3 0.122, c6 0.182, c9 0.249, c12 0.316, c15 0.376, c18 0.424, c21 0.460, c24 0.484, c27 0.500, c30 0.508

- small objects: keep lift 0.175, median leak ratio 0.899
- medium objects: keep lift 0.405, median leak ratio 0.654
- large objects: keep lift 0.528, median leak ratio 0.461

### ddim

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.236 | 0.796 | 0.216 | 0.795 | 0.342 |
| d0.1 | 64 | 0.366 | 0.677 | 0.325 | 0.663 | 0.155 |
| d1.0 | 32 | 0.328 | 0.746 | 0.284 | 0.688 | 0.395 |
| d1.1 | 32 | 0.360 | 0.722 | 0.306 | 0.689 | 0.250 |
| d2.0 | 16 | 0.260 | 0.835 | 0.218 | 0.771 | 0.400 |
| d2.1 | 16 | 0.247 | 0.843 | 0.196 | 0.774 | 0.519 |
| mid.0 | 8 | 0.183 | 0.956 | 0.145 | 0.891 | 0.471 |
| u1.0 | 16 | 0.312 | 0.807 | 0.264 | 0.710 | 0.513 |
| u1.1 | 16 | 0.436 | 0.636 | 0.364 | 0.522 | 0.210 |
| u1.2 | 16 | 0.286 | 0.838 | 0.223 | 0.727 | 0.495 |
| u2.0 | 32 | 0.530 | 0.385 | 0.490 | 0.326 | 0.189 |
| u2.1 | 32 | 0.553 | 0.389 | 0.473 | 0.365 | 0.254 |
| u2.2 | 32 | 0.398 | 0.659 | 0.324 | 0.620 | 0.452 |
| u3.0 | 64 | 0.635 | 0.293 | 0.575 | 0.321 | 0.088 |
| u3.1 | 64 | 0.630 | 0.298 | 0.525 | 0.355 | 0.268 |
| u3.2 | 64 | 0.454 | 0.510 | 0.373 | 0.534 | 0.380 |

Keep lift by call (exact, mean over modules and heads): c0 0.129, c3 0.165, c6 0.223, c9 0.284, c12 0.343, c15 0.394, c18 0.434, c21 0.464, c24 0.485, c27 0.498, c30 0.507

- small objects: keep lift 0.193, median leak ratio 0.882
- medium objects: keep lift 0.428, median leak ratio 0.630
- large objects: keep lift 0.543, median leak ratio 0.447

## 3. How many propagation steps? (y_t = A^t seed, head-mean A, averaged over the 11 calls)

### renoise_n0

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.873 | 0.883 | 0.880 | 0.876 | 0.872 | 0.867 | 0.862 | 0.857 | 0.852 |
| 16 | AP | 0.615 | 0.660 | 0.659 | 0.651 | 0.644 | 0.637 | 0.630 | 0.623 | 0.615 |
| 32 | AUROC | 0.875 | 0.889 | 0.888 | 0.884 | 0.877 | 0.868 | 0.860 | 0.852 | 0.845 |
| 32 | AP | 0.620 | 0.675 | 0.673 | 0.667 | 0.658 | 0.646 | 0.634 | 0.625 | 0.614 |
| 64 | AUROC | 0.875 | 0.891 | 0.889 | 0.883 | 0.876 | 0.868 | 0.860 | 0.852 | 0.845 |
| 64 | AP | 0.620 | 0.674 | 0.666 | 0.652 | 0.638 | 0.621 | 0.608 | 0.596 | 0.583 |

C16 path A64·up(A16^t c16): AUROC t=1 0.889, t=2 0.884, t=3 0.879, t=4 0.875; AP t=1 0.682, t=2 0.680, t=3 0.676, t=4 0.669

- small: AP side 16 0.311 0.328 0.316 0.286 0.263 | side 64 0.321 0.335 0.301 0.244 0.172 (t = 0,1,2,4,8)
- medium: AP side 16 0.689 0.751 0.748 0.728 0.674 | side 64 0.692 0.780 0.779 0.744 0.651 (t = 0,1,2,4,8)
- large: AP side 16 0.840 0.895 0.906 0.910 0.902 | side 64 0.839 0.900 0.911 0.919 0.918 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.807 h1 0.889 h2 0.726 h3 0.865 h4 0.880 h5 0.866 h6 0.795 h7 0.884
- per-head AUROC at t=2, side 64: h0 0.889 h1 0.887 h2 0.881 h3 0.875 h4 0.868 h5 0.887 h6 0.888 h7 0.884

### ddim

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.899 | 0.906 | 0.901 | 0.896 | 0.891 | 0.887 | 0.882 | 0.878 | 0.873 |
| 16 | AP | 0.662 | 0.702 | 0.691 | 0.681 | 0.673 | 0.666 | 0.658 | 0.650 | 0.644 |
| 32 | AUROC | 0.901 | 0.912 | 0.907 | 0.897 | 0.889 | 0.880 | 0.872 | 0.865 | 0.858 |
| 32 | AP | 0.666 | 0.724 | 0.709 | 0.690 | 0.676 | 0.664 | 0.651 | 0.640 | 0.626 |
| 64 | AUROC | 0.901 | 0.914 | 0.907 | 0.897 | 0.889 | 0.880 | 0.872 | 0.863 | 0.856 |
| 64 | AP | 0.667 | 0.724 | 0.702 | 0.683 | 0.664 | 0.647 | 0.632 | 0.615 | 0.601 |

C16 path A64·up(A16^t c16): AUROC t=1 0.909, t=2 0.904, t=3 0.898, t=4 0.893; AP t=1 0.721, t=2 0.713, t=3 0.703, t=4 0.699

- small: AP side 16 0.366 0.387 0.356 0.315 0.273 | side 64 0.376 0.414 0.348 0.273 0.183 (t = 0,1,2,4,8)
- medium: AP side 16 0.749 0.805 0.792 0.775 0.730 | side 64 0.754 0.833 0.825 0.773 0.683 (t = 0,1,2,4,8)
- large: AP side 16 0.863 0.907 0.917 0.920 0.921 | side 64 0.863 0.917 0.926 0.936 0.928 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.833 h1 0.910 h2 0.734 h3 0.887 h4 0.902 h5 0.892 h6 0.811 h7 0.901
- per-head AUROC at t=2, side 64: h0 0.911 h1 0.904 h2 0.897 h3 0.888 h4 0.880 h5 0.908 h6 0.912 h7 0.899

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
| n0_V0R2 | 0.516 | 0.754 | 46% | 3% | 51% | 38% | 62% |
| ddim_C16 | 0.560 | 0.794 | 50% | 4% | 46% | 39% | 61% |
| ddim_V0 | 0.526 | 0.795 | 44% | 4% | 53% | 38% | 62% |

Instance coverage (share of each instance's pixels predicted), by instance size as a share of the image:

| variant | <1% | 1–5% | 5–20% | >20% | instances missed (<10% covered) |
|---|---:|---:|---:|---:|---:|
| n0_V0R2 | 0.49 (n=149) | 0.68 (n=145) | 0.76 (n=148) | 0.77 (n=78) | 20% of 520 |
| ddim_C16 | 0.52 (n=149) | 0.75 (n=145) | 0.82 (n=148) | 0.81 (n=78) | 17% of 520 |
| ddim_V0 | 0.58 (n=149) | 0.79 (n=145) | 0.82 (n=148) | 0.80 (n=78) | 15% of 520 |

Most FP pixels on another object (n0_V0R2), target → other class: 
chair→diningtable 10%, person→car 6%, bottle→person 6%, person→bus 6%, person→train 5%, person→diningtable 5%, bicycle→person 4%, pottedplant→diningtable 4%, car→person 3%, chair→sofa 3%, chair→person 3%, diningtable→person 2%

