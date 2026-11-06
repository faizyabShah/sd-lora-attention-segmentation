# SG-real-atlas alignment diagnostics on VOC-Sim (generated, regenerated live) (490 pairs, 490 images)

Exploratory. The real mask is only used to measure the stored attention. AUROC: 0.5 = chance, 1 = the map ranks every object pixel above every non-object pixel; it ignores the threshold. The SA window is calls 10–20 (t ≈ 661→331). Object-size thirds cut at 0.238 / 0.420 of the image.

## 1. Cross-attention atlas (class token)

### generate

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.954

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.574 | 0.592 | 0.592 | 0.459 | 0.754 (h6) | 0.441 (h3) |
| d0.1 | 64 | 0.518 | 0.521 | 0.523 | 0.491 | 0.678 (h6) | 0.416 (h5) |
| d1.0 | 32 | 0.627 | 0.572 | 0.576 | 0.415 | 0.893 (h2) | 0.385 (h7) |
| d1.1 | 32 | 0.590 | 0.530 | 0.536 | 0.365 | 0.808 (h0) | 0.272 (h5) |
| d2.0 | 16 | 0.701 | 0.647 | 0.642 | 0.278 | 0.856 (h2) | 0.252 (h7) |
| d2.1 | 16 | 0.632 | 0.622 | 0.623 | 0.456 | 0.855 (h4) | 0.406 (h1) |
| mid.0 | 8 | 0.718 | 0.727 | 0.722 | 0.513 | 0.868 (h5) | 0.625 (h7) |
| u1.0 | 16 | 0.652 | 0.683 | 0.663 | 0.721 | 0.816 (h0) | 0.493 (h4) |
| u1.1 | 16 | 0.620 | 0.648 | 0.624 | 0.716 | 0.758 (h4) | 0.446 (h5) |
| u1.2 | 16 | 0.631 | 0.639 | 0.630 | 0.633 | 0.723 (h6) | 0.544 (h1) |
| u2.0 | 32 | 0.584 | 0.487 | 0.486 | 0.527 | 0.770 (h1) | 0.473 (h4) |
| u2.1 | 32 | 0.741 | 0.645 | 0.649 | 0.415 | 0.800 (h7) | 0.607 (h5) |
| u2.2 | 32 | 0.688 | 0.642 | 0.639 | 0.570 | 0.770 (h5) | 0.552 (h6) |
| u3.0 | 64 | 0.533 | 0.536 | 0.530 | 0.553 | 0.617 (h4) | 0.444 (h5) |
| u3.1 | 64 | 0.590 | 0.487 | 0.494 | 0.429 | 0.758 (h2) | 0.500 (h6) |
| u3.2 | 64 | 0.534 | 0.524 | 0.525 | 0.518 | 0.583 (h7) | 0.469 (h3) |

By call (prob, mean over modules and heads): c0 0.575, c3 0.593, c6 0.609, c9 0.614, c12 0.617, c15 0.621, c18 0.625, c21 0.628, c24 0.631, c27 0.634, c30 0.646

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.904, K=5: 0.903, K=20: 0.898, K=100: 0.872 — versus the A3 seed 0.954.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.99. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.34.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.506; small 0.356, medium 0.473, large 0.691.

Token AUROC (head-mean probability, window, all modules): SOT 0.280, a 0.528, photograph 0.512, of 0.501, a2 0.522, cls 0.760, EOT 0.659, pad1 0.659, pad2 0.660, pad3 0.661, pad4 0.662

By size third (prob, all modules): small 0.642 (seed 0.970), medium 0.626 (seed 0.962), large 0.594 (seed 0.929)

### renoise_n0

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.952

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.574 | 0.590 | 0.590 | 0.458 | 0.748 (h6) | 0.439 (h3) |
| d0.1 | 64 | 0.515 | 0.520 | 0.521 | 0.493 | 0.676 (h6) | 0.413 (h5) |
| d1.0 | 32 | 0.625 | 0.569 | 0.574 | 0.415 | 0.889 (h2) | 0.396 (h7) |
| d1.1 | 32 | 0.593 | 0.530 | 0.537 | 0.365 | 0.811 (h0) | 0.284 (h5) |
| d2.0 | 16 | 0.697 | 0.642 | 0.637 | 0.286 | 0.855 (h2) | 0.253 (h7) |
| d2.1 | 16 | 0.628 | 0.618 | 0.619 | 0.456 | 0.854 (h4) | 0.401 (h1) |
| mid.0 | 8 | 0.717 | 0.725 | 0.720 | 0.512 | 0.866 (h5) | 0.621 (h7) |
| u1.0 | 16 | 0.650 | 0.682 | 0.662 | 0.722 | 0.812 (h0) | 0.492 (h4) |
| u1.1 | 16 | 0.617 | 0.645 | 0.620 | 0.720 | 0.758 (h4) | 0.446 (h5) |
| u1.2 | 16 | 0.632 | 0.639 | 0.630 | 0.632 | 0.724 (h6) | 0.539 (h1) |
| u2.0 | 32 | 0.581 | 0.486 | 0.484 | 0.526 | 0.767 (h1) | 0.462 (h4) |
| u2.1 | 32 | 0.738 | 0.642 | 0.645 | 0.417 | 0.800 (h7) | 0.606 (h5) |
| u2.2 | 32 | 0.686 | 0.642 | 0.638 | 0.575 | 0.767 (h5) | 0.547 (h6) |
| u3.0 | 64 | 0.529 | 0.535 | 0.528 | 0.557 | 0.623 (h4) | 0.438 (h5) |
| u3.1 | 64 | 0.590 | 0.489 | 0.496 | 0.431 | 0.751 (h2) | 0.493 (h6) |
| u3.2 | 64 | 0.536 | 0.528 | 0.529 | 0.517 | 0.591 (h7) | 0.464 (h0) |

By call (prob, mean over modules and heads): c0 0.578, c3 0.590, c6 0.604, c9 0.610, c12 0.615, c15 0.619, c18 0.624, c21 0.627, c24 0.631, c27 0.634, c30 0.645

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.904, K=5: 0.904, K=20: 0.897, K=100: 0.871 — versus the A3 seed 0.952.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.99. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.34.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.504; small 0.351, medium 0.470, large 0.691.

Token AUROC (head-mean probability, window, all modules): SOT 0.283, a 0.525, photograph 0.511, of 0.498, a2 0.519, cls 0.759, EOT 0.656, pad1 0.655, pad2 0.656, pad3 0.656, pad4 0.657

By size third (prob, all modules): small 0.641 (seed 0.967), medium 0.624 (seed 0.959), large 0.593 (seed 0.928)

### ddim

A3 seed AUROC (the pipeline's CA seed, 64 px): 0.957

Mean AUROC over pairs, window calls, heads — by signal and module:

| module | side | prob | logit | cos | |q| | best head (prob) | worst head (prob) |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.568 | 0.589 | 0.588 | 0.447 | 0.770 (h6) | 0.385 (h1) |
| d0.1 | 64 | 0.511 | 0.514 | 0.516 | 0.485 | 0.672 (h6) | 0.396 (h5) |
| d1.0 | 32 | 0.637 | 0.575 | 0.580 | 0.411 | 0.898 (h2) | 0.391 (h7) |
| d1.1 | 32 | 0.588 | 0.527 | 0.533 | 0.367 | 0.821 (h0) | 0.279 (h5) |
| d2.0 | 16 | 0.704 | 0.649 | 0.644 | 0.285 | 0.860 (h2) | 0.245 (h7) |
| d2.1 | 16 | 0.629 | 0.620 | 0.622 | 0.462 | 0.854 (h4) | 0.406 (h1) |
| mid.0 | 8 | 0.717 | 0.726 | 0.721 | 0.517 | 0.867 (h5) | 0.628 (h7) |
| u1.0 | 16 | 0.654 | 0.682 | 0.662 | 0.722 | 0.818 (h0) | 0.493 (h4) |
| u1.1 | 16 | 0.625 | 0.650 | 0.625 | 0.711 | 0.764 (h4) | 0.448 (h5) |
| u1.2 | 16 | 0.631 | 0.638 | 0.628 | 0.630 | 0.723 (h6) | 0.546 (h1) |
| u2.0 | 32 | 0.581 | 0.484 | 0.482 | 0.532 | 0.770 (h1) | 0.467 (h4) |
| u2.1 | 32 | 0.744 | 0.647 | 0.650 | 0.416 | 0.804 (h0) | 0.606 (h5) |
| u2.2 | 32 | 0.685 | 0.637 | 0.633 | 0.579 | 0.758 (h5) | 0.555 (h6) |
| u3.0 | 64 | 0.527 | 0.525 | 0.519 | 0.554 | 0.577 (h4) | 0.460 (h6) |
| u3.1 | 64 | 0.594 | 0.481 | 0.489 | 0.418 | 0.763 (h2) | 0.508 (h0) |
| u3.2 | 64 | 0.525 | 0.527 | 0.526 | 0.533 | 0.574 (h6) | 0.458 (h0) |

By call (prob, mean over modules and heads): c0 0.603, c3 0.610, c6 0.613, c9 0.614, c12 0.617, c15 0.620, c18 0.623, c21 0.626, c24 0.628, c27 0.631, c30 0.645

Held-out (grouped folds) mean AUROC of the top-K single (module, call, head) units picked on the other folds: K=1: 0.906, K=5: 0.906, K=20: 0.903, K=100: 0.875 — versus the A3 seed 0.957.

Head ranking (module × head, mean AUROC) split-half consistency across photos: ρ = 0.99. Do the heads that dominate the output (large ‖v W_o‖) align better? ρ(‖v W_o‖, AUROC) = +0.35.

R-precision of prob (share of the top-|M| pixels inside the mask): mean 0.506; small 0.355, medium 0.472, large 0.692.

Token AUROC (head-mean probability, window, all modules): SOT 0.281, a 0.531, photograph 0.508, of 0.499, a2 0.524, cls 0.754, EOT 0.663, pad1 0.663, pad2 0.664, pad3 0.665, pad4 0.666

By size third (prob, all modules): small 0.640 (seed 0.974), medium 0.624 (seed 0.966), large 0.596 (seed 0.930)

## 2. Self-attention propagation atlas (real mask, keep / leak)

keep = share of a mask pixel's attention that lands inside the mask; leak = the same for a non-mask pixel. Chance for both is the mask area a. Reported: keep lift (keep − a)/(1 − a) (1 = perfect), leak ratio leak / a (0 = no leak, 1 = chance), for the exact, eroded (interior) and dilated (+2 px at 64) masks.

### generate

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.413 | 0.669 | 0.325 | 0.689 | 0.308 |
| d0.1 | 64 | 0.609 | 0.431 | 0.481 | 0.486 | 0.149 |
| d1.0 | 32 | 0.569 | 0.507 | 0.448 | 0.523 | 0.426 |
| d1.1 | 32 | 0.621 | 0.474 | 0.487 | 0.533 | 0.244 |
| d2.0 | 16 | 0.474 | 0.635 | 0.367 | 0.633 | 0.397 |
| d2.1 | 16 | 0.426 | 0.671 | 0.317 | 0.653 | 0.779 |
| mid.0 | 8 | 0.318 | 0.848 | 0.236 | 0.806 | 0.730 |
| u1.0 | 16 | 0.494 | 0.693 | 0.400 | 0.625 | 0.675 |
| u1.1 | 16 | 0.638 | 0.509 | 0.511 | 0.422 | 0.224 |
| u1.2 | 16 | 0.459 | 0.725 | 0.338 | 0.611 | 0.590 |
| u2.0 | 32 | 0.744 | 0.217 | 0.671 | 0.209 | 0.185 |
| u2.1 | 32 | 0.754 | 0.219 | 0.640 | 0.256 | 0.259 |
| u2.2 | 32 | 0.617 | 0.462 | 0.475 | 0.531 | 0.423 |
| u3.0 | 64 | 0.852 | 0.133 | 0.765 | 0.234 | 0.060 |
| u3.1 | 64 | 0.844 | 0.127 | 0.712 | 0.266 | 0.187 |
| u3.2 | 64 | 0.702 | 0.265 | 0.552 | 0.409 | 0.354 |

Keep lift by call (exact, mean over modules and heads): c0 0.116, c3 0.235, c6 0.410, c9 0.509, c12 0.566, c15 0.603, c18 0.627, c21 0.644, c24 0.656, c27 0.665, c30 0.671

- small objects: keep lift 0.590, median leak ratio 0.507
- medium objects: keep lift 0.630, median leak ratio 0.421
- large objects: keep lift 0.568, median leak ratio 0.427

### renoise_n0

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.400 | 0.684 | 0.319 | 0.699 | 0.301 |
| d0.1 | 64 | 0.591 | 0.450 | 0.471 | 0.498 | 0.145 |
| d1.0 | 32 | 0.555 | 0.519 | 0.442 | 0.531 | 0.419 |
| d1.1 | 32 | 0.605 | 0.490 | 0.479 | 0.541 | 0.235 |
| d2.0 | 16 | 0.463 | 0.643 | 0.360 | 0.642 | 0.387 |
| d2.1 | 16 | 0.417 | 0.677 | 0.312 | 0.661 | 0.772 |
| mid.0 | 8 | 0.317 | 0.855 | 0.237 | 0.809 | 0.724 |
| u1.0 | 16 | 0.491 | 0.699 | 0.399 | 0.638 | 0.674 |
| u1.1 | 16 | 0.634 | 0.514 | 0.508 | 0.431 | 0.226 |
| u1.2 | 16 | 0.453 | 0.728 | 0.334 | 0.625 | 0.588 |
| u2.0 | 32 | 0.736 | 0.223 | 0.665 | 0.216 | 0.184 |
| u2.1 | 32 | 0.746 | 0.228 | 0.636 | 0.265 | 0.258 |
| u2.2 | 32 | 0.609 | 0.475 | 0.471 | 0.541 | 0.421 |
| u3.0 | 64 | 0.843 | 0.142 | 0.760 | 0.240 | 0.061 |
| u3.1 | 64 | 0.833 | 0.140 | 0.706 | 0.274 | 0.188 |
| u3.2 | 64 | 0.692 | 0.278 | 0.548 | 0.412 | 0.350 |

Keep lift by call (exact, mean over modules and heads): c0 0.171, c3 0.257, c6 0.394, c9 0.489, c12 0.553, c15 0.595, c18 0.622, c21 0.641, c24 0.654, c27 0.663, c30 0.668

- small objects: keep lift 0.579, median leak ratio 0.517
- medium objects: keep lift 0.619, median leak ratio 0.432
- large objects: keep lift 0.561, median leak ratio 0.439

### ddim

| module | side | keep lift exact | leak ratio exact | keep lift eroded | leak ratio dilated | head spread (keep lift) |
|---|---:|---:|---:|---:|---:|---:|
| d0.0 | 64 | 0.436 | 0.646 | 0.341 | 0.661 | 0.299 |
| d0.1 | 64 | 0.622 | 0.415 | 0.493 | 0.468 | 0.156 |
| d1.0 | 32 | 0.573 | 0.511 | 0.449 | 0.525 | 0.410 |
| d1.1 | 32 | 0.630 | 0.460 | 0.491 | 0.514 | 0.237 |
| d2.0 | 16 | 0.481 | 0.630 | 0.373 | 0.626 | 0.399 |
| d2.1 | 16 | 0.441 | 0.661 | 0.327 | 0.645 | 0.770 |
| mid.0 | 8 | 0.323 | 0.846 | 0.239 | 0.802 | 0.715 |
| u1.0 | 16 | 0.500 | 0.684 | 0.404 | 0.614 | 0.671 |
| u1.1 | 16 | 0.643 | 0.498 | 0.514 | 0.409 | 0.218 |
| u1.2 | 16 | 0.465 | 0.711 | 0.343 | 0.594 | 0.590 |
| u2.0 | 32 | 0.749 | 0.213 | 0.674 | 0.205 | 0.183 |
| u2.1 | 32 | 0.759 | 0.214 | 0.644 | 0.254 | 0.256 |
| u2.2 | 32 | 0.621 | 0.454 | 0.478 | 0.523 | 0.421 |
| u3.0 | 64 | 0.856 | 0.129 | 0.766 | 0.228 | 0.061 |
| u3.1 | 64 | 0.848 | 0.123 | 0.713 | 0.261 | 0.193 |
| u3.2 | 64 | 0.705 | 0.267 | 0.551 | 0.412 | 0.361 |

Keep lift by call (exact, mean over modules and heads): c0 0.317, c3 0.387, c6 0.470, c9 0.533, c12 0.578, c15 0.609, c18 0.630, c21 0.644, c24 0.655, c27 0.663, c30 0.667

- small objects: keep lift 0.598, median leak ratio 0.495
- medium objects: keep lift 0.636, median leak ratio 0.415
- large objects: keep lift 0.576, median leak ratio 0.423

## 3. How many propagation steps? (y_t = A^t seed, head-mean A, averaged over the 11 calls)

### generate

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.952 | 0.968 | 0.967 | 0.965 | 0.963 | 0.960 | 0.958 | 0.956 | 0.954 |
| 16 | AP | 0.910 | 0.938 | 0.935 | 0.927 | 0.922 | 0.918 | 0.914 | 0.912 | 0.909 |
| 32 | AUROC | 0.953 | 0.978 | 0.980 | 0.980 | 0.979 | 0.977 | 0.976 | 0.974 | 0.972 |
| 32 | AP | 0.913 | 0.962 | 0.967 | 0.965 | 0.961 | 0.958 | 0.954 | 0.951 | 0.948 |
| 64 | AUROC | 0.954 | 0.979 | 0.981 | 0.980 | 0.979 | 0.977 | 0.975 | 0.973 | 0.970 |
| 64 | AP | 0.913 | 0.967 | 0.970 | 0.969 | 0.964 | 0.960 | 0.955 | 0.951 | 0.947 |

C16 path A64·up(A16^t c16): AUROC t=1 0.981, t=2 0.980, t=3 0.978, t=4 0.976; AP t=1 0.971, t=2 0.968, t=3 0.962, t=4 0.959

- small: AP side 16 0.861 0.904 0.893 0.858 0.837 | side 64 0.865 0.950 0.953 0.936 0.896 (t = 0,1,2,4,8)
- medium: AP side 16 0.915 0.942 0.944 0.942 0.930 | side 64 0.918 0.972 0.977 0.975 0.966 (t = 0,1,2,4,8)
- large: AP side 16 0.955 0.968 0.967 0.965 0.960 | side 64 0.956 0.979 0.981 0.982 0.978 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.907 h1 0.961 h2 0.834 h3 0.944 h4 0.967 h5 0.963 h6 0.903 h7 0.966
- per-head AUROC at t=2, side 64: h0 0.976 h1 0.980 h2 0.977 h3 0.976 h4 0.973 h5 0.977 h6 0.977 h7 0.979

### renoise_n0

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.950 | 0.967 | 0.967 | 0.965 | 0.963 | 0.961 | 0.958 | 0.956 | 0.954 |
| 16 | AP | 0.906 | 0.936 | 0.932 | 0.925 | 0.921 | 0.917 | 0.913 | 0.911 | 0.908 |
| 32 | AUROC | 0.951 | 0.976 | 0.978 | 0.978 | 0.976 | 0.974 | 0.972 | 0.971 | 0.969 |
| 32 | AP | 0.908 | 0.958 | 0.962 | 0.960 | 0.957 | 0.953 | 0.949 | 0.946 | 0.944 |
| 64 | AUROC | 0.952 | 0.978 | 0.979 | 0.978 | 0.977 | 0.975 | 0.973 | 0.971 | 0.969 |
| 64 | AP | 0.908 | 0.963 | 0.966 | 0.963 | 0.959 | 0.955 | 0.951 | 0.948 | 0.944 |

C16 path A64·up(A16^t c16): AUROC t=1 0.980, t=2 0.979, t=3 0.978, t=4 0.976; AP t=1 0.967, t=2 0.963, t=3 0.958, t=4 0.955

- small: AP side 16 0.853 0.901 0.887 0.855 0.834 | side 64 0.856 0.944 0.944 0.923 0.893 (t = 0,1,2,4,8)
- medium: AP side 16 0.910 0.941 0.944 0.943 0.932 | side 64 0.912 0.967 0.974 0.972 0.962 (t = 0,1,2,4,8)
- large: AP side 16 0.954 0.967 0.966 0.964 0.958 | side 64 0.955 0.978 0.981 0.980 0.977 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.909 h1 0.962 h2 0.835 h3 0.944 h4 0.966 h5 0.962 h6 0.901 h7 0.966
- per-head AUROC at t=2, side 64: h0 0.975 h1 0.978 h2 0.976 h3 0.974 h4 0.971 h5 0.976 h6 0.975 h7 0.977

### ddim

| side | metric | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | AUROC | 0.955 | 0.967 | 0.966 | 0.964 | 0.962 | 0.960 | 0.958 | 0.957 | 0.955 |
| 16 | AP | 0.918 | 0.939 | 0.935 | 0.929 | 0.923 | 0.919 | 0.916 | 0.913 | 0.910 |
| 32 | AUROC | 0.957 | 0.977 | 0.978 | 0.978 | 0.976 | 0.975 | 0.973 | 0.971 | 0.968 |
| 32 | AP | 0.920 | 0.964 | 0.967 | 0.963 | 0.960 | 0.956 | 0.952 | 0.948 | 0.945 |
| 64 | AUROC | 0.957 | 0.978 | 0.979 | 0.978 | 0.976 | 0.974 | 0.972 | 0.969 | 0.967 |
| 64 | AP | 0.920 | 0.968 | 0.970 | 0.966 | 0.961 | 0.956 | 0.952 | 0.948 | 0.944 |

C16 path A64·up(A16^t c16): AUROC t=1 0.980, t=2 0.979, t=3 0.977, t=4 0.975; AP t=1 0.970, t=2 0.967, t=3 0.962, t=4 0.958

- small: AP side 16 0.873 0.906 0.893 0.860 0.835 | side 64 0.876 0.952 0.952 0.931 0.894 (t = 0,1,2,4,8)
- medium: AP side 16 0.922 0.944 0.946 0.945 0.934 | side 64 0.925 0.974 0.977 0.974 0.963 (t = 0,1,2,4,8)
- large: AP side 16 0.957 0.966 0.965 0.963 0.960 | side 64 0.958 0.977 0.979 0.978 0.974 (t = 0,1,2,4,8)
- per-head AUROC at t=2, side 16: h0 0.909 h1 0.961 h2 0.839 h3 0.944 h4 0.966 h5 0.962 h6 0.906 h7 0.965
- per-head AUROC at t=2, side 64: h0 0.975 h1 0.977 h2 0.976 h3 0.974 h4 0.970 h5 0.976 h6 0.976 h7 0.977

## 4. Spectral structure of the SA graphs (VOC-Sim, renoise/n0)

490 pairs. E_k = share of the mask's energy in the k smoothest non-trivial eigenvectors (rolled mask = same shape, wrong place = chance).

| side | graph | E_k mask (k=1,2,4,8,16,32) | E_k rolled | E_k seed | best eigvec AUROC (oracle, j≤16) | seed-picked eigvec AUROC | seed low-pass AUROC (k=4,8,16,32,64) | seed AUROC |
|---|---|---|---|---|---:|---:|---|---:|
| 16 | mean | 0.53 0.65 0.71 0.77 0.82 0.86 | 0.17 0.21 0.26 0.34 0.43 0.52 | 0.61 0.75 0.82 0.87 0.91 0.94 | 0.965 | 0.947 | 0.96 0.96 0.96 0.96 0.96 | 0.955 |
| 16 | head | 0.38 0.52 0.61 0.67 0.73 0.77 | 0.13 0.18 0.23 0.30 0.37 0.45 | 0.44 0.62 0.70 0.77 0.82 0.85 | 0.924 | 0.899 | 0.92 0.93 0.94 0.94 0.94 | 0.955 |
| 32 | mean | 0.47 0.67 0.75 0.80 0.84 0.87 | 0.13 0.18 0.23 0.29 0.37 0.45 | 0.47 0.68 0.78 0.84 0.88 0.92 | 0.961 | 0.939 | 0.96 0.96 0.96 0.96 0.96 | 0.953 |
| 32 | head | 0.41 0.60 0.70 0.76 0.80 0.84 | 0.11 0.17 0.22 0.27 0.34 0.41 | 0.42 0.62 0.73 0.79 0.84 0.88 | 0.948 | 0.923 | 0.95 0.96 0.96 0.96 0.95 | 0.953 |
| 64 | mean | 0.42 0.61 0.70 0.75 0.79 0.82 | 0.09 0.14 0.19 0.23 0.29 0.35 | 0.38 0.56 0.67 0.73 0.78 0.83 | 0.937 | 0.910 | 0.94 0.95 0.95 0.95 0.95 | 0.952 |

Seed low-pass AUROC on the head-mean graph by object size (k = 4 / 8 / 16, side 32 and 64) vs seed:
- small (n=163): side 32: 0.972 0.975 0.973 (seed 0.968); side 64: 0.949 0.963 0.964 (seed 0.967)
- medium (n=164): side 32: 0.973 0.972 0.967 (seed 0.960); side 64: 0.950 0.959 0.959 (seed 0.959)
- large (n=163): side 32: 0.944 0.944 0.940 (seed 0.930); side 64: 0.924 0.931 0.931 (seed 0.928)

Pairs where a 16 px graph had no valid in-mask cell (AUROC undefined, dropped from those means): 0

## 5. Where the errors are

Pixel shares at the OOF thresholds of analysis 1. FP split by what the wrong pixel actually is; FN by whether it lies within 8 px (≈ one latent cell) of the object boundary.

| variant | precision | recall | FP: other object | FP: background ≤8 px | FP: background far | FN: boundary band | FN: interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| gen_V3R2 | 0.890 | 0.910 | 0% | 42% | 58% | 21% | 79% |
| gen_V0R2 | 0.961 | 0.688 | 0% | 46% | 54% | 20% | 80% |
| gen_C16 | 0.938 | 0.749 | 0% | 61% | 39% | 21% | 79% |
| n0_V0R2 | 0.956 | 0.688 | 0% | 44% | 56% | 20% | 80% |
| ddim_C16 | 0.938 | 0.753 | 0% | 60% | 40% | 20% | 80% |
| ddim_V0 | 0.949 | 0.731 | 0% | 54% | 46% | 21% | 79% |

Instance coverage (share of each instance's pixels predicted), by instance size as a share of the image:

| variant | <1% | 1–5% | 5–20% | >20% | instances missed (<10% covered) |
|---|---:|---:|---:|---:|---:|
| gen_V3R2 | nan (n=0) | 0.81 (n=4) | 0.95 (n=131) | 0.93 (n=355) | 0% of 490 |
| gen_V0R2 | nan (n=0) | 0.73 (n=4) | 0.83 (n=131) | 0.72 (n=355) | 0% of 490 |
| gen_C16 | nan (n=0) | 0.94 (n=4) | 0.91 (n=131) | 0.78 (n=355) | 0% of 490 |
| n0_V0R2 | nan (n=0) | 0.72 (n=4) | 0.83 (n=131) | 0.72 (n=355) | 1% of 490 |
| ddim_C16 | nan (n=0) | 0.95 (n=4) | 0.92 (n=131) | 0.78 (n=355) | 0% of 490 |
| ddim_V0 | nan (n=0) | 0.90 (n=4) | 0.88 (n=131) | 0.76 (n=355) | 0% of 490 |

VOC-Sim masks have one object class, so 'other object' is always 0.

## 6. Map scores on VOC-Sim (class-macro mIoU over 20 classes)

| condition | map | best τ | mIoU at best τ | OOF mIoU | mIoU at τ 0.55 | mIoU at τ 0.65 |
|---|---|---:|---:|---:|---:|---:|
| generate | A3 | 0.400 | 0.705 | 0.705 | 0.611 | 0.466 |
| generate | C16 | 0.525 | 0.803 | 0.802 | 0.803 | 0.763 |
| generate | V0 | 0.450 | 0.811 | 0.810 | 0.791 | 0.705 |
| generate | V1 | 0.525 | 0.808 | 0.808 | 0.807 | 0.771 |
| generate | V3 | 0.550 | 0.817 | 0.816 | 0.817 | 0.801 |
| generate | V0+R2 | 0.500 | 0.816 | 0.816 | 0.796 | 0.701 |
| generate | V3+R2 | 0.550 | 0.830 | 0.830 | 0.830 | 0.807 |
| renoise_n0 | A3 | 0.400 | 0.695 | 0.695 | 0.602 | 0.467 |
| renoise_n0 | C16 | 0.500 | 0.797 | 0.796 | 0.794 | 0.758 |
| renoise_n0 | V0 | 0.450 | 0.806 | 0.806 | 0.781 | 0.696 |
| renoise_n0 | V1 | 0.525 | 0.801 | 0.801 | 0.800 | 0.767 |
| renoise_n0 | V3 | 0.550 | 0.813 | 0.812 | 0.813 | 0.794 |
| renoise_n0 | V0+R2 | 0.500 | 0.813 | 0.813 | 0.788 | 0.696 |
| renoise_n0 | V3+R2 | 0.550 | 0.828 | 0.828 | 0.828 | 0.802 |
| ddim | A3 | 0.375 | 0.719 | 0.719 | 0.589 | 0.430 |
| ddim | C16 | 0.525 | 0.808 | 0.807 | 0.807 | 0.769 |
| ddim | V0 | 0.450 | 0.815 | 0.812 | 0.794 | 0.705 |
| ddim | V1 | 0.525 | 0.810 | 0.808 | 0.810 | 0.776 |
| ddim | V3 | 0.575 | 0.821 | 0.821 | 0.820 | 0.804 |
| ddim | V0+R2 | 0.500 | 0.818 | 0.818 | 0.797 | 0.699 |
| ddim | V3+R2 | 0.550 | 0.833 | 0.833 | 0.833 | 0.808 |

