# Per-location CA layer reliability gate and signed temporal CA contrast (`PREREG.md`)

`scripts/jc_layergate.py`. One live capture per image (runs/20 corrected prompts, safety checker off). Class-macro mIoU. *VOC-sel.* = 5-fold OOF on VOC-Sim; full-VOC-Sim best threshold applied unchanged on COCO-Cap. Δ with image-bootstrap 95% interval (50). Exploratory.

## Data checks

- VOC-Sim: 1 images. Per-layer reconstruction vs runtime seed: max |Δ| 0.0e+00. Image vs runs/20 capture: identical on 1/1, median max |Δ| 0. BRACE hierarchy map vs runs/20: median max |Δ| 1.2e-03.
- COCO-Cap: 2 images. Per-layer reconstruction vs runtime seed: max |Δ| 0.0e+00. Image vs runs/20 capture: identical on 0/2, median max |Δ| 26. BRACE hierarchy map vs runs/20: median max |Δ| 3.9e-03.

## Diagnostics (pixel level; ΔAUC = augmented − base, VOC-Sim 5-fold OOF, COCO-Cap fit on VOC-Sim)

| set | target | pixels | error rate | AUC base | AUC +reliability (Exp. 1) | ΔAUC Exp. 1 | AUC +early/late (Exp. 2) | ΔAUC Exp. 2 | AUC D=E−L | AUC −D |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| VOC-Sim | FP vs TP | 550 | 0.667 | 0.5000 | 0.5000 | +0.0000 [+0.0000, +0.0000] | 0.5000 | +0.0000 [+0.0000, +0.0000] | 0.2412 | 0.7588 |
| VOC-Sim | FN vs TN | 474 | 0.000 | nan | nan | +nan [+nan, +nan] | nan | +nan [+nan, +nan] | nan | nan |
| COCO-Cap | FP vs TP | 503 | 0.060 | 0.8944 | 0.8970 | +0.0025 [+0.0025, +0.0026] | 0.9273 | +0.0329 [+0.0329, +0.0405] | 0.4634 | 0.5366 |
| COCO-Cap | FN vs TN | 1545 | 0.021 | 0.5000 | 0.5000 | +0.0000 [+0.0000, +0.0000] | 0.5000 | +0.0000 [+0.0000, +0.0000] | 0.4911 | 0.5089 |

**Diagnostic pass (VOC-Sim ΔAUC ≥ 0.01, lower bound > 0, either target):** Exp. 1 **fail**, Exp. 2 **fail**.

Agreeing-wrong check (≥ 6/11 layers fire at ≥ 0.5): VOC-Sim FP 0.000 vs TP 0.060, mean R̄ FP 0.390 vs TP 0.320; COCO-Cap FP 0.000 vs TP 0.008, mean R̄ FP 0.339 vs TP 0.281.

## Components, VOC-Sim (1 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.273 | 0.273 | 0.196 | 0.304 |
| UN: uniform layer weights | 0.275 | 0.246 | 0.188 | 0.265 |
| Gpx: gate, pixels | 0.307 | 0.296 | 0.218 | 0.316 |
| Gpxx: gate, pixels, shuffled | 0.326 | 0.268 | 0.187 | 0.291 |
| Gs16: gate, SA k-means k=16 | 0.324 | 0.329 | 0.254 | 0.362 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.288 | 0.288 | 0.213 | 0.299 |
| Gs64: gate, SA k-means k=64 | 0.387 | 0.339 | 0.234 | 0.381 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.359 | 0.304 | 0.188 | 0.327 |
| Gs256: gate, SA k-means k=256 | 0.362 | 0.317 | 0.224 | 0.353 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.343 | 0.298 | 0.204 | 0.307 |
| Gg16: gate, grid 4x4 | 0.360 | 0.325 | 0.232 | 0.366 |
| Gg16x: gate, grid 4x4, shuffled | 0.319 | 0.322 | 0.217 | 0.355 |
| Gg64: gate, grid 8x8 | 0.324 | 0.296 | 0.223 | 0.326 |
| Gg64x: gate, grid 8x8, shuffled | 0.292 | 0.277 | 0.188 | 0.314 |
| Gg256: gate, grid 16x16 | 0.307 | 0.289 | 0.223 | 0.321 |
| Gg256x: gate, grid 16x16, shuffled | 0.284 | 0.290 | 0.198 | 0.326 |
| TE: early calls 0-9 | 0.246 | 0.263 | 0.197 | 0.266 |
| TM: middle calls 10-20 | 0.356 | 0.327 | 0.229 | 0.356 |
| TL: late calls 21-30 | 0.362 | 0.256 | 0.201 | 0.246 |
| TD+: seed + early - late | 0.268 | 0.270 | 0.210 | 0.269 |
| TD-: seed + late - early | 0.261 | 0.221 | 0.213 | 0.231 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.460 | 0.405 | 0.335 | 0.436 |
| UN: uniform layer weights | 0.370 | 0.396 | 0.297 | 0.436 |
| Gpx: gate, pixels | 0.524 | 0.414 | 0.342 | 0.433 |
| Gpxx: gate, pixels, shuffled | 0.360 | 0.408 | 0.331 | 0.435 |
| Gs16: gate, SA k-means k=16 | 0.510 | 0.426 | 0.363 | 0.485 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.391 | 0.417 | 0.338 | 0.432 |
| Gs64: gate, SA k-means k=64 | 0.556 | 0.457 | 0.363 | 0.495 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.519 | 0.437 | 0.309 | 0.490 |
| Gs256: gate, SA k-means k=256 | 0.560 | 0.425 | 0.356 | 0.475 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.234 | 0.414 | 0.336 | 0.428 |
| Gg16: gate, grid 4x4 | 0.570 | 0.453 | 0.368 | 0.459 |
| Gg16x: gate, grid 4x4, shuffled | 0.506 | 0.436 | 0.352 | 0.483 |
| Gg64: gate, grid 8x8 | 0.606 | 0.426 | 0.352 | 0.454 |
| Gg64x: gate, grid 8x8, shuffled | 0.472 | 0.415 | 0.325 | 0.586 |
| Gg256: gate, grid 16x16 | 0.578 | 0.413 | 0.349 | 0.449 |
| Gg256x: gate, grid 16x16, shuffled | 0.463 | 0.408 | 0.336 | 0.453 |
| TE: early calls 0-9 | 0.409 | 0.409 | 0.337 | 0.434 |
| TM: middle calls 10-20 | 0.481 | 0.415 | 0.348 | 0.464 |
| TL: late calls 21-30 | 0.039 | 0.380 | 0.301 | 0.416 |
| TD+: seed + early - late | 0.508 | 0.424 | 0.349 | 0.431 |
| TD-: seed + late - early | 0.008 | 0.367 | 0.300 | 0.390 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.276 | 0.272 | 0.197 | 0.301 |
| UN: uniform layer weights | 0.276 | 0.246 | 0.190 | 0.265 |
| Gpx: gate, pixels | 0.307 | 0.294 | 0.219 | 0.310 |
| Gpxx: gate, pixels, shuffled | 0.318 | 0.268 | 0.189 | 0.290 |
| Gs16: gate, SA k-means k=16 | 0.321 | 0.321 | 0.254 | 0.353 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.288 | 0.287 | 0.214 | 0.298 |
| Gs64: gate, SA k-means k=64 | 0.383 | 0.329 | 0.234 | 0.372 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.354 | 0.299 | 0.189 | 0.319 |
| Gs256: gate, SA k-means k=256 | 0.359 | 0.312 | 0.226 | 0.345 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.335 | 0.295 | 0.205 | 0.304 |
| Gg16: gate, grid 4x4 | 0.361 | 0.319 | 0.233 | 0.360 |
| Gg16x: gate, grid 4x4, shuffled | 0.320 | 0.317 | 0.218 | 0.347 |
| Gg64: gate, grid 8x8 | 0.328 | 0.293 | 0.223 | 0.323 |
| Gg64x: gate, grid 8x8, shuffled | 0.295 | 0.276 | 0.189 | 0.310 |
| Gg256: gate, grid 16x16 | 0.311 | 0.286 | 0.224 | 0.317 |
| Gg256x: gate, grid 16x16, shuffled | 0.281 | 0.288 | 0.199 | 0.318 |
| TE: early calls 0-9 | 0.247 | 0.263 | 0.198 | 0.267 |
| TM: middle calls 10-20 | 0.351 | 0.319 | 0.229 | 0.347 |
| TL: late calls 21-30 | 0.353 | 0.252 | 0.201 | 0.248 |
| TD+: seed + early - late | 0.272 | 0.270 | 0.211 | 0.268 |
| TD-: seed + late - early | 0.252 | 0.220 | 0.211 | 0.230 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.459 | 0.402 | 0.327 | 0.445 |
| UN: uniform layer weights | 0.388 | 0.392 | 0.298 | 0.443 |
| Gpx: gate, pixels | 0.482 | 0.413 | 0.334 | 0.443 |
| Gpxx: gate, pixels, shuffled | 0.421 | 0.406 | 0.323 | 0.444 |
| Gs16: gate, SA k-means k=16 | 0.500 | 0.427 | 0.354 | 0.490 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.385 | 0.419 | 0.330 | 0.443 |
| Gs64: gate, SA k-means k=64 | 0.535 | 0.463 | 0.354 | 0.503 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.473 | 0.447 | 0.307 | 0.494 |
| Gs256: gate, SA k-means k=256 | 0.536 | 0.433 | 0.347 | 0.484 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.306 | 0.413 | 0.329 | 0.435 |
| Gg16: gate, grid 4x4 | 0.535 | 0.456 | 0.359 | 0.456 |
| Gg16x: gate, grid 4x4, shuffled | 0.499 | 0.439 | 0.343 | 0.490 |
| Gg64: gate, grid 8x8 | 0.536 | 0.432 | 0.343 | 0.452 |
| Gg64x: gate, grid 8x8, shuffled | 0.456 | 0.421 | 0.318 | 0.569 |
| Gg256: gate, grid 16x16 | 0.541 | 0.413 | 0.341 | 0.457 |
| Gg256x: gate, grid 16x16, shuffled | 0.428 | 0.408 | 0.328 | 0.459 |
| TE: early calls 0-9 | 0.401 | 0.406 | 0.329 | 0.438 |
| TM: middle calls 10-20 | 0.456 | 0.418 | 0.340 | 0.470 |
| TL: late calls 21-30 | 0.169 | 0.376 | 0.301 | 0.423 |
| TD+: seed + early - late | 0.499 | 0.428 | 0.341 | 0.439 |
| TD-: seed + late - early | 0.069 | 0.365 | 0.297 | 0.398 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR | 0.629 | 0.493 | 0.389 | 0.526 |
| UN | 0.505 | 0.561 | 0.374 | 0.619 |
| Gpx | 0.531 | 0.504 | 0.396 | 0.504 |
| Gpxx | 0.604 | 0.504 | 0.387 | 0.534 |
| Gs16 | 0.645 | 0.602 | 0.419 | 0.671 |
| Gs16x | 0.398 | 0.417 | 0.393 | 0.452 |
| Gs64 | 0.623 | 0.620 | 0.420 | 0.671 |
| Gs64x | 0.604 | 0.640 | 0.378 | 0.708 |
| Gs256 | 0.627 | 0.553 | 0.415 | 0.660 |
| Gs256x | 0.443 | 0.512 | 0.392 | 0.543 |
| Gg16 | 0.570 | 0.564 | 0.434 | 0.666 |
| Gg16x | 0.546 | 0.555 | 0.408 | 0.644 |
| Gg64 | 0.606 | 0.492 | 0.412 | 0.599 |
| Gg64x | 0.502 | 0.759 | 0.384 | 0.749 |
| Gg256 | 0.578 | 0.492 | 0.407 | 0.570 |
| Gg256x | 0.463 | 0.492 | 0.392 | 0.517 |
| TE | 0.568 | 0.501 | 0.393 | 0.493 |
| TM | 0.529 | 0.557 | 0.405 | 0.684 |
| TL | 0.395 | 0.426 | 0.376 | 0.612 |
| TD+ | 0.581 | 0.505 | 0.410 | 0.524 |
| TD- | 0.261 | 0.398 | 0.375 | 0.464 |

## Components, COCO-Cap (2 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.637 | 0.748 | 0.724 | 0.691 |
| UN: uniform layer weights | 0.787 | 0.771 | 0.720 | 0.667 |
| Gpx: gate, pixels | 0.611 | 0.755 | 0.781 | 0.749 |
| Gpxx: gate, pixels, shuffled | 0.696 | 0.720 | 0.721 | 0.667 |
| Gs16: gate, SA k-means k=16 | 0.556 | 0.833 | 0.754 | 0.764 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.744 | 0.871 | 0.715 | 0.752 |
| Gs64: gate, SA k-means k=64 | 0.546 | 0.737 | 0.773 | 0.726 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.644 | 0.748 | 0.735 | 0.702 |
| Gs256: gate, SA k-means k=256 | 0.610 | 0.745 | 0.774 | 0.740 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.835 | 0.735 | 0.730 | 0.659 |
| Gg16: gate, grid 4x4 | 0.604 | 0.701 | 0.744 | 0.618 |
| Gg16x: gate, grid 4x4, shuffled | 0.468 | 0.695 | 0.727 | 0.624 |
| Gg64: gate, grid 8x8 | 0.583 | 0.772 | 0.776 | 0.721 |
| Gg64x: gate, grid 8x8, shuffled | 0.575 | 0.689 | 0.718 | 0.680 |
| Gg256: gate, grid 16x16 | 0.553 | 0.751 | 0.782 | 0.738 |
| Gg256x: gate, grid 16x16, shuffled | 0.588 | 0.700 | 0.720 | 0.604 |
| TE: early calls 0-9 | 0.543 | 0.577 | 0.667 | 0.491 |
| TM: middle calls 10-20 | 0.849 | 0.814 | 0.753 | 0.819 |
| TL: late calls 21-30 | 0.877 | 0.850 | 0.764 | 0.825 |
| TD+: seed + early - late | 0.443 | 0.523 | 0.643 | 0.399 |
| TD-: seed + late - early | 0.775 | 0.830 | 0.792 | 0.765 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.756 | 0.916 | 0.844 | 0.912 |
| UN: uniform layer weights | 0.700 | 0.913 | 0.905 | 0.905 |
| Gpx: gate, pixels | 0.542 | 0.892 | 0.895 | 0.868 |
| Gpxx: gate, pixels, shuffled | 0.568 | 0.919 | 0.844 | 0.914 |
| Gs16: gate, SA k-means k=16 | 0.683 | 0.909 | 0.905 | 0.870 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.682 | 0.896 | 0.797 | 0.904 |
| Gs64: gate, SA k-means k=64 | 0.430 | 0.900 | 0.897 | 0.853 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.601 | 0.861 | 0.798 | 0.845 |
| Gs256: gate, SA k-means k=256 | 0.447 | 0.888 | 0.875 | 0.832 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.655 | 0.915 | 0.858 | 0.899 |
| Gg16: gate, grid 4x4 | 0.604 | 0.876 | 0.854 | 0.854 |
| Gg16x: gate, grid 4x4, shuffled | 0.653 | 0.872 | 0.800 | 0.866 |
| Gg64: gate, grid 8x8 | 0.583 | 0.907 | 0.901 | 0.911 |
| Gg64x: gate, grid 8x8, shuffled | 0.692 | 0.898 | 0.799 | 0.874 |
| Gg256: gate, grid 16x16 | 0.553 | 0.907 | 0.899 | 0.884 |
| Gg256x: gate, grid 16x16, shuffled | 0.588 | 0.906 | 0.801 | 0.885 |
| TE: early calls 0-9 | 0.685 | 0.796 | 0.774 | 0.774 |
| TM: middle calls 10-20 | 0.771 | 0.918 | 0.870 | 0.932 |
| TL: late calls 21-30 | 0.663 | 0.891 | 0.897 | 0.889 |
| TD+: seed + early - late | 0.592 | 0.745 | 0.751 | 0.680 |
| TD-: seed + late - early | 0.398 | 0.876 | 0.908 | 0.837 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.639 | 0.746 | 0.718 | 0.691 |
| UN: uniform layer weights | 0.733 | 0.768 | 0.729 | 0.685 |
| Gpx: gate, pixels | 0.677 | 0.780 | 0.773 | 0.823 |
| Gpxx: gate, pixels, shuffled | 0.644 | 0.729 | 0.715 | 0.670 |
| Gs16: gate, SA k-means k=16 | 0.508 | 0.831 | 0.764 | 0.775 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.684 | 0.850 | 0.701 | 0.745 |
| Gs64: gate, SA k-means k=64 | 0.565 | 0.757 | 0.770 | 0.744 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.651 | 0.776 | 0.724 | 0.789 |
| Gs256: gate, SA k-means k=256 | 0.605 | 0.763 | 0.765 | 0.752 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.723 | 0.725 | 0.728 | 0.652 |
| Gg16: gate, grid 4x4 | 0.556 | 0.701 | 0.737 | 0.609 |
| Gg16x: gate, grid 4x4, shuffled | 0.577 | 0.696 | 0.715 | 0.628 |
| Gg64: gate, grid 8x8 | 0.659 | 0.812 | 0.782 | 0.737 |
| Gg64x: gate, grid 8x8, shuffled | 0.580 | 0.701 | 0.703 | 0.713 |
| Gg256: gate, grid 16x16 | 0.565 | 0.784 | 0.777 | 0.757 |
| Gg256x: gate, grid 16x16, shuffled | 0.572 | 0.723 | 0.708 | 0.636 |
| TE: early calls 0-9 | 0.561 | 0.619 | 0.663 | 0.540 |
| TM: middle calls 10-20 | 0.817 | 0.799 | 0.753 | 0.807 |
| TL: late calls 21-30 | 0.790 | 0.840 | 0.766 | 0.805 |
| TD+: seed + early - late | 0.507 | 0.594 | 0.638 | 0.446 |
| TD-: seed + late - early | 0.605 | 0.824 | 0.820 | 0.804 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR: BRACE seed (global GOCA-style layer weights) | 0.724 | 0.891 | 0.851 | 0.903 |
| UN: uniform layer weights | 0.684 | 0.898 | 0.874 | 0.905 |
| Gpx: gate, pixels | 0.559 | 0.883 | 0.879 | 0.870 |
| Gpxx: gate, pixels, shuffled | 0.549 | 0.892 | 0.850 | 0.905 |
| Gs16: gate, SA k-means k=16 | 0.643 | 0.866 | 0.887 | 0.852 |
| Gs16x: gate, SA k-means k=16, shuffled | 0.653 | 0.872 | 0.811 | 0.890 |
| Gs64: gate, SA k-means k=64 | 0.454 | 0.887 | 0.879 | 0.865 |
| Gs64x: gate, SA k-means k=64, shuffled | 0.590 | 0.873 | 0.833 | 0.858 |
| Gs256: gate, SA k-means k=256 | 0.467 | 0.876 | 0.875 | 0.846 |
| Gs256x: gate, SA k-means k=256, shuffled | 0.617 | 0.891 | 0.866 | 0.894 |
| Gg16: gate, grid 4x4 | 0.556 | 0.866 | 0.868 | 0.855 |
| Gg16x: gate, grid 4x4, shuffled | 0.660 | 0.868 | 0.832 | 0.876 |
| Gg64: gate, grid 8x8 | 0.583 | 0.901 | 0.886 | 0.896 |
| Gg64x: gate, grid 8x8, shuffled | 0.646 | 0.867 | 0.820 | 0.882 |
| Gg256: gate, grid 16x16 | 0.565 | 0.903 | 0.880 | 0.887 |
| Gg256x: gate, grid 16x16, shuffled | 0.572 | 0.879 | 0.832 | 0.887 |
| TE: early calls 0-9 | 0.645 | 0.793 | 0.808 | 0.773 |
| TM: middle calls 10-20 | 0.741 | 0.908 | 0.875 | 0.927 |
| TL: late calls 21-30 | 0.638 | 0.895 | 0.884 | 0.903 |
| TD+: seed + early - late | 0.575 | 0.741 | 0.780 | 0.679 |
| TD-: seed + late - early | 0.427 | 0.872 | 0.905 | 0.858 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| BR | 0.847 | 0.919 | 0.905 | 0.915 |
| UN | 0.842 | 0.914 | 0.909 | 0.911 |
| Gpx | 0.845 | 0.892 | 0.899 | 0.869 |
| Gpxx | 0.883 | 0.922 | 0.906 | 0.919 |
| Gs16 | 0.789 | 0.914 | 0.907 | 0.874 |
| Gs16x | 0.788 | 0.916 | 0.890 | 0.905 |
| Gs64 | 0.841 | 0.900 | 0.902 | 0.877 |
| Gs64x | 0.847 | 0.898 | 0.889 | 0.894 |
| Gs256 | 0.872 | 0.888 | 0.899 | 0.870 |
| Gs256x | 0.840 | 0.915 | 0.906 | 0.924 |
| Gg16 | 0.782 | 0.899 | 0.902 | 0.887 |
| Gg16x | 0.796 | 0.904 | 0.899 | 0.895 |
| Gg64 | 0.835 | 0.908 | 0.903 | 0.911 |
| Gg64x | 0.803 | 0.909 | 0.902 | 0.910 |
| Gg256 | 0.872 | 0.907 | 0.900 | 0.907 |
| Gg256x | 0.870 | 0.915 | 0.904 | 0.915 |
| TE | 0.705 | 0.909 | 0.906 | 0.854 |
| TM | 0.921 | 0.921 | 0.905 | 0.932 |
| TL | 0.884 | 0.915 | 0.905 | 0.889 |
| TD+ | 0.635 | 0.850 | 0.906 | 0.821 |
| TD- | 0.781 | 0.892 | 0.908 | 0.906 |

## Selection and contrasts (hierarchy + DenseCRF)

- SA regions: VOC-Sim OOF Gs16 0.254, Gs64 0.234, Gs256 0.224 → **Gs16**
- grid: VOC-Sim OOF Gg16 0.232, Gg64 0.223, Gg256 0.223 → **Gg16**

| Contrast | VOC-Sim, VOC-sel. | COCO-Cap, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---|---|---|---|
| BR − UN (GOCA weighting) | +0.008 [+0.008, +0.008] | +0.004 [-0.012, +0.019] | +0.038 [+0.038, +0.038] | -0.061 [-0.120, -0.002] |
| Gpx − BR | +0.022 [+0.022, +0.022] | +0.057 [+0.009, +0.106] | +0.006 [+0.006, +0.006] | +0.052 [-0.010, +0.113] |
| Gs16 − BR | +0.058 [+0.058, +0.058] | +0.030 [+0.013, +0.046] | +0.028 [+0.028, +0.028] | +0.061 [-0.006, +0.129] |
| Gg16 − BR | +0.036 [+0.036, +0.036] | +0.021 [-0.000, +0.041] | +0.033 [+0.033, +0.033] | +0.010 [-0.006, +0.026] |
| TD+ − BR | +0.014 [+0.014, +0.014] | -0.080 [-0.152, -0.009] | +0.014 [+0.014, +0.014] | -0.093 [-0.096, -0.090] |
| TD- − BR | +0.017 [+0.017, +0.017] | +0.068 [+0.021, +0.115] | -0.035 [-0.035, -0.035] | +0.065 [-0.009, +0.138] |
| Gpx − Gpxx (shuffled) | +0.031 [+0.031, +0.031] | +0.060 [+0.010, +0.111] | +0.011 [+0.011, +0.011] | +0.051 [-0.010, +0.112] |
| Gs16 − Gs16x (shuffled) | +0.041 [+0.041, +0.041] | +0.039 [+0.034, +0.043] | +0.025 [+0.025, +0.025] | +0.108 [-0.005, +0.220] |
| Gg16 − Gg16x (shuffled) | +0.016 [+0.016, +0.016] | +0.018 [+0.006, +0.030] | +0.016 [+0.016, +0.016] | +0.054 [-0.003, +0.111] |
| Gs16 − Gg16 (regions vs grid) | +0.022 [+0.022, +0.022] | +0.009 [+0.005, +0.013] | -0.005 [-0.005, -0.005] | +0.051 [-0.001, +0.103] |

The same contrasts at the other propagations (VOC-sel. + DenseCRF; VOC-Sim / COCO-Cap):

| Contrast | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---|---|---|---|
| BR − UN (GOCA weighting) | -0.002 / -0.150 | +0.027 / -0.023 | +0.008 / +0.004 | +0.039 / +0.023 |
| Gpx − BR | +0.034 / -0.026 | +0.023 / +0.007 | +0.022 / +0.057 | +0.012 / +0.058 |
| Gs16 − BR | +0.051 / -0.081 | +0.056 / +0.085 | +0.058 / +0.030 | +0.058 / +0.073 |
| Gg16 − BR | +0.087 / -0.033 | +0.052 / -0.047 | +0.036 / +0.021 | +0.062 / -0.073 |
| TD+ − BR | -0.005 / -0.195 | -0.004 / -0.225 | +0.014 / -0.080 | -0.035 / -0.291 |
| TD- − BR | -0.012 / +0.138 | -0.052 / +0.083 | +0.017 / +0.068 | -0.073 / +0.075 |
| Gpx − Gpxx (shuffled) | -0.020 / -0.086 | +0.028 / +0.035 | +0.031 / +0.060 | +0.024 / +0.082 |
| Gs16 − Gs16x (shuffled) | +0.035 / -0.188 | +0.041 / -0.038 | +0.041 / +0.039 | +0.063 / +0.012 |
| Gg16 − Gg16x (shuffled) | +0.041 / +0.136 | +0.003 / +0.005 | +0.016 / +0.018 | +0.011 / -0.006 |
| Gs16 − Gg16 (regions vs grid) | -0.036 / -0.048 | +0.004 / +0.132 | +0.022 / +0.009 | -0.004 / +0.146 |

**Reading** (helps = COCO Δ vs BR interval > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0; gates also need Δ vs shuffled interval > 0 on COCO):

- Gpx (gate, pixels): COCO +0.057 (lower +0.009), VOC +0.022, COCO best-thr. -0.006 → **calibration only**; diagnostic failed
- Gs16 (gate, SA k-means k=16): COCO +0.030 (lower +0.013), VOC +0.058, COCO best-thr. +0.002 → **helps**; diagnostic failed
- Gg16 (gate, grid 4x4): COCO +0.021 (lower -0.000), VOC +0.036, COCO best-thr. -0.003 → **does not help**; diagnostic failed
- TD+ (seed + early - late): COCO -0.080 (lower -0.152), VOC +0.014, COCO best-thr. +0.001 → **does not help**; diagnostic failed
- TD- (seed + late - early): COCO +0.068 (lower +0.021), VOC +0.017, COCO best-thr. +0.003 → **helps**; diagnostic failed

**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**

| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BR | BRACE hierarchy | 0.724 | 0.619 | nan | 0.829 | 0.000 | 0.0021 | 0.994 | 0.728 |
| BR | NERVE walk | 0.691 | 0.756 | nan | 0.625 | 0.000 | 0.0013 | 0.995 | 0.693 |
| UN | BRACE hierarchy | 0.720 | 0.631 | nan | 0.810 | 0.000 | 0.0019 | 0.994 | 0.724 |
| UN | NERVE walk | 0.667 | 0.808 | nan | 0.527 | 0.000 | 0.0009 | 0.996 | 0.668 |
| Gpx | BRACE hierarchy | 0.781 | 0.628 | nan | 0.934 | 0.000 | 0.0037 | 0.990 | 0.791 |
| Gpx | NERVE walk | 0.749 | 0.637 | nan | 0.861 | 0.000 | 0.0038 | 0.989 | 0.757 |
| Gs16 | BRACE hierarchy | 0.754 | 0.632 | nan | 0.875 | 0.000 | 0.0031 | 0.991 | 0.761 |
| Gs16 | NERVE walk | 0.764 | 0.840 | nan | 0.688 | 0.000 | 0.0021 | 0.992 | 0.768 |
| Gg16 | BRACE hierarchy | 0.744 | 0.619 | nan | 0.870 | 0.000 | 0.0030 | 0.991 | 0.751 |
| Gg16 | NERVE walk | 0.618 | 0.598 | nan | 0.638 | 0.000 | 0.0015 | 0.994 | 0.620 |
| TD+ | BRACE hierarchy | 0.643 | 0.611 | nan | 0.676 | 0.000 | 0.0016 | 0.994 | 0.646 |
| TD+ | NERVE walk | 0.399 | 0.634 | nan | 0.165 | 0.500 | 0.0000 | 1.000 | 0.399 |
| TD- | BRACE hierarchy | 0.792 | 0.640 | nan | 0.944 | 0.000 | 0.0032 | 0.992 | 0.800 |
| TD- | NERVE walk | 0.765 | 0.643 | nan | 0.888 | 0.000 | 0.0041 | 0.988 | 0.775 |

## Per class, VOC-sel. + CRF, hierarchy (BR / Gpx / Gs16)

VOC-Sim: dining-table 0.20 / 0.22 / 0.25

COCO-Cap: cat 0.83 / 0.93 / 0.88, tv 0.62 / 0.63 / 0.63

