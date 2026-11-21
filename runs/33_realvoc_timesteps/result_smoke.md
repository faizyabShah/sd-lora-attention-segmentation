# Run 33: noise levels and profile sources on real PASCAL VOC photos (`PREREG.md`)

**Exploratory; second look at VOC val** (run 32 reported it first). VOC 2012 val: 4 photos, 4 items. Thresholds chosen on VOC train (4 items, raw maps). One capture: three SD1.5 calls per item at t = 100, 300, 500 with one shared noise draw; arms read one call or all three. Class-macro mIoU over items; 95% photo-bootstrap intervals (50 samples, paired).


## 1. Primary contrasts (val, + DenseCRF, train-selected threshold)

| # | Contrast | Δ mIoU [95% CI] | Reading |
|---|---|---|---|
| P1 | multi − t100, full stack (source A) | +0.232 [+0.023, +0.441] | **helps** |
| P2 | multi − t100, BRACE (capped hierarchy, no gate) | +0.169 [-0.018, +0.358] | not resolved |
| P3 | hierarchy, average effect, multi, source A | +0.243 [+0.080, +0.418] | **helps** |
| P4 | C − B: real vs generated fit images (both captions), t = 100, full stack | +0.161 [+0.000, +0.317] | not resolved |
| P5 | S − C: labels vs captions (real, 490 each), t = 100, full stack | -0.009 [-0.019, +0.000] | not resolved |
| P6 | A − S: 2,170 vs 490 fit items (labels), t = 100, full stack | -0.033 [-0.067, +0.000] | not resolved |

## 2. Noise levels

| Map | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| all off (uniform, fine-only, no gate) | 0.544 | 0.382 | 0.690 | 0.413 |
| BRACE (capped hierarchy) | 0.524 | 0.659 | 0.713 | 0.693 |
| SA weights + hierarchy | 0.493 | 0.617 | 0.719 | 0.662 |
| full stack, source A | 0.460 | 0.635 | 0.710 | 0.692 |
| full stack, source S | 0.493 | 0.617 | 0.719 | 0.662 |
| full stack, source B | 0.342 | 0.499 | 0.696 | 0.535 |
| full stack, source C | 0.503 | 0.717 | 0.721 | 0.716 |

Δ vs t = 100 (primary rule):

| Map | t = 300 | t = 500 | multi |
|---|---|---|---|
| BRACE | +0.136 [-0.015, +0.288] | +0.189 [-0.012, +0.391] | +0.169 [-0.018, +0.358] |
| full stack, A | +0.175 [+0.007, +0.342] | +0.250 [+0.002, +0.498] | +0.232 [+0.023, +0.441] |
| full stack, C | +0.214 [-0.001, +0.430] | +0.218 [-0.014, +0.449] | +0.213 [-0.017, +0.442] |
| all off | -0.163 [-0.493, +0.128] | +0.146 [-0.051, +0.347] | -0.131 [-0.507, +0.202] |

Average effect of each component per noise level (primary rule):

| Component | Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---|---|---|---|---|
| SA-guided layer weights | A | +0.053 [+0.003, +0.101] | +0.107 [+0.022, +0.201] | -0.062 [-0.133, +0.001] | +0.017 [-0.045, +0.079] |
| hierarchy (no cap) | A | -0.094 [-0.166, -0.022] | +0.069 [-0.050, +0.207] | +0.125 [+0.017, +0.247] | +0.243 [+0.080, +0.418] |
| 4 absent-class profiles | A | -0.030 [-0.060, +0.000] | +0.056 [+0.000, +0.113] | +0.005 [+0.000, +0.011] | +0.017 [+0.000, +0.034] |
| 4 absent-class profiles | B | -0.159 [-0.321, +0.004] | -0.053 [-0.143, +0.038] | -0.040 [-0.076, -0.008] | -0.066 [-0.155, +0.021] |
| 4 absent-class profiles | C | +0.002 [-0.000, +0.004] | +0.095 [+0.025, +0.165] | +0.062 [+0.016, +0.115] | +0.105 [+0.045, +0.171] |

## 3. Profile sources

Full stack (SA weights + hierarchy + 4 profiles), primary rule:

| Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| no gate | 0.493 | 0.617 | 0.719 | 0.662 |
| A: VOC train labels | 0.460 | 0.635 | 0.710 | 0.692 |
| S: VOC train labels, cut to 490 | 0.493 | 0.617 | 0.719 | 0.662 |
| C: real COCO photos, captions | 0.503 | 0.717 | 0.721 | 0.716 |
| B: VOC-Sim generated, captions | 0.342 | 0.499 | 0.696 | 0.535 |

Paired source contrasts, full stack, multi (secondary):

| Contrast | Δ [95% CI] | Reading |
|---|---|---|
| C − B | +0.181 [+0.009, +0.353] | **helps** |
| S − C | -0.053 [-0.127, +0.020] | not resolved |
| A − S | +0.030 [+0.000, +0.059] | not resolved |

### Factorial, t = 100, source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.544 | 0.599 | 0.628 | 0.644 | 0.600 |
|  |  | ✓ | 0.491 | 0.574 | 0.577 | 0.621 | 0.600 |
|  | ✓ |  | 0.509 | 0.632 | 0.552 | 0.623 | 0.700 |
|  | ✓ | ✓ | 0.486 | 0.657 | 0.533 | 0.649 | 0.700 |
| ✓ |  |  | 0.650 | 0.647 | 0.686 | 0.679 | 0.575 |
| ✓ |  | ✓ | 0.639 | 0.645 | 0.682 | 0.680 | 0.575 |
| ✓ | ✓ |  | 0.493 | 0.669 | 0.549 | 0.661 | 0.700 |
| ✓ | ✓ | ✓ | 0.460 | 0.678 | 0.528 | 0.673 | 0.700 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.106 [+0.004, +0.202] | -0.026 [-0.054, +0.003] | +0.053 [+0.003, +0.101] |
| hierarchy (no cap) | -0.036 [-0.063, -0.008] | -0.179 [-0.321, -0.036] | -0.094 [-0.166, -0.022] |
| 4 absent-class profiles | -0.053 [-0.107, +0.000] | -0.033 [-0.067, +0.000] | -0.030 [-0.060, +0.000] |

### Factorial, t = 100, source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.544 | 0.599 | 0.628 | 0.644 | 0.600 |
|  |  | ✓ | 0.372 | 0.455 | 0.498 | 0.563 | 0.600 |
|  | ✓ |  | 0.509 | 0.632 | 0.552 | 0.623 | 0.700 |
|  | ✓ | ✓ | 0.344 | 0.559 | 0.401 | 0.601 | 0.700 |
| ✓ |  |  | 0.650 | 0.647 | 0.686 | 0.679 | 0.575 |
| ✓ |  | ✓ | 0.501 | 0.501 | 0.588 | 0.588 | 0.550 |
| ✓ | ✓ |  | 0.493 | 0.669 | 0.549 | 0.661 | 0.700 |
| ✓ | ✓ | ✓ | 0.342 | 0.588 | 0.410 | 0.626 | 0.700 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.106 [+0.004, +0.202] | -0.001 [-0.005, +0.003] | +0.055 [+0.008, +0.101] |
| hierarchy (no cap) | -0.036 [-0.063, -0.008] | -0.159 [-0.304, -0.014] | -0.095 [-0.173, -0.016] |
| 4 absent-class profiles | -0.172 [-0.345, +0.000] | -0.151 [-0.302, +0.000] | -0.159 [-0.321, +0.004] |

### Factorial, t = 100, source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.544 | 0.599 | 0.628 | 0.644 | 0.600 |
|  |  | ✓ | 0.530 | 0.592 | 0.633 | 0.680 | 0.600 |
|  | ✓ |  | 0.509 | 0.632 | 0.552 | 0.623 | 0.700 |
|  | ✓ | ✓ | 0.516 | 0.636 | 0.582 | 0.635 | 0.700 |
| ✓ |  |  | 0.650 | 0.647 | 0.686 | 0.679 | 0.575 |
| ✓ |  | ✓ | 0.655 | 0.655 | 0.718 | 0.718 | 0.550 |
| ✓ | ✓ |  | 0.493 | 0.669 | 0.549 | 0.661 | 0.700 |
| ✓ | ✓ | ✓ | 0.503 | 0.673 | 0.582 | 0.675 | 0.700 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.106 [+0.004, +0.202] | -0.013 [-0.031, +0.005] | +0.050 [-0.012, +0.109] |
| hierarchy (no cap) | -0.036 [-0.063, -0.008] | -0.152 [-0.293, -0.011] | -0.090 [-0.174, -0.005] |
| 4 absent-class profiles | -0.014 [-0.027, +0.000] | +0.009 [+0.000, +0.019] | +0.002 [-0.000, +0.004] |

### Factorial, multi (100+300+500), source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.413 | 0.701 | 0.563 | 0.723 | 0.700 |
|  |  | ✓ | 0.422 | 0.699 | 0.569 | 0.724 | 0.700 |
|  | ✓ |  | 0.708 | 0.701 | 0.714 | 0.688 | 0.600 |
|  | ✓ | ✓ | 0.704 | 0.710 | 0.716 | 0.700 | 0.600 |
| ✓ |  |  | 0.463 | 0.694 | 0.598 | 0.723 | 0.700 |
| ✓ |  | ✓ | 0.498 | 0.699 | 0.639 | 0.724 | 0.700 |
| ✓ | ✓ |  | 0.662 | 0.713 | 0.630 | 0.705 | 0.400 |
| ✓ | ✓ | ✓ | 0.692 | 0.713 | 0.657 | 0.707 | 0.400 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.050 [-0.023, +0.122] | -0.012 [-0.059, +0.036] | +0.017 [-0.045, +0.079] |
| hierarchy (no cap) | +0.295 [+0.099, +0.511] | +0.194 [+0.060, +0.334] | +0.243 [+0.080, +0.418] |
| 4 absent-class profiles | +0.009 [+0.000, +0.017] | +0.030 [+0.000, +0.059] | +0.017 [+0.000, +0.034] |

### Factorial, multi (100+300+500), source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.413 | 0.701 | 0.563 | 0.723 | 0.700 |
|  |  | ✓ | 0.386 | 0.574 | 0.497 | 0.637 | 0.700 |
|  | ✓ |  | 0.708 | 0.701 | 0.714 | 0.688 | 0.600 |
|  | ✓ | ✓ | 0.519 | 0.614 | 0.575 | 0.653 | 0.700 |
| ✓ |  |  | 0.463 | 0.694 | 0.598 | 0.723 | 0.700 |
| ✓ |  | ✓ | 0.543 | 0.574 | 0.618 | 0.636 | 0.600 |
| ✓ | ✓ |  | 0.662 | 0.713 | 0.630 | 0.705 | 0.400 |
| ✓ | ✓ | ✓ | 0.535 | 0.639 | 0.612 | 0.665 | 0.700 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.050 [-0.023, +0.122] | +0.016 [+0.005, +0.027] | +0.044 [-0.014, +0.105] |
| hierarchy (no cap) | +0.295 [+0.099, +0.511] | -0.009 [-0.027, +0.012] | +0.155 [+0.061, +0.259] |
| 4 absent-class profiles | -0.027 [-0.054, +0.000] | -0.128 [-0.284, +0.019] | -0.066 [-0.155, +0.021] |

### Factorial, multi (100+300+500), source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.413 | 0.701 | 0.563 | 0.723 | 0.700 |
|  |  | ✓ | 0.592 | 0.705 | 0.573 | 0.734 | 0.375 |
|  | ✓ |  | 0.708 | 0.701 | 0.714 | 0.688 | 0.600 |
|  | ✓ | ✓ | 0.674 | 0.702 | 0.653 | 0.694 | 0.500 |
| ✓ |  |  | 0.463 | 0.694 | 0.598 | 0.723 | 0.700 |
| ✓ |  | ✓ | 0.687 | 0.719 | 0.682 | 0.767 | 0.425 |
| ✓ | ✓ |  | 0.662 | 0.713 | 0.630 | 0.705 | 0.400 |
| ✓ | ✓ | ✓ | 0.716 | 0.716 | 0.728 | 0.716 | 0.575 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.050 [-0.023, +0.122] | +0.042 [-0.000, +0.084] | +0.035 [+0.006, +0.070] |
| hierarchy (no cap) | +0.295 [+0.099, +0.511] | +0.029 [-0.007, +0.065] | +0.151 [+0.046, +0.260] |
| 4 absent-class profiles | +0.178 [+0.044, +0.315] | +0.053 [-0.020, +0.127] | +0.105 [+0.045, +0.171] |

## 4. Descriptive

Seed only (raw, train-selected):

| Seed | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| uniform | 0.333 | 0.274 | 0.207 | 0.194 |
| SA weights | 0.263 | 0.216 | 0.267 | 0.245 |
| SA weights + gate A | 0.245 | 0.226 | 0.264 | 0.245 |
| SA weights + gate C | 0.517 | 0.629 | 0.630 | 0.631 |

Per class IoU, primary rule, full stack per source:

| Class | items | all off, t100 | A, t100 | S, t100 | C, t100 | B, t100 | A, multi | C, multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 1 | 0.837 | 0.597 | 0.731 | 0.731 | 0.126 | 0.760 | 0.830 |
| bicycle | 1 | 0.354 | 0.339 | 0.339 | 0.377 | 0.339 | 0.287 | 0.317 |
| potted plant | 1 | 0.901 | 0.902 | 0.902 | 0.902 | 0.902 | 1.000 | 0.929 |
| tv | 1 | 0.085 | 0.002 | 0.002 | 0.002 | 0.002 | 0.722 | 0.788 |

Run 32 (different capture, descriptive only, primary rule): all off 0.596, SA weights + hierarchy 0.610, full stack A 0.646, full stack B 0.623. Compare with the t = 100 column above.

Fit set sizes (items): {'A_all': 6, 'A_f0': 4, 'A_f1': 2, 'S_all': 1, 'S_f0': 1, 'S_f1': 2, 'B_all': 6, 'C_all': 6}. Profiles with no absent-class pool fall back to g = 1.

