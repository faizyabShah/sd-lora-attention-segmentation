# Run 33: noise levels and profile sources on real PASCAL VOC photos (`PREREG.md`)

**Exploratory; second look at VOC val** (run 32 reported it first). VOC 2012 val: 1449 photos, 2148 items. Thresholds chosen on VOC train (2170 items, raw maps). One capture: three SD1.5 calls per item at t = 100, 300, 500 with one shared noise draw; arms read one call or all three. Class-macro mIoU over items; 95% photo-bootstrap intervals (2000 samples, paired).


## 1. Primary contrasts (val, + DenseCRF, train-selected threshold)

| # | Contrast | Δ mIoU [95% CI] | Reading |
|---|---|---|---|
| P1 | multi − t100, full stack (source A) | -0.028 [-0.039, -0.017] | **hurts** |
| P2 | multi − t100, BRACE (capped hierarchy, no gate) | -0.024 [-0.035, -0.014] | **hurts** |
| P3 | hierarchy, average effect, multi, source A | -0.006 [-0.013, -0.001] | **hurts** |
| P4 | C − B: real vs generated fit images (both captions), t = 100, full stack | +0.023 [+0.014, +0.032] | **helps** |
| P5 | S − C: labels vs captions (real, 490 each), t = 100, full stack | -0.003 [-0.009, +0.003] | not resolved |
| P6 | A − S: 2,170 vs 490 fit items (labels), t = 100, full stack | +0.004 [-0.000, +0.007] | not resolved |

## 2. Noise levels

| Map | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| all off (uniform, fine-only, no gate) | 0.596 | 0.578 | 0.512 | 0.573 |
| BRACE (capped hierarchy) | 0.608 | 0.603 | 0.526 | 0.584 |
| SA weights + hierarchy | 0.610 | 0.604 | 0.526 | 0.586 |
| full stack, source A | 0.646 | 0.631 | 0.549 | 0.618 |
| full stack, source S | 0.642 | 0.629 | 0.547 | 0.613 |
| full stack, source B | 0.623 | 0.618 | 0.543 | 0.601 |
| full stack, source C | 0.645 | 0.632 | 0.544 | 0.616 |

Δ vs t = 100 (primary rule):

| Map | t = 300 | t = 500 | multi |
|---|---|---|---|
| BRACE | -0.005 [-0.016, +0.004] | -0.082 [-0.097, -0.068] | -0.024 [-0.035, -0.014] |
| full stack, A | -0.014 [-0.026, -0.003] | -0.096 [-0.112, -0.082] | -0.028 [-0.039, -0.017] |
| full stack, C | -0.014 [-0.026, -0.001] | -0.101 [-0.116, -0.087] | -0.030 [-0.041, -0.019] |
| all off | -0.018 [-0.029, -0.006] | -0.083 [-0.100, -0.069] | -0.022 [-0.034, -0.012] |

Average effect of each component per noise level (primary rule):

| Component | Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---|---|---|---|---|
| SA-guided layer weights | A | +0.027 [+0.023, +0.031] | +0.022 [+0.018, +0.025] | +0.018 [+0.014, +0.022] | +0.018 [+0.014, +0.021] |
| hierarchy (no cap) | A | +0.002 [-0.005, +0.009] | +0.003 [-0.004, +0.009] | -0.006 [-0.013, -0.000] | -0.006 [-0.013, -0.001] |
| 4 absent-class profiles | A | +0.022 [+0.013, +0.032] | +0.029 [+0.019, +0.038] | +0.028 [+0.021, +0.035] | +0.036 [+0.027, +0.044] |
| 4 absent-class profiles | B | +0.002 [-0.008, +0.012] | +0.013 [+0.005, +0.021] | +0.015 [+0.008, +0.021] | +0.015 [+0.008, +0.023] |
| 4 absent-class profiles | C | +0.022 [+0.012, +0.032] | +0.025 [+0.016, +0.034] | +0.021 [+0.014, +0.027] | +0.030 [+0.022, +0.037] |

## 3. Profile sources

Full stack (SA weights + hierarchy + 4 profiles), primary rule:

| Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| no gate | 0.610 | 0.604 | 0.526 | 0.586 |
| A: VOC train labels | 0.646 | 0.631 | 0.549 | 0.618 |
| S: VOC train labels, cut to 490 | 0.642 | 0.629 | 0.547 | 0.613 |
| C: real COCO photos, captions | 0.645 | 0.632 | 0.544 | 0.616 |
| B: VOC-Sim generated, captions | 0.623 | 0.618 | 0.543 | 0.601 |

Paired source contrasts, full stack, multi (secondary):

| Contrast | Δ [95% CI] | Reading |
|---|---|---|
| C − B | +0.014 [+0.007, +0.021] | **helps** |
| S − C | -0.003 [-0.009, +0.003] | not resolved |
| A − S | +0.005 [+0.002, +0.008] | **helps** |

### Factorial, t = 100, source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.602 | 0.600 | 0.610 | 0.611 | 0.525 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.616 | 0.605 | 0.625 | 0.608 | 0.625 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.642 | 0.633 | 0.650 | 0.645 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.646 | 0.639 | 0.652 | 0.642 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.029 [+0.023, +0.036] | +0.027 [+0.023, +0.031] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.004 [-0.005, +0.012] | +0.002 [-0.005, +0.009] |
| 4 absent-class profiles | +0.006 [-0.006, +0.019] | +0.036 [+0.026, +0.046] | +0.022 [+0.013, +0.032] |

### Factorial, t = 100, source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.589 | 0.581 | 0.593 | 0.590 | 0.500 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.593 | 0.589 | 0.601 | 0.592 | 0.600 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.622 | 0.608 | 0.628 | 0.619 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.623 | 0.618 | 0.630 | 0.622 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.030 [+0.022, +0.036] | +0.025 [+0.021, +0.029] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.000 [-0.008, +0.009] | -0.002 [-0.009, +0.006] |
| 4 absent-class profiles | -0.006 [-0.020, +0.006] | +0.013 [+0.004, +0.022] | +0.002 [-0.008, +0.012] |

### Factorial, t = 100, source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.603 | 0.602 | 0.609 | 0.610 | 0.525 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.619 | 0.606 | 0.627 | 0.607 | 0.625 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.636 | 0.627 | 0.642 | 0.639 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.645 | 0.637 | 0.652 | 0.639 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.026 [+0.020, +0.032] | +0.024 [+0.020, +0.029] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.009 [-0.000, +0.018] | +0.004 [-0.004, +0.011] |
| 4 absent-class profiles | +0.008 [-0.006, +0.021] | +0.036 [+0.026, +0.045] | +0.022 [+0.012, +0.032] |

### Factorial, multi (100+300+500), source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.608 | 0.606 | 0.617 | 0.613 | 0.575 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.605 | 0.586 | 0.613 | 0.586 | 0.650 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.632 | 0.631 | 0.637 | 0.638 | 0.525 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.618 | 0.607 | 0.626 | 0.607 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.013 [+0.008, +0.017] | +0.018 [+0.014, +0.021] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.014 [-0.021, -0.007] | -0.006 [-0.013, -0.001] |
| 4 absent-class profiles | +0.035 [+0.024, +0.046] | +0.032 [+0.023, +0.040] | +0.036 [+0.027, +0.044] |

### Factorial, multi (100+300+500), source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.582 | 0.582 | 0.590 | 0.590 | 0.550 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.585 | 0.573 | 0.594 | 0.574 | 0.650 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.613 | 0.610 | 0.619 | 0.618 | 0.525 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.601 | 0.595 | 0.610 | 0.596 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.016 [+0.011, +0.022] | +0.020 [+0.016, +0.024] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.012 [-0.019, -0.004] | -0.004 [-0.011, +0.002] |
| 4 absent-class profiles | +0.009 [-0.001, +0.020] | +0.015 [+0.007, +0.023] | +0.015 [+0.008, +0.023] |

### Factorial, multi (100+300+500), source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.604 | 0.603 | 0.612 | 0.609 | 0.575 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.597 | 0.580 | 0.607 | 0.580 | 0.675 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.623 | 0.623 | 0.630 | 0.630 | 0.550 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.616 | 0.599 | 0.624 | 0.599 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.019 [+0.014, +0.023] | +0.018 [+0.014, +0.022] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.008 [-0.015, -0.000] | -0.006 [-0.012, +0.000] |
| 4 absent-class profiles | +0.031 [+0.020, +0.041] | +0.029 [+0.022, +0.037] | +0.030 [+0.022, +0.037] |

## 4. Descriptive

Seed only (raw, train-selected):

| Seed | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| uniform | 0.539 | 0.516 | 0.485 | 0.530 |
| SA weights | 0.558 | 0.540 | 0.502 | 0.546 |
| SA weights + gate A | 0.575 | 0.577 | 0.538 | 0.586 |
| SA weights + gate C | 0.566 | 0.566 | 0.525 | 0.574 |

Per class IoU, primary rule, full stack per source:

| Class | items | all off, t100 | A, t100 | S, t100 | C, t100 | B, t100 | A, multi | C, multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 90 | 0.761 | 0.838 | 0.831 | 0.843 | 0.828 | 0.777 | 0.775 |
| bicycle | 79 | 0.308 | 0.349 | 0.341 | 0.334 | 0.342 | 0.298 | 0.288 |
| bird | 103 | 0.751 | 0.741 | 0.728 | 0.732 | 0.705 | 0.708 | 0.696 |
| boat | 72 | 0.644 | 0.732 | 0.728 | 0.728 | 0.726 | 0.679 | 0.664 |
| bottle | 96 | 0.505 | 0.419 | 0.445 | 0.435 | 0.380 | 0.427 | 0.423 |
| bus | 74 | 0.820 | 0.813 | 0.813 | 0.824 | 0.800 | 0.811 | 0.823 |
| car | 127 | 0.451 | 0.530 | 0.512 | 0.525 | 0.493 | 0.478 | 0.478 |
| cat | 119 | 0.818 | 0.843 | 0.849 | 0.852 | 0.842 | 0.819 | 0.816 |
| chair | 123 | 0.292 | 0.304 | 0.302 | 0.320 | 0.270 | 0.267 | 0.268 |
| cow | 71 | 0.855 | 0.877 | 0.876 | 0.876 | 0.841 | 0.839 | 0.839 |
| dining table | 75 | 0.415 | 0.498 | 0.485 | 0.448 | 0.522 | 0.534 | 0.513 |
| dog | 128 | 0.785 | 0.828 | 0.833 | 0.846 | 0.859 | 0.768 | 0.776 |
| horse | 79 | 0.770 | 0.830 | 0.828 | 0.830 | 0.804 | 0.763 | 0.759 |
| motorbike | 76 | 0.718 | 0.756 | 0.749 | 0.754 | 0.752 | 0.725 | 0.725 |
| person | 446 | 0.263 | 0.516 | 0.489 | 0.525 | 0.331 | 0.490 | 0.490 |
| potted plant | 85 | 0.288 | 0.324 | 0.296 | 0.288 | 0.318 | 0.312 | 0.324 |
| sheep | 57 | 0.844 | 0.834 | 0.853 | 0.858 | 0.768 | 0.771 | 0.741 |
| sofa | 90 | 0.538 | 0.709 | 0.699 | 0.710 | 0.724 | 0.670 | 0.676 |
| train | 84 | 0.706 | 0.720 | 0.740 | 0.736 | 0.703 | 0.772 | 0.781 |
| tv | 74 | 0.380 | 0.453 | 0.444 | 0.445 | 0.441 | 0.453 | 0.457 |

Run 32 (different capture, descriptive only, primary rule): all off 0.596, SA weights + hierarchy 0.610, full stack A 0.646, full stack B 0.623. Compare with the t = 100 column above.

Fit set sizes (items): {'A_all': 2170, 'A_f0': 1095, 'A_f1': 1075, 'S_all': 490, 'S_f0': 490, 'S_f1': 490, 'B_all': 490, 'C_all': 490}. Profiles with no absent-class pool fall back to g = 1.

