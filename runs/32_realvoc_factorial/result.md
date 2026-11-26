# Run 32: component factorial on real PASCAL VOC photos, one noising pass (`PREREG.md`)

**Exploratory.** VOC 2012 val: 1449 photos, 2148 (photo, class) items. Thresholds chosen on VOC train (2170 items, raw maps). One SD1.5 pass per item at t = 100, prompt `A photograph of a {class}`. Class-macro mIoU over items (our benchmark metric, not the VOC21 dataset metric). 95% photo-bootstrap intervals (2000 samples). DenseCRF is applied at the threshold selected on raw train maps.


## Profile source A: VOC train labels

| SA-guided weights | Hierarchy | 4 absent-class profiles | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 | train threshold |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.602 | 0.600 | 0.610 | 0.611 | 0.525 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.616 | 0.605 | 0.625 | 0.608 | 0.625 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.642 | 0.633 | 0.650 | 0.645 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.646 | 0.639 | 0.652 | 0.642 | 0.600 |

Effect of each component (Δ mIoU). *Alone*: only it on, minus all off. *In the stack*: all on, minus all on without it. *Average*: mean of the 4 paired differences that change only it.

| Component | Effect | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 |
|---|---|---|---|---|---|
| SA-guided layer weights | alone | +0.023 [+0.016, +0.029] | +0.022 [+0.015, +0.029] | +0.025 [+0.019, +0.030] | +0.024 [+0.018, +0.030] |
| SA-guided layer weights | in the stack | +0.029 [+0.023, +0.036] | +0.034 [+0.027, +0.042] | +0.027 [+0.022, +0.033] | +0.034 [+0.027, +0.041] |
| SA-guided layer weights | average | +0.027 [+0.023, +0.031] | +0.027 [+0.022, +0.031] | +0.027 [+0.023, +0.031] | +0.027 [+0.023, +0.031] |
| hierarchy (no cap) | alone | -0.002 [-0.011, +0.007] | -0.011 [-0.023, +0.002] | +0.002 [-0.007, +0.010] | -0.016 [-0.028, -0.004] |
| hierarchy (no cap) | in the stack | +0.004 [-0.005, +0.012] | +0.006 [-0.006, +0.018] | +0.003 [-0.006, +0.010] | -0.004 [-0.015, +0.007] |
| hierarchy (no cap) | average | +0.002 [-0.005, +0.009] | -0.004 [-0.014, +0.007] | +0.003 [-0.004, +0.010] | -0.011 [-0.021, -0.002] |
| 4 absent-class profiles | alone | +0.006 [-0.006, +0.019] | +0.007 [-0.006, +0.020] | +0.011 [-0.001, +0.023] | +0.012 [-0.001, +0.023] |
| 4 absent-class profiles | in the stack | +0.036 [+0.026, +0.046] | +0.040 [+0.031, +0.048] | +0.036 [+0.027, +0.045] | +0.041 [+0.033, +0.050] |
| 4 absent-class profiles | average | +0.022 [+0.013, +0.032] | +0.022 [+0.013, +0.031] | +0.024 [+0.015, +0.033] | +0.025 [+0.017, +0.033] |

Reading (primary rule, average effect; fixed in PREREG): SA-guided layer weights **contributes**; hierarchy (no cap) not resolved; 4 absent-class profiles **contributes**.


## Profile source B: VOC-Sim captions (label-free)

| SA-guided weights | Hierarchy | 4 absent-class profiles | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 | train threshold |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.589 | 0.581 | 0.593 | 0.590 | 0.500 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.593 | 0.589 | 0.601 | 0.592 | 0.600 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.622 | 0.608 | 0.628 | 0.619 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.623 | 0.618 | 0.630 | 0.622 | 0.600 |

Effect of each component (Δ mIoU). *Alone*: only it on, minus all off. *In the stack*: all on, minus all on without it. *Average*: mean of the 4 paired differences that change only it.

| Component | Effect | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 |
|---|---|---|---|---|---|
| SA-guided layer weights | alone | +0.023 [+0.016, +0.029] | +0.022 [+0.015, +0.029] | +0.025 [+0.019, +0.030] | +0.024 [+0.018, +0.030] |
| SA-guided layer weights | in the stack | +0.030 [+0.022, +0.036] | +0.029 [+0.022, +0.035] | +0.029 [+0.023, +0.036] | +0.030 [+0.023, +0.035] |
| SA-guided layer weights | average | +0.025 [+0.021, +0.029] | +0.024 [+0.020, +0.028] | +0.026 [+0.022, +0.030] | +0.025 [+0.021, +0.029] |
| hierarchy (no cap) | alone | -0.002 [-0.011, +0.007] | -0.011 [-0.023, +0.002] | +0.002 [-0.007, +0.010] | -0.016 [-0.028, -0.004] |
| hierarchy (no cap) | in the stack | +0.000 [-0.008, +0.009] | +0.010 [-0.002, +0.021] | +0.002 [-0.006, +0.010] | +0.002 [-0.009, +0.013] |
| hierarchy (no cap) | average | -0.002 [-0.009, +0.006] | -0.002 [-0.012, +0.008] | +0.001 [-0.006, +0.008] | -0.009 [-0.018, +0.001] |
| 4 absent-class profiles | alone | -0.006 [-0.020, +0.006] | -0.012 [-0.026, +0.001] | -0.006 [-0.018, +0.006] | -0.009 [-0.022, +0.004] |
| 4 absent-class profiles | in the stack | +0.013 [+0.004, +0.022] | +0.018 [+0.009, +0.027] | +0.014 [+0.005, +0.022] | +0.021 [+0.013, +0.029] |
| 4 absent-class profiles | average | +0.002 [-0.008, +0.012] | +0.002 [-0.007, +0.011] | +0.003 [-0.006, +0.012] | +0.004 [-0.004, +0.013] |

Reading (primary rule, average effect; fixed in PREREG): SA-guided layer weights **contributes**; hierarchy (no cap) not resolved; 4 absent-class profiles not resolved.


## Descriptive

| Map | raw, train-selected | raw, 0.55 |
|---|---:|---:|
| seed only, uniform weights | 0.539 | 0.335 |
| seed only, SA-guided weights | 0.558 | 0.375 |
| SA-guided weights + capped hierarchy (BRACE, one pass) | 0.616 | 0.595 |
| SA-guided + hierarchy + 1 absent-class profile, source A | 0.645 | 0.635 |
| SA-guided + hierarchy + 4 absent-class profiles, source A | 0.652 | 0.642 |
| SA-guided + hierarchy + 1 absent-class profile, source B | 0.616 | 0.610 |
| SA-guided + hierarchy + 4 absent-class profiles, source B | 0.630 | 0.622 |

Per class IoU, primary rule (all off / all on, source A / all on, source B):

| Class | items | all off | all on (A) | all on (B) |
|---|---:|---:|---:|---:|
| aeroplane | 90 | 0.761 | 0.838 | 0.828 |
| bicycle | 79 | 0.308 | 0.349 | 0.342 |
| bird | 103 | 0.751 | 0.741 | 0.705 |
| boat | 72 | 0.644 | 0.732 | 0.726 |
| bottle | 96 | 0.505 | 0.419 | 0.380 |
| bus | 74 | 0.820 | 0.813 | 0.800 |
| car | 127 | 0.451 | 0.530 | 0.493 |
| cat | 119 | 0.818 | 0.843 | 0.842 |
| chair | 123 | 0.292 | 0.304 | 0.270 |
| cow | 71 | 0.855 | 0.877 | 0.841 |
| dining table | 75 | 0.415 | 0.498 | 0.522 |
| dog | 128 | 0.785 | 0.828 | 0.859 |
| horse | 79 | 0.770 | 0.830 | 0.804 |
| motorbike | 76 | 0.718 | 0.756 | 0.752 |
| person | 446 | 0.263 | 0.516 | 0.331 |
| potted plant | 85 | 0.288 | 0.324 | 0.318 |
| sheep | 57 | 0.844 | 0.834 | 0.768 |
| sofa | 90 | 0.538 | 0.709 | 0.724 |
| train | 84 | 0.706 | 0.720 | 0.703 |
| tv | 74 | 0.380 | 0.453 | 0.441 |

Checks: profiles with no absent-class pool fall back to g = 1; gate misses per item are in the capture files.

