# Run 32: component factorial on real PASCAL VOC photos, one noising pass (`PREREG.md`)

**Exploratory.** VOC 2012 val: 4 photos, 4 (photo, class) items. Thresholds chosen on VOC train (4 items, raw maps). One SD1.5 pass per item at t = 100, prompt `A photograph of a {class}`. Class-macro mIoU over items (our benchmark metric, not the VOC21 dataset metric). 95% photo-bootstrap intervals (50 samples). DenseCRF is applied at the threshold selected on raw train maps.


## Profile source A: VOC train labels

| SA-guided weights | Hierarchy | 4 absent-class profiles | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 | train threshold |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.544 | 0.599 | 0.628 | 0.644 | 0.600 |
|  |  | ✓ | 0.491 | 0.574 | 0.577 | 0.621 | 0.600 |
|  | ✓ |  | 0.509 | 0.632 | 0.552 | 0.623 | 0.700 |
|  | ✓ | ✓ | 0.486 | 0.657 | 0.533 | 0.649 | 0.700 |
| ✓ |  |  | 0.650 | 0.647 | 0.686 | 0.679 | 0.575 |
| ✓ |  | ✓ | 0.639 | 0.645 | 0.682 | 0.680 | 0.575 |
| ✓ | ✓ |  | 0.493 | 0.669 | 0.549 | 0.661 | 0.700 |
| ✓ | ✓ | ✓ | 0.460 | 0.678 | 0.528 | 0.673 | 0.700 |

Effect of each component (Δ mIoU). *Alone*: only it on, minus all off. *In the stack*: all on, minus all on without it. *Average*: mean of the 4 paired differences that change only it.

| Component | Effect | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 |
|---|---|---|---|---|---|
| SA-guided layer weights | alone | +0.106 [+0.004, +0.202] | +0.048 [-0.017, +0.113] | +0.058 [-0.047, +0.166] | +0.035 [-0.053, +0.123] |
| SA-guided layer weights | in the stack | -0.026 [-0.054, +0.003] | +0.021 [+0.005, +0.037] | -0.005 [-0.027, +0.019] | +0.024 [+0.003, +0.045] |
| SA-guided layer weights | average | +0.053 [+0.003, +0.101] | +0.044 [-0.006, +0.094] | +0.039 [-0.016, +0.093] | +0.039 [-0.025, +0.104] |
| hierarchy (no cap) | alone | -0.036 [-0.063, -0.008] | +0.033 [-0.095, +0.172] | -0.075 [-0.131, -0.011] | -0.022 [-0.192, +0.166] |
| hierarchy (no cap) | in the stack | -0.179 [-0.321, -0.036] | +0.033 [-0.037, +0.109] | -0.154 [-0.280, -0.028] | -0.007 [-0.113, +0.109] |
| hierarchy (no cap) | average | -0.094 [-0.166, -0.022] | +0.043 [-0.049, +0.140] | -0.102 [-0.184, -0.011] | -0.005 [-0.138, +0.138] |
| 4 absent-class profiles | alone | -0.053 [-0.107, +0.000] | -0.025 [-0.049, +0.000] | -0.051 [-0.102, +0.000] | -0.023 [-0.047, +0.000] |
| 4 absent-class profiles | in the stack | -0.033 [-0.067, +0.000] | +0.009 [+0.000, +0.017] | -0.021 [-0.042, +0.000] | +0.012 [+0.000, +0.024] |
| 4 absent-class profiles | average | -0.030 [-0.060, +0.000] | +0.002 [+0.000, +0.003] | -0.024 [-0.047, +0.000] | +0.004 [+0.000, +0.008] |

Reading (primary rule, average effect; fixed in PREREG): SA-guided layer weights **contributes**; hierarchy (no cap) **hurts**; 4 absent-class profiles not resolved.


## Profile source B: VOC-Sim captions (label-free)

| SA-guided weights | Hierarchy | 4 absent-class profiles | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 | train threshold |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.544 | 0.599 | 0.628 | 0.644 | 0.600 |
|  |  | ✓ | 0.372 | 0.455 | 0.498 | 0.563 | 0.600 |
|  | ✓ |  | 0.509 | 0.632 | 0.552 | 0.623 | 0.700 |
|  | ✓ | ✓ | 0.344 | 0.559 | 0.401 | 0.601 | 0.700 |
| ✓ |  |  | 0.650 | 0.647 | 0.686 | 0.679 | 0.575 |
| ✓ |  | ✓ | 0.501 | 0.501 | 0.588 | 0.588 | 0.550 |
| ✓ | ✓ |  | 0.493 | 0.669 | 0.549 | 0.661 | 0.700 |
| ✓ | ✓ | ✓ | 0.342 | 0.588 | 0.410 | 0.626 | 0.700 |

Effect of each component (Δ mIoU). *Alone*: only it on, minus all off. *In the stack*: all on, minus all on without it. *Average*: mean of the 4 paired differences that change only it.

| Component | Effect | + DenseCRF, train-selected threshold (primary) | + DenseCRF, fixed 0.55 | raw, train-selected threshold | raw, fixed 0.55 |
|---|---|---|---|---|---|
| SA-guided layer weights | alone | +0.106 [+0.004, +0.202] | +0.048 [-0.017, +0.113] | +0.058 [-0.047, +0.166] | +0.035 [-0.053, +0.123] |
| SA-guided layer weights | in the stack | -0.001 [-0.005, +0.003] | +0.028 [+0.005, +0.052] | +0.009 [+0.001, +0.019] | +0.024 [+0.003, +0.045] |
| SA-guided layer weights | average | +0.055 [+0.008, +0.101] | +0.040 [-0.006, +0.086] | +0.039 [-0.021, +0.099] | +0.031 [-0.025, +0.086] |
| hierarchy (no cap) | alone | -0.036 [-0.063, -0.008] | +0.033 [-0.095, +0.172] | -0.075 [-0.131, -0.011] | -0.022 [-0.192, +0.166] |
| hierarchy (no cap) | in the stack | -0.159 [-0.304, -0.014] | +0.087 [-0.017, +0.190] | -0.178 [-0.352, -0.004] | +0.038 [-0.078, +0.153] |
| hierarchy (no cap) | average | -0.095 [-0.173, -0.016] | +0.062 [-0.028, +0.151] | -0.122 [-0.197, -0.047] | +0.009 [-0.116, +0.138] |
| 4 absent-class profiles | alone | -0.172 [-0.345, +0.000] | -0.144 [-0.288, +0.000] | -0.130 [-0.259, +0.000] | -0.081 [-0.162, +0.000] |
| 4 absent-class profiles | in the stack | -0.151 [-0.302, +0.000] | -0.081 [-0.162, +0.000] | -0.139 [-0.278, +0.000] | -0.036 [-0.071, +0.000] |
| 4 absent-class profiles | average | -0.159 [-0.321, +0.004] | -0.111 [-0.222, +0.000] | -0.130 [-0.263, +0.006] | -0.057 [-0.115, +0.000] |

Reading (primary rule, average effect; fixed in PREREG): SA-guided layer weights **contributes**; hierarchy (no cap) **hurts**; 4 absent-class profiles not resolved.


## Descriptive

| Map | raw, train-selected | raw, 0.55 |
|---|---:|---:|
| seed only, uniform weights | 0.333 | 0.294 |
| seed only, SA-guided weights | 0.263 | 0.329 |
| SA-guided weights + capped hierarchy (BRACE, one pass) | 0.565 | 0.663 |
| SA-guided + hierarchy + 1 absent-class profile, source A | 0.523 | 0.674 |
| SA-guided + hierarchy + 4 absent-class profiles, source A | 0.528 | 0.673 |
| SA-guided + hierarchy + 1 absent-class profile, source B | 0.412 | 0.627 |
| SA-guided + hierarchy + 4 absent-class profiles, source B | 0.410 | 0.626 |

Per class IoU, primary rule (all off / all on, source A / all on, source B):

| Class | items | all off | all on (A) | all on (B) |
|---|---:|---:|---:|---:|
| aeroplane | 1 | 0.837 | 0.597 | 0.126 |
| bicycle | 1 | 0.354 | 0.339 | 0.339 |
| potted plant | 1 | 0.901 | 0.902 | 0.902 |
| tv | 1 | 0.085 | 0.002 | 0.002 |

Checks: profiles with no absent-class pool fall back to g = 1; gate misses per item are in the capture files.

