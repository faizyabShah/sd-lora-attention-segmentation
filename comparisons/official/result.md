# Official comparison rows, side N (GoCA, NERVE, DiffSegmenter), paired with BRACE

Workplan Section 3 contract. Class-macro mIoU (per-class IoU pooled over images, then mean over classes). Scripts: `comparisons/official/` (`brace_repro.py`, `score_n.py`, `report.py`); metric, OOF and bootstrap code reused unchanged from `scripts/sg_class_comp.py` (`miou`, `oof`, `boot`); DenseCRF R2 = `src/segmentation/ovam_full.apply_ovam_densecrf` (OVAM parameters (10, 80, 13, 3, 3, 5), SimpleCRF 0.2.1.1). Threshold grid (27 values): 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95. Paired Δ: 2000-sample image bootstrap, 95% interval.

**Labels.** *VOC-sel.* = threshold selected on VOC-Sim: 5-fold out-of-fold on VOC-Sim (folds crc32(id) mod 5), and the full-VOC-Sim-best threshold applied unchanged to COCO-Cap (untuned transfer). *fixed* = fixed-threshold. *audit* = uses the evaluated set's labels, never a result. BRACE's VOC-Sim numbers are in-sample for the method design (VOC-Sim was used to develop BRACE, not the comparison methods). Everything here is exploratory in the sense of AGENTS.md: no row was pre-registered beyond the Section 3 contract and Entry 3.

**Inputs.** N methods read the stored OVAM benchmark images (`comparisons/manifests/`, 490 VOC-Sim, 707 COCO-Cap). BRACE reads attention during generation, so it is regenerated with the corrected prompts of runs/20 (`configs/new_reruns_prompts.json`, safety checker off): *same prompts, regenerated*. Pairing is on identical image ids and identical ground-truth masks.

## Data checks

- VOC-Sim: 490 images (50 with corrected prompt). BRACE regeneration vs stored official PNG: bit-identical 0/490; mean |Δ| < 1 grey level on 431, ≥ 2 (unresolved) on 36; median mean |Δ| 0.17. Ground-truth masks identical (BRACE live sample vs annotation PNG used by N): 490/490; identical across N methods: True; class labels match: True.
- COCO-Cap: 707 images (86 with corrected prompt). BRACE regeneration vs stored official PNG: bit-identical 0/707; mean |Δ| < 1 grey level on 583, ≥ 2 (unresolved) on 52; median mean |Δ| 0.27. Ground-truth masks identical (BRACE live sample vs annotation PNG used by N): 707/707; identical across N methods: True; class labels match: True.

## BRACE reproduction of runs/20 (runs/20 grid 0.30–0.70, runs/20 convention)

| Cell | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | VOC-Sim 0.55 | COCO-Cap 0.55 | runs/20 (same four) |
|---|---:|---:|---:|---:|---|
| BRACE hierarchy + DenseCRF | 0.831 | 0.708 | 0.832 | 0.708 | 0.832 / 0.709 / 0.833 / 0.709 |
| BRACE hierarchy, raw | 0.818 | 0.692 | 0.819 | 0.692 | 0.818 / 0.693 / 0.819 / 0.693 |
| fine-only, raw | 0.811 | 0.670 | 0.792 | 0.690 | 0.811 / 0.670 / 0.792 / 0.690 |
| seed alone, raw | 0.706 | 0.591 | 0.613 | 0.510 | 0.706 / 0.592 / 0.613 / 0.510 |

Full-VOC-Sim-best threshold (BRACE + DenseCRF): 0.550 (runs/20: 0.550). COCO-Cap best-threshold audit 0.719 (runs/20: 0.719).

## Official rows, raw (no DenseCRF)

| Method | Rule | VOC-Sim | COCO-Cap | Note |
|---|---|---:|---:|---|
| BRACE | VOC-sel. | 0.818 | 0.692 | τ full-VOC 0.550; OOF folds 0.550/0.550/0.550/0.550/0.575 |
| BRACE | fixed 0.5 | 0.812 | 0.664 | fixed-threshold |
| BRACE | fixed 0.55 (frozen) | 0.819 | 0.692 | fixed-threshold; BRACE's frozen rule |
| BRACE | best on set (audit) | 0.819 | 0.709 | in-sample audit: τ VOC-Sim 0.550, COCO-Cap 0.625 |
| GoCA SD1.5 | VOC-sel. | 0.738 | 0.610 | τ full-VOC 0.750; OOF folds 0.700/0.750/0.750/0.750/0.750 |
| GoCA SD1.5 | fixed 0.5 | 0.671 | 0.452 | fixed-threshold |
| GoCA SD1.5 | best on set (audit) | 0.740 | 0.614 | in-sample audit: τ VOC-Sim 0.750, COCO-Cap 0.800 |
| GoCA SD1.5 | own rule (offset 0) | 0.671 | 0.452 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SD1.5 | own rule, VOC-sel. offset | 0.739 | 0.586 | offset full-VOC +0.4; OOF folds +0.4/+0.4/+0.4/+0.4/+0.4 |
| GoCA SD1.5 | own rule, best offset on set (audit) | 0.739 | 0.586 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim +0.4, COCO-Cap +0.4 |
| GoCA SDXL | VOC-sel. | 0.787 | 0.602 | τ full-VOC 0.475; OOF folds 0.475/0.450/0.475/0.475/0.475 |
| GoCA SDXL | fixed 0.5 | 0.788 | 0.607 | fixed-threshold |
| GoCA SDXL | best on set (audit) | 0.789 | 0.609 | in-sample audit: τ VOC-Sim 0.475, COCO-Cap 0.525 |
| GoCA SDXL | own rule (offset 0) | 0.787 | 0.607 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SDXL | own rule, VOC-sel. offset | 0.785 | 0.596 | offset full-VOC -0.1; OOF folds -0.1/-0.1/+0.0/+0.0/-0.1 |
| GoCA SDXL | own rule, best offset on set (audit) | 0.788 | 0.607 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim -0.1, COCO-Cap +0.0 |
| GoCA SD2.1 | VOC-sel. | 0.719 | 0.586 | τ full-VOC 0.575; OOF folds 0.575/0.575/0.575/0.550/0.575 |
| GoCA SD2.1 | fixed 0.5 | 0.710 | 0.546 | fixed-threshold |
| GoCA SD2.1 | best on set (audit) | 0.720 | 0.595 | in-sample audit: τ VOC-Sim 0.575, COCO-Cap 0.625 |
| GoCA SD2.1 | own rule (offset 0) | 0.710 | 0.546 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SD2.1 | own rule, VOC-sel. offset | 0.720 | 0.576 | offset full-VOC +0.1; OOF folds +0.1/+0.1/+0.1/+0.1/+0.1 |
| GoCA SD2.1 | own rule, best offset on set (audit) | 0.720 | 0.594 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim +0.1, COCO-Cap +0.3 |
| DiffSegmenter | VOC-sel. | 0.772 | 0.668 | τ full-VOC 0.375; OOF folds 0.375/0.375/0.375/0.350/0.350 |
| DiffSegmenter | fixed 0.5 | 0.713 | 0.653 | fixed-threshold |
| DiffSegmenter | best on set (audit) | 0.772 | 0.674 | in-sample audit: τ VOC-Sim 0.375, COCO-Cap 0.425 |
| DiffSegmenter | own rule (in-sample bg sweep) | 0.772 | 0.674 | DiffSegmenter's evaluation script: background score swept 0.30–0.74 on the evaluated set (here step 0.025); in-sample, uses labels |
| NERVE A | VOC-sel. | 0.832 | 0.744 | τ full-VOC 0.550; OOF folds 0.550/0.550/0.550/0.525/0.525 |
| NERVE A | fixed 0.5 | 0.830 | 0.729 | fixed-threshold |
| NERVE A | best on set (audit) | 0.832 | 0.747 | in-sample audit: τ VOC-Sim 0.550, COCO-Cap 0.575 |
| NERVE A | own rule | 0.790 | 0.645 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |
| NERVE V | VOC-sel. | 0.823 | 0.741 | τ full-VOC 0.475; OOF folds 0.500/0.500/0.500/0.475/0.450 |
| NERVE V | fixed 0.5 | 0.824 | 0.743 | fixed-threshold |
| NERVE V | best on set (audit) | 0.824 | 0.744 | in-sample audit: τ VOC-Sim 0.475, COCO-Cap 0.525 |
| NERVE V | own rule | 0.799 | 0.687 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |
| NERVE W | VOC-sel. | 0.815 | 0.721 | τ full-VOC 0.425; OOF folds 0.425/0.450/0.425/0.450/0.425 |
| NERVE W | fixed 0.5 | 0.810 | 0.729 | fixed-threshold |
| NERVE W | best on set (audit) | 0.815 | 0.729 | in-sample audit: τ VOC-Sim 0.425, COCO-Cap 0.500 |
| NERVE W | own rule | 0.803 | 0.701 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |

## Official rows, with DenseCRF R2

| Method | Rule | VOC-Sim | COCO-Cap | Note |
|---|---|---:|---:|---|
| BRACE | VOC-sel. | 0.831 | 0.708 | τ full-VOC 0.550; OOF folds 0.550/0.525/0.550/0.550/0.550 |
| BRACE | fixed 0.5 | 0.829 | 0.682 | fixed-threshold |
| BRACE | fixed 0.55 (frozen) | 0.832 | 0.708 | fixed-threshold; BRACE's frozen rule |
| BRACE | best on set (audit) | 0.832 | 0.719 | in-sample audit: τ VOC-Sim 0.550, COCO-Cap 0.600 |
| GoCA SD1.5 | VOC-sel. | 0.753 | 0.599 | τ full-VOC 0.700; OOF folds 0.700/0.700/0.700/0.700/0.700 |
| GoCA SD1.5 | fixed 0.5 | 0.694 | 0.464 | fixed-threshold |
| GoCA SD1.5 | best on set (audit) | 0.753 | 0.621 | in-sample audit: τ VOC-Sim 0.700, COCO-Cap 0.800 |
| GoCA SD1.5 | own rule (offset 0) | 0.694 | 0.464 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SD1.5 | own rule, VOC-sel. offset | 0.753 | 0.599 | offset full-VOC +0.4; OOF folds +0.4/+0.4/+0.4/+0.4/+0.4 |
| GoCA SD1.5 | own rule, best offset on set (audit) | 0.753 | 0.599 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim +0.4, COCO-Cap +0.4 |
| GoCA SDXL | VOC-sel. | 0.797 | 0.606 | τ full-VOC 0.450; OOF folds 0.450/0.450/0.450/0.475/0.450 |
| GoCA SDXL | fixed 0.5 | 0.794 | 0.614 | fixed-threshold |
| GoCA SDXL | best on set (audit) | 0.799 | 0.614 | in-sample audit: τ VOC-Sim 0.450, COCO-Cap 0.500 |
| GoCA SDXL | own rule (offset 0) | 0.794 | 0.615 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SDXL | own rule, VOC-sel. offset | 0.799 | 0.606 | offset full-VOC -0.1; OOF folds -0.1/-0.1/-0.1/-0.1/-0.1 |
| GoCA SDXL | own rule, best offset on set (audit) | 0.799 | 0.615 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim -0.1, COCO-Cap +0.0 |
| GoCA SD2.1 | VOC-sel. | 0.740 | 0.597 | τ full-VOC 0.550; OOF folds 0.550/0.550/0.550/0.550/0.550 |
| GoCA SD2.1 | fixed 0.5 | 0.733 | 0.565 | fixed-threshold |
| GoCA SD2.1 | best on set (audit) | 0.740 | 0.612 | in-sample audit: τ VOC-Sim 0.550, COCO-Cap 0.625 |
| GoCA SD2.1 | own rule (offset 0) | 0.733 | 0.565 | GoCA argmax(class, background + offset), offset 0, no tuning |
| GoCA SD2.1 | own rule, VOC-sel. offset | 0.740 | 0.597 | offset full-VOC +0.1; OOF folds +0.1/+0.1/+0.1/+0.1/+0.1 |
| GoCA SD2.1 | own rule, best offset on set (audit) | 0.740 | 0.611 | in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset VOC-Sim +0.1, COCO-Cap +0.2 |
| DiffSegmenter | VOC-sel. | 0.795 | 0.688 | τ full-VOC 0.350; OOF folds 0.350/0.350/0.350/0.350/0.350 |
| DiffSegmenter | fixed 0.5 | 0.721 | 0.665 | fixed-threshold |
| DiffSegmenter | best on set (audit) | 0.795 | 0.697 | in-sample audit: τ VOC-Sim 0.350, COCO-Cap 0.400 |
| DiffSegmenter | own rule (in-sample bg sweep) | 0.795 | 0.697 | DiffSegmenter's evaluation script: background score swept 0.30–0.74 on the evaluated set (here step 0.025); in-sample, uses labels |
| NERVE A | VOC-sel. | 0.844 | 0.740 | τ full-VOC 0.500; OOF folds 0.500/0.500/0.500/0.500/0.500 |
| NERVE A | fixed 0.5 | 0.844 | 0.740 | fixed-threshold |
| NERVE A | best on set (audit) | 0.844 | 0.752 | in-sample audit: τ VOC-Sim 0.500, COCO-Cap 0.575 |
| NERVE A | own rule | 0.812 | 0.663 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |
| NERVE V | VOC-sel. | 0.839 | 0.747 | τ full-VOC 0.425; OOF folds 0.425/0.450/0.425/0.425/0.425 |
| NERVE V | fixed 0.5 | 0.834 | 0.751 | fixed-threshold |
| NERVE V | best on set (audit) | 0.840 | 0.751 | in-sample audit: τ VOC-Sim 0.425, COCO-Cap 0.500 |
| NERVE V | own rule | 0.821 | 0.705 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |
| NERVE W | VOC-sel. | 0.830 | 0.730 | τ full-VOC 0.400; OOF folds 0.400/0.425/0.425/0.425/0.400 |
| NERVE W | fixed 0.5 | 0.817 | 0.734 | fixed-threshold |
| NERVE W | best on set (audit) | 0.830 | 0.735 | in-sample audit: τ VOC-Sim 0.400, COCO-Cap 0.475 |
| NERVE W | own rule | 0.822 | 0.717 | NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed |

## Paired Δ (method − BRACE), same images, 95% bootstrap interval

Primary: both at VOC-sel. Secondary: each at its fixed default (method 0.5 vs BRACE's frozen 0.55), both at 0.5, and the method's own rule vs BRACE's frozen rule. Raw and DenseCRF rows are separate comparisons.

**Raw (both sides)**

| Method | Method rule | BRACE rule | VOC-Sim Δ | COCO-Cap Δ |
|---|---|---|---|---|
| GoCA SD1.5 | VOC-sel. | VOC-sel. | -0.079 [-0.095, -0.063] | -0.082 [-0.106, -0.058] |
| GoCA SD1.5 | fixed 0.5 | fixed 0.55 (frozen) | -0.148 [-0.167, -0.129] | -0.240 [-0.260, -0.220] |
| GoCA SD1.5 | fixed 0.5 | fixed 0.5 | -0.141 [-0.159, -0.123] | -0.212 [-0.231, -0.192] |
| GoCA SD1.5 | own rule (offset 0) | fixed 0.55 (frozen) | -0.148 [-0.167, -0.129] | -0.240 [-0.260, -0.220] |
| GoCA SD1.5 | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.080 [-0.096, -0.063] | -0.106 [-0.130, -0.082] |
| GoCA SDXL | VOC-sel. | VOC-sel. | -0.031 [-0.043, -0.018] | -0.090 [-0.113, -0.067] |
| GoCA SDXL | fixed 0.5 | fixed 0.55 (frozen) | -0.031 [-0.044, -0.019] | -0.085 [-0.108, -0.061] |
| GoCA SDXL | fixed 0.5 | fixed 0.5 | -0.025 [-0.039, -0.009] | -0.057 [-0.080, -0.034] |
| GoCA SDXL | own rule (offset 0) | fixed 0.55 (frozen) | -0.032 [-0.044, -0.019] | -0.085 [-0.108, -0.061] |
| GoCA SDXL | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.034 [-0.048, -0.022] | -0.097 [-0.118, -0.074] |
| GoCA SD2.1 | VOC-sel. | VOC-sel. | -0.098 [-0.117, -0.080] | -0.107 [-0.129, -0.084] |
| GoCA SD2.1 | fixed 0.5 | fixed 0.55 (frozen) | -0.109 [-0.128, -0.091] | -0.147 [-0.167, -0.125] |
| GoCA SD2.1 | fixed 0.5 | fixed 0.5 | -0.102 [-0.122, -0.084] | -0.118 [-0.139, -0.097] |
| GoCA SD2.1 | own rule (offset 0) | fixed 0.55 (frozen) | -0.109 [-0.128, -0.091] | -0.147 [-0.167, -0.125] |
| GoCA SD2.1 | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.100 [-0.117, -0.082] | -0.117 [-0.139, -0.094] |
| DiffSegmenter | VOC-sel. | VOC-sel. | -0.045 [-0.058, -0.032] | -0.025 [-0.045, -0.005] |
| DiffSegmenter | fixed 0.5 | fixed 0.55 (frozen) | -0.106 [-0.121, -0.089] | -0.040 [-0.065, -0.013] |
| DiffSegmenter | fixed 0.5 | fixed 0.5 | -0.099 [-0.116, -0.081] | -0.011 [-0.037, +0.016] |
| NERVE A | VOC-sel. | VOC-sel. | +0.014 [+0.000, +0.029] | +0.051 [+0.026, +0.079] |
| NERVE A | fixed 0.5 | fixed 0.55 (frozen) | +0.011 [-0.002, +0.025] | +0.037 [+0.013, +0.063] |
| NERVE A | fixed 0.5 | fixed 0.5 | +0.018 [+0.004, +0.032] | +0.065 [+0.042, +0.092] |
| NERVE A | own rule | fixed 0.55 (frozen) | -0.029 [-0.048, -0.012] | -0.047 [-0.076, -0.017] |
| NERVE V | VOC-sel. | VOC-sel. | +0.006 [-0.009, +0.020] | +0.048 [+0.022, +0.076] |
| NERVE V | fixed 0.5 | fixed 0.55 (frozen) | +0.005 [-0.010, +0.019] | +0.051 [+0.025, +0.080] |
| NERVE V | fixed 0.5 | fixed 0.5 | +0.011 [-0.003, +0.026] | +0.079 [+0.052, +0.107] |
| NERVE V | own rule | fixed 0.55 (frozen) | -0.020 [-0.036, -0.003] | -0.006 [-0.031, +0.022] |
| NERVE W | VOC-sel. | VOC-sel. | -0.003 [-0.017, +0.010] | +0.028 [+0.002, +0.057] |
| NERVE W | fixed 0.5 | fixed 0.55 (frozen) | -0.009 [-0.023, +0.005] | +0.037 [+0.009, +0.067] |
| NERVE W | fixed 0.5 | fixed 0.5 | -0.002 [-0.016, +0.012] | +0.065 [+0.036, +0.095] |
| NERVE W | own rule | fixed 0.55 (frozen) | -0.016 [-0.031, -0.001] | +0.009 [-0.016, +0.037] |

**With DenseCRF R2 (both sides)**

| Method | Method rule | BRACE rule | VOC-Sim Δ | COCO-Cap Δ |
|---|---|---|---|---|
| GoCA SD1.5 | VOC-sel. | VOC-sel. | -0.078 [-0.095, -0.060] | -0.110 [-0.134, -0.084] |
| GoCA SD1.5 | fixed 0.5 | fixed 0.55 (frozen) | -0.139 [-0.159, -0.119] | -0.244 [-0.265, -0.222] |
| GoCA SD1.5 | fixed 0.5 | fixed 0.5 | -0.136 [-0.155, -0.116] | -0.218 [-0.239, -0.197] |
| GoCA SD1.5 | own rule (offset 0) | fixed 0.55 (frozen) | -0.139 [-0.159, -0.119] | -0.244 [-0.265, -0.222] |
| GoCA SD1.5 | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.080 [-0.097, -0.062] | -0.110 [-0.134, -0.084] |
| GoCA SDXL | VOC-sel. | VOC-sel. | -0.034 [-0.048, -0.021] | -0.102 [-0.124, -0.078] |
| GoCA SDXL | fixed 0.5 | fixed 0.55 (frozen) | -0.038 [-0.051, -0.025] | -0.094 [-0.118, -0.069] |
| GoCA SDXL | fixed 0.5 | fixed 0.5 | -0.035 [-0.051, -0.020] | -0.067 [-0.091, -0.042] |
| GoCA SDXL | own rule (offset 0) | fixed 0.55 (frozen) | -0.038 [-0.052, -0.025] | -0.093 [-0.118, -0.069] |
| GoCA SDXL | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.034 [-0.046, -0.021] | -0.102 [-0.124, -0.078] |
| GoCA SD2.1 | VOC-sel. | VOC-sel. | -0.091 [-0.110, -0.072] | -0.111 [-0.134, -0.086] |
| GoCA SD2.1 | fixed 0.5 | fixed 0.55 (frozen) | -0.099 [-0.120, -0.080] | -0.143 [-0.165, -0.119] |
| GoCA SD2.1 | fixed 0.5 | fixed 0.5 | -0.096 [-0.117, -0.077] | -0.116 [-0.138, -0.093] |
| GoCA SD2.1 | own rule (offset 0) | fixed 0.55 (frozen) | -0.099 [-0.120, -0.080] | -0.143 [-0.165, -0.119] |
| GoCA SD2.1 | own rule, VOC-sel. offset | fixed 0.55 (frozen) | -0.092 [-0.111, -0.073] | -0.111 [-0.134, -0.086] |
| DiffSegmenter | VOC-sel. | VOC-sel. | -0.036 [-0.049, -0.023] | -0.020 [-0.041, +0.001] |
| DiffSegmenter | fixed 0.5 | fixed 0.55 (frozen) | -0.111 [-0.128, -0.092] | -0.044 [-0.071, -0.015] |
| DiffSegmenter | fixed 0.5 | fixed 0.5 | -0.108 [-0.126, -0.088] | -0.017 [-0.044, +0.013] |
| NERVE A | VOC-sel. | VOC-sel. | +0.012 [-0.002, +0.028] | +0.032 [+0.007, +0.060] |
| NERVE A | fixed 0.5 | fixed 0.55 (frozen) | +0.011 [-0.003, +0.026] | +0.032 [+0.007, +0.060] |
| NERVE A | fixed 0.5 | fixed 0.5 | +0.014 [-0.000, +0.029] | +0.059 [+0.034, +0.086] |
| NERVE A | own rule | fixed 0.55 (frozen) | -0.021 [-0.040, -0.002] | -0.045 [-0.074, -0.012] |
| NERVE V | VOC-sel. | VOC-sel. | +0.008 [-0.007, +0.023] | +0.039 [+0.013, +0.068] |
| NERVE V | fixed 0.5 | fixed 0.55 (frozen) | +0.002 [-0.014, +0.017] | +0.043 [+0.016, +0.073] |
| NERVE V | fixed 0.5 | fixed 0.5 | +0.005 [-0.011, +0.021] | +0.069 [+0.042, +0.099] |
| NERVE V | own rule | fixed 0.55 (frozen) | -0.011 [-0.029, +0.006] | -0.003 [-0.029, +0.027] |
| NERVE W | VOC-sel. | VOC-sel. | -0.002 [-0.017, +0.012] | +0.022 [-0.005, +0.052] |
| NERVE W | fixed 0.5 | fixed 0.55 (frozen) | -0.015 [-0.031, -0.001] | +0.026 [-0.003, +0.057] |
| NERVE W | fixed 0.5 | fixed 0.5 | -0.012 [-0.028, +0.003] | +0.052 [+0.023, +0.084] |
| NERVE W | own rule | fixed 0.55 (frozen) | -0.011 [-0.027, +0.005] | +0.009 [-0.018, +0.038] |

**Sensitivity (VOC-sel. both sides): images whose BRACE regeneration is within 2 / within 1 grey level (mean |Δ|) of the stored PNG**

Subset sizes: VOC-Sim 454 / 431, COCO-Cap 655 / 583.

| Method | CRF | VOC-Sim < 2 | COCO-Cap < 2 | VOC-Sim < 1 | COCO-Cap < 1 |
|---|---|---|---|---|---|
| GoCA SD1.5 | raw | -0.082 [-0.100, -0.065] | -0.088 [-0.111, -0.062] | -0.079 [-0.096, -0.063] | -0.095 [-0.118, -0.066] |
| GoCA SD1.5 | R2 | -0.083 [-0.102, -0.065] | -0.114 [-0.138, -0.089] | -0.081 [-0.099, -0.062] | -0.119 [-0.144, -0.090] |
| GoCA SDXL | raw | -0.034 [-0.046, -0.022] | -0.092 [-0.113, -0.069] | -0.036 [-0.048, -0.022] | -0.094 [-0.117, -0.068] |
| GoCA SDXL | R2 | -0.038 [-0.051, -0.025] | -0.103 [-0.125, -0.080] | -0.040 [-0.053, -0.026] | -0.104 [-0.128, -0.079] |
| GoCA SD2.1 | raw | -0.100 [-0.119, -0.084] | -0.111 [-0.134, -0.085] | -0.098 [-0.116, -0.080] | -0.111 [-0.134, -0.086] |
| GoCA SD2.1 | R2 | -0.093 [-0.112, -0.076] | -0.115 [-0.138, -0.089] | -0.091 [-0.109, -0.073] | -0.113 [-0.138, -0.086] |
| DiffSegmenter | raw | -0.047 [-0.060, -0.035] | -0.030 [-0.049, -0.009] | -0.045 [-0.057, -0.032] | -0.030 [-0.050, -0.009] |
| DiffSegmenter | R2 | -0.039 [-0.052, -0.028] | -0.024 [-0.044, -0.002] | -0.037 [-0.049, -0.025] | -0.024 [-0.045, -0.001] |
| NERVE A | raw | +0.009 [-0.004, +0.022] | +0.048 [+0.023, +0.075] | +0.008 [-0.007, +0.022] | +0.044 [+0.019, +0.073] |
| NERVE A | R2 | +0.006 [-0.008, +0.020] | +0.028 [+0.004, +0.055] | +0.005 [-0.011, +0.019] | +0.027 [+0.002, +0.056] |
| NERVE V | raw | +0.001 [-0.014, +0.014] | +0.045 [+0.020, +0.073] | +0.000 [-0.015, +0.016] | +0.039 [+0.012, +0.070] |
| NERVE V | R2 | +0.002 [-0.012, +0.016] | +0.035 [+0.010, +0.063] | +0.002 [-0.014, +0.016] | +0.031 [+0.004, +0.062] |
| NERVE W | raw | -0.008 [-0.021, +0.006] | +0.024 [-0.002, +0.052] | -0.008 [-0.023, +0.005] | +0.019 [-0.008, +0.051] |
| NERVE W | R2 | -0.008 [-0.022, +0.007] | +0.017 [-0.009, +0.047] | -0.008 [-0.024, +0.006] | +0.014 [-0.014, +0.047] |

## Runtime

BRACE regeneration + capture + hierarchy (this run, Tesla V100-PCIE-16GB): median 17.3 s / image VOC-Sim, 17.3 s / image COCO-Cap; total 5.73 GPU-h; peak allocated 5.3 GiB. N methods: see Entry 4 (A100, per-item seconds in `outputs/segmentation/comparisons/n_side/<run>/per_item_counts.csv`). Different GPUs; not a cost comparison.

