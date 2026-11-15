# Regeneration pilot: does attention-constrained regeneration rank masks by correctness? (exploratory)

Plan and pass bars: `PILOT.md`. 20 images. Intervention from step 8, class-token floor at the 90% quantile, conditional branch only. D = whole-image squared error vs the unconstrained replay.

| set | class | object area | replay D | IoU @0.55 | selected τ → IoU | oracle τ IoU | ρ(D, IoU) over τ | ρ(resid D, IoU) all | gt beats: gt_shift gt_flip gt_dil8 gt_ero8 gt_dil16 gt_ero16 empty full |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| VOC-Sim | bottle | 0.198 | 2.6e-06 | 0.553 | 0.75 → 0.507 | 0.886 | -0.21 | -0.21 | ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ |
| VOC-Sim | cat | 0.658 | 5.3e-06 | 0.961 | 0.75 → 0.628 | 0.961 | +0.04 | -0.21 | · ✓ ✓ · ✓ · ✓ · |
| VOC-Sim | dining-table | 0.244 | 3.9e-06 | 0.446 | 0.60 → 0.477 | 0.801 | -0.07 | +0.18 | · · · ✓ · ✓ ✓ · |
| VOC-Sim | dog | 0.208 | 3.9e-06 | 0.899 | 0.50 → 0.898 | 0.899 | -0.11 | -0.20 | ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ |
| VOC-Sim | horse | 0.688 | 1.4e-06 | 0.515 | 0.50 → 0.674 | 0.949 | -0.89 | +0.09 | ✓ ✓ · ✓ · ✓ ✓ ✓ |
| VOC-Sim | motorbike | 0.329 | 7.2e-06 | 0.878 | 0.75 → 0.797 | 0.882 | +0.50 | +0.56 | ✓ ✓ · · · · · ✓ |
| VOC-Sim | potted-plant | 0.512 | 0.00021 | 0.367 | 0.75 → 0.175 | 0.676 | +0.75 | -0.24 | · ✓ ✓ ✓ ✓ ✓ · ✓ |
| VOC-Sim | train | 0.017 | 2.1e-05 | 0.145 | 0.55 → 0.145 | 0.453 | -0.50 | +0.42 | · · ✓ ✓ ✓ · · ✓ |
| VOC-Sim | tv | 0.430 | 7.1e-06 | 0.956 | 0.60 → 0.886 | 0.980 | +0.04 | -0.28 | ✓ ✓ ✓ · ✓ · ✓ ✓ |
| VOC-Sim | boat | 0.129 | 3.6e-06 | 0.891 | 0.55 → 0.891 | 0.902 | -0.68 | -0.35 | ✓ ✓ ✓ ✓ · ✓ ✓ ✓ |
| COCO-Cap | sheep | 0.044 | 4.2e-06 | 0.270 | 0.50 → 0.164 | 0.932 | -0.11 | -0.72 | ✓ ✓ · ✓ · ✓ ✓ ✓ |
| COCO-Cap | bottle | 0.102 | 1.4e-06 | 0.272 | 0.75 → 0.307 | 0.307 | -0.32 | -0.56 | ✓ ✓ ✓ · · · ✓ · |
| COCO-Cap | aeroplane | 0.024 | 4.2e-06 | 0.093 | 0.50 → 0.090 | 0.938 | +0.68 | -0.27 | ✓ ✓ ✓ ✓ · ✓ ✓ · |
| COCO-Cap | cow | 0.250 | 1.7e-06 | 0.979 | 0.45 → 0.974 | 0.980 | +0.00 | -0.28 | ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ |
| COCO-Cap | train | 0.424 | 2.4e-05 | 0.938 | 0.75 → 0.880 | 0.975 | -0.43 | -0.12 | ✓ ✓ ✓ · ✓ · · ✓ |
| COCO-Cap | person | 0.099 | 2.1e-06 | 0.902 | 0.35 → 0.804 | 0.913 | +0.29 | -0.50 | ✓ ✓ · ✓ · ✓ ✓ ✓ |
| COCO-Cap | sofa | 0.066 | 5.4e-05 | 0.247 | 0.60 → 0.293 | 0.781 | -0.29 | +0.30 | · ✓ ✓ · · · ✓ ✓ |
| COCO-Cap | bicycle | 0.116 | 6.9e-06 | 0.383 | 0.75 → 0.689 | 0.689 | -0.75 | -0.42 | ✓ ✓ · · · · ✓ · |
| COCO-Cap | bird | 0.039 | 1.3e-06 | 0.929 | 0.45 → 0.895 | 0.935 | +0.29 | -0.74 | ✓ ✓ · ✓ · ✓ ✓ · |
| COCO-Cap | motorbike | 0.164 | 6.8e-06 | 0.905 | 0.75 → 0.786 | 0.910 | +0.11 | -0.33 | ✓ ✓ ✓ · ✓ ✓ ✓ ✓ |

gt beats each control (of 20): gt_shift 15, gt_flip 18, gt_dil8 13, gt_ero8 12, gt_dil16 9, gt_ero16 12, empty 16, full 14

median ρ(D, IoU) over τ masks -0.09; median ρ(resid D, IoU) -0.26; selected τ vs 0.55: mean IoU gain -0.029, better on 5/20

## Pass bars (fixed in PILOT.md before the run)

- FAIL: 1. gt beats shift, flip, dil8, ero8 on >= 15/20 each
- FAIL: 2. median within-image Spearman(D, IoU) over the 7 tau masks <= -0.5
- FAIL: 3. selected tau beats 0.55: mean IoU gain > 0 and strictly better on >= 12/20
- FAIL: 4. median Spearman(D residual after area + KL, IoU) over all candidates <= -0.3

**Verdict: not viable as specified — close without tuning**

Figures: `fig/scatter.png` (IoU vs D per image; blue τ masks, green gt, orange gt controls, grey empty/full), `fig/regen_examples.jpg` (regenerated images under several masks; red = the imposed mask).
