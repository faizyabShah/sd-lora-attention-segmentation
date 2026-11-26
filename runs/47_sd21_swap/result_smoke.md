# Run 47: SD 2.1-base component swap between our method and iSeg (`runs/47_sd21_swap/PREREG.md`)

Run 44's items and protocol (class word given; one mask per primary item). Every cell on SD 2.1-base; seeds from their own pass (O = ours, I = iSeg's category-enhanced cross-attention; L = our MK4-LR gate); refinement V0 (ours) or ERS (iSeg); read-out bin (ours, 0.55) or comp (iSeg's cam_to_label over the photo's classes). Class-macro mIoU; 95% paired photo bootstrap (50).


## 1. Primary contrasts

| # | Contrast | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---|---|---|---|
| Q1 | iSeg refinement on our seed: OL ERS - OL V0 (bin) | -0.026 [-0.269, +0.216] not resolved | -0.057 [-0.078, -0.029] **hurts** | -0.006 [-0.321, +0.007] not resolved |
| Q2 | our gate on iSeg: IL ERS - I ERS (comp) | +0.004 [+0.003, +0.004] **helps** | -0.019 [-0.057, +0.030] not resolved | +0.114 [-0.009, +0.200] not resolved |
| Q3 | seed, iSeg refinement + read-out: OL ERS - I ERS (comp) | -0.076 [-0.104, -0.047] **hurts** | -0.246 [-0.259, -0.229] **hurts** | -0.166 [-0.414, -0.010] **hurts** |
| Q4 | competition for ours: OL V0 comp - OL V0 bin | -0.016 [-0.225, +0.192] not resolved | -0.052 [-0.054, -0.050] **hurts** | -0.130 [-0.170, -0.088] **hurts** |
| Q5 | backbone: ours SD 2.1 - ours SD 1.5 (OL V0 bin) | -0.336 [-0.353, -0.319] **hurts** | +0.050 [+0.011, +0.101] **helps** | -0.194 [-0.361, -0.039] **hurts** |

## 2. Every cell (mIoU)

| Seed | Refinement | Read-out | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---|---|---:|---:|---:|
| O- | V0 | bin | 0.527 | 0.204 | 0.288 |
| O- | V0 | comp | 0.517 | 0.251 | 0.170 |
| O- | ERS | bin | 0.521 | 0.160 | 0.346 |
| O- | ERS | comp | 0.821 | 0.218 | 0.403 |
| OL | V0 | bin | 0.553 | 0.210 | 0.355 |
| OL | V0 | comp | 0.537 | 0.157 | 0.225 |
| OL | ERS | bin | 0.527 | 0.153 | 0.349 |
| OL | ERS | comp | 0.755 | 0.124 | 0.400 |
| I | V0 | bin | 0.741 | 0.333 | 0.565 |
| I | V0 | comp | 0.659 | 0.343 | 0.314 |
| I | ERS | bin | 0.710 | 0.326 | 0.599 |
| I | ERS | comp | 0.831 | 0.371 | 0.566 |
| IL | V0 | bin | 0.764 | 0.354 | 0.602 |
| IL | V0 | comp | 0.680 | 0.332 | 0.319 |
| IL | ERS | bin | 0.656 | 0.301 | 0.581 |
| IL | ERS | comp | 0.835 | 0.351 | 0.679 |

References (run 44, same items): ours SD 1.5 (MK4-LR V0 bin) 0.889/0.160/0.550; NERVE 0.872/0.132/0.377; iSeg released, class given 0.827/0.344/0.563; iSeg released, own labels 0.827/0.176/0.519.


Reproduction: I ERS comp (this transcription, fp32) vs iSeg released, class given (run 44, fp16): PASCAL VOC (20) +0.004 [-0.000, +0.008] | PASCAL Context (59) +0.026 [+0.002, +0.045] | COCO-Object (80) +0.002 [+0.002, +0.003].


## 3. Secondary contrasts

| Contrast | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---|---|---|
| our gate, ours: OL - O- (V0 bin) | +0.026 [+0.021, +0.031] | +0.005 [+0.004, +0.006] | +0.067 [+0.004, +0.087] |
| our gate, ours: OL - O- (ERS comp) | -0.066 [-0.100, -0.032] | -0.094 [-0.106, -0.078] | -0.003 [-0.011, -0.003] |
| seed: I - O- (V0 bin) | +0.214 [+0.133, +0.296] | +0.128 [+0.054, +0.184] | +0.277 [+0.108, +0.305] |
| seed: IL - OL (V0 bin) | +0.211 [+0.123, +0.298] | +0.144 [+0.073, +0.198] | +0.247 [+0.103, +0.293] |
| refinement, iSeg seed: I ERS - I V0 (comp) | +0.172 [+0.034, +0.310] | +0.027 [+0.013, +0.038] | +0.251 [+0.195, +0.251] |
| competition, iSeg: I ERS comp - I ERS bin | +0.121 [-0.027, +0.269] | +0.044 [+0.007, +0.094] | -0.034 [-0.103, +0.126] |
| best ours-gated cell - iSeg released given | +0.007 [+0.003, +0.012] | +0.007 [-0.012, +0.032] | +0.116 [-0.006, +0.203] |
| OL ERS comp - NERVE | -0.116 [-0.307, +0.075] | -0.007 [-0.025, +0.006] | +0.023 [+0.023, +0.135] |
| IL ERS comp - NERVE | -0.037 [-0.200, +0.126] | +0.219 [+0.208, +0.234] | +0.303 [+0.270, +0.540] |

## 4. Subsets (Context / COCO-Object: 20 VOC classes vs the others)

- **context, 20 VOC classes** (3 items): OL|V0|bin 0.376, OL|ERS|comp 0.160, I|ERS|comp 0.384, IL|ERS|comp 0.384, sd15:OL|V0|bin 0.283, NERVE 0.308
- **context, other classes** (4 items): OL|V0|bin 0.085, OL|ERS|comp 0.097, I|ERS|comp 0.361, IL|ERS|comp 0.327, sd15:OL|V0|bin 0.067, NERVE 0.000
- **coco, 20 VOC classes** (2 items): OL|V0|bin 0.472, OL|ERS|comp 0.114, I|ERS|comp 0.835, IL|ERS|comp 0.757, sd15:OL|V0|bin 0.811, NERVE 0.000
- **coco, other classes** (3 items): OL|V0|bin 0.297, OL|ERS|comp 0.543, I|ERS|comp 0.431, IL|ERS|comp 0.641, sd15:OL|V0|bin 0.419, NERVE 0.565

## 5. Per class (OL V0 bin | OL ERS comp | I ERS comp | IL ERS comp | NERVE)


PASCAL VOC (20):

| Class | items | OL|V0|bin | OL|ERS|comp | I|ERS|comp | IL|ERS|comp | NERVE |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 1 | 0.552 | 0.848 | 0.895 | 0.899 | 0.773 |
| train | 1 | 0.554 | 0.663 | 0.767 | 0.770 | 0.970 |

PASCAL Context (59):

| Class | items | OL|V0|bin | OL|ERS|comp | I|ERS|comp | IL|ERS|comp | NERVE |
|---|---:|---:|---:|---:|---:|---:|
| building | 1 | 0.184 | 0.041 | 0.532 | 0.521 | 0.000 |
| dining table | 1 | 0.156 | 0.052 | 0.247 | 0.239 | 0.000 |
| platform | 1 | 0.069 | 0.226 | 0.345 | 0.313 | 0.000 |
| sky | 1 | 0.047 | 0.003 | 0.181 | 0.018 | 0.000 |
| train | 1 | 0.664 | 0.405 | 0.653 | 0.631 | 0.651 |
| tv | 1 | 0.307 | 0.025 | 0.252 | 0.281 | 0.272 |
| window | 1 | 0.041 | 0.120 | 0.384 | 0.454 | 0.000 |

COCO-Object (80):

| Class | items | OL|V0|bin | OL|ERS|comp | I|ERS|comp | IL|ERS|comp | NERVE |
|---|---:|---:|---:|---:|---:|---:|
| frisbee | 1 | 0.533 | 0.885 | 0.726 | 0.766 | 0.816 |
| handbag | 2 | 0.061 | 0.201 | 0.136 | 0.515 | 0.314 |
| person | 2 | 0.472 | 0.114 | 0.835 | 0.757 | 0.000 |

The paper 2 method stays frozen; this is a component analysis, nothing is promoted.
