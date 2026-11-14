# Idea 2: dose–response segmentation (pre-registered in `PREREG.md`)

Response of each pixel to turning the target token's cross-attention weight to 0.5× and 2× during generation (same seed), in final-latent or decoded-pixel space, from step 0 or 10. BRACE = frozen V3 + R2 from the same process. Class-macro mIoU; OOF = τ chosen on 4 folds (crc32 % 5); Δ with image-bootstrap 95% CI.

## VOC-Sim (490 images; base-vs-live mean |Δpixel| 0.31/255; 158.8 s/image)

| map | OOF + R2 | Δ vs BRACE OOF + R2 | prec / rec | OOF raw | @0.55 + R2 |
|---|---:|---|---|---:|---:|
| BRACE (frozen) | 0.830 | — | 0.890 / 0.910 | 0.816 | 0.830 |
| lat_s0 | 0.154 | -0.677 (-0.696…-0.658) | 0.745 / 0.143 | 0.212 | 0.002 |
| lat_s10 | 0.037 | -0.793 (-0.808…-0.778) | 0.746 / 0.032 | 0.114 | 0.001 |
| pix_s0 | 0.035 | -0.795 (-0.810…-0.779) | 0.792 / 0.032 | 0.123 | 0.001 |
| pix_s10 | 0.001 | -0.829 (-0.844…-0.816) | 0.859 / 0.001 | 0.031 | 0.000 |

Per class, OOF + R2 (BRACE / lat_s0 / lat_s10 / pix_s0 / pix_s10): aeroplane 0.89 / 0.22 / 0.09 / 0.08 / 0.00, bicycle 0.80 / 0.17 / 0.02 / 0.03 / 0.00, bird 0.92 / 0.19 / 0.07 / 0.05 / 0.00, boat 0.86 / 0.22 / 0.05 / 0.03 / 0.00, bottle 0.88 / 0.15 / 0.04 / 0.05 / 0.00, bus 0.92 / 0.31 / 0.04 / 0.09 / 0.00, car 0.85 / 0.13 / 0.01 / 0.01 / 0.00, cat 0.85 / 0.03 / 0.02 / 0.02 / 0.00, chair 0.79 / 0.21 / 0.02 / 0.06 / 0.00, cow 0.90 / 0.22 / 0.07 / 0.08 / 0.00, dining-table 0.42 / 0.09 / 0.02 / 0.03 / 0.00, dog 0.93 / 0.06 / 0.03 / 0.01 / 0.00, horse 0.90 / 0.10 / 0.03 / 0.03 / 0.00, motorbike 0.87 / 0.21 / 0.03 / 0.01 / 0.00, person 0.81 / 0.01 / 0.00 / 0.01 / 0.00, potted-plant 0.72 / 0.09 / 0.02 / 0.02 / 0.00, sheep 0.75 / 0.11 / 0.03 / 0.03 / 0.00, sofa 0.96 / 0.04 / 0.01 / 0.02 / 0.00, train 0.74 / 0.34 / 0.13 / 0.03 / 0.00, tv 0.85 / 0.17 / 0.01 / 0.03 / 0.00

## COCO-Cap (707 images; base-vs-live mean |Δpixel| 0.60/255; 158.2 s/image)

| map | OOF + R2 | Δ vs BRACE OOF + R2 | prec / rec | OOF raw | @0.55 + R2 |
|---|---:|---|---|---:|---:|
| BRACE (frozen) | 0.707 | — | 0.775 / 0.860 | 0.697 | 0.697 |
| lat_s0 | 0.153 | -0.554 (-0.580…-0.527) | 0.572 / 0.171 | 0.189 | 0.003 |
| lat_s10 | 0.039 | -0.667 (-0.690…-0.644) | 0.664 / 0.039 | 0.100 | 0.001 |
| pix_s0 | 0.039 | -0.668 (-0.690…-0.644) | 0.553 / 0.038 | 0.108 | 0.001 |
| pix_s10 | 0.002 | -0.705 (-0.728…-0.683) | 0.664 / 0.002 | 0.027 | 0.000 |

- token rule **class_word**: 437 images (62%); OOF + R2 on this subset: brace 0.698, lat_s0 0.164, lat_s10 0.039, pix_s0 0.041, pix_s10 0.001
- token rule **fallback**: 270 images (38%); OOF + R2 on this subset: brace 0.713, lat_s0 0.156, lat_s10 0.042, pix_s0 0.035, pix_s10 0.002

Per class, OOF + R2 (BRACE / lat_s0 / lat_s10 / pix_s0 / pix_s10): aeroplane 0.84 / 0.22 / 0.08 / 0.05 / 0.00, bicycle 0.64 / 0.21 / 0.04 / 0.05 / 0.00, bird 0.78 / 0.12 / 0.06 / 0.06 / 0.01, boat 0.62 / 0.24 / 0.08 / 0.05 / 0.00, bottle 0.53 / 0.18 / 0.04 / 0.08 / 0.00, bus 0.93 / 0.29 / 0.05 / 0.08 / 0.00, car 0.78 / 0.21 / 0.03 / 0.04 / 0.00, cat 0.87 / 0.10 / 0.05 / 0.02 / 0.00, chair 0.49 / 0.17 / 0.05 / 0.06 / 0.00, cow 0.86 / 0.18 / 0.05 / 0.04 / 0.00, dining table 0.39 / 0.07 / 0.01 / 0.02 / 0.00, dog 0.75 / 0.07 / 0.03 / 0.01 / 0.00, horse 0.73 / 0.11 / 0.04 / 0.02 / 0.00, motorbike 0.68 / 0.20 / 0.04 / 0.03 / 0.00, person 0.27 / 0.03 / 0.01 / 0.01 / 0.00, potted plant 0.80 / 0.06 / 0.01 / 0.01 / 0.00, sheep 0.87 / 0.18 / 0.05 / 0.07 / 0.00, sofa 0.72 / 0.06 / 0.01 / 0.00 / 0.00, train 0.78 / 0.24 / 0.05 / 0.05 / 0.00, tv 0.82 / 0.13 / 0.01 / 0.03 / 0.00

## Registered test

Chosen on VOC-Sim: **lat_s0** (OOF + R2 0.154 vs BRACE 0.830). COCO-Cap: lat_s0 0.153 vs BRACE 0.707, Δ -0.554 (CI -0.580…-0.527) → **no gain over BRACE**.

## Execution audit

All six shard processes exited with code 0; the report completed. The cache contains exactly the expected 490 VOC-Sim and 707 COCO-Cap image IDs, with finite, nonnegative confusion counts. The experiment script, pre-registration, and tracked files under `src/` were not modified. The missing COCO-Cap manifest was generated locally from `run_coco_cap_transfer.coco_rows`, preserving its 707 records exactly.

The full run used three processes at a time, VOC-Sim followed by COCO-Cap. Peak sampled GPU memory was 10,088 MiB per worker on the NVIDIA A40.

Smoke measurements (two images per set, one process):

| set / image ID | live capture (s) | branches (s) | total (s) | base-vs-live mean absolute pixel difference (/255) |
|---|---:|---:|---:|---:|
| VOC-Sim / ovam-voc-sim-0 | 37.8 | 45.7 | 138.7 | 0.19 |
| VOC-Sim / ovam-voc-sim-1 | 29.9 | 46.5 | 129.4 | 0.19 |
| COCO-Cap / ovam-coco-144317-aeroplane-2313808621 | 33.5 | 46.3 | 134.3 | 0.08 |
| COCO-Cap / ovam-coco-196984-aeroplane-3878063291 | 28.6 | 44.6 | 124.9 | 0.83 |

Peak sampled smoke GPU memory was 10,086 MiB per process on each set.

**Capture-match caveat:** despite the small dataset means reported above, 12/490 VOC-Sim images and 20/707 COCO-Cap images have base-vs-live mean absolute pixel difference greater than 2/255. Maxima are 6.395/255 and 127.963/255. The largest COCO-Cap difference is `ovam-coco-556255-dog-3429633183`; its log contains the Stable Diffusion safety-checker warning that a black image was returned immediately before that image's completion line. Other outliers include `ovam-coco-488074-bicycle-1514556934` (15.411/255) and `ovam-coco-166787-train-1839819348` (8.513/255). These capture discrepancies warrant review when interpreting the comparison. No images were excluded and no post hoc corrections were applied.

The completed-run logs contain no traceback or out-of-memory failure. They also contain the Transformers tokenizer future warning about `clean_up_tokenization_spaces`.
