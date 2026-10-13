# COCO-Cap untuned generalization

Evaluated **707 annotated images** from **20 classes**. All choices in our method were frozen from VOC-Sim before this evaluation.

## Overall results

| Method | mIoU (20-class macro) | Mean image IoU | Overall pixel IoU |
|---|---:|---:|---:|
| OVAM (paper COCO-Cap parameters) | 0.5719 | 0.6041 | 0.5589 |
| Our best configuration (frozen on VOC-Sim) | 0.6080 | 0.6393 | 0.5835 |

## Caption wording split

This split tests whether the method still localizes the annotated category when the exact category word is absent from the generation caption. Both methods receive the category as a separate attribution prompt, matching OVAM's COCO-Cap protocol.

| Method | Exact class word present | Exact class word absent |
|---|---:|---:|
| OVAM (paper COCO-Cap parameters) | 0.5666 (465 images) | 0.5857 (242 images) |
| Our best configuration (frozen on VOC-Sim) | 0.6016 (465 images) | 0.6170 (242 images) |

## Per-class foreground IoU

| Class | Images | OVAM | Frozen best VOC | Difference |
|---|---:|---:|---:|---:|
| aeroplane | 49 | 0.6458 | 0.6266 | -0.0193 |
| bicycle | 46 | 0.6386 | 0.5100 | -0.1286 |
| bird | 48 | 0.5511 | 0.7041 | +0.1531 |
| boat | 32 | 0.5157 | 0.4859 | -0.0298 |
| bottle | 21 | 0.3261 | 0.3683 | +0.0423 |
| bus | 37 | 0.8518 | 0.8502 | -0.0017 |
| car | 28 | 0.4729 | 0.4358 | -0.0371 |
| cat | 46 | 0.6511 | 0.8475 | +0.1964 |
| chair | 20 | 0.2801 | 0.3175 | +0.0374 |
| cow | 48 | 0.7651 | 0.8291 | +0.0639 |
| dining table | 14 | 0.8960 | 0.6191 | -0.2769 |
| dog | 44 | 0.5636 | 0.7448 | +0.1812 |
| horse | 38 | 0.7158 | 0.7053 | -0.0105 |
| motorbike | 41 | 0.6453 | 0.5883 | -0.0569 |
| person | 52 | 0.1941 | 0.2725 | +0.0784 |
| potted plant | 14 | 0.5443 | 0.6462 | +0.1019 |
| sheep | 38 | 0.7386 | 0.8601 | +0.1215 |
| sofa | 34 | 0.3668 | 0.4064 | +0.0396 |
| train | 33 | 0.5467 | 0.7278 | +0.1811 |
| tv | 24 | 0.5280 | 0.6152 | +0.0872 |

## Fixed evaluation settings

- OVAM uses the paper's COCO-Cap self-attention floor of 0.91 and threshold of 0.50, followed by its DenseCRF.
- Our method uses the VOC-selected 12 CA layers and top four CA heads per layer, final 64×64 SA block, four propagation passes, SA heads 0/1/2/4/5/6, denoising calls 15 onward, percentile 1–99 normalization, threshold 0.45, and 1% component removal.
- The generation caption and seed come from the released COCO-Cap data. The class name is encoded separately for attribution, as in OVAM's released COCO-Cap script.
