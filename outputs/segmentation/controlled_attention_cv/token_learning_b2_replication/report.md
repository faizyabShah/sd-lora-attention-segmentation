# B2 learned-token package

Frozen pipeline: conditional token CA, one SA pass, DenseCRF, threshold 0.475.

## VOC-Sim

| Configuration | mIoU | Images |
|---|---:|---:|
| `REL` | 0.8652 | 490 |
| `A1` | 0.8334 | 490 |
| `A2` | 0.8627 | 490 |
| `A3` | 0.8449 | 468 |
| `A12` | 0.8688 | 490 |
| `A13` | 0.8590 | 468 |
| `A23` | 0.8711 | 468 |
| `AALL` | 0.8685 | 490 |
| `B2B` | 0.8562 | 490 |

### S4 shot scaling

| Supports | mIoU | Range |
|---|---:|---:|
| 1 | 0.8470 | 0.8334–0.8627 |
| 2 | 0.8663 | 0.8590–0.8711 |
| all available | 0.8685 | — |

B2a vector comparison covers 57 released supports: mean cosine 0.8676.

## COCO-Cap transfer

| Configuration | mIoU |
|---|---:|
| `REL` | 0.6850 |
| `AALL` | 0.6926 |
