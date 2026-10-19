# SG9 reciprocal self-attention propagation

VOC-Sim 490 images (selection); COCO-Cap 707 images (transfer only).

| Track | Winner | Votes | OOF reference → selection | Gain | Frozen | COCO frozen/reference |
|---|---|---:|---:|---:|---|---:|
| W | `W/REF` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `W/REF` | 0.6178/0.6178 |
| T | `T/REF` | 5/5 | 0.8508 → 0.8508 | +0.0000 | `T/REF` | 0.6683/0.6683 |

Every cell:

| Cell | VOC threshold | VOC mIoU | VOC R/P | COCO mIoU | COCO R/P |
|---|---:|---:|---:|---:|---:|
| `W/REF` | 0.425 | 0.7888 | 0.906/0.853 | 0.6178 | 0.870/0.660 |
| `W/R1_geometric` | 0.425 | 0.7913 | 0.907/0.855 | 0.6187 | 0.867/0.664 |
| `W/R2_minimum` | 0.425 | 0.7904 | 0.903/0.857 | 0.6197 | 0.860/0.670 |
| `W/R3_soft_gate` | 0.400 | 0.7743 | 0.904/0.838 | 0.6079 | 0.853/0.662 |
| `T/REF` | 0.475 | 0.8508 | 0.942/0.896 | 0.6683 | 0.889/0.726 |
| `T/R1_geometric` | 0.475 | 0.8528 | 0.942/0.898 | 0.6703 | 0.888/0.728 |
| `T/R2_minimum` | 0.475 | 0.8524 | 0.941/0.899 | 0.6713 | 0.886/0.730 |
| `T/R3_soft_gate` | 0.475 | 0.8439 | 0.934/0.897 | 0.6717 | 0.877/0.736 |

AN1-style change/spatial counts and reciprocal-edge mechanism diagnostics are in `summary.json`.
