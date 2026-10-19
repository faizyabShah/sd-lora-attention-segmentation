**A12: retrained tokens in the frozen T-C pipeline (linear-objective variant of OVAM's recipe)**

| Token | Cell | VOC-Sim mIoU (T-C, 0.475) | Images |
|---|---|---:|---:|
| released (OVAM, 2-3 supports) | REL | 0.8653 | 490 |
| support 1 | A1 | 0.8333 | 490 |
| support 2 | A2 | 0.8630 | 490 |
| support 3 (19 classes) | A3 | 0.8451 | 468 |
| supports 1+2 | A12 | 0.8689 | 490 |
| supports 1+3 (19 classes) | A13 | 0.8589 | 468 |
| supports 2+3 (19 classes) | A23 | 0.8713 | 468 |
| all supports | AALL | 0.8684 | 490 |
| fold-safe one-support (B2b) | B2B | 0.8564 | 490 |
