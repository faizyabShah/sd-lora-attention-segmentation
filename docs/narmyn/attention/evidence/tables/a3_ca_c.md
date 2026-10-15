**CA-C: the combined confirmation (one run, in-sample values; fold votes from the 5-fold selection)**

| Configuration | mIoU@0.45 | Δ vs reference | mIoU@0.35 | mIoU@0.55 | Mean AP | Fold votes |
|---|---:|---:|---:|---:|---:|---|
| reference (16 modules, 8 heads) | 0.6653 | 0.0000 | 0.7057 | 0.5430 | 0.9218 | 0/5 |
| CA-R: without 32×32 (11 modules) | 0.6956 | 0.0303 | 0.7067 | 0.5877 | 0.9131 | 4/5 |
| CA-L: without modules 10, 11 | 0.6921 | 0.0267 | 0.7035 | 0.5938 | 0.9201 | 1/5 |
| CA-H: without head 1 | 0.6761 | 0.0108 | 0.7056 | 0.5623 | 0.9205 | 0/5 |
| CA-R + CA-H | 0.6960 | 0.0307 | 0.7058 | 0.5879 | 0.9127 | 0/5 |
| CA-L + CA-H | 0.6924 | 0.0270 | 0.7020 | 0.5948 | 0.9190 | 0/5 |
