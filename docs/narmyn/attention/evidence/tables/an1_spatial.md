**AN1 part 2: where each pass's new pixels land (threshold 0.45, annotation resolution, summed over images)**

| Transition | New TP (> 16 px from previous mask) | New FP (separate; > 16 px from object) | FP removed | New FP per new TP | Net per object pixel |
|---|---|---|---|---:|---:|
| VOC-Sim, word, 1→2 | 1.09 M (26%) | 0.70 M (7%; 61%) | 0.62 M | 0.6437 | 0.0159 |
| VOC-Sim, word, 2→3 | 0.46 M (12%) | 0.51 M (8%; 66%) | 0.26 M | 1.1227 | 0.0005 |
| VOC-Sim, token, 1→2 | 0.34 M (15%) | 0.51 M (3%; 54%) | 0.53 M | 1.4793 | 0.0033 |
| COCO-Cap, word, 1→2 | 1.20 M (23%) | 2.73 M (14%; 83%) | 0.61 M | 2.2679 | -0.0251 |
| COCO-Cap, word, 2→3 | 0.44 M (15%) | 1.53 M (18%; 82%) | 0.33 M | 3.5165 | -0.0212 |
| COCO-Cap, token, 1→2 | 0.36 M (18%) | 1.65 M (17%; 83%) | 0.70 M | 4.6210 | -0.0195 |
