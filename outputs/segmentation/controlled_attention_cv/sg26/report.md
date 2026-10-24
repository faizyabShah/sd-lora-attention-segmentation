# SG26 results

A3 CA frozen; same-call coarse gates; VOC five-fold selection; frozen COCO.

## Raw

| Cell | VOC forced OOF | Δ OOF | Held-out wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.8097 | +0.0000 | 0/5 | 0.8114 | 0.450 | 0.6599 |
| S1 | 0.8020 | -0.0077 | 0/5 | 0.8032 | 0.525 | 0.6869 |
| S2 | 0.8089 | -0.0008 | 1/5 | 0.8106 | 0.475 | 0.6695 |
| S3 | 0.7711 | -0.0387 | 0/5 | 0.7725 | 0.450 | 0.6502 |
| S4 | 0.7748 | -0.0349 | 0/5 | 0.7755 | 0.400 | 0.6474 |
| S5 | 0.7999 | -0.0099 | 0/5 | 0.8009 | 0.475 | 0.6701 |
| S6 | 0.8021 | -0.0077 | 0/5 | 0.8021 | 0.450 | 0.6672 |

## After R2

| Cell | VOC forced OOF | Δ OOF | Held-out wins | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| S0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 | 0.425 | 0.6621 |
| S1+R2 | 0.8170 | -0.0058 | 1/5 | 0.8183 | 0.500 | 0.6975 |
| S2+R2 | 0.8217 | -0.0012 | 3/5 | 0.8228 | 0.450 | 0.6717 |
| S3+R2 | 0.7983 | -0.0245 | 0/5 | 0.7992 | 0.450 | 0.6817 |
| S4+R2 | 0.7993 | -0.0235 | 0/5 | 0.7993 | 0.400 | 0.6746 |
| S5+R2 | 0.8185 | -0.0043 | 1/5 | 0.8201 | 0.450 | 0.6821 |
| S6+R2 | 0.8160 | -0.0068 | 1/5 | 0.8193 | 0.425 | 0.6773 |

## Mechanism at fixed raw threshold .45

| Dataset | Cell | TP gained | TP lost | FP gained | FP removed | FP removed / TP lost | FP >16 gained/removed | FP >64 gained/removed | Separate FP gained/removed |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| voc | S3 | 941571 | 1140715 | 2836116 | 770332 | 0.675 | 720363/509570 | 60338/213440 | 5199/37216 |
| voc | S4 | 517723 | 2883124 | 1442741 | 1277711 | 0.443 | 344408/677111 | 32746/302309 | 3930/40530 |
| voc | S5 | 580980 | 347625 | 1239698 | 382213 | 1.099 | 232794/269127 | 32417/121271 | 3348/22980 |
| voc | S6 | 320230 | 696552 | 695174 | 530237 | 0.761 | 140055/319016 | 21766/142179 | 2861/22093 |
| coco | S3 | 607585 | 1660770 | 3096138 | 4347169 | 2.618 | 993352/3700864 | 298090/2525401 | 122775/741488 |
| coco | S4 | 438432 | 2716207 | 1582318 | 5240135 | 1.929 | 516687/4214359 | 216038/2852674 | 136140/812549 |
| coco | S5 | 393023 | 404098 | 1430743 | 1743192 | 4.314 | 387458/1486400 | 141694/1018121 | 53346/400464 |
| coco | S6 | 275883 | 550583 | 825338 | 2013209 | 3.657 | 264887/1659010 | 117481/1133426 | 69296/412411 |

## Edge diagnostics

Per-image means on a deterministic balanced 64-foreground/64-background query/key sample; ratios compare each gated fine edge to its original value.

| Dataset | Coarse gate | Strongly suppressed (<0.5×) | Nearly unchanged (0.9–1.1×) | Row entropy before→after | Effective neighbors before→after |
|---|---:|---:|---:|---:|---:|
| voc | 16 | 0.648 | 0.049 | 6.698→5.954 | 1220.0→654.6 |
| voc | 32 | 0.811 | 0.025 | 6.698→5.535 | 1220.0→504.8 |
| coco | 16 | 0.670 | 0.044 | 6.570→5.707 | 1097.8→531.4 |
| coco | 32 | 0.820 | 0.023 | 6.570→5.294 | 1097.8→399.4 |

All per-class IoUs: `per_class_iou.csv`. Per-class mask-change counts: `mechanism.csv`. Full GT-edge-type, per-class edge and bootstrap diagnostics: `results.json`.
