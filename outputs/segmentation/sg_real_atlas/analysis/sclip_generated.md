# SCLIP vs BRACE on generated images (class-given binary protocol, as in the paper)

BRACE = frozen V3+R2 at τ 0.55 (paper protocol). SCLIP ViT-B/16 (our port, reproduces the repo on VOC val). *bin* = SCLIP's own closed-vocabulary prediction == target class (no τ); *prob* = SCLIP's target-class probability, min-max, thresholded like BRACE (OOF = τ chosen on 4 folds, crc32(image_id) % 5). R2 = the same DenseCRF as BRACE. SCLIP seed → SD = the SCLIP VOC21 target probability used as the seed of the frozen SD paths of the image's own generation run. Δ = row − BRACE, image-bootstrap 95% CI.

## VOC-Sim (490 images)

| pipeline | mIoU | prec / rec | Δ vs BRACE @0.55 | CI |
|---|---:|---|---:|---|
| **BRACE V3+R2 @0.55 (paper)** | **0.830** | 0.890 / 0.910 | — | — |
| BRACE V3+R2 OOF | 0.830 | 0.890 / 0.910 | +0.000 | +0.000…+0.000 |
| SD fine-only P1+R2 OOF | 0.810 | 0.870 / 0.912 | -0.020 | -0.028…-0.012 |
| SCLIP VOC21 vocab, bin | 0.712 | 0.772 / 0.926 | -0.118 | -0.136…-0.103 |
| SCLIP VOC21 vocab, bin + R2 | 0.752 | 0.804 / 0.934 | -0.078 | -0.096…-0.063 |
| SCLIP VOC21 vocab, prob OOF | 0.704 | 0.813 / 0.867 | -0.126 | -0.143…-0.111 |
| SCLIP VOC21 vocab, prob + R2 OOF | 0.749 | 0.850 / 0.882 | -0.081 | -0.098…-0.065 |
| SCLIP COCO-Object vocab, bin | 0.683 | 0.786 / 0.857 | -0.148 | -0.165…-0.132 |
| SCLIP COCO-Object vocab, bin + R2 | 0.722 | 0.819 / 0.867 | -0.108 | -0.126…-0.092 |
| SCLIP COCO-Object vocab, prob OOF | 0.649 | 0.846 / 0.761 | -0.181 | -0.199…-0.163 |
| SCLIP COCO-Object vocab, prob + R2 OOF | 0.692 | 0.889 / 0.777 | -0.138 | -0.158…-0.118 |
| SCLIP seed → SD C16 OOF | 0.824 | 0.885 / 0.923 | -0.006 | -0.020…+0.007 |
| SCLIP seed → SD V3+R2 OOF | 0.845 | 0.892 / 0.934 | +0.015 | +0.001…+0.027 |
| SCLIP seed → SD P1+R2 OOF | 0.819 | 0.880 / 0.928 | -0.011 | -0.027…+0.003 |

Per class (BRACE @0.55 / best SCLIP-only = SCLIP VOC21 vocab, bin + R2 / SCLIP seed → SD C16):

aeroplane 0.88 / 0.71 / 0.86, bicycle 0.79 / 0.65 / 0.78, bird 0.92 / 0.88 / 0.88, boat 0.86 / 0.58 / 0.84, bottle 0.88 / 0.74 / 0.80, bus 0.92 / 0.87 / 0.92, car 0.85 / 0.84 / 0.85, cat 0.85 / 0.85 / 0.91, chair 0.79 / 0.58 / 0.70, cow 0.90 / 0.88 / 0.94, dining-table 0.42 / 0.49 / 0.49, dog 0.93 / 0.86 / 0.91, horse 0.90 / 0.78 / 0.90, motorbike 0.87 / 0.80 / 0.86, person 0.81 / 0.84 / 0.81, potted-plant 0.72 / 0.73 / 0.73, sheep 0.75 / 0.88 / 0.94, sofa 0.96 / 0.80 / 0.87, train 0.74 / 0.68 / 0.84, tv 0.85 / 0.61 / 0.66

Regeneration replay: BRACE V3+R2 @0.55 from the new capture = 0.830 (stored 0.830).

SCLIP runtime: 0.08 s/image for both vocabularies (GPU, excl. CRF).

## COCO-Cap (707 images)

| pipeline | mIoU | prec / rec | Δ vs BRACE @0.55 | CI |
|---|---:|---|---:|---|
| **BRACE V3+R2 @0.55 (paper)** | **0.696** | 0.747 / 0.885 | — | — |
| BRACE V3+R2 OOF | 0.707 | 0.775 / 0.860 | +0.010 | +0.006…+0.015 |
| SD fine-only P1+R2 OOF | 0.683 | 0.750 / 0.860 | -0.013 | -0.026…-0.002 |
| SCLIP VOC21 vocab, bin | 0.574 | 0.710 / 0.797 | -0.122 | -0.148…-0.096 |
| SCLIP VOC21 vocab, bin + R2 | 0.608 | 0.755 / 0.802 | -0.088 | -0.116…-0.059 |
| SCLIP VOC21 vocab, prob OOF | 0.567 | 0.759 / 0.736 | -0.130 | -0.154…-0.103 |
| SCLIP VOC21 vocab, prob + R2 OOF | 0.601 | 0.812 / 0.746 | -0.095 | -0.120…-0.066 |
| SCLIP COCO-Object vocab, bin | 0.523 | 0.745 / 0.695 | -0.174 | -0.201…-0.146 |
| SCLIP COCO-Object vocab, bin + R2 | 0.552 | 0.796 / 0.703 | -0.144 | -0.173…-0.115 |
| SCLIP COCO-Object vocab, prob OOF | 0.537 | 0.781 / 0.662 | -0.159 | -0.185…-0.131 |
| SCLIP COCO-Object vocab, prob + R2 OOF | 0.573 | 0.839 / 0.670 | -0.123 | -0.151…-0.092 |
| SCLIP seed → SD C16 OOF | 0.692 | 0.836 / 0.825 | -0.004 | -0.030…+0.022 |
| SCLIP seed → SD V3+R2 OOF | 0.713 | 0.840 / 0.848 | +0.017 | -0.008…+0.042 |
| SCLIP seed → SD P1+R2 OOF | 0.691 | 0.810 / 0.849 | -0.005 | -0.031…+0.020 |

Per class (BRACE @0.55 / best SCLIP-only = SCLIP VOC21 vocab, bin + R2 / SCLIP seed → SD C16):

aeroplane 0.80 / 0.46 / 0.68, bicycle 0.63 / 0.63 / 0.73, bird 0.78 / 0.70 / 0.80, boat 0.59 / 0.38 / 0.50, bottle 0.50 / 0.42 / 0.53, bus 0.92 / 0.78 / 0.85, car 0.76 / 0.72 / 0.73, cat 0.87 / 0.72 / 0.79, chair 0.47 / 0.53 / 0.66, cow 0.87 / 0.81 / 0.86, dining table 0.45 / 0.54 / 0.47, dog 0.75 / 0.72 / 0.75, horse 0.72 / 0.59 / 0.76, motorbike 0.67 / 0.63 / 0.68, person 0.28 / 0.64 / 0.61, potted plant 0.74 / 0.48 / 0.60, sheep 0.87 / 0.81 / 0.88, sofa 0.70 / 0.63 / 0.75, train 0.75 / 0.60 / 0.65, tv 0.79 / 0.36 / 0.57

Regeneration replay: BRACE V3+R2 @0.55 from the new capture = 0.697 (stored 0.696).

SCLIP runtime: 0.08 s/image for both vocabularies (GPU, excl. CRF).

