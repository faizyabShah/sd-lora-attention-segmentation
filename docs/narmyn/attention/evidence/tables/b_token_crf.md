**OVAM optimized token and DenseCRF on the earlier final method (exploratory)**

| Dataset | Configuration | mIoU |
|---|---|---:|
| VOC-Sim | ovam_implemented | 0.6945 |
| VOC-Sim | plain_top4/sa_iter1 | 0.7802 |
| VOC-Sim | plain_top4/sa_iter1+crf | 0.7957 |
| VOC-Sim | opt_top4/sa_iter1 | 0.8347 |
| VOC-Sim | opt_top4/sa_iter1+crf | 0.8525 |
| VOC-Sim | opt_all16/sa_iter1+crf | 0.8553 |
| VOC-Sim | plain_top4/sa_iter4 | 0.8000 |
| VOC-Sim | opt_top4/sa_iter4+crf | 0.8521 |
| COCO-Cap | ovam_implemented | 0.5717 |
| COCO-Cap | plain_top4/sa_iter1 | 0.6264 |
| COCO-Cap | plain_top4/sa_iter1+crf | 0.6432 |
| COCO-Cap | opt_top4/sa_iter1 | 0.6419 |
| COCO-Cap | opt_top4/sa_iter1+crf | 0.6612 |
| COCO-Cap | opt_all16/sa_iter1+crf | 0.6552 |
| COCO-Cap | plain_top4/sa_iter4 | 0.6077 |
| COCO-Cap | opt_top4/sa_iter4+crf | 0.6369 |
| VOC-Sim | released OVAM token-optimized masks | 0.8247 |
| COCO-Cap | OVAM token-optimized (paper) | 0.6920 |
