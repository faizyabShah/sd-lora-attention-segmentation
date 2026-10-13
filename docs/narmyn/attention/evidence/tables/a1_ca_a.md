**CA-A: the 12 attribution conventions (in-sample; reference = conditional, padded, all)**

| Query branch | Softmax | Class pieces | mIoU@0.45 | Δ vs reference | mIoU@0.35 | mIoU@0.55 | Mean AP |
|---|---|---|---:|---:|---:|---:|---:|
| conditional | unpadded | all | 0.6678 | 0.0018 | 0.7054 | 0.5480 | 0.9245 |
| conditional | padded | all | 0.6661 | 0.0000 | 0.7057 | 0.5435 | 0.9218 |
| unconditional | padded | all | 0.6612 | -0.0048 | 0.6992 | 0.5489 | 0.9222 |
| unconditional | unpadded | all | 0.6610 | -0.0050 | 0.6986 | 0.5506 | 0.9241 |
| conditional | unpadded | last | 0.6597 | -0.0063 | 0.6998 | 0.5414 | 0.9246 |
| conditional | padded | last | 0.6579 | -0.0082 | 0.7000 | 0.5365 | 0.9219 |
| unconditional | padded | last | 0.6541 | -0.0119 | 0.6938 | 0.5436 | 0.9223 |
| unconditional | unpadded | last | 0.6539 | -0.0122 | 0.6935 | 0.5454 | 0.9243 |
| unconditional | unpadded | first | 0.6505 | -0.0156 | 0.6970 | 0.5252 | 0.9127 |
| conditional | unpadded | first | 0.6479 | -0.0182 | 0.6962 | 0.5172 | 0.9107 |
| unconditional | padded | first | 0.6474 | -0.0187 | 0.6967 | 0.5199 | 0.9106 |
| conditional | padded | first | 0.6451 | -0.0210 | 0.6953 | 0.5112 | 0.9080 |
