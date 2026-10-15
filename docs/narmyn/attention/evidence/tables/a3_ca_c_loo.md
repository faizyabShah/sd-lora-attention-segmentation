**CA-C leave-one-out of the combined candidate (in-sample)**

| Combined candidate | mIoU@0.45 | Δ vs combined | Reading |
|---|---:|---:|---|
| combined CA-R + CA-H | 0.6960 | 0.0000 | — |
| without CA-R (= CA-L + CA-H) | 0.6924 | -0.0036 | CA-R contributes |
| without CA-H (= CA-R alone) | 0.6956 | -0.0004 | within the 0.002 tie: CA-H is not kept |
| without CA-L | 0.6960 | 0.0000 | CA-L is contained in CA-R |
