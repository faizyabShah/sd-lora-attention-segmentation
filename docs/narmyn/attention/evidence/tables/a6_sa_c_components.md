**SA-C: removing one component from the full combination (in-sample)**

| Component | With | Without | Δ mIoU@0.45 |
|---|---|---|---:|
| T within T+L+H | T + L + H | L + H | -0.0046 |
| L within T+L+H | T + L + H | T + H | 0.0093 |
| H within T+L+H | T + L + H | T + L | 0.0005 |
| R within T+R+H | T + R + H | T + H | -0.0040 |
