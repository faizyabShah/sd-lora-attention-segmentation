**A12: which attention objective reproduces OVAM's released optimization (protocol Section 69)**

| Attention objective (unconditional, no middle block) | Epoch-0 loss vs OVAM, median abs. diff. (57 images) | Token cosine to OVAM after 800 epochs (6 images) |
|---|---|---:|
| token softmax (OVAM's library default) | 1.8e-06 | 0.9961 |
| linear (B2 recipe) | 1.8e-02 | 0.8731 |
