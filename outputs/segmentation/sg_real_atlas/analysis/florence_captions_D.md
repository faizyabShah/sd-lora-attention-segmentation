# Florence-2 captions as SD prompts on real VOC D (exploratory, 2026-11-09)

`scripts/sg_caption.py`. Captions: Florence-2-large `<CAPTION>` (label-free). Conditions per pair:
- **base**: inversion with "", capture prompt "a photograph of a {cls}" (the stored D pipeline)
- **inv**: inversion with the caption, capture prompt as base
- **cap**: inversion with "", capture prompt = the caption ("{caption}, a {cls}" if the class word is missing)
- **both**: caption for inversion and capture

Pairs: 307; class word already in the caption for 102 (33%). Replay of base vs stored D maps: max 1.3e-03.

OOF class-macro mIoU (τ chosen on 4 folds, applied to the 5th), Δ vs base with photo-bootstrap 95% CI:

| map | base | inv | cap | both | Δ inv | Δ cap | Δ both |
|---|---:|---:|---:|---:|---|---|---|
| A3 seed (BRACE) | 0.488 | 0.474 | 0.488 | 0.476 | -0.014 (-0.021…-0.005) | +0.000 (-0.023…+0.025) | -0.012 (-0.035…+0.015) |
| BRACE V3+R2 | 0.538 | 0.505 | 0.534 | 0.525 | -0.033 (-0.045…-0.020) | -0.004 (-0.034…+0.028) | -0.013 (-0.042…+0.019) |
| A3 → C16 | 0.553 | 0.534 | 0.565 | 0.542 | -0.020 (-0.029…-0.007) | +0.012 (-0.022…+0.051) | -0.012 (-0.041…+0.026) |
| A3 → P1+R2 | 0.547 | 0.523 | 0.552 | 0.537 | -0.024 (-0.034…-0.009) | +0.005 (-0.023…+0.044) | -0.011 (-0.040…+0.028) |
| CLIP seed → C16 | 0.735 | 0.729 | 0.729 | 0.728 | -0.006 (-0.012…+0.001) | -0.006 (-0.016…+0.004) | -0.008 (-0.016…+0.002) |
| CLIP seed → V3+R2 | 0.720 | 0.710 | 0.726 | 0.712 | -0.010 (-0.019…+0.000) | +0.006 (-0.003…+0.014) | -0.008 (-0.016…+0.004) |

Same, split by whether the class word was in the caption (OOF predictions from the full run):

| map | subset | n | base | inv | cap | both |
|---|---|---:|---:|---:|---:|---:|
| A3 seed (BRACE) | in caption | 102 | 0.574 | 0.547 | 0.583 | 0.568 |
| A3 seed (BRACE) | appended | 205 | 0.429 | 0.426 | 0.408 | 0.398 |
| BRACE V3+R2 | in caption | 102 | 0.627 | 0.590 | 0.657 | 0.638 |
| BRACE V3+R2 | appended | 205 | 0.500 | 0.482 | 0.487 | 0.477 |
| A3 → C16 | in caption | 102 | 0.666 | 0.645 | 0.693 | 0.669 |
| A3 → C16 | appended | 205 | 0.506 | 0.496 | 0.498 | 0.482 |
| A3 → P1+R2 | in caption | 102 | 0.649 | 0.611 | 0.664 | 0.637 |
| A3 → P1+R2 | appended | 205 | 0.497 | 0.491 | 0.488 | 0.478 |
| CLIP seed → C16 | in caption | 102 | 0.758 | 0.752 | 0.755 | 0.751 |
| CLIP seed → C16 | appended | 205 | 0.688 | 0.685 | 0.675 | 0.677 |
| CLIP seed → V3+R2 | in caption | 102 | 0.742 | 0.732 | 0.747 | 0.742 |
| CLIP seed → V3+R2 | appended | 205 | 0.680 | 0.677 | 0.684 | 0.668 |

Seed separation (A3 seed, mean over pairs): AUROC target vs other objects / vs background

- base: 0.724 / 0.924
- inv: 0.719 / 0.920
- cap: 0.813 / 0.837
- both: 0.805 / 0.836

Example captions: `2007_000170`: A man holding a glass of beer in a bar.; `2007_000250`: A plate of fruit and a bottle of wine on a table.; `2007_000504`: A white cow standing next to a pile of trash.; `2007_000793`: A statue of a man with a bicycle on a city street.; `2007_000836`: A person on a horse jumping over an obstacle.
