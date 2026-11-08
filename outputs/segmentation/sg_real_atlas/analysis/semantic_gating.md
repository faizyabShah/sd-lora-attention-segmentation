# Semantic gating inside the capped hierarchy (real VOC D, exploratory)

307 pairs, 200 photos. Definitions and pass rules were fixed before the run (see `scripts/sg_sem_gate.py` docstring): gate g = (c+ε)/(c+b+2ε), c = target A3 seed, b = max of the 8 generic-word seeds (each min-max), λ = 1. `mid` gates the 16 px map that feeds A32, the V3 cap and the 16→64 C16 path; `out` multiplies the final maps by g; `shift` is `mid` with the gate rolled by (8, 8); `comp` = marg.5/mm/gen seed. OOF τ over 5 photo-grouped folds; Δ CIs photo-bootstrap 95%. Max replay |Δ| vs the stored maps: 1.3e-03.

| variant | V3+R2 OOF | C16 OOF | V3 OOF | P1+R2 OOF | V3+R2 prec / rec / FP-on-other | V3 AUROC tgt-other / tgt-bg | C16 AUROC tgt-other / tgt-bg |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.538 (τ≈0.70) | 0.553 (τ≈0.68) | 0.533 (τ≈0.72) | 0.547 (τ≈0.60) | 0.528 / 0.815 / 45% | 0.665 / 0.939 | 0.688 / 0.944 |
| comp | 0.601 (τ≈0.60) | 0.596 (τ≈0.50) | 0.597 (τ≈0.60) | 0.574 (τ≈0.40) | 0.567 / 0.829 / 49% | 0.729 / 0.943 | 0.745 / 0.945 |
| out | 0.534 (τ≈0.60) | 0.537 (τ≈0.60) | 0.524 (τ≈0.62) | 0.558 (τ≈0.50) | 0.515 / 0.798 / 41% | 0.765 / 0.941 | 0.765 / 0.942 |
| mid | 0.553 (τ≈0.65) | 0.568 (τ≈0.62) | 0.555 (τ≈0.68) | 0.547 (τ≈0.60) | 0.529 / 0.833 / 44% | 0.716 / 0.945 | 0.741 / 0.947 |
| comp+out | 0.569 (τ≈0.50) | 0.567 (τ≈0.47) | 0.564 (τ≈0.53) | 0.565 (τ≈0.30) | 0.514 / 0.835 / 42% | 0.774 / 0.945 | 0.777 / 0.945 |
| comp+mid | 0.592 (τ≈0.55) | 0.595 (τ≈0.47) | 0.592 (τ≈0.55) | 0.574 (τ≈0.40) | 0.561 / 0.824 / 48% | 0.745 / 0.945 | 0.761 / 0.946 |
| shift | 0.235 (τ≈0.55) | 0.243 (τ≈0.53) | 0.239 (τ≈0.57) | 0.547 (τ≈0.60) | 0.263 / 0.584 / 19% | 0.629 / 0.721 | 0.620 / 0.735 |

## Registered comparisons

| test | comparison | map | Δ | 95% CI | pass |
|---|---|---|---:|---|---|
| M | mid − out | V3+R2 | +0.019 | -0.005…+0.050 | no |
| M | mid − out | C16 | +0.030 | +0.014…+0.048 | yes |
| M | comp+mid − comp+out | V3+R2 | +0.024 | +0.009…+0.036 | yes |
| M | comp+mid − comp+out | C16 | +0.029 | +0.017…+0.040 | yes |
| X | mid − shift | V3+R2 | +0.318 | +0.274…+0.358 | yes |
| X | mid − shift | C16 | +0.325 | +0.283…+0.363 | yes |
| P | best gated (comp+mid C16) − comp C16 (registered) | — | -0.000 | -0.007…+0.007 | no |
| P | best gated (comp+mid C16) − comp V3+R2 | — | -0.005 | -0.022…+0.014 | no |

**Mechanism (M):** PASS. **Spatial control (X):** PASS. **Practical (P, vs comp/C16):** FAIL. **Line continues:** no — close.

## Per-class V3+R2 OOF IoU (base → comp → mid → comp+mid)

aeroplane 0.69→0.80→0.70→0.80, bicycle 0.34→0.32→0.32→0.32, bird 0.42→0.74→0.52→0.69, boat 0.62→0.71→0.69→0.72, bottle 0.21→0.20→0.21→0.20, bus 0.69→0.70→0.68→0.67, car 0.40→0.42→0.37→0.41, cat 0.88→0.92→0.87→0.90, chair 0.26→0.27→0.26→0.26, cow 0.79→0.86→0.85→0.83, dining-table 0.56→0.65→0.60→0.68, dog 0.67→0.72→0.68→0.70, horse 0.53→0.68→0.52→0.66, motorbike 0.69→0.82→0.73→0.82, person 0.33→0.34→0.33→0.33, potted-plant 0.15→0.13→0.15→0.13, sheep 0.75→0.82→0.81→0.82, sofa 0.50→0.49→0.53→0.53, train 0.79→0.83→0.75→0.81, tv 0.48→0.58→0.49→0.56

## By object size (V3+R2 OOF predictions, class-macro within bin)

| bin | n | base | comp | out | mid | comp+out | comp+mid | shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <5% | 101 | 0.140 | 0.224 | 0.203 | 0.157 | 0.242 | 0.227 | 0.047 |
| 5–20% | 104 | 0.550 | 0.603 | 0.553 | 0.548 | 0.571 | 0.591 | 0.222 |
| >20% | 102 | 0.781 | 0.772 | 0.701 | 0.790 | 0.742 | 0.766 | 0.367 |
