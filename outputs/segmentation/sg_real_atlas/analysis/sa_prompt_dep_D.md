# Does C16 depend on the prompt word of the SD run? (D, dev check, 2026-11-09)

`scripts/sg_sa_prompt_dep.py`. Scope: the 77 D photos with ≥2 annotated classes (184 pairs). For each pair, the SCLIP VOC21 seed of class k (mm, frozen clip C16, τ 0.575) is propagated two ways:
- through the stored SA of k's own run ("a photograph of a k");
- through the SA of every other annotated class j's run on the same photo.

The batched multi-channel C16 matches `sg27_fusion.base_paths` (max rel err 3.4e-7).

| SA source | class-macro IoU |
|---|---:|
| own prompt | 0.4949 |
| other class's prompt | 0.4877 |

The mean pair Δ is −0.008; the median |Δ| is 0.0045 and the 90th percentile is 0.053.

**Conclusion.** The SA graphs are nearly prompt-independent for propagation. One shared SD run per photo (the prompt word is SCLIP's top class) is a faithful approximation for the multi-class protocol. It costs about 0.01 against the per-class runs.
