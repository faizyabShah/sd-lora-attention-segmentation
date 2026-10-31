# SG-A3G

Historical (V) vs GoCA-oriented (G) A3 seed from the same diffusion run; FINE-P2 (0) and V3 (3).
40 class shards, 490 VOC and 707 COCO images. VOC-only selection; COCO thresholds frozen.

| Cell | VOC forced OOF | Δ vs V0 | Fold wins | Full VOC (thr) | Frozen COCO | COCO best-thr Δ | Worst VOC class |
|---|---:|---:|---:|---:|---:|---:|---|
| V0 | 0.8097 | +0.0000 | 0/5 | 0.8113 (0.450) | 0.6600 | +0.0000 | — |
| V0+R2 | 0.8228 | +0.0000 | 0/5 | 0.8242 (0.425) | 0.6623 | +0.0000 | — |
| V3 | 0.8155 | +0.0059 | 5/5 | 0.8173 (0.550) | 0.6821 | +0.0171 | sheep -0.069 |
| V3+R2 | 0.8292 | +0.0063 | 5/5 | 0.8302 (0.550) | 0.6967 | +0.0174 | sheep -0.092 |
| G0 | 0.8098 | +0.0001 | 4/5 | 0.8114 (0.450) | 0.6598 | -0.0007 | chair -0.000 |
| G0+R2 | 0.8228 | -0.0000 | 3/5 | 0.8242 (0.425) | 0.6617 | -0.0007 | train -0.002 |
| G3 | 0.8156 | +0.0060 | 5/5 | 0.8174 (0.550) | 0.6815 | +0.0169 | sheep -0.069 |
| G3+R2 | 0.8304 | +0.0075 | 5/5 | 0.8304 (0.550) | 0.6961 | +0.0173 | sheep -0.091 |

| Paired contrast | VOC forced OOF Δ | Fold wins | Paired mean-image CI |
|---|---:|---:|---|
| G0_vs_V0 | +0.0001 | 4/5 | [-0.0001, +0.0003] |
| G0+R2_vs_V0+R2 | -0.0000 | 3/5 | [-0.0001, +0.0001] |
| G3_vs_V3 | +0.0001 | 4/5 | [-0.0000, +0.0002] |
| G3+R2_vs_V3+R2 | +0.0012 | 4/5 | [+0.0002, +0.0021] |
| G3_vs_G0 | +0.0059 | 5/5 | [+0.0007, +0.0121] |
| G3+R2_vs_G0+R2 | +0.0075 | 5/5 | [+0.0020, +0.0128] |
| V3_vs_V0 | +0.0059 | 5/5 | [+0.0006, +0.0121] |
| V3+R2_vs_V0+R2 | +0.0063 | 5/5 | [+0.0010, +0.0115] |

Control check (Section 107): G0+R2 − V0+R2 = -0.0000 VOC forced OOF; inside the ±0.002 tie band; frozen COCO -0.0006.
Raw selector: V0 {'V0': 4, 'G3': 1}
R2 selector: V0+R2 {'V0+R2': 5}
Seed difference (VOC): {"mean_abs_seed_difference": 0.0008142444199694757, "max_abs_seed_difference": 0.0042032515630126, "seed_pearson_mean": 0.9999925789784412, "seed_pearson_min": 0.9998247027397156, "call_mean_abs_layer_weight_difference": 0.0009725738975471266}

AP, best-threshold, size-bin and per-class values are in results.json.
