# Run 31, Part C: query-space diagnostics on the dev dump (`PREREG.md`)

Regime X (VOC-Sim + COCO-Cap dev), cross-fitted profiles. HGFP = background pixels with NC gate > 0.5; medians over (class, module) cells with [IQR]. Ground truth only labels regions.

## Go / no-go (fixed in PREREG)

- Median negative-pool silhouette at K = 4: 0.113 [0.097, 0.145] (n=220) (rule ≥ 0.10)
- Median dA vs dB matched-centroid cosine at K = 4: 0.992 [0.983, 0.999] (n=220) (rule ≥ 0.90)
- **Structure present: yes**

## C1 are the remaining false positives clustered (vs the true-positive null)?

| K | HGFP silhouette | high-gated object (null) silhouette | HGFP inertia drop | null inertia drop |
|---|---|---|---|---|
| 2 | 0.148 [0.120, 0.185] (n=216) | 0.148 [0.119, 0.189] (n=216) | 0.083 [0.065, 0.110] (n=216) | 0.081 [0.061, 0.115] (n=216) |
| 4 | 0.129 [0.083, 0.163] (n=216) | 0.122 [0.082, 0.180] (n=216) | 0.165 [0.135, 0.213] (n=216) | 0.166 [0.123, 0.223] (n=216) |
| 8 | 0.118 [0.064, 0.159] (n=211) | 0.117 [0.061, 0.162] (n=211) | 0.241 [0.195, 0.303] (n=216) | 0.237 [0.172, 0.312] (n=216) |

## C2 reproducibility

- HGFP clusters, split-half matched cosine (K = 4): 0.965 [0.931, 0.987] (n=207)
- Negative pools, dA vs dB matched cosine (K = 4): 0.992 [0.983, 0.999] (n=220)
- Cross-module ARI of nearest-prototype assignments on shared HGFP pixels (64-px module pairs): 0.053 [0.024, 0.154] (n=200)

## C3 does one negative mean blend several modes?

| K | negative-pool silhouette |
|---|---|
| 2 | 0.130 [0.108, 0.151] (n=220) |
| 4 | 0.113 [0.097, 0.145] (n=220) |
| 8 | 0.088 [0.073, 0.154] (n=220) |

Generic all-pixel pools (MG4), silhouette K = 4: 0.068 [0.055, 0.087] (n=11)

Effect of each gate relative to NC (pooled over all cells): share of HGFP pixels pushed below 0.5 (background removed) and share of object pixels with NC ≥ 0.5 pushed below 0.5 (object wrongly removed).

| Gate | background removed | object wrongly removed |
|---|---:|---:|
| MK2 | 0.123 | 0.126 |
| MK4 | 0.343 | 0.274 |
| MK8 | 0.476 | 0.382 |
| MG4 | 0.884 | 0.694 |
| MS4 | 0.041 | 0.019 |

## C1 vs C0: do the gate's prototypes cover the actual errors?

- Max cosine of each HGFP centroid to the absent-class centroids (MK4): 0.904 [0.840, 0.977] (n=216)
- ... to the generic centroids (MG4): 0.929 [0.872, 0.993] (n=216)
- Share of HGFP pixels closer to some MK4 centroid than to the object profile: 0.223 [0.083, 0.330] (n=216)

## C4 classes harmed vs helped by the gate (pooled over modules)

| Class | object gate median | object g<0.5 | cos(object, negative mean) | cos(object, object profile) | object closer to a negative centroid | background removed by MK4 | object wrongly removed by MK4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tv | 0.74 | 0.26 | 0.748 | 0.794 | 0.31 | 0.339 | 0.310 |
| train | 0.83 | 0.15 | 0.795 | 0.840 | 0.32 | 0.274 | 0.354 |
| potted plant | 0.83 | 0.22 | 0.793 | 0.828 | 0.26 | 0.327 | 0.247 |
| person | 0.96 | 0.07 | 0.722 | 0.789 | 0.19 | 0.371 | 0.286 |
| dining table | 0.65 | 0.32 | 0.786 | 0.805 | 0.35 | 0.285 | 0.360 |
| aeroplane | 0.87 | 0.10 | 0.780 | 0.852 | 0.20 | 0.355 | 0.314 |

Top source classes of the absent-class clusters (K = 4) that absorb object pixels, for the harmed classes (64-px module 0 shown):

- tv (down_blocks.0.attentions.0.transformer_blocks.0.attn2): object pixels per cluster [6556, 4140, 2024, 1469]; cluster sources [[('dog', 0.11), ('sheep', 0.1), ('person', 0.1)], [('aeroplane', 0.1), ('bicycle', 0.1), ('train', 0.09)], [('bus', 0.12), ('motorbike', 0.09), ('car', 0.09)], [('aeroplane', 0.08), ('motorbike', 0.07), ('person', 0.07)]]
- train (down_blocks.0.attentions.0.transformer_blocks.0.attn2): object pixels per cluster [1185, 1736, 6041, 256]; cluster sources [[('person', 0.08), ('bus', 0.08), ('tv', 0.08)], [('bus', 0.16), ('car', 0.11), ('motorbike', 0.1)], [('bicycle', 0.11), ('cat', 0.08), ('aeroplane', 0.07)], [('dog', 0.09), ('bird', 0.09), ('person', 0.08)]]
- potted plant (down_blocks.0.attentions.0.transformer_blocks.0.attn2): object pixels per cluster [868, 1720, 1085, 4413]; cluster sources [[('bicycle', 0.11), ('aeroplane', 0.11), ('train', 0.09)], [('bus', 0.11), ('cow', 0.09), ('motorbike', 0.09)], [('cat', 0.13), ('sofa', 0.11), ('person', 0.1)], [('sheep', 0.13), ('bird', 0.1), ('dog', 0.09)]]

Caveat: 'absent' uses the caption, so an unmentioned class may still be visible (likely for person and car).

