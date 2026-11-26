# Run 31, Part C: query-space diagnostics on the dev dump (`PREREG.md`)

Regime X (VOC-Sim + COCO-Cap dev), cross-fitted profiles. HGFP = background pixels with NC gate > 0.5; medians over (class, module) cells with [IQR]. Ground truth only labels regions.

## Go / no-go (fixed in PREREG)

- Median negative-pool silhouette at K = 4: 0.136 [0.111, 0.157] (n=10) (rule ≥ 0.10)
- Median dA vs dB matched-centroid cosine at K = 4: 0.979 [0.972, 0.982] (n=10) (rule ≥ 0.90)
- **Structure present: yes**

## C1 are the remaining false positives clustered (vs the true-positive null)?

| K | HGFP silhouette | high-gated object (null) silhouette | HGFP inertia drop | null inertia drop |
|---|---|---|---|---|
| 2 | 0.164 [0.141, 0.209] (n=10) | 0.148 [0.132, 0.197] (n=10) | 0.087 [0.072, 0.105] (n=10) | 0.081 [0.070, 0.114] (n=10) |
| 4 | 0.132 [0.127, 0.156] (n=10) | 0.142 [0.119, 0.194] (n=10) | 0.171 [0.154, 0.221] (n=10) | 0.158 [0.155, 0.209] (n=10) |
| 8 | 0.128 [0.111, 0.153] (n=10) | 0.112 [0.107, 0.133] (n=9) | 0.250 [0.232, 0.327] (n=10) | 0.233 [0.218, 0.240] (n=10) |

## C2 reproducibility

- HGFP clusters, split-half matched cosine (K = 4): 0.960 [0.960, 0.960] (n=1)
- Negative pools, dA vs dB matched cosine (K = 4): 0.979 [0.972, 0.982] (n=10)
- Cross-module ARI of nearest-prototype assignments on shared HGFP pixels (64-px module pairs): 0.046 [0.025, 0.068] (n=14)

## C3 does one negative mean blend several modes?

| K | negative-pool silhouette |
|---|---|
| 2 | 0.170 [0.139, 0.202] (n=10) |
| 4 | 0.136 [0.111, 0.157] (n=10) |
| 8 | 0.109 [0.098, 0.130] (n=10) |

Generic all-pixel pools (MG4), silhouette K = 4: 0.075 [0.066, 0.077] (n=5)

Effect of each gate relative to NC (pooled over all cells): share of HGFP pixels pushed below 0.5 (background removed) and share of object pixels with NC ≥ 0.5 pushed below 0.5 (object wrongly removed).

| Gate | background removed | object wrongly removed |
|---|---:|---:|
| MK2 | 0.210 | 0.188 |
| MK4 | 0.228 | 0.264 |
| MK8 | 0.253 | 0.238 |
| MG4 | 0.859 | 0.758 |
| MS4 | 0.048 | 0.002 |

## C1 vs C0: do the gate's prototypes cover the actual errors?

- Max cosine of each HGFP centroid to the absent-class centroids (MK4): 0.967 [0.943, 0.972] (n=10)
- ... to the generic centroids (MG4): 0.984 [0.974, 0.989] (n=10)
- Share of HGFP pixels closer to some MK4 centroid than to the object profile: 0.099 [0.045, 0.138] (n=10)

## C4 classes harmed vs helped by the gate (pooled over modules)

| Class | object gate median | object g<0.5 | cos(object, negative mean) | cos(object, object profile) | object closer to a negative centroid | background removed by MK4 | object wrongly removed by MK4 |
|---|---:|---:|---:|---:|---:|---:|---:|

Top source classes of the absent-class clusters (K = 4) that absorb object pixels, for the harmed classes (64-px module 0 shown):


Caveat: 'absent' uses the caption, so an unmentioned class may still be visible (likely for person and car).

