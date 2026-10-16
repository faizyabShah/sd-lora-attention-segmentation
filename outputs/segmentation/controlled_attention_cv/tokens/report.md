# Controlled propagation package

490 images, 20 classes, 27 configurations, 17 thresholds.
Method and threshold are selected on four folds and evaluated on the held-out fold.

| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Full threshold | Full mIoU | Why |
|---|---|---:|---:|---:|---|---:|---:|---|
| B1 released token, OVAM form (unconditional query) | `TU/REF` | 5/5 | 0.7873 → 0.8428 | +0.0555 | `TU/REF` | 0.475 | 0.8435 | fold vote and OOF guard passed |
| B1 released token, conditional query | `TC/REF` | 5/5 | 0.7873 → 0.8507 | +0.0634 | `TC/REF` | 0.475 | 0.8507 | fold vote and OOF guard passed |

## B1 released token, OVAM form (unconditional query)

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `TU/REF` | 0.475 | 0.425 | 0.8468/0.7900 | 0.8281/0.7854 | promote |
| 1 | `TU/REF` | 0.475 | 0.425 | 0.8370/0.7824 | 0.8658/0.8115 | promote |
| 2 | `TU/REF` | 0.500 | 0.425 | 0.8438/0.7900 | 0.8426/0.7859 | promote |
| 3 | `TU/REF` | 0.475 | 0.425 | 0.8448/0.7911 | 0.8371/0.7802 | promote |
| 4 | `TU/REF` | 0.500 | 0.450 | 0.8448/0.7908 | 0.8330/0.7752 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `TU/REF` | 0.8402 |

## B1 released token, conditional query

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `TC/REF` | 0.475 | 0.425 | 0.8538/0.7900 | 0.8366/0.7854 | promote |
| 1 | `TC/REF` | 0.475 | 0.425 | 0.8448/0.7824 | 0.8695/0.8115 | promote |
| 2 | `TC/REF` | 0.475 | 0.425 | 0.8500/0.7900 | 0.8531/0.7859 | promote |
| 3 | `TC/REF` | 0.475 | 0.425 | 0.8526/0.7911 | 0.8426/0.7802 | promote |
| 4 | `TC/REF` | 0.475 | 0.450 | 0.8515/0.7908 | 0.8420/0.7752 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `TC/REF` | 0.8480 |
