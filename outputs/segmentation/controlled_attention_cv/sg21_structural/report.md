# SG21 structural head weighting

Frozen: SG18/T1; promoted: False; fold choices: ['SG18/T1', 'SG18/T1', 'SG18/T1', 'SG18/T1', 'SG18/T1']
Reference T1 OOF mIoU: 0.7939

| Method | VOC forced OOF | Δ vs T1 | Fold votes | Full VOC | VOC threshold | Frozen COCO |
|---|---:|---:|---:|---:|---:|---:|
| SG18/T1 | 0.7939 | +0.0000 | 5/5 | 0.7959 | 0.425 | 0.6285 |
| SG21-LM | 0.7884 | -0.0055 | 0/5 | 0.7903 | 0.400 | 0.6193 |
| SG21-LM-RANK | 0.7901 | -0.0039 | 0/5 | 0.7918 | 0.400 | 0.6200 |
| SG21-ENT | 0.7872 | -0.0067 | 0/5 | 0.7892 | 0.400 | 0.6201 |
| SG21-DIST | 0.7941 | +0.0002 | 0/5 | 0.7941 | 0.425 | 0.6287 |
| W/P1 (historical) | 0.7997 | — | — | 0.8006 | 0.450 | 0.6245 |

W/P1 is external context from SG18, generated in a separate run. Full VOC and COCO precision/recall, pixel and mean-image IoU, all 20 class IoUs/deltas, 20,000-resample paired intervals and better/worse/tied image counts are in `results.json`. Fixed-0.45 CA-relative events, per-call effective head counts, winning physical heads, weight spread and any promoted weight-quartile diagnostics are also there.
