# SG16 onward: shared result evidence

This index distinguishes completed result packages from proposed experiments.
Claude can use the paths below to audit both aggregate scores and compact
per-image evidence. The per-image files contain confusion counts and reduced
diagnostics, not generated images or attention maps.

| Study | Aggregate report | Per-image evidence |
|---|---|---|
| Pre-SG16/17 diagnostics | `outputs/segmentation/analysis/pre_sg16_sg17/results.json` | 1,197 compact `voc/shards/` and `coco/shards/` region/histogram records. |
| SG15 oracle visual audit (Section 93) | `outputs/segmentation/analysis/sg15_label_oracle_20/results.json` | `per_image_iou.csv` and 20 presentation sheets in `figures/`. |
| SG17 | `outputs/segmentation/controlled_attention_cv/sg17/results.json` | `sg17/per_image_selected.csv` and 1,197 compact `sg17/screen/{voc,coco}/*.npz` scoring records. |
| SG18 temporal windows | `outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed/results.json` | `sg18_temporal_windowed/per_image_selected.csv` and 1,197 compact `screen/{voc,coco}/*.npz` records. |
| SG18 T1 + CRF | `outputs/segmentation/controlled_attention_cv/sg18_t1_crf/results.json` | 40 compact `voc/shards/` and `coco/shards/` metric files. |
| Pre-SG18/21 diagnostics | `outputs/segmentation/analysis/pre_sg18_sg21/` | 1,197 compact `voc/shards/` and `coco/shards/` descriptor/histogram records. |
| SG18 late-call diagnostic | `outputs/segmentation/analysis/sg18_late_diagnostic/` | 1,197 small `per_image/{voc,coco}/*.json` event records. |
| Pre-SG19 diagnostics | `outputs/segmentation/analysis/pre_sg19_*/` | Published reduced tables; heavy map caches remain local. |
| SG21 | `outputs/segmentation/controlled_attention_cv/sg21_structural/results.json` | 1,197 compact `sg21_structural/cache/{voc,coco}/class_*/*.npz` count/descriptor records. |
| SG22/23 | `outputs/segmentation/controlled_attention_cv/sg22/` and `sg23/` | 40 class shards per study, each with a metric and diagnostic file in `voc/shards/` or `coco/shards/`. |
| SG24-A/B | `outputs/segmentation/controlled_attention_cv/sg24_a/` and `sg24_b/` | 40 class shards per study, each with metrics and reduced CA layer-weight descriptors. |
| SG25 and Pre-SG26 | `outputs/segmentation/controlled_attention_cv/sg25/` | 40 class shards, each with a metric and arrival file; `first_arrival.csv` and `caption_presence_per_image.csv` are also shared. |
| SG26 | `outputs/segmentation/controlled_attention_cv/sg26/` | 40 class shards, each with a metric and diagnostic file. |
| SG27 | `configs/sg27_unique.json` and `outputs/segmentation/controlled_attention_cv/sg27/duplicate_scan.json` | Production array 236777 and merge 236778 are pending. Publish complete per-image metric shards and summaries after the merge passes. |

The SG16 and SG19 labels referred to proposals/diagnostics; no completed
standalone SG16 or SG19 method result package exists. See
`docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` for experimental locks,
job IDs, and review ownership. Large generation/attention caches remain
ignored because the compact score records suffice to reproduce the reported
selection and diagnostics.
