# Comparison-method reruns (side N, JC review workplan)

Side N's reruns of published training-free methods for JC item #3 and #5
(`docs/JC_REVIEW_WORKPLAN.md`). BRACE code is not touched here.

## What is here

| Folder | Content |
|---|---|
| `goca/` | GoCA (Meng et al., CVPR 2026) class-word driver, full-run driver and SLURM script, manifest builder, pre-registration for the backbones |
| `nerve/` | NERVE (Mahatha et al., WACV 2026) class-word driver, the patch applied to its released code, SLURM script, pre-registration, annotation-gap screen |
| `diffsegmenter/` | DiffSegmenter (Wang et al., TIP 2025) class-word driver, SLURM script, environment freeze |
| `manifests/` | The 490 VOC-Sim and 707 COCO-Cap items every method was run on (stored OVAM benchmark images, no regeneration) |
| `notes/` | Long-form notes behind the workplan entries (#2 mapping, #4 token provenance, #6 consistency) |
| `verify_runs.py` | Independent rescoring of every saved map (missing/NaN/duplicate maps, leakage check, centre-prior control) |
| `export_for_repo.py` | Writes the per-item counts in `outputs/segmentation/comparisons/n_side/` |

Results: `outputs/segmentation/comparisons/n_side/<run>/` with `sanity_summary.json` and
`per_item_counts.csv` (TP/FP/FN at the fixed 0.5 threshold, runtime, peak memory per item).

## External code (not vendored)

| Method | Repository | Commit | Changes |
|---|---|---|---|
| GoCA | github.com/Darkbblue/goca | `475892f` | none; installed with its `install/components` copied into generic-diffusion-feature (github.com/Darkbblue/generic-diffusion-feature `adf80e1`), as its README says |
| NERVE | github.com/kunal-mahatha/nerve | `1ddc5f9` | `nerve/repo_patch.diff`: K=1 guard (variant A), SD 1.5 head size, offline model id; listed as deviations in `nerve/PREREG_sd15.md` |
| DiffSegmenter | github.com/vcg-team/diffsegmenter | `b3a8686` | none (model id only, in the driver) |
| SPARK | none released | — | no-go: no public code, and the method takes no class query |

The drivers expect the clones at `~/TRDP/external/<repo>` and write maps to a local, git-ignored
folder. Soft maps (float16, 512×512) are kept locally and are not committed.

## Status of the numbers

All numbers here are **sanity numbers at a fixed threshold of 0.5 on the min-max map, without
DenseCRF**. They are not the official comparison: official scores use Entry 1's shared scorer,
VOC-Sim-selected thresholds and the frozen DenseCRF, once Entry 1 exists.
