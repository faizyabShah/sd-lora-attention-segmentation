# Seed × propagation swap between BRACE and NERVE (pre-registration)

Workplan: `docs/JC_REVIEW_WORKPLAN.md`. Requested by the owner (Faizyab) on
16 November 2026 after side N's sanity rerun of NERVE (Entry 4: 0.830 VOC-Sim,
0.729 COCO-Cap at a fixed threshold, no DenseCRF). This run uses side N's
patched NERVE code unchanged and does not replace N's official NERVE row.
Written and committed before any image of this run is scored. BRACE is not
changed by this run, whatever the outcome.

## Question

NERVE differs from BRACE in two things at once: where the class evidence comes
from (CLIP instead of cross-attention) and how it is spread (a closed-form
random walk on one noised pass instead of a capped coarse-to-fine hierarchy
during generation). Which of the two explains NERVE's COCO-Cap lead?

## Cells

Seeds (class word only, one class per image):

- **CA**: BRACE's frozen cross-attention seed, from the live generation.
- **CLIP**: NERVE's own seed. NERVE's modified CLIP ViT-L/14 with the
  vocabulary {class word} + NERVE's VOC background words (side N's variant A),
  computed on the stored benchmark image; the class column, bilinear to 64×64.

Propagations:

- **none**: the seed alone.
- **fine-only**: BRACE's two-pass 64-px propagation.
- **hierarchy**: BRACE's capped 16→32→64 hierarchy (frozen code,
  `src/segmentation/sg_combined_v2.py`), self-attention of generation calls
  10–20. Seeds enter min–max normalised, as in the earlier SCLIP-seeded runs.
- **walk**: NERVE's refinement exactly as released (`Pipeline.refinement` of
  `ovs/pipeline_clean_mean_final_run_correct.py`, commit `1ddc5f9` with side
  N's patch): stored image, one noised pass at scheduler index 45 of 50, empty
  prompt, two 64-px self-attention layers, NERVE's preprocessing and
  global/local fusion, closed-form infinite walk with decay 0.9. Noise seed 0.

All eight cells (2 seeds × 4 propagations) are scored. The 2×2 of interest is
{CA, CLIP} × {hierarchy, walk}. CA × hierarchy is BRACE; CLIP × walk is NERVE
(variant A).

With a single class, NERVE's entropy head weights are undefined and side N's
patch makes them uniform. Both seeds therefore go through the *same* fused
transition matrix in the walk cells.

## Settings that are not identical across cells (disclosed)

- The hierarchy reads self-attention from the generation; the walk reads it
  from the finished image with added noise. This difference is part of each
  method and is not removed.
- The CLIP seed and the walk use the **stored** benchmark image. The hierarchy
  uses a **regeneration** of it with the frozen runtime. About half the
  regenerated images are bit-identical to the stored ones; the rest differ by
  GPU nondeterminism (mean |Δpixel| about 0.6/255). The contrasts are also
  reported on the bit-identical subset.
- DenseCRF and ground truth use the regenerated image and the stored
  annotation, for every cell.
- NERVE uses CLIP ViT-L/14, a second model. CLIP-seed cells are not SD-only.

## Data

All annotated VOC-Sim (490) and COCO-Cap (707) images. One live generation per
image supplies the CA seed and the hierarchy for both seeds.

## Scoring

- Class-macro mIoU of the min–max map, bilinear to the annotation size.
  Thresholds 0.30–0.70 in steps of 0.025, raw and with the frozen DenseCRF.
- **Primary number:** threshold selected on VOC-Sim, with DenseCRF. On VOC-Sim
  this is 5-fold out-of-fold (fold = crc32(image id) mod 5); on COCO-Cap the
  full-VOC-Sim-best threshold of that cell is applied unchanged. The cells have
  different map scales, so a common fixed threshold is not the primary number.
- Also reported: the same without DenseCRF; fixed threshold 0.5 raw (side N's
  sanity convention); BRACE's fixed 0.55 with DenseCRF for the BRACE cell.
- Differences with a 2000-sample image bootstrap, 95% interval.

## Registered contrasts (primary number)

| ID | Contrast | Reads as |
|---|---|---|
| P-CLIP | walk − hierarchy, CLIP seed | propagation effect on NERVE's seed |
| P-CA | walk − hierarchy, CA seed | propagation effect on BRACE's seed |
| S-H | CLIP − CA, hierarchy | seed effect under BRACE's propagation |
| S-W | CLIP − CA, walk | seed effect under NERVE's propagation |
| I | (CLIP walk − CLIP hierarchy) − (CA walk − CA hierarchy) | interaction |

Reading rule, on COCO-Cap:

- **The seed explains NERVE's lead** if S-H and S-W are both positive with
  intervals above zero.
- **The walk explains NERVE's lead** if P-CLIP and P-CA are both positive with
  intervals above zero.
- Both can hold. If neither holds, the result is reported as mixed, with the
  interaction.

VOC-Sim contrasts are reported the same way without a rule. No method is
selected or changed from this run; any follow-up needs its own registration.

## Fidelity checks (reported, not tests)

- CA × hierarchy at 0.55 with DenseCRF against the paper's 0.830 / 0.696.
- CLIP × walk at fixed 0.5 raw against side N's committed per-image counts
  (`outputs/segmentation/comparisons/n_side/nerve_sd15_A/per_item_counts.csv`),
  computed with side N's own upsample-then-min–max convention.

## Files

- `scripts/jc_swap_nerve.py` (NERVE environment): `seed`, `walk`.
- `scripts/jc_swap.py` (BRACE environment): `capture`, `score`, `report`.
- `jobs/nerve_gpu.slurm`.
- Maps: `<BRACE_CACHE_DIR>/jc_swap/`. Report: `runs/17_seed_walk_swap/result.md`.
