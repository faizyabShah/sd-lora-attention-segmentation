# Component ladder with the class word and with OVAM's optimized token (pre-registration)

For JC's comments #4 (BRACE with optimized tokens), #7 (Table 1 with COCO-Cap
values and Δ) and #1 (where the gains come from). Workplan:
`docs/JC_REVIEW_WORKPLAN.md`. Written and committed before any image of this
run is scored.

## Question

Does each component of the frozen pipeline still help when the cross-attention
seed comes from OVAM's optimized token instead of the class word? BRACE is not
changed by this run, whatever the outcome.

## Design

One live generation per image with the frozen runtime (`scripts/sg27_runtime.py`).
From that single generation, two query columns are computed, so every cell of an
image shares its trajectory and its self-attention:

- **Class word (W):** the frozen attribution, unchanged.
- **Optimized token (T):** OVAM's released VOC-Sim token for the class
  (`configs/ovam/tokens_VOC.npy`, the mean of the 2–3 per-image tokens in
  `voc_sim_train/tokens/`). Attribution keys are `[<SoT>, token]`, softmax over
  those two keys, conditional query, head mean. This is the token form fixed
  earlier in the project (protocol Section 50, "TC"). The same VOC-Sim tokens
  are used on COCO-Cap, as OVAM did; there are no COCO-Cap tokens.

Everything else is identical between the columns: the 11 frozen cross-attention
layers, the 31 calls, the layer weights, the self-attention calls 10–20 and the
propagation code (`src/segmentation/sg_combined_v2.py`). The self-attention-guided
layer weights do not depend on the text query, so both columns use the same
weights by construction.

Rows, for each column:

| Row | Seed | Propagation | Post-processing |
|---|---|---|---|
| 1 | cross-attention, uniform layer weights | none | none |
| 2 | cross-attention, self-attention-guided layer weights | none | none |
| 3 | as row 2 | fine-only, two passes (64) | none |
| 4 | as row 2 | 16→32→64 hierarchy, no cap | none |
| 5 | as row 2 | hierarchy with one-sided cap at 16→32 | none |
| 6 | as row 2 | as row 5 | DenseCRF (frozen settings) |
| 3c | as row 2 | as row 3 | DenseCRF |
| 6u | as row 1 (uniform weights) | as row 5 | DenseCRF |

Row 6 with the class word is BRACE. Rows 3c and 6u exist so that the hierarchy
and the layer weighting can each be compared with the same post-processing.

## Data

All annotated images of VOC-Sim (490) and COCO-Cap (707), the paper's evaluation
sets. These sets were used to develop the frozen method, so class-word results
are in-sample for the method design. The tokens were optimized on separate
VOC-Sim training images (no evaluation seed appears among the 57 token seeds;
filename check, to be confirmed independently by the other side).

## Scoring

- Class-macro mIoU, image bootstrap with 2000 resamples, 95% interval.
- **Primary number:** fixed threshold 0.55 for every cell, both columns.
- **Secondary number:** threshold selected on VOC-Sim. On VOC-Sim this is
  5-fold out-of-fold (fold = crc32(image id) mod 5). On COCO-Cap the threshold
  that maximises VOC-Sim mIoU for that cell is applied unchanged. No threshold
  is ever chosen on COCO-Cap.
- Raw rows are compared with raw rows and DenseCRF rows with DenseCRF rows. The
  only raw-to-DenseCRF difference reported is row 6 − row 5, labelled as the
  DenseCRF effect.
- Folds won (x/5): on VOC-Sim, the number of held-out folds in which a row beats
  its reference.

## Registered comparisons (optimized-token column, fixed 0.55)

| ID | Comparison | Reads as |
|---|---|---|
| H | row 6 − row 3c | capped hierarchy vs fine-only, same DenseCRF |
| L | row 6 − row 6u | self-attention-guided layer weights vs uniform |
| C | row 5 − row 4 | cap vs no cap |
| Q | row 6 (T) − row 6 (W) | optimized token vs class word at the full pipeline |

Decision rule for H, L and C, each separately: the component **holds with the
optimized token** if the COCO-Cap difference is positive with its interval above
zero and the VOC-Sim difference is not below −0.005. Otherwise it **does not
hold**, and the row-by-row table is used to say where the token's gain replaces
the component's. Q is descriptive (no rule).

The same four comparisons are reported for the class-word column for reference;
they are a rerun of earlier results, not a new test.

## Reporting

- Both columns, both benchmarks, every row, with Δ to the stated reference row.
- Failed comparisons are reported as failed.
- OVAM's own optimized-token scores from its paper are quoted as an unpaired
  reference, not as a row.
- The optimized-token column is few-shot (the tokens need annotated images) and
  is labelled so wherever it appears.
- Fidelity: each image's class-word row 5 and row 3 maps are compared with the
  stored maps of the earlier capture, and the count of bit-identical regenerated
  images is reported.

## Files

- `scripts/jc_ladder.py`: `capture` (GPU, sharded), `score` (CPU), `report`.
- Maps: `<BRACE_CACHE_DIR>/jc_ladder/<set>/<image id>.npz`.
- Report: `runs/15_token_ladder/result.md`.
