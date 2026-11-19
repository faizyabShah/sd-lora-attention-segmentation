# Table 1 ladder on both benchmarks, one capture (side N, JC #7): pre-registration

Written before any image of this run is captured or scored (2026-11-19). Exploratory reproduction, not a selection:
nothing in BRACE changes, and no row is chosen from these numbers.

## Why

JC #7 asks for COCO-Cap values on the first three rows of Table 1, a COCO-Cap Δ column, folds won per row, an
explicit reference for each Δ, and the hidden .691 → .789 jump made visible. The paper's first three rows come from
early VOC-only screens (protocol Sections 27, 33, 35; fixed threshold 0.45) and runs/15 and runs/25 do not contain
them. This run rebuilds every row of Table 1 from one live generation per image, so every Δ is paired.

## Rows (each row adds one decision to the row above; definitions from protocol Section 11, "frozen settings")

| Row | Map | Seed | Propagation | DenseCRF |
|---|---|---|---|---|
| 1 | CA reference: all 16 CA modules, calls 0–30, uniform, conditional padded softmax, all class-word pieces | — | none | no |
| 2 | + drop the five 32×32 CA modules (11 modules) | row 1 → 11 modules | none | no |
| 3a | (reference for row 3) one pass `mean_i A_i c` through all 16 SA modules, calls 0–30, seed area-downsampled to each module, upsampled back, equal weights | row 2 | 1 pass, 16 modules | no |
| 3 | + one pass through the 64×64 SA module 15 only, calls 0–30 | row 2 | 1 pass | no |
| 4 | + two passes `mean_i A_i² c` | row 2 | 2 passes, calls 0–30 | no |
| 5 | + DenseCRF R2 | row 2 | as row 4 | yes |
| 6 | + SA calls 10–20 only | row 2 | 2 passes, calls 10–20 (= fine-only V0) | yes |
| 7 | + SA-guided CA layer weights (A3, the frozen seed) = **fine-only reference** | A3 | V0 | yes |
| 8 | + coarse-to-fine hierarchy with the one-sided cap at 16→32 (V3) = **BRACE** | A3 | V3 | yes |

Every map is min-max normalised, bilinearly upsampled to 512, thresholded. Raw versions of rows 5–8 are scored too.

## Generation and data

Frozen runtime (`scripts/sg27_runtime.py`, SD 1.5 fp32, the manifest's 30 steps, CFG 7.5, 512 px), runs/20
corrected prompts (`scripts/jc_newrerun.rows`), safety checker off; all 490 VOC-Sim and 707 COCO-Cap images
(COCO rows from the rebuilt `ovam_coco_cap.csv`, as in `comparisons/official/`). The A3 seed must equal the
runtime's seed exactly (the job stops otherwise). Row 8 is compared with side N's runs/20 reproduction.

## Scoring and reporting

- Thresholds 0.30–0.70, step 0.025 (runs/20). Class-macro mIoU.
- **Primary convention (Table 1):** VOC-sel. = 5-fold OOF on VOC-Sim (folds crc32(id) mod 5); the full-VOC-Sim best
  threshold applied unchanged to COCO-Cap. Also reported: fixed 0.45 (the threshold of the early screens) and fixed
  0.55 (BRACE's frozen threshold, the shared-threshold column).
- Δ = row minus the row above, paired image bootstrap 95% interval (2000), on both benchmarks; row 3 also against 3a.
- Folds won: for each VOC-Sim fold, the row's OOF mIoU on that held-out fold vs the row above's (thresholds chosen on
  the other four folds each).
- Reproduction check (not a pass/fail rule): VOC-Sim rows at fixed 0.45 should be near the paper's screen values
  (.665, .691, .756/.789, .800, .813); rows 6–8 near .817/.823/.829.
- Class-word results are in-sample for the method design.

## Files

`comparisons/drafts/table1_ladder/t1_ladder.py` (`capture` GPU sharded → `score` CPU → `report`); maps and scores in
`~/TRDP/external/n_side/table1_ladder/`; report `comparisons/drafts/table1_ladder/result.md`.
