# runs/25 (JC #4 on corrected data): independent side-N reproduction of side F's result

**Headline result is side F's** (`runs/25_token_ladder_nr/result.md`, commit a64aff6: BRACE class word 0.832 / 0.708
and optimized token 0.864 / 0.693 at 0.55; F holds for both queries, H fails for both). This file only reports
whether an independent run of the same pre-registered script on side N gives the same numbers. It does.

## What was run

- F's pre-registration `runs/25_token_ladder_nr/PREREG.md` and committed script `scripts/jc_ladder_nr.py`,
  unchanged, through the thin wrapper `run_nr.py`. The wrapper redirects only the cache
  (`~/TRDP/external/n_side/token_ladder_nr/cache/`) and the report path, and reads COCO-Cap rows from the rebuilt
  `ovam_coco_cap.csv`, as in `comparisons/official/`.
- Capture: all 490 VOC-Sim and 707 COCO-Cap images, 2 × V100 on erdos, `gen` environment (torch 2.4.1, diffusers
  0.36.0, transformers 5.6.2). Jobs 253886_0/1, 3.1 h.
- Scoring:
  - VOC-Sim uses F's `score_one` unchanged on the full threshold grid.
  - COCO-Cap DenseCRF counts were computed only at the thresholds the report reads: the full-VOC-Sim-best threshold
    of each map, and 0.55. Every other COCO-Cap DenseCRF threshold is stored as −1 and is never read
    (`comparisons/drafts/fast_score.py`).
  - Raw counts are on the full grid.
  - 177 COCO-Cap images had already been scored on the full grid by F's scorer before the switch.
  - This changes no reported number. It cut DenseCRF CPU time by about 5×, from 4.3 s per call.
- Report: F's `report()` unchanged → `result_n_full.md`.

## Agreement with F's result (a64aff6)

- **All 3-decimal numbers.** Of the 181 three-decimal cells in the stage tables and the registered comparisons,
  **172 are identical** and 9 differ by 0.001. All 9 are on COCO-Cap:
  - 4 stage means, in the secondary tables (raw BRACE class word 0.693 → 0.692; uniform-weight fine SA 0.668 →
    0.669 and 0.708 → 0.709);
  - 5 Δ values or interval bounds.
  - Every primary DenseCRF stage cell is identical.
- **Verdicts.** Every verdict is the same:
  - F holds for both queries;
  - H does not hold for either query.
- **Per-class values.** Two-decimal per-class values agree except for 2 classes, which differ by 0.01 in the token
  column.
- **Images and maps.** Compared with side N's runs/20 reproduction (`comparisons/official/brace_repro.py`, V100 on
  wald), this capture is bit-identical on 490/490 VOC-Sim and 707/707 COCO-Cap images. Its class-word BRACE maps
  differ by 0 (`fidelity.py`, `~/TRDP/external/n_side/token_ladder_nr/fidelity.json`).
- **Residual difference.** F's capture matches F's runs/20 on 164/490 and 177/707 images; the rest differ by GPU
  nondeterminism (F's data check). That is the source of the ±0.001.

Key cells, class word / optimized token, CA + hierarchy (BRACE), DenseCRF:

| Run | VOC-Sim 0.55 | COCO-Cap 0.55 | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. |
|---|---|---|---|---|
| F (a64aff6) | 0.832 / 0.864 | 0.708 / 0.693 | 0.831 / 0.864 | 0.708 / 0.693 |
| N (this run) | 0.832 / 0.864 | 0.708 / 0.693 | 0.831 / 0.864 | 0.708 / 0.693 |

## Files

| File | Content |
|---|---|
| `run_nr.py` | wrapper |
| `capture.slurm`, `score.slurm` | job scripts; the CPU-saving scorer is `../drafts/fast_score.py` with `../drafts/fast_score.slurm` |
| `fidelity.py` | comparison with side N's runs/20 reproduction |
| `result_n_full.md` | F's report format, N's numbers |

The data check lines in that report are replaced by a note, because F's runs/20 cache is not readable here.
