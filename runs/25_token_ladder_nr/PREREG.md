# Stage ladder with the class word and OVAM's optimized token, on the corrected data (pre-registration)

Owner request (2026-11-18), for the professor (JC #4: BRACE with optimized tokens). Written and committed before any
image of this run is captured or scored. Rerun of runs/15_token_ladder on the runs/20 corrected generation prompts
(`configs/new_reruns_prompts.json`) with the safety checker off; runs/15 numbers are superseded, not replaced.
BRACE is not changed by this run.

## Design

As runs/15: one live generation per image with the frozen runtime; two query columns from that generation, so every
cell of an image shares trajectory and self-attention.

- **W, class word**: frozen attribution (corrected class word for VOC "dining table" / "potted plant").
- **T, optimized token**: OVAM's released VOC-Sim token (`configs/ovam/tokens_VOC.npy`), keys [<SoT>, token],
  softmax over the two, conditional query, head mean (protocol Section 50 "TC"). Same VOC-Sim tokens on COCO-Cap.
  Few-shot (needs annotated images); labelled so everywhere.
- Layer weights: SA-guided (GOCA-style, BRACE's) and uniform; text-independent, shared by both columns.

Stages, for each column (owner's table):

| Stage | Seed | Propagation | Post-processing |
|---|---|---|---|
| CA | CA seed, SA-guided weights | none | DenseCRF R2 |
| CA + fine SA | as above | fine-only, two passes (V0) | DenseCRF R2 |
| CA + hierarchy (BRACE) | as above | 16→32→64 hierarchy, one-sided cap at 16→32 (V3) | DenseCRF R2 |

Also reported, secondary: the same three stages raw, and with uniform layer weights.

## Data and scoring

- All 490 VOC-Sim and 707 COCO-Cap images. Class-word results are in-sample for the method design.
- Thresholds 0.30–0.70 (step 0.025), as runs/20. Two numbers per cell, both reported:
  **VOC-sel.** (5-fold OOF on VOC-Sim by crc32(id) mod 5; the full-VOC-Sim best threshold applied unchanged to
  COCO-Cap) and **fixed 0.55**. No threshold is chosen on COCO-Cap.
- Class-macro mIoU; paired image-bootstrap 95% intervals (2000).
- Fidelity: class-word BRACE seed equals the runtime seed exactly (job stops otherwise); class-word hierarchy map
  compared with the runs/20 capture; image differences reported.

## Registered comparisons (all with DenseCRF, both thresholds; 0.55 is primary as in runs/15)

| ID | Comparison |
|---|---|
| F | CA + fine − CA (fine SA effect) |
| H | CA + hierarchy − CA + fine (hierarchy vs fine) |
| Q | T − W at each stage |

For F and H, per column: **holds** if the COCO-Cap Δ is positive with its interval above zero and the VOC-Sim Δ is
not below −0.005 (runs/15 rule). Q is descriptive. Failed comparisons are reported as failed.

## Files

`scripts/jc_ladder_nr.py` (`capture` GPU sharded → `score` CPU → `report`); maps in
`<BRACE_CACHE_DIR>/jc_ladder_nr/`; report `runs/25_token_ladder_nr/result.md`. Smoke on 1 VOC + 1 COCO image,
tag `_smoke`, kept.
