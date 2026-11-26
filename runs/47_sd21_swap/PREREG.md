# Run 47: SD 2.1-base component swap between our method and iSeg (pre-registration)

Committed after the smoke (mechanics, identity and reproduction checks; numbers not used) and **before any full capture
or scoring**. Drivers: `scripts/jc_sd21_47.py` (profiles on SD 2.1) and `scripts/jc_swap47.py` (passes, cells, scoring,
report). The paper 2 method stays frozen; this is a component analysis, and nothing is promoted.

## 1. Questions

Run 44 (same items): iSeg given every class of the photo scores 0.759 / 0.521 / 0.549, against ours at 0.713 / 0.447 /
0.486. iSeg also uses SD 2.1-base while ours uses SD 1.5. **With one backbone for everything (SD 2.1-base), which
component explains the gap, and are the components complementary?**

1. Does iSeg's refinement improve our seed?
2. Does our gate improve iSeg?
3. Is it the seed?
4. Is it class competition?
5. What does the backbone alone change?

## 2. Items and protocol (= run 44)

- **Columns:** PASCAL VOC val (1,947 primary items / 1,409 photos), PASCAL Context val (22,691 / 5,104) and COCO-Object
  (9,816 / 4,571).
- **Task:** class word given; one mask per (photo, class) item with class area ≥ 1%.
- **Metric:** class-macro mIoU; 2,000-sample paired photo bootstrap; 95% CI.
- **No tuning:** nothing is tuned for any cell. Every setting is ours as frozen or iSeg's as released, per column.

## 3. Passes (SD 2.1-base, the sd2-community snapshot iSeg and NERVE use; fp32; t = 100)

Both passes are recorded with the same recorder: `jc_realvoc.Proc` on SD 2.1, plus the full cross-attention of iSeg's
16/32 modules.

**P_ours**, one pass per item: our pass exactly as run 44 except for the model.
- Photo bicubic to 512, VAE mean, CPU noise seed 0, prompt "A photograph of a {word}".

**P_iseg**, one pass per photo: iSeg's released pass, transcribed.
- **Image:** photo / 255, bilinear to 512, 2x − 1, VAE mean × 0.18215.
- **Noise:** CUDA noise seed 1101, drawn in fp16 as the release does.
- **Prompt:** "a photograph of ⟨c1⟩ and … and other object and background" + bg_context (as the released code evaluates
  it). It lists **every primary class of the photo** (= run 44's "iSeg, class given").
- **Category enhancement:** class-token embeddings × 1.6.
- **Checked on every photo:** the prompt text and class ids must equal run 44's saved iSeg outputs.

**Profiles for our gate:** rebuilt on SD 2.1 from the same photos with the same recipe and caption rules.
- 1,600 size-20 photos for the 80 COCO words (run 42);
- 2,260 jobs for the 33 Context words (run 44).

## 4. Cells: seed × refinement × read-out (4 × 2 × 2 = 16)

| Factor | Levels |
|---|---|
| Seed | **O-** our uniform 11-module cross-attention seed, no gate (P_ours) · **OL** O- with our MK4-LR gate, γ = 1 (P_ours) · **I** iSeg's category-enhanced cross-attention: 16/32 modules at 0.7/0.3, head subset on VOC, per-token normalisation, DFC_KL clustering, class-token mean (P_iseg) · **IL** I × our MK4-LR gate, using the 64 × 64 consensus from P_iseg's own queries |
| Refinement (on the seed's own pass) | **V0** ours: two 64 × 64 self-attention steps from the min-max seed · **ERS** iSeg's entropy-reduced self-attention (up_blocks.3.attentions.1 attn1, heads 1–2), released iter / ent per column (VOC 8 / 0.015, Context 5 / 0.002, COCO 10 / 0.2) |
| Read-out | **bin** ours: min-max, bilinear to the photo, ≥ 0.55 · **comp** iSeg's `cam_to_label` over the photo's classes, released background rule (VOC / Context (1 − max)², COCO threshold 0.45) |

- **iSeg as released = I · ERS · comp.**
- **Ours as frozen (on SD 2.1) = OL · V0 · bin.**

## 5. Primary contrasts (every column; Δ mIoU [95% CI]; CI > 0 helps, CI < 0 hurts, otherwise not resolved)

| # | Contrast | Question |
|---|---|---|
| Q1 | OL·ERS·bin − OL·V0·bin | does iSeg's refinement improve our seed? |
| Q2 | IL·ERS·comp − I·ERS·comp | does our gate improve iSeg? |
| Q3 | OL·ERS·comp − I·ERS·comp | with iSeg's refinement and read-out, is our seed better or worse? |
| Q4 | OL·V0·comp − OL·V0·bin | does class competition help our method? |
| Q5 | OL·V0·bin (SD 2.1) − ours on SD 1.5 (run 44) | the backbone alone |

**Secondary (descriptive):**
- Every cell.
- Our gate within each refinement and read-out.
- Seed I vs O-.
- Refinement and competition on iSeg's seed.
- Cells vs NERVE and vs released iSeg.
- Context / COCO-Object subsets (20 VOC classes vs the rest).
- Per class.

## 6. Checks (smoke, before submission)

1. **Identity:** P_ours O-·V0 and OL·V0 equal run 44's capture code on SD 2.1 (`jc_ours44.capture`) on 3 items
   (|Δ| < 2e-3, fp16 storage).
2. **iSeg transcription:** the prompt text and class ids equal run 44's saved iSeg outputs. The pixel agreement of
   I·ERS·comp with the released iSeg label map is reported for 3 photos per column (fp32 here, fp16 in the release).
**Smoke result (job 263684; 2-3 items or photos per column; profiles were SD 1.5 copies, so mechanics only; no number
used):**
- **Identity:** O-·V0 and OL·V0 vs `jc_ours44.capture` on SD 2.1: max |Δ| = 0.0 on all 6 maps (3 VOC items). LR
  consensus vs an independent mean: 6.0e-08; V0 vs `jc_realvoc.propagate`: 0.0.
- **iSeg transcription:** prompt text and ids equal on every smoke photo. Pixel agreement with the released label map:
  - VOC 0.998 / 0.999 / 0.999;
  - Context 0.997 / 0.967 / 0.994;
  - COCO 0.994 / 0.989 / 1.000.
- **Fix before this smoke:** SD 2.1 uses 64-dim heads (5 at 64 × 64, vs SD 1.5's 8 heads), and the first smoke (job
  263681) stopped on the frozen helper's 8-head shape check. `jc_sd21_47.setup()` installs a copy without that check.
  On SD 1.5 it is identical.

3. **Full-run reproduction:** I·ERS·comp − released iSeg (class given, run 44) is reported per column.
   - **If |Δ| > 0.01 on any column**, the iSeg cells are read as "iSeg transcribed", with that difference stated.

## 7. Notes and limits

- **Class information is unequal by design:** iSeg's pass and the comp read-out use every primary class of the photo.
  O-/OL with bin use only the queried word.
- **Profiles:** our gate on SD 2.1 uses profiles built with the template prompt. On iSeg's pass the gate reads that
  pass's own queries, from a multi-class enhanced prompt.
- **Mixed backbones:** run 44's SD 1.5 numbers come from a different backbone and pass (Q5 only), on the same items and
  the same scoring.
- **Cost:**
  - profiles: 3,860 passes;
  - P_ours: 34,454 passes;
  - P_iseg: 11,084 passes;
  - about 8–10 GPU-hours, at most 4 GPUs at a time.
