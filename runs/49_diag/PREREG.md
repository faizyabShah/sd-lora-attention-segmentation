# Run 49: small-set component diagnostics (ours vs DiffSegmenter vs iSeg), pre-registration

**Status: exploratory.** Nothing is promoted from this run, and the paper 2 method stays frozen. Driver:
`scripts/jc_diag49.py`. It runs on the VPU-lab machine; every map compared comes from one machine and one capture.

## 1. Questions

1. Which parts of DiffSegmenter and of iSeg give them an edge over us: backbone, prompt, cross-attention aggregation,
   refinement, or read-out?
2. What goes wrong in our maps: ranking (AUROC), threshold (oracle gap), distractor objects, missed instances, edges, or
   small objects?
3. Which changes to our method are worth a confirmation run?

## 2. Data: development photos only

- **Source:** COCO **train2017**. Excluded: every photo of the run 42 profiles (all sizes), the run 44 Context profiles
  and the run 40 fits (9,639 photos), leaving a pool of 107,650.
- **Rounds:** the pool is shuffled once (seed 49). Round r takes the next 30 photos with at least one primary item, so
  rounds never overlap.
- **Items:** every class with area ≥ 1% (COCO-Object rule: crowd regions ignored, the 80 COCO classes).
  - Round 0: 63 items, 31 classes.
  - Round 1: 61 items, 32 classes.
- **No test photo:** no photo from COCO val2017, PASCAL VOC or PASCAL Context is used.

## 3. Cells: everything listed in the driver docstring, all fixed now

| Factor | Levels |
|---|---|
| Backbone | SD 1.5 · SD 2.1-base |
| Prompt | T template · E template with the word's embeddings × 1.6 · C DiffSegmenter's BLIP-large conditional caption |
| Seed | U- ours, no gate · UL ours, MK4-LR gate · D- DiffSegmenter's aggregation · DL D- × gate map |
| Refinement | none · V0 ours · DSR DiffSegmenter · ERS iSeg |
| Read-out | t55 ours · t50 DiffSegmenter's · comp iSeg's competition |

- **iSeg's own pass,** on SD 2.1, has seeds I- and IL, with the same four refinements.
- **Profiles:**
  - SD 1.5: the cluster's run 42 fit, shipped.
  - SD 2.1: built on this machine with `jc_sd21_47` prof / dump42 / fit42 (the same recipe).
- **No tuning:** every setting is ours as frozen, or the competitor's as released. "best" (one in-sample threshold) and
  "orc" (an oracle threshold per item) are diagnostics only.
- **Not used for selection:** comp uses the photo's class list (label information). It measures what competition
  would give, and no change that relies on it can be selected.

## 4. Reported

- **Reference pipelines:**
  - ours = `15|T|UL|V0` @t55;
  - DiffSegmenter transcribed = `15|C|D-|DSR` @t50;
  - iSeg transcribed = `21|iseg|I-|ERS` @comp.
- **Swaps:**
  - one change to ours;
  - one change to each competitor toward ours;
  - factor averages.
- **Error measures** for the reference cells.
- **Every cell.**
- **The largest per-item gaps,** with photo panels.
- **Metric:** class-macro mIoU over the classes present; 95% paired photo bootstrap.

## 5. What counts as a lead (fixed now)

**A label-free change to ours is a lead if, on two pooled rounds (60 photos), its Δ vs ours is ≥ +0.02 and its CI lower
bound is > 0.**

- A lead is not a result. It goes to a separately pre-registered confirmation: fresh development rounds first, then the
  run 44 test columns.
- Single-round numbers (30 photos) are directions only.
- Transcription limits:
  - DiffSegmenter's parts are rebuilt inside our pass (bicubic and fp32 rather than its bilinear and fp16; the same
    noise stream), not run from its release;
  - iSeg's pass is run 47's transcription (pixel agreement with the release 0.967–1.000 on the cluster smoke).
