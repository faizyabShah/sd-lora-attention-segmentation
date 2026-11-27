# Run 50: what in iSeg beats us? Its seed taken apart on 100 hard COCO photos (pre-registration)

**Status: exploratory.** Development photos only. Nothing is promoted, and the paper 2 method stays frozen. Driver:
`scripts/jc_iseg50.py`. It runs on the VPU-lab machine, and every map compared comes from that machine.

## 1. Why this run

Earlier runs located iSeg's edge in its **seed**, not in its backbone or refinement:
- Run 47 Q3: our seed in place of iSeg's: −0.096 on Context, −0.027 on COCO.
- Run 47 Q5 and run 49: SD 2.1 hurts us.
- Run 47 Q1 and run 49: iSeg's refinement on our seed: 0 to −0.085.

Run 44 suggests that most of the seed's edge is the **photo's class list in the prompt**:
- iSeg given the classes: 0.759 / 0.521 / 0.549;
- iSeg with its own labels: 0.705 / 0.401 / 0.476;
- ours: 0.713 / 0.447 / 0.486.

iSeg's seed bundles five ingredients that have never been measured separately: competing words in the prompt (the class
list), fixed background words, ×1.6, its aggregation (16/32 modules, per-token normalisation, clustering), and its pass
(preprocessing, noise). This run takes them apart, and tests label-free stand-ins for the class list.

## 2. Photos (fixed before any capture)

- **Pool:** run 49's COCO train2017 development pool (107,650 photos, profile and fit photos excluded), in run 49's
  permutation (seed 49), **from position 2,000 on**. Run 49's rounds use the start of that order, so the two runs never
  share a photo.
- **"Hard", from annotations only:** at least 3 primary classes (area ≥ 1%) and at least one primary item < 5% of the
  photo. The selection never looks at any method's output, so no score regresses to the mean through it.
- **Size:** the first 100 such photos, with every primary item: **364 items (178 small), 59 classes** (positions
  2,000–2,393; computed on the cluster, CPU).

## 3. Cells

All passes except iSeg's own use our pass code (bicubic 512, VAE mean, CPU noise seed 0, fp32, t = 100), so each prompt
step changes only the prompt. Class names and token selection are iSeg's (`data/coco14/coco_name80.py`).

| Prompt | Text | Label use |
|---|---|---|
| T | "A photograph of a {word}" (ours, frozen) | none |
| TE | T, word embeddings × 1.6 | none |
| S1 | "a photograph of {c} and other object and background", × 1.6 | none |
| S2 | S1 + iSeg's released COCO background words | none |
| S2n | S2 without × 1.6 | none |
| B | S2 format; target + every COCO class named in a BLIP-large **unconditional** caption of the photo | none (image-derived) |
| F | S2 format; target + fixed generic list (person, car, chair, dining table, cup, bottle; first 4 ≠ target) | none |
| K | S2 format; every primary class of the photo = iSeg's prompt text | **class list** |
| Kn | K without × 1.6 | **class list** |
| iseg | iSeg's own pass on SD 2.1 (its preprocessing, CUDA noise seed 1101; text asserted equal to K) | **class list** |

- **Background words:** used verbatim as released, which concatenates them to "background" ("backgroundground, land,
  …").
- **Caption matching for B:** a class is named when its COCO name, our word, iSeg's name or a plural of these occurs as
  a word. "person" also matches common person nouns (man, woman, child, player…), as listed in the driver. The matching
  rule is fixed now.

**Seeds** (from each pass, for the target's tokens):
- U- / UL: ours, without / with the MK4-LR gate;
- I- / IL: iSeg's aggregation, without / with the 64 × 64 gate map;
- In: iSeg's aggregation without clustering.

**Refinements:** none, V0 (ours), ERS (iSeg's COCO settings).

**Backbones:** SD 1.5 (ours) and SD 2.1-base (iSeg's).

**Read-outs:**
- t55 (ours);
- t50;
- comp: iSeg's competition over the photo's classes, which uses the class list; diagnostic only;
- best: one in-sample threshold; diagnostic only.

**Profiles:** SD 1.5 = the shipped run 42 fit; SD 2.1 = the run 49 build on the VPU machine.

**Built-in checks** (in the capture; any failure stops the run):
- the new recorder's seed equals `make_seed` on every T pass;
- iSeg's aggregation from head means equals its per-head computation on the first K pass;
- the selected tokens decode to the class name;
- iSeg's own text equals K's.

## 4. Reported (class-macro mIoU, 95% paired photo bootstrap, all items and small items separately)

1. **Reference cells:** ours, iSeg's prompt through our pass, and iSeg's own pass.
2. **Ladder from ours to iSeg** (SD 2.1 and SD 1.5), one ingredient per step:
   - iSeg's format;
   - background words;
   - iSeg's aggregation (gated);
   - ERS;
   - class list;
   - drop our gate;
   - competition read-out;
   - (SD 2.1 only) iSeg's own pass.
   Each step's Δ is reported.
3. **Is it the class list?** K − S2 (labels) against the label-free B − S2 and F − S2, per seed and refinement.
4. **One ingredient removed from iSeg** (`21|K|I-|ERS`), at comp and at t55.
5. **Factor averages,** for all items, small items and other items.
6. **Error measures** (run 49's columns) for the main cells.
7. **Every cell,** and the items where iSeg beats us most, with panels.

## 5. What counts (fixed now)

- **Attribution:** an ingredient "explains" iSeg's edge when its ladder step is ≥ +0.02 with CI > 0. It is reported
  separately for label-free ingredients and for those that use labels.
- **Lead:** a label-free change to ours on SD 1.5 (prompt TE, S1, S2, S2n, B or F; seed UL or IL; V0 or ERS; t55) with
  Δ vs `15|T|UL|V0` ≥ +0.02 and CI low > 0.
  - Because the photos are selected to be hard, a lead must then pass a separately pre-registered confirmation on random
    development photos, then on the run 44 test columns.
  - Nothing is selected on comp or best.
