# Run 44 handoff: real-photo comparison on PASCAL VOC, PASCAL Context and COCO-Object

For ChatGPT. Status on 2026-11-26: **11 of the 12 methods are fully scored. OVDiff is still predicting**; its column, and a
rescored report, follow automatically. This file fills paper 2's Table 3 (`tab:real`). Internal names (MK4, LR, V0) are
used for precision only, not for the manuscript.

- Pre-registration: `runs/44_real_benchmarks/PREREG.md` (commit `a7a10b4cd`, before any full capture or scoring).
- Full report (11 methods): `runs/44_real_benchmarks/result.md`. It is copied verbatim in the appendix below and kept as
  `result_pre_ovdiff.md`.
- Plan and owner decisions: `runs/44_real_benchmarks/PLAN.md`.

---

## 1. Bottom line

**Against the other SD-attention methods, ours is clearly better than DAAM and OVAM and on par with DiffSegmenter and iSeg.
It beats most CLIP methods on VOC and COCO-Object and beats no gate everywhere. But NERVE beats us on VOC and Context, and
COCO-Object is not resolved.** NERVE's lead comes from (a) non-VOC classes, mostly stuff classes, and (b) a few VOC classes
(dining table, car, potted plant, tv, bottle, chair). Ours wins big on person.

| Pre-registered contrast | VOC | Context | COCO-Object |
|---|---|---|---|
| ours − no gate (P1–P3) | **+0.086** [+0.075, +0.097] helps | **+0.090** [+0.076, +0.095] helps | **+0.041** [+0.034, +0.047] helps |
| ours − NERVE (P4–P6) | **−0.036** [−0.052, −0.020] trails | **−0.054** [−0.064, −0.038] trails | −0.016 [−0.028, +0.003] not resolved |

---

## 2. Protocol (fixed in advance)

- **Task:** the class word is given; one binary mask per (photo, class) item. The positive items are the classes present
  in the photo. Absent-class queries are not scored.
- **Primary items:** class area ≥ 1% of the photo, on all three columns (owner decision). All items are secondary.

  | Column | Classes | Photos | Primary items | Ground truth |
  |---|---:|---:|---:|---|
  | PASCAL VOC 2012 val | 20 | 1,409 | 1,947 | `SegmentationClass`, 255 = void, excluded |
  | PASCAL Context val (standard 5,104 photos) | 59 | 5,104 | 22,691 | 459 → 59 classes with iSeg's released converter |
  | COCO-Object (val2017 minus the 200-image §136 set) | 80 | 4,571 | 9,816 | instance masks, crowd excluded |

- **Metric:** class-macro mIoU per column (tp / fp / fn pooled per class), with a 2,000-sample paired photo bootstrap (the
  same resamples for every method).
- **Nothing is tuned** for any method on any column. Our maps are also reported at every threshold from 0.30 to 0.70,
  descriptively only.
- **Prior exposure:**
  - VOC val was seen 3 times in development (runs 32–37).
  - COCO val2017's 20 VOC classes were used in runs 40–42; its other 60 classes were used only in run 42 (confirmatory).
  - PASCAL Context was never used before.

## 3. Our method (frozen; the same as Table 2 / runs 42–43)

- One SD 1.5 pass per item at t = 100, noise seed 0, prompt "A photograph of a {word}".
- Uniform cross-attention seed. **MK4 gate**: per-module contrast of an object profile against the 4 nearest negative
  centroids. **MK4-LR**: the mean of the gates of the six modules below 64×64, applied at every module, γ = 1.
- **Fine-only propagation** (two 64×64 self-attention steps), min-max, **fixed threshold 0.55** (raw is primary).
  DenseCRF R2 is secondary.
- **Profiles: COCO train2017 captions only** (no labels or masks; none of the evaluated photos).
  - 20 positive photos whose captions mention the word and 480 negatives whose captions don't.
  - The object profile comes from the top 5% of queries; the 4 negative centroids come from the top 1% of activations on
    the negatives.
  - The 80 COCO words reuse run 42's profiles. The 33 Context words that aren't COCO words use the same recipe with caption
    rules written down in advance (SYN44, e.g. road: road/street/highway; sky; wall; grass: grass/lawn…). These rules are
    noisy by design ("dirt bike" counts as ground).
  - Profile size doesn't matter on COCO's unseen classes: in run 42, 5 / 20 / 100 photos gave 0.458 / 0.459 / 0.459. It
    was not tested on the Context words.
- The Context word "dining table" is queried as "table", since Context's class covers every table.

## 4. Baselines (each run as released, on the same photos)

| Method | Code / backbone | What it sees | Read-out (primary) |
|---|---|---|---|
| DAAM | castorini/daam; SD 1.5; one noised pass at t = 100 (our latent) | the queried word | word heat map, map / max > 0.4 |
| OVAM (no token optimisation) | vpulab/ovam; the same pass | the queried word | map / max × self-attention in [0.85, 1] > 0.4 (OVAM's VOC-Sim settings for all columns) |
| DiffSegmenter | reviewed class-word driver; SD 1.5, t = 100, BLIP-large caption prompt | the queried word | 64×64 map, min-max, > 0.5 fixed |
| iSeg | linsun449/iseg.code; SD 2.1-base | **its own TagCLIP image labels** | `cam_to_label`. Secondary row: the queried class given |
| SCLIP, ClearCLIP, NACLIP, ProxyCLIP | official repos, ViT-B/16 | the full class list in each repo's VOC21 / Context60 / COCO-Object config, with background | argmax label map |
| NERVE | official, SD 2.1-base | released VOC21 / Context60 / COCO5k configs and vocabularies | `cam_to_label` with its class-score gate |
| OVDiff | karazijal/ovdiff; SD 1.5 support sets (32 per class), SD + CLIP + DINO prototypes, CutLER | the full class list | argmax label map |

- **VOC21 reproductions** with each CLIP repo's own runner: SCLIP 59.64 (published 59.1), ClearCLIP 51.80 (51.8),
  NACLIP 64.06 (64.1), ProxyCLIP 61.24 (61.3).
  - ClearCLIP's released `base_config.py` gives 32.0. The paper's q-q attention (`model_type='ClearCLIP'`) reproduces
    51.8, so that setting is used.
- **Protocol asymmetry, which must be stated in the paper:**
  - The multi-class methods (the CLIP methods, NERVE and OVDiff) and iSeg get competition between classes for free: every
    pixel goes to exactly one class.
  - Our method, DAAM, OVAM and DiffSegmenter produce one map per queried word, with no competition between classes.
  - The class-given iSeg row is the only one where iSeg is told the class.

## 5. Results (mIoU, primary items; ours raw at 0.55)

| Family | Method | VOC (20) | Context (59) | COCO-Object (80) |
|---|---|---:|---:|---:|
| Diffusion attention | DAAM | 0.417 | 0.234 | 0.353 |
| | OVAM (no token opt.) | 0.460 | 0.260 | 0.309 |
| | DiffSegmenter | 0.658 | 0.433 | 0.500 |
| | iSeg | 0.705 | 0.401 | 0.476 |
| CLIP | SCLIP | 0.629 | 0.410 | 0.411 |
| | ClearCLIP | 0.545 | 0.424 | 0.383 |
| | NACLIP | 0.677 | 0.470 | 0.444 |
| | ProxyCLIP | 0.645 | 0.459 | 0.438 |
| Random walk | **NERVE** | **0.750** | **0.501** | **0.502** |
| Prototypes | OVDiff | *pending* | *pending* | *pending* |
| Ours | No gate (ablation) | 0.627 | 0.357 | 0.445 |
| | MK4 (ablation) | 0.656 | 0.385 | 0.460 |
| | **Ours (MK4-LR), raw** | 0.713 | 0.447 | 0.486 |
| | Ours + DenseCRF | 0.702 | 0.437 | 0.482 |
| Secondary | OVAM + DenseCRF | 0.468 | 0.264 | 0.325 |
| | No gate + DenseCRF | 0.619 | 0.355 | 0.442 |
| | iSeg, class given | 0.759 | 0.521 | 0.549 |

### Ours − each baseline [95% CI] (positive = ours better)

| Baseline | VOC | Context | COCO-Object |
|---|---|---|---|
| DAAM | +0.296 [+0.281, +0.310] | +0.213 [+0.202, +0.219] | +0.133 [+0.124, +0.144] |
| OVAM | +0.254 [+0.240, +0.267] | +0.186 [+0.174, +0.191] | +0.178 [+0.168, +0.188] |
| DiffSegmenter | +0.056 [+0.042, +0.069] | +0.014 [−0.000, +0.018] | **−0.014** [−0.021, −0.005] |
| iSeg | +0.008 [−0.012, +0.028] | +0.046 [+0.038, +0.057] | +0.010 [−0.001, +0.028] |
| SCLIP | +0.085 [+0.069, +0.101] | +0.037 [+0.028, +0.053] | +0.076 [+0.065, +0.093] |
| ClearCLIP | +0.168 [+0.151, +0.187] | +0.023 [+0.013, +0.039] | +0.104 [+0.092, +0.122] |
| NACLIP | +0.036 [+0.021, +0.053] | **−0.023** [−0.031, −0.011] | +0.042 [+0.030, +0.059] |
| ProxyCLIP | +0.069 [+0.051, +0.087] | **−0.012** [−0.021, −0.004] | +0.049 [+0.036, +0.066] |
| NERVE | **−0.036** [−0.052, −0.020] | **−0.054** [−0.064, −0.038] | −0.016 [−0.028, +0.003] |
| MK4 (ablation) | +0.057 | +0.062 | +0.026 (all CI > 0) |
| iSeg, class given | −0.046 | −0.074 | −0.062 (all CI < 0) |

Note: several intervals are asymmetric around the point estimate (e.g. Context +0.090 [+0.076, +0.095]). This is how a
class-macro photo bootstrap behaves when a few large classes dominate; it is not a bug that was checked for.

## 6. Where the gap to NERVE comes from (descriptive; partly post hoc)

### 6a. Seen vs other classes (subsets in the report)

| Subset | Items | Ours | No gate | NERVE | ProxyCLIP | iSeg | ours − no gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Context, 20 VOC classes | 7,925 | **0.663** | 0.589 | 0.632 | 0.600 | 0.620 | +0.074 [+0.068, +0.080] |
| Context, other 39 | 14,766 | 0.336 | 0.237 | **0.433** | 0.387 | 0.289 | +0.099 [+0.077, +0.104] |
| COCO-Object, 20 VOC classes | 5,414 | **0.568** | 0.476 | 0.533 | 0.474 | 0.502 | +0.092 [+0.083, +0.100] |
| COCO-Object, other 60 | 4,402 | 0.459 | 0.435 | **0.492** | 0.426 | 0.468 | +0.024 [+0.015, +0.031] |

- **On the 20 VOC classes ours leads NERVE in Context and COCO-Object, but not on PASCAL VOC itself** (0.713 vs 0.750;
  ours wins only 7 of 20 classes).
- **On COCO-Object's other 60 classes, the gate gain over no gate (+0.024) is about a quarter of the seen-class gain
  (+0.092).** This matches run 42 on the same photos (+0.022 vs +0.092).
- On Context's other 39 the gate gain is large (+0.099), but the no-gate baseline is very low there (0.237).

### 6b. Context's other 39 split into stuff and things (post hoc; OVDiff's released stuff/thing table)

| Group | Classes | Ours | No gate | NERVE | ours − NERVE |
|---|---:|---:|---:|---:|---:|
| Stuff (sky, wall, floor, road, grass, ground, water, ceiling…) | 19 | 0.335 | 0.232 | 0.469 | **−0.134** |
| Things (bag, bed, book, building, cup, door, tree…) | 20 | 0.336 | 0.242 | 0.400 | −0.064 |

### 6c. Largest per-class gaps (ours − NERVE)

| Column | NERVE far ahead | Ours far ahead |
|---|---|---|
| VOC | dining table −0.24, car −0.20, potted plant −0.17, tv −0.13, bottle −0.13, chair −0.11 | aeroplane +0.24, horse +0.13, boat +0.08, person +0.04 |
| Context | sky −0.66, grass −0.43, water −0.37, road −0.36, bag −0.33, floor −0.29, tree −0.28, sidewalk −0.24 | person +0.54, horse +0.25, aeroplane +0.24, bed +0.23 |
| COCO-Object | toaster −0.33, mouse −0.32, orange −0.31, toothbrush −0.31, refrigerator −0.28, fork −0.26, traffic light −0.24, car −0.23 | person +0.56, tennis racket +0.37, bowl +0.37, tv +0.33, cell phone +0.30 |

- **Our stuff maps are weak with or without the gate.** Context sky is 0.137 (no gate 0.048), road 0.072, wall 0.195,
  tree 0.206, while the CLIP methods and NERVE reach 0.5–0.8 on sky, grass and water.
- **The gate hurts some classes** (ours < no gate by more than 0.02):
  - VOC: bottle (−0.078).
  - COCO-Object: cake (−0.128), orange (−0.121), sports ball, tie, baseball bat, remote, spoon, scissors, elephant,
    giraffe… These are the same ambiguous words and sports items as in run 42.

### 6d. Candidate explanations (none of them tested)

1. **Development bias:** the gate's design was chosen on the VOC classes.
2. **Seed quality:** single-word SD cross-attention is weak for stuff and for small or rare objects, whereas CLIP-based
   seeds cover a broad vocabulary.
3. **No competition between classes:** a binary per-word map can leak onto neighbouring stuff (wall vs floor vs ceiling).
   The multi-class methods resolve that by argmax.
4. **Noisy caption rules** for the 33 Context words.
   - Profile size is ruled out on COCO's unseen classes (flat from 5 to 100 photos), but not on Context.

## 7. Anomalies checked

- **Person is about 0 for ClearCLIP and ProxyCLIP** in every column: VOC 0.015 / 0.026, COCO-Object 0.000 / 0.000.
  - Checked on 60 VOC person items: they label 91–92% of person pixels as background. SCLIP labels 74% correctly.
  - Likely cause: their configs list plain "person", while SCLIP and NACLIP add synonyms such as "person in shirt".
  - Our VOC21 reproductions match their published mIoU, so this is taken as their released behaviour. It is inferred, not
    verified: their per-class VOC21 numbers were not checked.
- **NERVE person on COCO-Object is 0.014** (Context 0.105, VOC 0.653). Not investigated; it is read as the behaviour of its
  released COCO5k vocabulary.
- **Bed = 0.000 for SCLIP and NACLIP on COCO-Object** because their released COCO-Object vocabulary also lists "bed" among
  the background synonyms, so bed pixels become background. This is as released. **ProxyCLIP's bed = 0.000 is not yet
  explained:** its background line doesn't contain "bed".
- **Person drives part of our advantage** wherever the multi-class methods collapse. It is one class out of 20/59/80 in
  class-macro mIoU.

## 8. Infrastructure problems during the run (memory and bugs only; no method or threshold changed)

| Method | Problem | Fix |
|---|---|---|
| SCLIP, COCO-Object | Its postprocess forms a classes × queries × H × W tensor (about 18 GB on large photos). The 16 GB V100s ran out of memory at photo 89/4,571, and the A100s were taken (FIFO queue). | That photo's postprocess reruns on the CPU (fp32 instead of fp16) after a GPU out-of-memory; it happened on 69 photos. |
| DAAM | Memory built up over items (our driver ran the VAE encode with autograd on). | VAE encode under `no_grad` and a per-item cache release, plus a resume-and-retry wrapper. The values are unchanged. |
| OVDiff, support sets | The released batch of 8 needs a 4 GB attention softmax, which runs out of memory on a V100. | Batch size 2. The noise comes from a CPU generator, so it is the same noise stream. |
| OVDiff, predict | (1) `--prots` swallowed the positional arguments (the README's own argument order); (2) a release bug: `gt_mask` is unset under SLURM arrays but sizes the output; (3) a release bug: `similarities.items()` on `None` without `--save_sim`. | Positionals first; output sized to the image; a `None` guard. All recorded in `jc_ovdiff44.py prep`; tested on 3 VOC photos. |
| OVDiff, scoring | It predicts at its resized input (short side 448). | The scorer resizes back to the ground-truth size with nearest-neighbour. |

- **Other OVDiff notes:**
  - Its Context and COCO-Object class lists were added in the release's own format; the release itself implements VOC
    only.
  - Its context mode skips background prototypes for the 19 Context stuff classes. That is its design and was left as
    released.
- **Coverage:** every method has every primary item on every column.

## 9. Not yet established and open questions

1. **OVDiff:** prediction is running, and its column plus the rescored report follow automatically. Untested risks: memory
   problems on Context or COCO-Object.
2. **Why NERVE wins on stuff and non-VOC classes.** No ablation exists. Candidate checks:
   - a CPU-only, post hoc split by class frequency and area;
   - our pipeline with class competition across the column's vocabulary (a new pre-registered run);
   - a stuff-aware or multi-word profile.
3. **How to present the protocol asymmetry** (multi-class argmax vs per-word binary) in the paper.
4. **Framing:** we beat NERVE on generated benchmarks (Table 2, COCO-Cap +0.033) but trail it on real photos (VOC −0.036,
   Context −0.054). The paper needs an honest framing. Candidate: training-free class-word maps that hold up across
   generated and real images, competitive with the strongest SD and CLIP methods, with a clear stuff-class limitation.
5. **DenseCRF hurts slightly on real photos** (−0.004 to −0.011). Whether to report it as secondary only.

---

## Appendix: full run 44 report (11 methods; OVDiff pending)

The same numbers as above, plus every threshold (§5) and every class (§6).

## Run 44: real-photo comparison on PASCAL VOC, PASCAL Context and COCO-Object (`PREREG.md`)

Class word given; one binary mask per (photo, class) primary item (class area >= 1%). Every method as released on the same photos (PREREG section 4); ours: MK4-LR, gamma 1, fine-only, fixed 0.55. Class-macro mIoU; 95% paired photo bootstrap (2000 samples). Items: PASCAL VOC (20) 1947 (1409 photos), PASCAL Context (59) 22691 (5104 photos), COCO-Object (80) 9816 (4571 photos).


### 1. Primary contrasts (ours = MK4-LR, V0, raw 0.55)

| # | Contrast | Column | Delta mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | ours - no gate | PASCAL VOC (20) | +0.086 [+0.075, +0.097] | **helps** |
| P2 | ours - no gate | PASCAL Context (59) | +0.090 [+0.076, +0.095] | **helps** |
| P3 | ours - no gate | COCO-Object (80) | +0.041 [+0.034, +0.047] | **helps** |
| P4 | ours - NERVE | PASCAL VOC (20) | -0.036 [-0.052, -0.020] | **hurts** |
| P5 | ours - NERVE | PASCAL Context (59) | -0.054 [-0.064, -0.038] | **hurts** |
| P6 | ours - NERVE | COCO-Object (80) | -0.016 [-0.028, +0.003] | not resolved |

### 2. Main table (mIoU)

| Method | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---:|---:|---:|
| DAAM | 0.417 | 0.234 | 0.353 |
| OVAM (no token opt.) | 0.460 | 0.260 | 0.309 |
| DiffSegmenter | 0.658 | 0.433 | 0.500 |
| iSeg | 0.705 | 0.401 | 0.476 |
| SCLIP | 0.629 | 0.410 | 0.411 |
| ClearCLIP | 0.545 | 0.424 | 0.383 |
| NACLIP | 0.677 | 0.470 | 0.444 |
| ProxyCLIP | 0.645 | 0.459 | 0.438 |
| NERVE | 0.750 | 0.501 | 0.502 |
| OVDiff | — | — | — |
| No gate (ours, ablation) | 0.627 | 0.357 | 0.445 |
| MK4 (ours, ablation) | 0.656 | 0.385 | 0.460 |
| **Ours (MK4-LR), raw** | 0.713 | 0.447 | 0.486 |
| **Ours + DenseCRF** | 0.702 | 0.437 | 0.482 |
| OVAM + DenseCRF | 0.468 | 0.264 | 0.325 |
| No gate + DenseCRF | 0.619 | 0.355 | 0.442 |
| iSeg, class given | 0.759 | 0.521 | 0.549 |

### 3. Ours - each baseline (raw 0.55)

| Baseline | PASCAL VOC (20) | PASCAL Context (59) | COCO-Object (80) |
|---|---|---|---|
| DAAM | +0.296 [+0.281, +0.310] | +0.213 [+0.202, +0.219] | +0.133 [+0.124, +0.144] |
| OVAM (no token opt.) | +0.254 [+0.240, +0.267] | +0.186 [+0.174, +0.191] | +0.178 [+0.168, +0.188] |
| DiffSegmenter | +0.056 [+0.042, +0.069] | +0.014 [-0.000, +0.018] | -0.014 [-0.021, -0.005] |
| iSeg | +0.008 [-0.012, +0.028] | +0.046 [+0.038, +0.057] | +0.010 [-0.001, +0.028] |
| SCLIP | +0.085 [+0.069, +0.101] | +0.037 [+0.028, +0.053] | +0.076 [+0.065, +0.093] |
| ClearCLIP | +0.168 [+0.151, +0.187] | +0.023 [+0.013, +0.039] | +0.104 [+0.092, +0.122] |
| NACLIP | +0.036 [+0.021, +0.053] | -0.023 [-0.031, -0.011] | +0.042 [+0.030, +0.059] |
| ProxyCLIP | +0.069 [+0.051, +0.087] | -0.012 [-0.021, -0.004] | +0.049 [+0.036, +0.066] |
| NERVE | -0.036 [-0.052, -0.020] | -0.054 [-0.064, -0.038] | -0.016 [-0.028, +0.003] |
| OVDiff | — | — | — |
| No gate (ours, ablation) | +0.086 [+0.075, +0.097] | +0.090 [+0.076, +0.095] | +0.041 [+0.034, +0.047] |
| MK4 (ours, ablation) | +0.057 [+0.048, +0.066] | +0.062 [+0.057, +0.066] | +0.026 [+0.020, +0.032] |
| OVAM + DenseCRF | +0.245 [+0.229, +0.260] | +0.183 [+0.170, +0.189] | +0.161 [+0.151, +0.172] |
| No gate + DenseCRF | +0.094 [+0.082, +0.106] | +0.092 [+0.079, +0.098] | +0.044 [+0.037, +0.051] |
| iSeg, class given | -0.046 [-0.062, -0.030] | -0.074 [-0.088, -0.065] | -0.062 [-0.076, -0.049] |

### 4. Subsets (raw 0.55)

- **Context, 20 VOC classes** (7925 items): ours 0.663; no gate 0.589; NERVE 0.632; ProxyCLIP 0.600; iSeg 0.620; ours - no gate +0.074 [+0.068, +0.080]
- **Context, other 39** (14766 items): ours 0.336; no gate 0.237; NERVE 0.433; ProxyCLIP 0.387; iSeg 0.289; ours - no gate +0.099 [+0.077, +0.104]
- **COCO-Object, 20 VOC classes** (5414 items): ours 0.568; no gate 0.476; NERVE 0.533; ProxyCLIP 0.474; iSeg 0.502; ours - no gate +0.092 [+0.083, +0.100]
- **COCO-Object, other 60** (4402 items): ours 0.459; no gate 0.435; NERVE 0.492; ProxyCLIP 0.426; iSeg 0.468; ours - no gate +0.024 [+0.015, +0.031]

### 5. Every threshold (ours, raw; descriptive)

| Column | Map | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASCAL VOC (20) | uL|V0|raw | 0.558 | 0.585 | 0.611 | 0.634 | 0.657 | 0.677 | 0.693 | 0.705 | 0.713 | 0.715 | 0.713 | 0.707 | 0.696 | 0.678 | 0.655 | 0.624 | 0.586 |
| PASCAL VOC (20) | u-|V0|raw | 0.479 | 0.505 | 0.530 | 0.553 | 0.575 | 0.592 | 0.606 | 0.617 | 0.626 | 0.629 | 0.627 | 0.618 | 0.605 | 0.585 | 0.559 | 0.525 | 0.485 |
| PASCAL Context (59) | uL|V0|raw | 0.343 | 0.359 | 0.373 | 0.388 | 0.405 | 0.418 | 0.429 | 0.437 | 0.443 | 0.446 | 0.447 | 0.445 | 0.439 | 0.429 | 0.417 | 0.400 | 0.380 |
| PASCAL Context (59) | u-|V0|raw | 0.285 | 0.296 | 0.307 | 0.317 | 0.327 | 0.336 | 0.343 | 0.350 | 0.354 | 0.357 | 0.357 | 0.354 | 0.348 | 0.339 | 0.326 | 0.309 | 0.289 |
| COCO-Object (80) | uL|V0|raw | 0.352 | 0.372 | 0.391 | 0.408 | 0.425 | 0.440 | 0.454 | 0.466 | 0.475 | 0.482 | 0.486 | 0.487 | 0.483 | 0.475 | 0.463 | 0.446 | 0.425 |
| COCO-Object (80) | u-|V0|raw | 0.301 | 0.321 | 0.340 | 0.358 | 0.376 | 0.392 | 0.407 | 0.420 | 0.431 | 0.440 | 0.445 | 0.447 | 0.445 | 0.438 | 0.427 | 0.411 | 0.391 |

### 6. Per class (raw 0.55)


PASCAL VOC (20):

| Class | items | DAAM | OVAM (no token opt.) | DiffSegmenter | iSeg | SCLIP | ClearCLIP | NACLIP | ProxyCLIP | NERVE | No gate (ours, ablation) | MK4 (ours, ablation) | Ours (MK4-LR), raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 82 | 0.616 | 0.567 | 0.837 | 0.921 | 0.537 | 0.439 | 0.657 | 0.511 | 0.617 | 0.776 | 0.794 | 0.853 |
| bicycle | 64 | 0.375 | 0.356 | 0.371 | 0.553 | 0.331 | 0.413 | 0.394 | 0.455 | 0.444 | 0.412 | 0.476 | 0.485 |
| bird | 91 | 0.562 | 0.548 | 0.786 | 0.726 | 0.729 | 0.605 | 0.765 | 0.728 | 0.850 | 0.803 | 0.867 | 0.867 |
| boat | 64 | 0.568 | 0.582 | 0.770 | 0.814 | 0.544 | 0.430 | 0.599 | 0.604 | 0.675 | 0.731 | 0.718 | 0.756 |
| bottle | 71 | 0.413 | 0.427 | 0.600 | 0.638 | 0.587 | 0.583 | 0.673 | 0.635 | 0.667 | 0.617 | 0.441 | 0.539 |
| bus | 70 | 0.543 | 0.696 | 0.839 | 0.911 | 0.831 | 0.773 | 0.815 | 0.879 | 0.890 | 0.848 | 0.825 | 0.835 |
| car | 100 | 0.326 | 0.406 | 0.577 | 0.794 | 0.739 | 0.670 | 0.799 | 0.807 | 0.837 | 0.546 | 0.603 | 0.638 |
| cat | 118 | 0.501 | 0.557 | 0.722 | 0.937 | 0.839 | 0.800 | 0.854 | 0.898 | 0.934 | 0.803 | 0.866 | 0.895 |
| chair | 112 | 0.164 | 0.168 | 0.461 | 0.328 | 0.337 | 0.302 | 0.420 | 0.406 | 0.569 | 0.328 | 0.282 | 0.456 |
| cow | 69 | 0.519 | 0.646 | 0.821 | 0.921 | 0.791 | 0.762 | 0.827 | 0.889 | 0.929 | 0.839 | 0.845 | 0.859 |
| dining table | 73 | 0.269 | 0.236 | 0.466 | 0.472 | 0.586 | 0.542 | 0.625 | 0.624 | 0.691 | 0.405 | 0.374 | 0.447 |
| dog | 127 | 0.370 | 0.449 | 0.798 | 0.901 | 0.824 | 0.765 | 0.825 | 0.842 | 0.910 | 0.778 | 0.847 | 0.876 |
| horse | 77 | 0.451 | 0.577 | 0.775 | 0.878 | 0.615 | 0.595 | 0.718 | 0.706 | 0.736 | 0.787 | 0.862 | 0.870 |
| motorbike | 74 | 0.667 | 0.701 | 0.757 | 0.648 | 0.640 | 0.621 | 0.690 | 0.675 | 0.767 | 0.732 | 0.764 | 0.778 |
| person | 392 | 0.168 | 0.248 | 0.391 | 0.446 | 0.711 | 0.015 | 0.607 | 0.026 | 0.653 | 0.279 | 0.433 | 0.694 |
| potted plant | 66 | 0.282 | 0.160 | 0.492 | 0.538 | 0.562 | 0.603 | 0.611 | 0.622 | 0.682 | 0.363 | 0.381 | 0.517 |
| sheep | 54 | 0.476 | 0.671 | 0.841 | 0.918 | 0.750 | 0.568 | 0.839 | 0.837 | 0.944 | 0.838 | 0.881 | 0.899 |
| sofa | 90 | 0.270 | 0.306 | 0.656 | 0.505 | 0.571 | 0.598 | 0.665 | 0.727 | 0.736 | 0.548 | 0.659 | 0.725 |
| train | 83 | 0.491 | 0.599 | 0.726 | 0.735 | 0.616 | 0.549 | 0.702 | 0.759 | 0.855 | 0.729 | 0.761 | 0.805 |
| tv | 70 | 0.308 | 0.291 | 0.469 | 0.528 | 0.434 | 0.266 | 0.460 | 0.268 | 0.605 | 0.377 | 0.449 | 0.474 |

PASCAL Context (59):

| Class | items | DAAM | OVAM (no token opt.) | DiffSegmenter | iSeg | SCLIP | ClearCLIP | NACLIP | ProxyCLIP | NERVE | No gate (ours, ablation) | MK4 (ours, ablation) | Ours (MK4-LR), raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 286 | 0.546 | 0.510 | 0.741 | 0.777 | 0.566 | 0.548 | 0.595 | 0.524 | 0.528 | 0.704 | 0.687 | 0.763 |
| bicycle | 239 | 0.410 | 0.492 | 0.576 | 0.567 | 0.567 | 0.594 | 0.580 | 0.646 | 0.597 | 0.499 | 0.545 | 0.576 |
| bird | 285 | 0.518 | 0.549 | 0.713 | 0.664 | 0.655 | 0.656 | 0.717 | 0.677 | 0.748 | 0.734 | 0.757 | 0.794 |
| boat | 202 | 0.459 | 0.494 | 0.621 | 0.567 | 0.557 | 0.514 | 0.584 | 0.563 | 0.555 | 0.614 | 0.562 | 0.589 |
| bottle | 250 | 0.367 | 0.306 | 0.529 | 0.527 | 0.492 | 0.586 | 0.574 | 0.582 | 0.649 | 0.511 | 0.391 | 0.541 |
| bus | 180 | 0.532 | 0.662 | 0.807 | 0.822 | 0.747 | 0.751 | 0.720 | 0.813 | 0.818 | 0.802 | 0.773 | 0.786 |
| car | 484 | 0.293 | 0.385 | 0.532 | 0.572 | 0.706 | 0.688 | 0.714 | 0.713 | 0.732 | 0.497 | 0.571 | 0.602 |
| cat | 500 | 0.470 | 0.553 | 0.701 | 0.890 | 0.819 | 0.777 | 0.868 | 0.851 | 0.875 | 0.785 | 0.855 | 0.865 |
| chair | 611 | 0.184 | 0.190 | 0.469 | 0.397 | 0.327 | 0.327 | 0.330 | 0.373 | 0.484 | 0.343 | 0.299 | 0.425 |
| cow | 121 | 0.522 | 0.656 | 0.796 | 0.816 | 0.769 | 0.751 | 0.802 | 0.827 | 0.834 | 0.822 | 0.812 | 0.824 |
| dining table | 699 | 0.121 | 0.117 | 0.317 | 0.312 | 0.249 | 0.310 | 0.373 | 0.405 | 0.397 | 0.226 | 0.234 | 0.268 |
| dog | 594 | 0.328 | 0.406 | 0.754 | 0.853 | 0.795 | 0.764 | 0.835 | 0.822 | 0.863 | 0.739 | 0.801 | 0.834 |
| horse | 215 | 0.437 | 0.572 | 0.743 | 0.828 | 0.645 | 0.612 | 0.706 | 0.704 | 0.582 | 0.744 | 0.807 | 0.833 |
| motorbike | 219 | 0.628 | 0.687 | 0.726 | 0.485 | 0.609 | 0.684 | 0.545 | 0.650 | 0.728 | 0.699 | 0.722 | 0.732 |
| person | 1792 | 0.154 | 0.266 | 0.341 | 0.565 | 0.198 | 0.115 | 0.104 | 0.022 | 0.105 | 0.289 | 0.436 | 0.640 |
| potted plant | 243 | 0.271 | 0.201 | 0.468 | 0.469 | 0.512 | 0.542 | 0.535 | 0.563 | 0.594 | 0.378 | 0.369 | 0.511 |
| sheep | 133 | 0.452 | 0.628 | 0.788 | 0.834 | 0.767 | 0.742 | 0.813 | 0.813 | 0.868 | 0.743 | 0.809 | 0.817 |
| sofa | 402 | 0.256 | 0.314 | 0.631 | 0.528 | 0.488 | 0.500 | 0.577 | 0.635 | 0.684 | 0.564 | 0.575 | 0.695 |
| train | 229 | 0.461 | 0.566 | 0.670 | 0.619 | 0.572 | 0.568 | 0.590 | 0.639 | 0.611 | 0.689 | 0.691 | 0.719 |
| tv | 241 | 0.248 | 0.281 | 0.425 | 0.303 | 0.186 | 0.217 | 0.089 | 0.177 | 0.398 | 0.405 | 0.434 | 0.453 |
| bag | 105 | 0.158 | 0.067 | 0.205 | 0.410 | 0.433 | 0.451 | 0.509 | 0.407 | 0.547 | 0.196 | 0.153 | 0.221 |
| bed | 61 | 0.236 | 0.329 | 0.520 | 0.236 | 0.387 | 0.478 | 0.410 | 0.468 | 0.259 | 0.479 | 0.439 | 0.487 |
| bedclothes | 152 | 0.247 | 0.206 | 0.514 | 0.320 | 0.274 | 0.158 | 0.205 | 0.165 | 0.457 | 0.458 | 0.497 | 0.600 |
| bench | 35 | 0.175 | 0.145 | 0.458 | 0.165 | 0.123 | 0.265 | 0.198 | 0.309 | 0.551 | 0.335 | 0.298 | 0.472 |
| book | 136 | 0.158 | 0.078 | 0.218 | 0.099 | 0.085 | 0.264 | 0.184 | 0.213 | 0.209 | 0.185 | 0.071 | 0.195 |
| building | 1385 | 0.158 | 0.275 | 0.342 | 0.039 | 0.078 | 0.073 | 0.084 | 0.044 | 0.200 | 0.267 | 0.298 | 0.344 |
| cabinet | 274 | 0.116 | 0.168 | 0.329 | 0.230 | 0.229 | 0.232 | 0.311 | 0.292 | 0.350 | 0.214 | 0.237 | 0.350 |
| ceiling | 267 | 0.147 | 0.145 | 0.462 | 0.275 | 0.431 | 0.376 | 0.502 | 0.369 | 0.527 | 0.328 | 0.266 | 0.444 |
| cloth | 345 | 0.143 | 0.141 | 0.274 | 0.244 | 0.133 | 0.196 | 0.235 | 0.153 | 0.208 | 0.226 | 0.185 | 0.293 |
| computer | 49 | 0.143 | 0.099 | 0.191 | 0.223 | 0.183 | 0.178 | 0.160 | 0.196 | 0.321 | 0.209 | 0.151 | 0.201 |
| cup | 104 | 0.173 | 0.086 | 0.221 | 0.231 | 0.206 | 0.188 | 0.232 | 0.212 | 0.240 | 0.186 | 0.106 | 0.214 |
| curtain | 172 | 0.220 | 0.194 | 0.470 | 0.405 | 0.462 | 0.495 | 0.587 | 0.486 | 0.658 | 0.351 | 0.387 | 0.562 |
| door | 215 | 0.140 | 0.126 | 0.313 | 0.290 | 0.288 | 0.325 | 0.357 | 0.370 | 0.402 | 0.200 | 0.202 | 0.303 |
| fence | 438 | 0.194 | 0.189 | 0.411 | 0.255 | 0.247 | 0.290 | 0.333 | 0.347 | 0.425 | 0.370 | 0.374 | 0.406 |
| floor | 862 | 0.120 | 0.178 | 0.606 | 0.385 | 0.556 | 0.577 | 0.690 | 0.676 | 0.658 | 0.322 | 0.316 | 0.372 |
| flower | 52 | 0.184 | 0.164 | 0.338 | 0.347 | 0.381 | 0.417 | 0.381 | 0.465 | 0.491 | 0.297 | 0.216 | 0.426 |
| food | 101 | 0.383 | 0.265 | 0.430 | 0.535 | 0.364 | 0.407 | 0.450 | 0.411 | 0.368 | 0.440 | 0.463 | 0.528 |
| grass | 1238 | 0.063 | 0.216 | 0.229 | 0.616 | 0.672 | 0.653 | 0.753 | 0.758 | 0.756 | 0.095 | 0.327 | 0.328 |
| ground | 1389 | 0.080 | 0.172 | 0.420 | 0.134 | 0.172 | 0.154 | 0.171 | 0.131 | 0.136 | 0.121 | 0.286 | 0.227 |
| keyboard | 42 | 0.335 | 0.211 | 0.359 | 0.412 | 0.454 | 0.513 | 0.638 | 0.619 | 0.709 | 0.398 | 0.513 | 0.500 |
| light | 155 | 0.064 | 0.048 | 0.182 | 0.310 | 0.294 | 0.299 | 0.316 | 0.351 | 0.304 | 0.066 | 0.060 | 0.118 |
| mountain | 264 | 0.275 | 0.244 | 0.531 | 0.373 | 0.336 | 0.387 | 0.486 | 0.461 | 0.577 | 0.334 | 0.294 | 0.412 |
| mouse | 2 | 0.081 | 0.056 | 0.123 | 0.809 | 0.640 | 0.589 | 0.755 | 0.518 | 0.607 | 0.063 | 0.719 | 0.769 |
| plate | 111 | 0.116 | 0.077 | 0.233 | 0.352 | 0.264 | 0.334 | 0.380 | 0.387 | 0.264 | 0.145 | 0.096 | 0.197 |
| platform | 71 | 0.048 | 0.111 | 0.096 | 0.000 | 0.039 | 0.038 | 0.044 | 0.025 | 0.158 | 0.105 | 0.090 | 0.064 |
| road | 532 | 0.064 | 0.146 | 0.521 | 0.203 | 0.523 | 0.483 | 0.585 | 0.562 | 0.431 | 0.106 | 0.128 | 0.072 |
| rock | 161 | 0.249 | 0.194 | 0.414 | 0.352 | 0.557 | 0.504 | 0.589 | 0.586 | 0.589 | 0.304 | 0.281 | 0.420 |
| shelves | 175 | 0.114 | 0.110 | 0.228 | 0.153 | 0.150 | 0.107 | 0.118 | 0.124 | 0.287 | 0.149 | 0.132 | 0.167 |
| sidewalk | 160 | 0.057 | 0.091 | 0.223 | 0.264 | 0.370 | 0.348 | 0.464 | 0.429 | 0.411 | 0.075 | 0.081 | 0.170 |
| sign | 125 | 0.231 | 0.151 | 0.407 | 0.326 | 0.073 | 0.139 | 0.349 | 0.256 | 0.475 | 0.359 | 0.268 | 0.407 |
| sky | 1485 | 0.052 | 0.164 | 0.233 | 0.008 | 0.556 | 0.667 | 0.805 | 0.722 | 0.801 | 0.048 | 0.232 | 0.137 |
| snow | 94 | 0.311 | 0.389 | 0.547 | 0.613 | 0.620 | 0.622 | 0.716 | 0.734 | 0.722 | 0.539 | 0.662 | 0.711 |
| track | 166 | 0.199 | 0.174 | 0.296 | 0.000 | 0.050 | 0.028 | 0.111 | 0.063 | 0.122 | 0.197 | 0.213 | 0.218 |
| tree | 1497 | 0.068 | 0.219 | 0.267 | 0.269 | 0.377 | 0.511 | 0.532 | 0.523 | 0.486 | 0.148 | 0.302 | 0.206 |
| truck | 34 | 0.199 | 0.199 | 0.281 | 0.402 | 0.426 | 0.367 | 0.486 | 0.465 | 0.489 | 0.238 | 0.271 | 0.271 |
| wall | 1381 | 0.060 | 0.168 | 0.331 | 0.105 | 0.177 | 0.229 | 0.373 | 0.225 | 0.091 | 0.121 | 0.209 | 0.195 |
| water | 415 | 0.075 | 0.230 | 0.347 | 0.203 | 0.751 | 0.742 | 0.815 | 0.804 | 0.787 | 0.113 | 0.395 | 0.422 |
| window | 347 | 0.131 | 0.161 | 0.364 | 0.332 | 0.265 | 0.318 | 0.347 | 0.376 | 0.433 | 0.266 | 0.162 | 0.349 |
| wood | 169 | 0.157 | 0.144 | 0.277 | 0.336 | 0.354 | 0.388 | 0.417 | 0.396 | 0.394 | 0.198 | 0.218 | 0.319 |

COCO-Object (80):

| Class | items | DAAM | OVAM (no token opt.) | DiffSegmenter | iSeg | SCLIP | ClearCLIP | NACLIP | ProxyCLIP | NERVE | No gate (ours, ablation) | MK4 (ours, ablation) | Ours (MK4-LR), raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 82 | 0.498 | 0.428 | 0.692 | 0.737 | 0.506 | 0.423 | 0.559 | 0.352 | 0.612 | 0.603 | 0.698 | 0.730 |
| bicycle | 90 | 0.355 | 0.285 | 0.473 | 0.485 | 0.558 | 0.538 | 0.530 | 0.604 | 0.622 | 0.341 | 0.394 | 0.440 |
| bird | 69 | 0.478 | 0.444 | 0.554 | 0.578 | 0.481 | 0.318 | 0.558 | 0.282 | 0.597 | 0.557 | 0.579 | 0.636 |
| boat | 88 | 0.443 | 0.395 | 0.500 | 0.523 | 0.444 | 0.275 | 0.474 | 0.457 | 0.550 | 0.443 | 0.450 | 0.454 |
| bottle | 169 | 0.205 | 0.099 | 0.364 | 0.194 | 0.251 | 0.307 | 0.428 | 0.297 | 0.252 | 0.250 | 0.216 | 0.332 |
| bus | 169 | 0.505 | 0.585 | 0.715 | 0.743 | 0.604 | 0.603 | 0.597 | 0.660 | 0.774 | 0.676 | 0.677 | 0.709 |
| car | 298 | 0.274 | 0.248 | 0.359 | 0.430 | 0.550 | 0.537 | 0.578 | 0.539 | 0.582 | 0.242 | 0.293 | 0.355 |
| cat | 174 | 0.476 | 0.480 | 0.734 | 0.864 | 0.792 | 0.757 | 0.837 | 0.814 | 0.839 | 0.732 | 0.768 | 0.801 |
| chair | 451 | 0.131 | 0.109 | 0.319 | 0.189 | 0.180 | 0.179 | 0.241 | 0.188 | 0.271 | 0.182 | 0.168 | 0.255 |
| cow | 73 | 0.540 | 0.600 | 0.754 | 0.739 | 0.703 | 0.647 | 0.722 | 0.709 | 0.749 | 0.767 | 0.723 | 0.744 |
| dining table | 442 | 0.198 | 0.237 | 0.242 | 0.147 | 0.268 | 0.248 | 0.318 | 0.269 | 0.184 | 0.250 | 0.295 | 0.303 |
| dog | 147 | 0.364 | 0.388 | 0.736 | 0.740 | 0.674 | 0.646 | 0.716 | 0.667 | 0.727 | 0.688 | 0.699 | 0.767 |
| horse | 108 | 0.427 | 0.499 | 0.652 | 0.729 | 0.630 | 0.628 | 0.679 | 0.720 | 0.672 | 0.609 | 0.695 | 0.717 |
| motorbike | 121 | 0.553 | 0.562 | 0.647 | 0.585 | 0.632 | 0.522 | 0.524 | 0.503 | 0.734 | 0.569 | 0.643 | 0.662 |
| person | 2243 | 0.161 | 0.235 | 0.338 | 0.181 | 0.256 | 0.000 | 0.346 | 0.000 | 0.014 | 0.258 | 0.402 | 0.573 |
| potted plant | 123 | 0.200 | 0.104 | 0.318 | 0.211 | 0.332 | 0.374 | 0.406 | 0.342 | 0.253 | 0.247 | 0.234 | 0.368 |
| sheep | 56 | 0.397 | 0.539 | 0.722 | 0.772 | 0.705 | 0.624 | 0.734 | 0.745 | 0.798 | 0.673 | 0.724 | 0.736 |
| sofa | 188 | 0.227 | 0.212 | 0.517 | 0.425 | 0.482 | 0.488 | 0.586 | 0.608 | 0.673 | 0.386 | 0.461 | 0.610 |
| train | 151 | 0.442 | 0.532 | 0.644 | 0.579 | 0.548 | 0.477 | 0.527 | 0.554 | 0.599 | 0.628 | 0.672 | 0.669 |
| tv | 172 | 0.250 | 0.222 | 0.457 | 0.182 | 0.138 | 0.145 | 0.162 | 0.175 | 0.160 | 0.416 | 0.434 | 0.493 |
| apple | 52 | 0.244 | 0.254 | 0.317 | 0.393 | 0.478 | 0.394 | 0.512 | 0.446 | 0.497 | 0.288 | 0.273 | 0.295 |
| backpack | 77 | 0.129 | 0.066 | 0.177 | 0.300 | 0.433 | 0.379 | 0.480 | 0.403 | 0.377 | 0.124 | 0.145 | 0.190 |
| banana | 80 | 0.498 | 0.462 | 0.650 | 0.681 | 0.616 | 0.615 | 0.643 | 0.716 | 0.726 | 0.630 | 0.650 | 0.678 |
| baseball bat | 22 | 0.212 | 0.076 | 0.213 | 0.379 | 0.321 | 0.321 | 0.289 | 0.328 | 0.233 | 0.246 | 0.183 | 0.187 |
| baseball glove | 20 | 0.250 | 0.140 | 0.385 | 0.522 | 0.086 | 0.147 | 0.133 | 0.185 | 0.390 | 0.275 | 0.288 | 0.265 |
| bear | 46 | 0.439 | 0.549 | 0.805 | 0.825 | 0.802 | 0.743 | 0.817 | 0.822 | 0.816 | 0.766 | 0.789 | 0.797 |
| bed | 141 | 0.302 | 0.399 | 0.510 | 0.491 | 0.000 | 0.493 | 0.000 | 0.000 | 0.580 | 0.497 | 0.528 | 0.570 |
| bench | 153 | 0.224 | 0.203 | 0.449 | 0.306 | 0.222 | 0.333 | 0.301 | 0.357 | 0.524 | 0.378 | 0.370 | 0.500 |
| book | 129 | 0.193 | 0.102 | 0.281 | 0.340 | 0.104 | 0.202 | 0.197 | 0.175 | 0.293 | 0.243 | 0.139 | 0.291 |
| bowl | 206 | 0.259 | 0.280 | 0.340 | 0.123 | 0.107 | 0.061 | 0.121 | 0.090 | 0.065 | 0.357 | 0.323 | 0.436 |
| broccoli | 63 | 0.459 | 0.463 | 0.571 | 0.508 | 0.458 | 0.492 | 0.476 | 0.550 | 0.493 | 0.519 | 0.568 | 0.584 |
| cake | 109 | 0.489 | 0.515 | 0.623 | 0.601 | 0.574 | 0.473 | 0.593 | 0.482 | 0.531 | 0.571 | 0.417 | 0.443 |
| carrot | 58 | 0.399 | 0.373 | 0.397 | 0.380 | 0.425 | 0.439 | 0.515 | 0.527 | 0.516 | 0.308 | 0.341 | 0.328 |
| cell phone | 61 | 0.327 | 0.176 | 0.521 | 0.460 | 0.270 | 0.229 | 0.213 | 0.239 | 0.220 | 0.425 | 0.432 | 0.520 |
| clock | 77 | 0.448 | 0.451 | 0.710 | 0.609 | 0.495 | 0.395 | 0.562 | 0.502 | 0.549 | 0.627 | 0.623 | 0.596 |
| cup | 222 | 0.168 | 0.102 | 0.267 | 0.211 | 0.180 | 0.143 | 0.216 | 0.148 | 0.124 | 0.183 | 0.163 | 0.228 |
| donut | 53 | 0.547 | 0.604 | 0.707 | 0.654 | 0.628 | 0.553 | 0.651 | 0.602 | 0.706 | 0.688 | 0.728 | 0.714 |
| elephant | 83 | 0.680 | 0.734 | 0.789 | 0.845 | 0.777 | 0.758 | 0.790 | 0.837 | 0.869 | 0.787 | 0.747 | 0.741 |
| fire hydrant | 50 | 0.651 | 0.243 | 0.806 | 0.610 | 0.603 | 0.586 | 0.646 | 0.647 | 0.790 | 0.770 | 0.757 | 0.741 |
| fork | 67 | 0.134 | 0.111 | 0.227 | 0.291 | 0.317 | 0.335 | 0.375 | 0.395 | 0.371 | 0.129 | 0.155 | 0.111 |
| frisbee | 24 | 0.615 | 0.504 | 0.660 | 0.369 | 0.476 | 0.321 | 0.594 | 0.522 | 0.665 | 0.575 | 0.758 | 0.548 |
| giraffe | 95 | 0.637 | 0.641 | 0.741 | 0.788 | 0.681 | 0.648 | 0.715 | 0.744 | 0.783 | 0.744 | 0.721 | 0.710 |
| hair drier | 5 | 0.254 | 0.065 | 0.303 | 0.611 | 0.337 | 0.316 | 0.178 | 0.042 | 0.415 | 0.304 | 0.375 | 0.495 |
| handbag | 116 | 0.110 | 0.052 | 0.196 | 0.180 | 0.286 | 0.250 | 0.320 | 0.246 | 0.274 | 0.098 | 0.097 | 0.148 |
| hot dog | 41 | 0.535 | 0.199 | 0.678 | 0.482 | 0.511 | 0.406 | 0.508 | 0.499 | 0.488 | 0.633 | 0.647 | 0.657 |
| keyboard | 82 | 0.381 | 0.290 | 0.390 | 0.393 | 0.453 | 0.403 | 0.501 | 0.471 | 0.617 | 0.415 | 0.576 | 0.565 |
| kite | 46 | 0.386 | 0.332 | 0.498 | 0.479 | 0.337 | 0.232 | 0.391 | 0.364 | 0.537 | 0.565 | 0.553 | 0.565 |
| knife | 56 | 0.201 | 0.094 | 0.248 | 0.080 | 0.122 | 0.116 | 0.137 | 0.141 | 0.231 | 0.153 | 0.176 | 0.245 |
| laptop | 139 | 0.281 | 0.225 | 0.495 | 0.448 | 0.429 | 0.374 | 0.401 | 0.361 | 0.564 | 0.361 | 0.448 | 0.467 |
| microwave | 32 | 0.275 | 0.198 | 0.453 | 0.460 | 0.314 | 0.164 | 0.248 | 0.320 | 0.385 | 0.286 | 0.396 | 0.407 |
| mouse | 25 | 0.277 | 0.122 | 0.395 | 0.403 | 0.262 | 0.269 | 0.376 | 0.425 | 0.585 | 0.256 | 0.409 | 0.262 |
| orange | 61 | 0.425 | 0.424 | 0.553 | 0.689 | 0.644 | 0.651 | 0.713 | 0.718 | 0.691 | 0.506 | 0.354 | 0.385 |
| oven | 104 | 0.247 | 0.214 | 0.376 | 0.330 | 0.362 | 0.311 | 0.373 | 0.405 | 0.401 | 0.292 | 0.319 | 0.310 |
| parking meter | 28 | 0.329 | 0.125 | 0.542 | 0.458 | 0.415 | 0.121 | 0.319 | 0.329 | 0.623 | 0.471 | 0.569 | 0.567 |
| pizza | 130 | 0.621 | 0.665 | 0.783 | 0.692 | 0.681 | 0.592 | 0.684 | 0.657 | 0.670 | 0.774 | 0.744 | 0.747 |
| refrigerator | 92 | 0.256 | 0.273 | 0.494 | 0.412 | 0.437 | 0.453 | 0.451 | 0.526 | 0.676 | 0.364 | 0.397 | 0.394 |
| remote | 35 | 0.165 | 0.092 | 0.404 | 0.345 | 0.307 | 0.180 | 0.360 | 0.313 | 0.397 | 0.277 | 0.133 | 0.219 |
| sandwich | 84 | 0.543 | 0.557 | 0.668 | 0.687 | 0.590 | 0.491 | 0.594 | 0.566 | 0.595 | 0.630 | 0.674 | 0.682 |
| scissors | 15 | 0.435 | 0.564 | 0.614 | 0.258 | 0.235 | 0.260 | 0.209 | 0.301 | 0.660 | 0.592 | 0.549 | 0.542 |
| sink | 115 | 0.200 | 0.176 | 0.370 | 0.204 | 0.326 | 0.304 | 0.383 | 0.413 | 0.415 | 0.312 | 0.361 | 0.349 |
| skateboard | 56 | 0.407 | 0.329 | 0.408 | 0.464 | 0.223 | 0.204 | 0.244 | 0.258 | 0.234 | 0.486 | 0.553 | 0.525 |
| skis | 36 | 0.201 | 0.164 | 0.213 | 0.270 | 0.061 | 0.102 | 0.074 | 0.094 | 0.058 | 0.216 | 0.201 | 0.193 |
| snowboard | 17 | 0.285 | 0.142 | 0.287 | 0.181 | 0.046 | 0.018 | 0.013 | 0.033 | 0.131 | 0.241 | 0.283 | 0.292 |
| spoon | 47 | 0.222 | 0.137 | 0.205 | 0.212 | 0.277 | 0.232 | 0.395 | 0.318 | 0.343 | 0.177 | 0.190 | 0.129 |
| sports ball | 14 | 0.394 | 0.067 | 0.572 | 0.457 | 0.355 | 0.371 | 0.481 | 0.633 | 0.468 | 0.318 | 0.196 | 0.244 |
| stop sign | 39 | 0.447 | 0.391 | 0.808 | 0.579 | 0.613 | 0.620 | 0.771 | 0.796 | 0.846 | 0.828 | 0.838 | 0.840 |
| suitcase | 81 | 0.333 | 0.310 | 0.530 | 0.466 | 0.460 | 0.410 | 0.405 | 0.503 | 0.547 | 0.520 | 0.519 | 0.554 |
| surfboard | 74 | 0.426 | 0.352 | 0.528 | 0.596 | 0.326 | 0.224 | 0.369 | 0.347 | 0.461 | 0.524 | 0.520 | 0.598 |
| teddy bear | 82 | 0.401 | 0.420 | 0.666 | 0.685 | 0.616 | 0.578 | 0.621 | 0.669 | 0.744 | 0.673 | 0.630 | 0.696 |
| tennis racket | 79 | 0.392 | 0.243 | 0.511 | 0.555 | 0.159 | 0.288 | 0.295 | 0.433 | 0.125 | 0.512 | 0.529 | 0.496 |
| tie | 59 | 0.239 | 0.097 | 0.385 | 0.523 | 0.150 | 0.135 | 0.161 | 0.153 | 0.162 | 0.263 | 0.153 | 0.195 |
| toaster | 4 | 0.246 | 0.207 | 0.514 | 0.399 | 0.443 | 0.573 | 0.518 | 0.612 | 0.810 | 0.389 | 0.354 | 0.483 |
| toilet | 133 | 0.413 | 0.415 | 0.544 | 0.571 | 0.208 | 0.236 | 0.264 | 0.300 | 0.402 | 0.539 | 0.576 | 0.551 |
| toothbrush | 9 | 0.366 | 0.294 | 0.306 | 0.491 | 0.385 | 0.327 | 0.454 | 0.519 | 0.582 | 0.293 | 0.347 | 0.277 |
| traffic light | 61 | 0.265 | 0.070 | 0.349 | 0.515 | 0.309 | 0.367 | 0.474 | 0.508 | 0.573 | 0.257 | 0.371 | 0.333 |
| truck | 173 | 0.332 | 0.339 | 0.478 | 0.468 | 0.372 | 0.394 | 0.436 | 0.439 | 0.431 | 0.389 | 0.406 | 0.439 |
| umbrella | 137 | 0.397 | 0.382 | 0.726 | 0.644 | 0.546 | 0.510 | 0.610 | 0.635 | 0.755 | 0.650 | 0.638 | 0.704 |
| vase | 67 | 0.305 | 0.216 | 0.523 | 0.382 | 0.231 | 0.183 | 0.181 | 0.197 | 0.276 | 0.478 | 0.482 | 0.484 |
| wine glass | 61 | 0.204 | 0.144 | 0.356 | 0.481 | 0.464 | 0.448 | 0.480 | 0.489 | 0.427 | 0.214 | 0.254 | 0.345 |
| zebra | 79 | 0.635 | 0.669 | 0.751 | 0.838 | 0.790 | 0.715 | 0.771 | 0.782 | 0.817 | 0.729 | 0.676 | 0.701 |

### 7. Coverage

- PASCAL VOC (20): methods with missing items: none
- PASCAL Context (59): methods with missing items: none
- COCO-Object (80): methods with missing items: none

BRACE stays frozen; nothing here is a promotion.
