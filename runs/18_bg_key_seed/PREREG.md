# Background-aware cross-attention seed (pre-registration)

Written and committed before any image of this run is scored. BRACE is not
changed by this run, whatever the outcome.

## Motivation (post hoc, exploratory)

The seed × propagation swap (`runs/17_seed_walk_swap`) and its breakdown showed
that on COCO-Cap the frozen pipeline loses to NERVE on small objects and on
images where the map floods far beyond the object (79 of 707 images below 0.3
IoU; 58 of them in the smallest object-size third; recall 0.75 with a
false-positive area about 16 times the object). The flood is already in the
cross-attention seed (0.18 IoU on those images; NERVE's CLIP seed 0.56).
NERVE's seed is a softmax of the class against a fixed list of background
words. The frozen seed has no such reference.

## Question

Does a cross-attention seed in which the class word competes with fixed,
class-agnostic background words inside the softmax improve the frozen pipeline
on COCO-Cap without hurting VOC-Sim? Stable Diffusion only; no CLIP image
encoder, no labels, no training.

## Design

One live generation per image with the frozen runtime
(`scripts/sg27_runtime.py`). All seeds come from that single generation and
share the trajectory, the 11 frozen cross-attention layers, the 31 calls, the
self-attention-guided layer weights (text-independent), the conditional query
and the head mean. Only the attribution keys change. Every seed is min–max
normalised and passed to the frozen propagation code
(`src/segmentation/sg_combined_v2.py`) unchanged.

Key vectors: each word is encoded on its own by the frozen text encoder
(`<SoT> word`); the output at the word's piece positions gives its keys. A
multi-piece word contributes one key per piece. The class map is the sum of
the softmax columns of the class word's pieces.

| Seed | Keys in the softmax | Class map |
|---|---|---|
| W | frozen attribution (reference, BRACE) | unchanged |
| S1 | `<SoT>`, class word | class columns |
| B8 | `<SoT>`, class word, 8 generic words | class columns |
| R8 | class word, 8 generic words (no `<SoT>`) | class columns (class share) |
| BN | `<SoT>`, class word, NERVE's 25 background words | class columns |
| RN | class word, NERVE's 25 background words (no `<SoT>`) | class columns |

- Generic words (fixed earlier in this project, `scripts/sg_class_comp.py`):
  wall, floor, ground, sky, grass, tree, building, road.
- NERVE's background words (its released `cls_voc21.txt`, the list behind the
  CLIP seed in `runs/17`): ground, land, grass, tree, building, wall, sky,
  lake, water, river, sea, railway, railroad, keyboard, helmet, cloud, house,
  mountain, ocean, road, rock, street, valley, bridge, sign.
- Neither list depends on the image, the class or the benchmark labels. S1
  isolates the change of key form from the effect of background words.
- Known asymmetry: a two-piece class word has two columns. Reported, not
  corrected.

Cells per seed: seed alone, fine-only (two passes), capped hierarchy, each raw;
fine-only and capped hierarchy with DenseCRF (frozen settings).

## Data

All annotated images of VOC-Sim (490) and COCO-Cap (707). Both sets were used
to develop the frozen method. VOC-Sim is the development set here; COCO-Cap is
used once, for the test below, and nothing is selected on it.

## Scoring

- Class-macro mIoU; image bootstrap, 2000 resamples, 95% interval.
- Thresholds 0.30 to 0.70, step 0.025. *VOC-selected*: 5-fold out-of-fold on
  VOC-Sim (fold = crc32(image id) mod 5); on COCO-Cap the threshold that
  maximises full-VOC-Sim mIoU for that cell, applied unchanged.
- Fixed threshold 0.55 is reported next to it as a secondary number.

## Selection and test

1. **Selection (VOC-Sim only).** Among S1, B8, R8, BN, RN, the candidate with
   the highest out-of-fold VOC-Sim mIoU of the full pipeline (capped hierarchy
   + DenseCRF).
2. **Test (COCO-Cap, once).** Selected candidate minus W, full pipeline, each
   at its VOC-selected threshold.
3. **Rule.** The background-aware seed **passes** if the COCO-Cap difference
   is positive with its interval above zero and the VOC-Sim out-of-fold
   difference is not below −0.005. Otherwise it **fails**.

A pass makes the seed a candidate for a later confirmation on fresh images; it
does not change BRACE.

## Registered secondary readouts (all candidates, descriptive, no selection)

- The full table for every seed and cell on both benchmarks, with Δ to W.
- COCO-Cap, full pipeline: per-image IoU by object-size third; share of images
  below 0.3 IoU; class-macro precision and recall; per-class IoU.
- Whether the capped hierarchy still beats fine-only on each seed (both with
  DenseCRF).
- COCO-Cap numbers of the non-selected candidates are diagnostic only and are
  labelled so.

## Amendment 1 (17 November 2026, before any image was scored)

Requested by the owner. The smoke jobs of the first version were cancelled
during capture; no scoring had run.

Every seed above is also computed with **uniform layer weights** (the plain
mean over the 11 cross-attention layers, no self-attention-guided weighting):
W0 (frozen class-word attribution, uniform) and S1u, B8u, R8u, BNu, RNu. They
come from the same generation and go through the same cells. This shows what
the hierarchy adds on the most basic seed, with and without background words.

- These six seeds are **descriptive**. They do not enter the selection or the
  test, which stay as registered above.
- Added readouts for every seed, on both benchmarks, at VOC-selected
  thresholds: capped hierarchy minus seed alone (raw); capped hierarchy minus
  fine-only (both with DenseCRF); self-attention-guided minus uniform layer
  weights at the full pipeline.
- Scoring may be sharded over images; this does not change any number.

## Fidelity

W must reproduce the frozen pipeline (0.830 / 0.696 at 0.55 with DenseCRF);
the run checks that W's seed equals the runtime's seed exactly and reports the
count of bit-identical regenerated images.

## Files

- `scripts/jc_bgkey.py`: `capture` (GPU, sharded), `score` (CPU), `report`.
- Maps: `<BRACE_CACHE_DIR>/jc_bgkey/<set>/<image id>.npz`.
- Report: `runs/18_bg_key_seed/result.md`.
