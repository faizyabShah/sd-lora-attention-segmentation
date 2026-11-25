# Run 44: real-photo comparison on PASCAL VOC, PASCAL Context and COCO-Object (pre-registration)

Committed before any full capture, baseline run or scoring. Smoke outputs (2–3 photos per column per method,
PLAN.md §8b) were used only to check that every driver runs and that every class maps by name; their numbers are not
used and they are moved to `*_smoke` before the full run. BRACE stays frozen; this fills paper2 Table 3 (`tab:real`).
Nothing is promoted. Setup details and owner decisions: `PLAN.md` (§2, §6, §8).

## 1. Question

With the class word given and one binary mask per (photo, class) item, how does the paper2 pipeline compare with
diffusion-attention, CLIP, random-walk, prototype and iterative-refinement baselines, each run as released on the same
photos, on the three object benchmarks that diffusion-based methods report (PASCAL VOC, PASCAL Context, COCO-Object)?

## 2. Items

Primary items: class area ≥ 1% of the photo, on all three columns (owner decision). All items are secondary.

| Column | Classes | Photos | Primary items | Ground truth (scoring only) |
|---|---:|---:|---:|---|
| PASCAL VOC 2012 val | 20 | 1,409 | 1,947 | `SegmentationClass`, 255 = void (excluded) |
| PASCAL Context val (the standard 5,104-photo list) | 59 | 5,104 | 22,691 | 459 → 59 with iSeg's released converter (checked by name, PLAN §8) |
| COCO-Object (val2017 minus the Section 136 set) | 80 | 4,571 | 9,816 | run 42 instance masks, crowd excluded |

VOC val was seen in development (runs 32–37) and COCO val2017's 20 VOC classes in runs 40–42: stated with the results.
PASCAL Context was never used before.

## 3. Our method (frozen; = Table 2 / runs 42–43)

- One SD 1.5 pass per item at t = 100, noise seed 0, prompt "A photograph of a {word}" (`jc_realvoc.one_pass`).
- Uniform CA seed; MK4 gate; **MK4-LR** at γ = 1; **fine-only V0**; min-max; **fixed threshold 0.55** (raw, primary);
  DenseCRF R2 at 0.55 (secondary). Also stored: no gate (`u-`), MK4 (`uM`), seed-only maps.
- Words: the canonical class name (VOC20 / CLS80 words as runs 42–43); PASCAL Context "table" is queried as "table"
  (its class covers every table).
- **Profiles: captions of COCO train2017 photos only** (no labels, masks or evaluated photos):
  - the 80 COCO words: run 42's size-20 profiles (`jc_ov42/fit.pt`), unchanged;
  - the 33 Context words that are not COCO words: same recipe (20 positives whose captions match the word, 480
    negatives whose captions do not; object profile from the top 5% queries, 4 negative centroids from the top 1%),
    caption rules fixed here (`scripts/jc_ours44.py` SYN44, regex alternatives + optional s/es):

    bag: bag, backpack, handbag, purse · bedclothes: sheet, blanket, comforter, duvet, bedspread, bedding · building:
    building, skyscraper · cabinet: cabinet, cupboard · ceiling · cloth: cloth, towel, tablecloth · computer: computer,
    laptop · curtain: curtain, drape · door · fence · floor · flower · food: food, meal · grass: grass, lawn · ground:
    ground, dirt · light: light, lamp · mountain: mountain, hill · plate · platform · road: road, street, highway ·
    rock: rock, boulder, stone · shelves: shelf, shelves, bookshelf, bookshelves · sidewalk: sidewalk, pavement · sign ·
    sky · snow · table: table, desk · track: track, rail, railroad, railway · tree · wall · water: water, ocean, lake,
    river, sea · window · wood: wood, wooden.

    Every word reaches 20 positives and 480 negatives (items44 log). Caption rules are noisy by design (e.g. "dirt
    bike" matches ground); this is the label-free setting and is reported as such.

## 4. Baselines (as released; drivers and environments in PLAN.md §8b)

| Method | Code | Input / backbone | Read-out (primary) |
|---|---|---|---|
| DAAM | castorini/daam c30493e | one noised SD 1.5 pass at t = 100 (our latent), [uncond, cond] batch | word map, bilinear, min-max, > 0.4 |
| OVAM (no token opt.) | vpulab/ovam 12afd87 | same pass; OVAM queries + self-attention hooker | map / max × SA ∈ [0.85, 1] > 0.4 (OVAM's VOC-Sim settings, every column) |
| DiffSegmenter | reviewed class-word driver (comparisons/) | SD 1.5, t = 100, BLIP-large caption prompt | 64×64 map, bilinear, min-max, > **0.5 fixed** (its in-sample background sweep = audit row only) |
| iSeg | linsun449/iseg.code 57e9a17 | SD 2.1-base, released per-dataset settings | **its TagCLIP image labels** (released step) + `cam_to_label`; pixels = class. Secondary: queried class given |
| SCLIP, ClearCLIP, NACLIP, ProxyCLIP | official repos (SCLIP 3608360, ClearCLIP ad68a40, NACLIP 7f2bb0a, ProxyCLIP 361e173) | photo, ViT-B/16 | each repo's VOC21 / Context60 / COCO-Object config (with background), label map; ClearCLIP with the paper's q-q attention (`model_type='ClearCLIP'`; released base config reproduces 32.0, the paper's 51.8 is reproduced exactly) |
| NERVE | official 1ddc5f9 (run 43 driver) | photo, SD 2.1-base | released VOC21 / Context60 / COCO5k configs and vocabularies, `cam_to_label` with its class-score gate |
| OVDiff | karazijal/ovdiff 218566b | SD 1.5 support sets (32 / class), SD + CLIP + DINO prototypes, CutLER | README pipeline; Context / COCO-Object class lists added in the release's own format; argmax label map |

VOC21 reproductions with each CLIP repo's own runner: SCLIP 59.64 (published 59.1), ClearCLIP 51.80 (51.8), NACLIP
64.06 (64.1, with its released PAMR), ProxyCLIP 61.24 (61.3). Multi-class baselines see all class names of their config
(their protocol); ours, DAAM, OVAM and DiffSegmenter see the queried word; iSeg sees its own predicted labels.

## 5. Scoring

Class-macro mIoU over the column's classes (per-class pooled tp / fp / fn); 95% intervals from a 2,000-sample photo
bootstrap with the same resamples for every method (paired). No threshold, γ or setting is tuned for any method on any
column. Our maps also report every threshold on the 0.30–0.70 grid (descriptive only).

## 6. Primary contrasts (fixed now)

Setting: ours = MK4-LR, V0, raw at 0.55.

| # | Contrast | Column |
|---|---|---|
| P1 / P2 / P3 | ours − no gate | VOC / Context / COCO-Object |
| P4 / P5 / P6 | ours − NERVE (own rule) | VOC / Context / COCO-Object |

Reading: CI > 0 helps / beats; CI < 0 hurts / trails; otherwise not resolved.

**Secondary (descriptive):** ours − every other baseline per column; + DenseCRF rows (ours, OVAM); MK4 and seed-only
maps; iSeg with given class; DiffSegmenter in-sample sweep (audit); every threshold (ours); per class; Context split into
VOC classes / other 39; COCO-Object split into 20 VOC / 60 other classes; the number of Context words whose profile
module set was incomplete.

## 7. Checks (smoke, done)

Eleven methods run on all three columns (smoke). **OVDiff's smoke is still pending** when this is committed: its bundled
CutLER needs detectron2 from git (rebuilt at 58e472e, 2023-01-21, per CutLER's INSTALL.md); OVDiff is run if its smoke passes
and otherwise reported as not run, with the reason. All 21 method × column vocabularies map one-to-one onto the canonical classes;
MK4-LR identity 1.8e-07, V0 identity 0; Context labels match the 459-class names for all 59 classes.

## 8. Files and cost

Scripts: `scripts/jc_real44.py` (items), `jc_ours44.py`, `jc_clip44.py`, `jc_nerve44.py`, `jc_iseg44.py`,
`jc_diffseg44.py`, `jc_daamovam44.py`, `jc_ovdiff44.py`, `jc_score44.py`, `jc_names44.py`. Data: `<CACHE>/jc_real44/`.
Report: `runs/44_real_benchmarks/result.md`. About 34,500 items: ours ≈ 4 GPU-h (+ 2,260 profile passes), DiffSegmenter
≈ 6 GPU-h, DAAM + OVAM ≈ 8 GPU-h, iSeg ≈ 3 GPU-h, CLIP methods + NERVE ≈ 3 GPU-h, OVDiff support sets ≈ 8 GPU-h;
at most 4 GPUs at a time.
