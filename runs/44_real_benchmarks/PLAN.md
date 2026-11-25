# Run 44 plan: real-photo comparison for paper2 Table 3 (`tab:real`). DRAFT, not a pre-registration

Status: plan for owner review. Nothing captured or scored. Once the open decisions (section 6) are settled, this becomes
`PREREG.md` and is committed before any full capture.

## 1. Question

On real photographs with the class word given (one binary mask per (photo, class) item), how does the new pipeline compare
with diffusion-attention, CLIP, random-walk and prototype baselines, each run as released on the same items?

## 2. Items (Table 1) — owner decision 2026-11-25: the three standard object benchmarks

Columns are the three benchmarks that nearly every diffusion-based method reports (OVDiff, iSeg, DiffSegmenter, FreeDA;
also the first three of SCLIP/NACLIP/NERVE): **PASCAL VOC, PASCAL Context, COCO-Object**. We use their images and class
lists under our protocol (class word given, one binary mask per (photo, class) item), so published multi-class numbers
are not comparable and every row is re-run. The seen / unseen COCO split moves to Table 5 only.

| Column | Classes | Photos | Items | Notes |
|---|---:|---:|---:|---|
| PASCAL VOC 2012 val | 20 | 1,409 | 1,947 | primary items (area ≥ 1%) of the 2,148 used in runs 32–37 (VOC val seen 3× in development: say so) |
| PASCAL Context val | 59 | 5,104 | to count | **new; data not on the cluster** (see §2a) |
| COCO-Object (val2017) | 80 | 4,571 | 9,816 | primary items (area ≥ 1%), val2017 minus the Section 136 set (runs 40–42) |

### 2a. PASCAL Context data (to obtain)
- Images: VOC2010 trainval images are a subset of our VOC2012 `JPEGImages` (check every val id exists).
- Labels: the 59-class val masks, built the standard way (mmseg `tools/dataset_converters/pascal_context.py` with
  `trainval_merged.json` and the Detail API), so the class list and ignore label match SCLIP / NERVE / NACLIP configs.
- Size: annotations are small (< 100 MB); no new images needed if the VOC2012 check passes.

### 2b. Item-rule decision
- **Decided (owner, 2026-11-25): primary items = class area ≥ 1% on all three columns.** VOC val: 1,947 of 2,148 items
  (1,409 photos; area over all pixels, `/home/shafa/sg_comp/tmp/voc_area.py`). All items are secondary.
- COCO-Object: val2017 minus the Section 136 set (4,571 of 5,000 photos) keeps runs 40–42 untouched-set logic; state it.

### 2c. Expected weakness
Context's 59 classes include many stuff words (sky, wall, ceiling, floor, road, grass). Caption-matched profile photos
may be few for some words; counts are reported per class, and classes with < 5 positive photos are listed.

## 3. Ours (frozen, same as Table 2)

- One SD 1.5 call per item at t = 100, prompt "A photograph of a {class}" (runs 32–42 recorder).
- Uniform CA seed; MK4-LR, γ = 1; fine-only V0; min-max; **fixed 0.55** (raw primary, DenseCRF R2 secondary).
- **Profiles: captioned COCO train2017 photos only (run 42 recipe)**, all three columns (Context words too, matched in
  COCO captions with the same rule): 20 positive photos per word
  (caption matches the word) + 480 negative photos (captions do not match), disjoint from the threshold-check photos and
  from every evaluated photo. VOC val uses the same COCO-caption profiles (no VOC labels or captions are needed).
- Threshold check (descriptive only): curve on the 1,200 COCO train items; 0.55 is not re-selected.
- Can reuse run 42's capture? **No**: run 42 used t = 100 too but only COCO val; VOC val needs a capture with these
  profiles, and Table 3 should come from one capture. Recapture all three columns (≈ 2,150 + Context + 9,800 items).

## 4. Baselines (each with its own released read-out)

| Method | Code | Real-photo input | Vocabulary / read-out | Effort |
|---|---|---|---|---|
| DAAM | `daam` (pip) | noised pass of the photo, prompt "A photograph of a {class}" (same t as ours) | class-word heat map, its τ = 0.4 (as OVAM reports) | small |
| OVAM (no token opt.) | `vpulab/ovam` | same noised pass | word attention + its SA refinement, τ 0.4, α 0.85 (OVAM §4.1) | small |
| DiffSegmenter | `comparisons/diffsegmenter` (Narmeen's driver) | as released (real-image method) | its own rule | small |
| iSeg (T-PAMI 2026) | `linsun449/iseg.code` @ 57e9a17, cloned to `other repositories/iseg` | photo, **SD 2.1-base** (its base model) | released per-dataset settings (VOC: 8 iterations, ent 0.015; Context: 5, 0.002, att_mean; COCO: its COCO-2014 setting) and its image-label step (TagCLIP-predicted labels, `ovs/predict_labels.py`) + `cam_to_label`; mask = pixels labelled as the class. Secondary: the queried class given as the image label (our protocol) | medium |
| SCLIP | `scripts/sg_clip_dense.py` (reproduces repo) | photo | VOC21, Context60 and COCO-Object configs (each with background), as released; pixels predicted as the class | small |
| ClearCLIP, NACLIP, ProxyCLIP | GitHub, to download | photo | their VOC21 / Context60 / COCO-Object configs, same read-out | medium (3 repos, mostly shared mmseg code) |
| NERVE | `other repositories/nerve` (run 43 driver) | photo, SD 2.1-base | released VOC21 and Context60 configs; COCO-Object: no released config → build one the same way (80 names + its background words), documented as the only deviation | small |
| OVDiff (ECCV 2024) | `karazijal/ovdiff` @ 218566b, cloned to `other repositories/ovdiff` | photo; support sets generated with SD 1.5 (32 images per class, 30 steps, seed 42), prototypes from SD, CLIP ViT-B/16 and DINO ViT-B/8, CutLER masks | released VOC pipeline (`sample_support_set` → `gen_vit_features` → `gen_proto_*` → `predict`); its own prediction. **The release implements VOC only** (`mmdata.get_dataset` raises NotImplementedError otherwise): Context and COCO-Object need class lists added in the same format (data-only deviation, documented) | **large** (old stack: torch 1.12, mmcv-full 1.7.1, mmseg 0.30, detectron2 0.6, diffusers 0.14; support generation for 20 + 59 + 80 classes) |

Multi-class baselines (CLIP methods, NERVE) see all class names of the dataset; the queried class's pixels form the mask.
This is their released protocol; ours sees only the queried word plus profiles. State it in the caption.

## 5. Primary contrasts (proposed)

Ours (raw, 0.55) − NERVE (own rule), and ours − best CLIP method, on each column. Everything else descriptive.

## 6. Open decisions for the owner

0. Columns: **decided** (VOC, Context, COCO-Object). Item rule: **decided**, ≥ 1% on all three (§2b).

1. **Scope: decided (owner, 2026-11-25): run iSeg and OVDiff ourselves.** Repos cloned (see §4). OVDiff needs its own
   environment and Context / COCO-Object class lists (release is VOC-only).
2. **NERVE on COCO:** accept a self-built COCO-Object vocabulary (documented deviation), or report NERVE on VOC val only?
3. **VOC val profiles:** COCO-caption profiles (proposed, label-free) vs VOC train with labels (not label-free).
4. **DAAM / OVAM on real photos:** noised single pass at t = 100 (proposed, same as ours) vs DDIM inversion.

## 7. Cost (estimate)

Ours: ≈ 12,000 items + 8,000 profile photos, ≈ 3 GPU-hours over 4 GPUs. CLIP methods: < 1 GPU-hour each. NERVE: ≈ 1 GPU-hour.
DAAM/OVAM: ≈ 2 GPU-hours. DiffSegmenter / iSeg: ≈ 2–4 GPU-hours each. OVDiff: support generation ≈ 4–8 GPU-hours.
