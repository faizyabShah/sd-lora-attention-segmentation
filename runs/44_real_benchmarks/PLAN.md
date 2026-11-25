# Run 44 plan: real-photo comparison for paper2 Table 3 (`tab:real`). DRAFT, not a pre-registration

Status: plan for owner review. Nothing captured or scored. Once the open decisions (section 6) are settled, this becomes
`PREREG.md` and is committed before any full capture.

## 1. Question

On real photographs with the class word given (one binary mask per (photo, class) item), how does the new pipeline compare
with diffusion-attention, CLIP, random-walk and prototype baselines, each run as released on the same items?

## 2. Items (Table 1)

| Column | Items | Photos | Notes |
|---|---:|---:|---|
| VOC 2012 val | 2,148 | 1,449 | all (photo, class) items, as runs 32–37 (VOC val already seen 3× in development: say so) |
| COCO val2017, 20 VOC classes | 5,414 | 3,582 | primary items (area ≥ 1%), val2017 minus the Section 136 set (runs 40–42) |
| COCO val2017, other 60 classes | 4,402 | 3,038 | same rule (run 42 unseen classes) |

Metric: class-macro mIoU over items; 95% photo bootstrap, 2,000 samples, paired across methods.

## 3. Ours (frozen, same as Table 2)

- One SD 1.5 call per item at t = 100, prompt "A photograph of a {class}" (runs 32–42 recorder).
- Uniform CA seed; MK4-LR, γ = 1; fine-only V0; min-max; **fixed 0.55** (raw primary, DenseCRF R2 secondary).
- **Profiles: captioned COCO train2017 photos only (run 42 recipe)**, all three columns: 20 positive photos per word
  (caption matches the word) + 480 negative photos (captions do not match), disjoint from the threshold-check photos and
  from every evaluated photo. VOC val uses the same COCO-caption profiles (no VOC labels or captions are needed).
- Threshold check (descriptive only): curve on the 1,200 COCO train items; 0.55 is not re-selected.
- Can reuse run 42's capture? **No**: run 42 used t = 100 too but only COCO val; VOC val needs a capture with these
  profiles, and Table 3 should come from one capture. Recapture all three columns (≈ 12,000 items).

## 4. Baselines (each with its own released read-out)

| Method | Code | Real-photo input | Vocabulary / read-out | Effort |
|---|---|---|---|---|
| DAAM | `daam` (pip) | noised pass of the photo, prompt "A photograph of a {class}" (same t as ours) | class-word heat map, its τ = 0.4 (as OVAM reports) | small |
| OVAM (no token opt.) | `vpulab/ovam` | same noised pass | word attention + its SA refinement, τ 0.4, α 0.85 (OVAM §4.1) | small |
| DiffSegmenter | `comparisons/diffsegmenter` (Narmeen's driver) | as released (real-image method) | its own rule | small |
| iSeg | GitHub, to download | as released (real-image method) | its own rule | medium |
| SCLIP | `scripts/sg_clip_dense.py` (reproduces repo) | photo | VOC21 config for VOC val; COCO-Object config (80 + background) for COCO; pixels predicted as the class | small |
| ClearCLIP, NACLIP, ProxyCLIP | GitHub, to download | photo | their VOC21 / COCO-Object configs, same read-out | medium (3 repos, mostly shared mmseg code) |
| NERVE | `other repositories/nerve` (run 43 driver) | photo, SD 2.1-base | VOC21 config for VOC val; COCO: no released COCO-Object config → build one the same way (80 names + its background words), documented as the only deviation | small |
| OVDiff | GitHub, to download | photo; builds prototypes from generated support images per class | its own rule | **large** (support-set generation for 80 classes) |

Multi-class baselines (CLIP methods, NERVE) see all class names of the dataset; the queried class's pixels form the mask.
This is their released protocol; ours sees only the queried word plus profiles. State it in the caption.

## 5. Primary contrasts (proposed)

Ours (raw, 0.55) − NERVE (own rule), and ours − best CLIP method, on each column. Everything else descriptive.

## 6. Open decisions for the owner

1. **Scope:** keep iSeg and OVDiff (most effort), or quote their published real-image numbers in a separate block?
2. **NERVE on COCO:** accept a self-built COCO-Object vocabulary (documented deviation), or report NERVE on VOC val only?
3. **VOC val profiles:** COCO-caption profiles (proposed, label-free) vs VOC train with labels (not label-free).
4. **DAAM / OVAM on real photos:** noised single pass at t = 100 (proposed, same as ours) vs DDIM inversion.

## 7. Cost (estimate)

Ours: ≈ 12,000 items + 8,000 profile photos, ≈ 3 GPU-hours over 4 GPUs. CLIP methods: < 1 GPU-hour each. NERVE: ≈ 1 GPU-hour.
DAAM/OVAM: ≈ 2 GPU-hours. DiffSegmenter / iSeg: ≈ 2–4 GPU-hours each. OVDiff: support generation ≈ 4–8 GPU-hours.
