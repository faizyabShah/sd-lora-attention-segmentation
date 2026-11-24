# Run 43: the new pipeline on the generated benchmarks, with NERVE and SCLIP in the same capture (pre-registration)

Committed before any full capture or scoring of this run. The smoke test regenerates a handful of images and runs
every step on them; its numbers are not used. BRACE stays frozen. This run fills the new paper's generated-benchmark
table (`paper2/main.tex`, `tab:generated`). Nothing is promoted.

## 1. Question

On VOC-Sim and COCO-Cap (OVAM's benchmarks), with a fixed threshold:
- does the new pipeline improve over the ungated map;
- how does it compare with NERVE and SCLIP run faithfully on the same images;
- does the gain hold for COCO-Cap images whose generation prompt does not contain the class word (OVAM's sense of
  querying a word that is not in the prompt)?

## 2. Images

- VOC-Sim (490) and COCO-Cap (707), regenerated once with OVAM's released seeds, the corrected prompts of runs/20
  (`configs/new_reruns_prompts.json`), SD 1.5, 30 steps, safety checker off (the frozen runtime, as runs/20 and 38).
- Each image is generated twice on the same code path: pass 1 (`dump`) only collects reference statistics for
  profiles; pass 2 (`capture`) produces every evaluated map and the PNG on which NERVE and SCLIP are run. All
  compared maps therefore come from one capture (pass 2).
- "Word in prompt" (COCO-Cap): OVAM's `word_included` column (242 of 707 images are `False`).

## 3. Our method (frozen)

| Part | Setting |
|---|---|
| Seed | generation-time cross-attention to the class token, **uniform** module weights |
| Gate | **MK4-LR**: per-module MK4 gates (object profile + 4 negative centroids, nearest-negative rule) at the modules below 64×64, averaged at 64×64 (`jc_gate36.consensus`, LR weights), applied at every module, **γ = 1** |
| Propagation | **fine-only (V0)**: two 64×64 self-attention steps |
| Read-out | min-max, **fixed threshold 0.55** (raw, primary); DenseCRF R2 at 0.55 secondary |

**Profiles** (part of the method; the evaluated image never contributes to its own profile; no masks are used):
- object profile for class c, per module: mean of the top 5% of queries by c's attention (keys from the word alone),
  on reference images generated for c;
- negatives for c: the top 1% of queries by c's attention on reference images generated for another class whose
  prompt does not mention c (`jc_trio.mentions`), clustered into 4 spherical k-means centroids (`jc_mneg.cluster`,
  which keeps fewer centroids when a cluster would hold fewer than 20 queries).

**Profile sources:**

| Code | Name | Reference images | Role |
|---|---|---|---|
| **H** | half split | each set split into halves (per class, by crc32 of the image id); an image in one half uses profiles from the other half of the **same** set | **primary** |
| F | few-shot | per evaluated image: **5** random images of its class and **100** random negative images (as defined above), both from the same set, excluding the image itself; random draw seeded by the image id. Equal in both sets. 100 negatives give at least 80 negative queries at the 8×8 module (1 per image), so 4 centroids are possible at every module. | backup |
| V | VOC-Sim to COCO-Cap | all of VOC-Sim; used for COCO-Cap images only (untuned transfer) | secondary |

**Maps per image** (one capture): seeds `u-` (no gate), `w-` (SA-guided weights, no gate), `u{H,F,V}M` (MK4),
`u{H,F,V}L` (MK4-LR), `wHL` (SA-guided weights + MK4-LR, H); each as seed only, V0 and V3 (capped hierarchy).
`V` arms exist for COCO-Cap only.

## 4. Baselines (same PNGs, run as in their papers and released code)

**NERVE** (WACV 2026; `kunal-mahatha/nerve` @ 1ddc5f9, copy of `other repositories/nerve`):
- as released: `scripts/config/vit-l-14/ovs_voc21.yaml` + `util/args.py` defaults (CLIP ViT-L/14, short side 336,
  logit scale 50, fused-attn, fuse_feature, SFSA refinement, layers −4/−6, time-step index 45 of 50, **SD 2.1-base**,
  which is the paper's main backbone), NERVE's 20 VOC class names and 25 background words (`voc_datasets.py`),
  15 `car_template.txt` templates, SFSA d = 64 (SD 2.1's head size);
- output: NERVE's own binarisation (`main_ovs.py:61-63`: `cam_to_label`, class-score gate 0.5, bkg_thre 0,
  min-max on); the mask is the pixels labelled as the queried class. **This is NERVE's row.**
- Infrastructure-only changes: the HF mirror endpoint line is disabled, and `stabilityai/stable-diffusion-2-1-base`
  (withdrawn from the Hub) is loaded from the `sd2-community/stable-diffusion-2-1-base` re-upload; the snapshot's
  file hashes are recorded. Noise: `torch.manual_seed(42)` before each image (NERVE seeds 42 once per run; we reseed
  per image so every image is reproducible). The code path is otherwise the released one (no K=1 or d patches).
- Secondary: NERVE's soft class map, min-max at 0.55; and the same rule without the class-score gate.

**SCLIP** (ECCV 2024; port `scripts/sg_clip_dense.py`, which reproduces the repo on VOC val: VOC20 81.49 vs 81.54,
VOC21 59.64 vs 59.63):
- as released for VOC21: CLIP ViT-B/16, correlative self-attention, 80 templates, `cls_voc21` names (26 background
  names + 20 classes), short side 336, sliding 224/112, logit scale 65, prob_thd 0.1, area_thd 0.1, no PAMR;
- output: the standalone VOC21 prediction; the mask is the pixels predicted as the queried class. **SCLIP's row.**
  SCLIP sees the 20 class names, which our method and NERVE's background-word setting also do in their own ways.
- Secondary: SCLIP's VOC21 probability for the queried class, min-max at 0.55.

Published rows (DAAM, Attn2Mask, OVAM, Grounded Diffusion, DatasetDM) are quoted from OVAM's Table 1, not rerun.

## 5. Scoring

Ground truth is OVAM's mask for the annotated class. Class-macro mIoU over the 20 classes; 95% intervals from a
2,000-sample image bootstrap, the same resamples for every map (paired). Threshold **0.55 fixed for every map of
ours** (no tuning). The threshold is described in the paper as determined empirically in earlier experiments
(on VOC-Sim, for the earlier pipeline) and not re-tuned.

## 6. Primary contrasts (fixed now)

Setting: source H, γ = 1, V0, raw at 0.55.

| # | Contrast | Set |
|---|---|---|
| P1 | MK4-LR − no gate | VOC-Sim |
| P2 | MK4-LR − no gate | COCO-Cap |
| P3 | MK4-LR − NERVE (own binarisation) | VOC-Sim |
| P4 | MK4-LR − NERVE (own binarisation) | COCO-Cap |
| P5 | MK4-LR − no gate | COCO-Cap, word **not** in prompt (242 images) |

Reading: CI > 0 helps / beats; CI < 0 hurts / trails; otherwise not resolved.

**Secondary (descriptive):** MK4; sources F and V; DenseCRF at 0.55; SCLIP; word in prompt vs not for every row;
seed only and V3; SA-guided weights; every threshold on the grid 0.30-0.70 (ours); the baselines' soft maps at 0.55;
per class; number of centroids kept per source.

## 7. Checks (smoke)

- `w-` seed equals the runtime's A3 seed (max |Δ| < 1e-2, as run 38); gated seed equals `jc_trio.ca_seeds` with the
  same gate (< 1e-5); batched V0/V3 equal `sg_combined_v2.aggregate` (< 1e-5 after min-max).
- MK4-LR equals an independent mean of the upsampled low-resolution MK4 gates (< 1e-6). All gates in [0, 1].
- No evaluated image is in its own reference set (asserted for H, F and V).
- NERVE: the SD 2.1 snapshot loads offline; head size of layers −4/−6 is 64.

## 8. Files and cost

- Scripts: `scripts/jc_eval43.py` (split, dump, fit, capture, sclip, score, crf, report; env `trdp`) and
  `scripts/jc_nerve43.py` (NERVE env).
- Report: `runs/43_generated_benchmarks/result.md`. Data: `<CACHE>/jc_eval43/`.
- About 1,200 images × 2 generations (about 6 GPU-hours over 4 GPUs), NERVE and SCLIP (about 1 GPU-hour), CPU fit
  and scoring. Smoke outputs are tagged `_smoke` and kept.
