# New reruns: every component on the prompt-corrected generated sets, plus a seed-calibration arm

Written and committed before any image of this run is generated or scored. Owner request (2026-11-17).

## Why

`runs/19_source_rule/addendum_image_mismatch.md`: our live generations, which every BRACE number so far uses, are not
the annotated OVAM images on 50/490 VOC-Sim and ~88/707 COCO-Cap images, because the manifest prompt text differs from
what OVAM generated with (VOC: "dining-table"/"potted-plant" hyphenated; COCO: trailing period stripped), and the
runtime's safety checker blacked 2 COCO-Cap outputs. This is a **data correction**: the method is unchanged (A3 seed,
V3 capped 16→32→64 hierarchy, DenseCRF R2, τ 0.55 remain the frozen BRACE). Numbers from this run supersede, and do
not silently replace, the earlier generated-set numbers; both stay on record.

## Data

- Corrected generation prompts: `configs/new_reruns_prompts.json` (50 VOC-Sim: hyphen → space in prompt, attribution
  prompt and token-matching class name; 86 COCO-Cap: trailing period restored, chosen label-free by pixel agreement
  with the official PNG). All other rows unchanged. Safety checker off.
- Everything else as the frozen runtime: SD 1.5 fp32, 30 steps, CFG 7.5, manifest seeds, `scripts/sg27_runtime.py`.
- Each image is checked against the official PNG (mean abs pixel difference). **Unresolved** = difference ≥ 2 grey
  levels (expected ≈ 21). Primary results include them; a sensitivity table excludes them.

## Pipeline (`scripts/jc_newrerun.py`)

1. `gen` (GPU, trdp): generate each image with the corrected prompt (the live runtime's exact call, no recording),
   save PNG. Used only as the input of NERVE's CLIP seed.
2. `jc_swap_nerve.py seed --image-dir` (GPU, NERVE env): NERVE's CLIP ViT-L/14 seed on those PNGs (as runs/17).
3. `bank` (CPU): sorted min–max-scaled NERVE CLIP seed values of every VOC-Sim image, with its fold
   (crc32(id) mod 5, the project's fold rule) and class. Used only by the template candidates.
4. `capture` (GPU, trdp): one live generation per image with the frozen runtime (corrected prompt, safety checker
   off). Image checked against the step-1 PNG. In the same capture: SCLIP ViT-B/16 VOC21 target probability (as the
   earlier CLIP-seeded BRACE) on the generated image; NERVE's noised pass (t = 101, empty prompt; ported, verified equal
   to NERVE's code) for the walk. All seeds below through: none, fine-only (V0), BRACE hierarchy (V3), NERVE walk
   (NERVE's layers on NERVE's pass).
5. `score` (CPU): class-macro counts at the 17 thresholds 0.30–0.70, raw and with DenseCRF R2, for every cell.
6. `report` (CPU): `runs/20_new_reruns/result.md`.

## Seeds

Component rows (confirmatory for the corrected data):

| ID | Seed |
|---|---|
| CAu | cross-attention, class word, frozen attribution, **uniform** layer weights (A0) |
| CAg | cross-attention, GOCA-style SA-guided layer weights (A3) = **BRACE's seed** |
| NV | NERVE's CLIP seed (ViT-L/14, class word + NERVE's background words), native grid → 64 |
| SC | SCLIP ViT-B/16 VOC21 target-class probability, 64 px |

CAg × V3 + DenseCRF at 0.55 is BRACE; NV × walk is NERVE.

Calibration arm (exploratory; all are monotone remaps of the min–max-scaled CAg seed, so its spatial ranking is
kept; hierarchy/walk inputs as above):

| ID | Remap | Uses CLIP? |
|---|---|---|
| hA | per-image NERVE CLIP sorted values placed by CAg's ranking (runs/19 hybrid A) | yes, every image |
| TG | one global template: mean of sorted VOC-Sim NERVE CLIP values; VOC image in fold f uses folds ≠ f; COCO uses all VOC | offline only |
| TC | per-class template, same fold rule (COCO class names mapped to VOC's) | offline only |
| PWγ | s^γ, γ ∈ {0.5, 1.5, 2, 3} | no |
| FLp | max(s − q_p(s), 0), p ∈ {50, 70, 80, 90} (percentile floor) | no |
| TKk | s · 1[s ≥ q_{1−k}(s)], k ∈ {5, 10, 20, 40}% | no |
| FO | max(s − Otsu(s), 0), parameter-free, per image | no |

(The earlier-mentioned area-matched template is replaced by FO before any result: a CA-area to CLIP-area match mixes two
scales; FO is the SD-only, per-image, parameter-free version.)

## Thresholds and conventions

Every cell is reported raw and with DenseCRF, at the **VOC-selected threshold** (5-fold OOF on VOC-Sim; full-VOC-Sim
best threshold applied unchanged on COCO-Cap) and at the **fixed 0.55**. Primary: VOC-selected with DenseCRF.
Also reported (audit, not selection): COCO-Cap best-threshold mIoU, to separate ranking gains from threshold
calibration (protocol §8 lesson: SG30/32 sharpening gains vanished at matched thresholds). Image-bootstrap 95%
intervals (2000) for every contrast.

## Registered contrasts (component rows; both sets; primary convention, also at 0.55)

- F: fine-only − seed alone, for CAu, CAg, NV, SC.
- H: hierarchy − fine-only, for each seed.
- G: CAg − CAu (GOCA weighting), at seed alone, fine-only, hierarchy.
- C: NV − CAg and SC − CAg, at each propagation.
- R: DenseCRF − raw, hierarchy, each seed.
- N: NERVE (NV × walk) − BRACE (CAg × V3), with DenseCRF; also both at their own conventions.

No promotion follows from these; they describe the corrected data.

## Calibration arm: selection and reading

- Within PW, FL, TK the parameter is selected on VOC-Sim by OOF mIoU of hierarchy + DenseCRF at the VOC-selected
  threshold (one per family; VOC numbers of the selected member are in-sample). hA, TG, TC, FO have no parameter.
- Each of hA, TG, TC, PW*, FL*, TK*, FO is compared with BRACE (CAg × V3 + DenseCRF), and with fine-only, on COCO-Cap.
- **Reading** (exploratory, because the direction came from COCO-Cap results in runs/19): a candidate "helps" if COCO-Cap
  Δ > 0 with interval above 0, VOC-Sim Δ ≥ 0, and the COCO-Cap best-threshold Δ is also > 0 (otherwise "calibration
  only"). An SD-only candidate (PW, FL, TK, FO) that helps would need a fresh annotated set before any claim.
- Descriptive: precision, recall, false-positive area, size thirds, share of images below 0.3 IoU.

## Fidelity checks (reported, not gates)

- Images whose prompt is unchanged: CAg × V3 map vs the stored paper capture; image vs the stored image.
- Every image: generated vs official PNG; capture image vs step-1 PNG.
- NV × walk vs runs/17's NERVE walk on images matched to the official PNG.

## Resources

Up to 4 GPUs (user's limit), CPU scoring sharded. Smoke on a few images first (smoke outputs use tag `_smoke`).
