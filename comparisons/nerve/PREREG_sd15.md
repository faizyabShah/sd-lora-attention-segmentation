# Pre-registration: NERVE rerun on SD 1.5 (side N, item #3)

Written 2026-11-16, before the full run. No full-run maps have been produced or looked at when this note is
written. The only numbers seen so far are the 12-item smoke (job 253158, `outputs/smoke_sd15/`), labelled
"smoke, fixed threshold, not a result".

## Method and code
- NERVE (WACV 2026), github.com/kunal-mahatha/nerve, commit `1ddc5f9`. Clean clone: `external/nerve`
  (untouched). Patched copy: `external/n_side/nerve/repo`. Full patch: `repo_patch.diff`
  (md5 085c0c873064a4ef06d5ebba3fa3212d).
- Driver: `nerve_classword.py` (md5 d3b3e4bac9df871a17711df7a6ed1055). The resumable version reproduces the
  smoke maps bit for bit (3 items × variants A and W, job 253162). It calls NERVE's own `Pipeline` from `ovs/pipeline_clean_mean_final_run_correct.py`
  (the class `main_ovs.py` imports) and copies the post-processing of `main_ovs.py:61-63`. It does not use mmseg
  or NERVE's dataset classes.
- Config: `scripts/config/vit-l-14/ovs_voc21.yaml` plus the `util/args.py` defaults, unchanged:
  - CLIP: OpenAI ViT-L/14, short side 336, logit_scale 50, fused-attn, fuse_feature on, `car_template.txt`
    (15 templates).
  - SD refinement: SFSA, self-attention layers −4 and −6 (`up_blocks[3].attentions[1/0].transformer_blocks[0].attn1`),
    empty prompt.
  - Random walk: closed form (1−δ)(I−δF)⁻¹ followed by F·(·). δ=0.9, β=0.8, α=3.0.
  - Binarisation settings: score_threshold 0.5, bkg_thre 0, min-max normalisation on.
- Backbone: SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, local cache). The paper's main setting is
  SD 2.1-base, so this SD version is itself a deviation (the workplan asks for SD 1.5).
- Timestep: index 45 of 50 PNDM steps, which is t = 101.
- Noise: seed 0, applied as `torch.manual_seed(0)` immediately before each image. NERVE's `main_ovs.py` uses a
  global seed of 42.
- Images: stored benchmark images at 512×512, taken from the manifests listed below. Maps are bilinearly resized
  to mask resolution (512×512). Ground truth is used for scoring only.

## Manifests (read only, written by the other n_side agent)
- `n_side/outputs/manifest_full_voc_sim.csv`: 490 rows, md5 57ad3fa63afc7b1ce7b3bfe6755ee7ff.
- `n_side/outputs/manifest_full_coco_cap.csv`: 707 rows, md5 a4e4e71742b16f292410efe1356048a3. All rows point
  to stored images. The smoke run used 4 regenerated COCO-Cap images instead.

## Variants (all three are run; which one is the official row is not decided)

| Variant | Foreground vocabulary | Text form | SFSA scale d | Deviations from NERVE as released |
|---|---|---|---|---|
| A | {queried class} | class word, hyphen → space | 40 (SD 1.5 head dim) | **D1**, **D2**, SD 1.5 |
| V | the 20 benchmark classes (VOC order), queried class's channel read | class word, hyphen → space ("dining table", "potted plant", "tv") | 40 | **D2**, SD 1.5 |
| W | NERVE's 20 `cls_voc21.txt` names verbatim ("diningtable", "bird avian", "chair seat", "tvmonitor screen", "person with clothes, people, human", …) | NERVE's | 64 (literal) | SD 1.5 only (as released) |

All variants use the same background: NERVE's VOC21 background list (`cls_voc21.txt` line 0, 25 words, split as
in `voc_datasets.py:69`). NERVE's own extra "mean of foreground" competitor is kept (`nerve_clip/model.py:176`).

### Deviations
- **D1. K=1 entropy guard** (variant A only). Patched-copy `ovs/pipeline_clean_mean_final_run_correct.py:40`;
  the original is line 38.
  - Before: `logK = math.log2(K)`.
  - After: `logK = math.log2(K) if K > 1 else float("inf")`.
  - Why: with a one-class vocabulary, K=1 makes logK 0 and every head's class entropy 0, so the weight is 0/0 = NaN.
    The patched form gives uniform head weights, which is the limit of NERVE's formula.
  - Effect: in variant A, NERVE's entropy-based head selection reduces to a plain mean over heads. For K>1 the code
    is unchanged, so V and W do not use this branch.
- **D2. SFSA attention scale d** (variants A and V). Patched-copy line 217; the original line 213 hard-codes `d=64`.
  - Change: `d=64` → `d=cfg.sfsa_head_dim`, set to 40, SD 1.5's head size (8 heads of size 40 in layers −4 and −6).
  - Why: 64 equals SD 2.1's head size, so NERVE's recomputed attention matches the UNet's own attention there. On SD 1.5,
    64 gives a softer attention than the UNet's.
  - Variant W keeps 64.
- **D3. Infrastructure only, no effect on the method.**
  - The HF mirror endpoint is disabled.
  - The repository id `runwayml/…` is mapped to the cached `stable-diffusion-v1-5/…`, which holds the same weights.

## Outputs per (image, class), in `outputs/full_sd15/variant_{A,V,W}/`
- `<key>__soft.npy`: the refined NERVE map for the queried class, float16, 512×512.
- `<key>__mask_t0.5.png`: 0.5 applied to the min-max-normalised soft map (fixed sanity threshold).
- `<key>__mask_nerve_own.png`: **NERVE's own binarisation**, exactly as in `main_ovs.py:61-63`. Each class map is
  rescaled as pred/amax·final_score. Classes with final_score ≤ 0.5 are dropped (class-score gate). Background
  score is (1 − max_c cam_c)², with cam min-max normalised. The pixel label is the argmax, and the mask is the
  pixels labelled with the queried class. In variants V and W, other classes compete.
- `<key>__mask_nerve_nogate.png`: the same rule applied to the queried class alone, with the gate disabled (the
  class is given). This is equivalent to min-max cam > 0.382.
- `records.jsonl`: one line per item with final_score, gate pass, runtime, peak GPU memory, NaN/constant flags and
  the intersection/union counts.

## Threshold and scoring policy
- The registered scoring is Entry 1's shared scorer, used unchanged once it exists. Thresholds are developed on
  VOC-Sim only and transferred to COCO-Cap without change. NERVE's own binarisation is reported alongside.
  DenseCRF follows the frozen R2 settings, and raw and CRF numbers are never compared with each other.
- Until Entry 1 exists, only fixed-0.5 and NERVE-own class-macro IoU are computed, per benchmark and labelled
  "sanity, fixed threshold, not a result". No pooled cross-benchmark row. No choice of variant, d or threshold is
  made from these numbers.
- Run: renyi (A100-40GB), env `nerve` (torch 1.13.1+cu117, diffusers 0.30.2), resumable. Job ids are recorded in
  the report.
