# Pre-registration: GoCA on SD 2.1-base and SDXL (JC #5, side N)

Written 2026-11-16, before any SD 2.1 or SDXL GoCA output was produced or seen.
Not to be edited after the runs start; amendments go in the dated section at the end.

## Common to every backbone

- **Method code:** GoCA @475892f, used unchanged. Feature extraction is generic-diffusion-feature
  @adf80e1, copied to `n_side/gdf` with `goca/install/components/*` patched in.
  Only one other change was made to that copy: the unused ControlNet import is wrapped in try/except.
- **Adapter:** the canonical `n_side/scripts/goca_classword.py`, with no backbone-specific changes.
  The resumable driver is `n_side/scripts/goca_full.py`, which imports the canonical adapter.
  The only per-backbone input is `--config`.
- **Prompt:** `"<class>, background"`. `<class>` is the manifest class name.
  The token-to-entity alignment is a port of `captioner.py:252-348`.
  The class tokens are the object, `background` is the supporting object, and the rescaling set is
  class tokens plus background (no `<sos>` or padding).
- **Images and items:** the stored OVAM benchmark images, the same items as the SD 1.5 full run:
  - `n_side/outputs/manifest_full_voc_sim.csv` (490 items)
  - `n_side/outputs/manifest_full_coco_cap.csv` (707 items)
- **Noise seed:** 0. `torch.manual_seed(0)` and `torch.cuda.manual_seed_all(0)` are called right before
  each `extract()`. The seed covers both the VAE posterior sample and the img2img noise.
- **Timestep:** each config's `t` (100 for both), passed as `extract(t=100)`.
  This is `get_timesteps(1000, 0.1)` on the backbone's own scheduler.
  The scheduler timestep actually used is logged in each run's meta file and reported as it comes out.
  For SD 1.5 (PNDM) it was 101.
- **Output:** the queried class's soft map. It is recorded after GoCA's head and layer aggregation,
  sum-1 rescaling + per-token renorm+, and the configured affinity post-process.
  It is resized bilinearly to the mask resolution, 512×512, as `downstream_dispatch.py:90` does.
  The file layout is the same as the SD 1.5 run:
  - post- and pre-affinity class maps
  - GoCA's merged background map
  - a mask from a fixed 0.5 threshold on the min-max-normalised map
  - a per-item record with runtime and peak GPU memory
- **Scoring and thresholds:** follow Entry 1's shared scorer, which is not yet available.
  - Threshold development is on VOC-Sim only; COCO-Cap gets the VOC-Sim threshold unchanged.
  - The fixed-0.5 class-macro IoU computed by the driver is a sanity number only, not a result.
  - No threshold tuning is done here.
  - DenseCRF: none in GoCA's configs. If it is reported, it uses the frozen R2 settings, as for every method.

## SD 2.1-base: BLOCKED (official weights removed)

The official repo `stabilityai/stable-diffusion-2-1-base` now returns 401/404, because Stability removed it.
No unofficial mirror is used: the user and side F have to agree on the weights first.
If it is unblocked, the plan, fixed here, is:

- **Config:** a new `n_side/configs/config_model_21.py`, copied from `config_model_15.py`, with only
  `name='21'` and `version='2-1'` changed. Everything else stays: `img_size 512`, `feat_size 64`, `t=100`,
  the head, layer and rescale methods, the 16 cross-attention layers, `ref_layer`
  `up-level2-repeat1-vit-block0-cross-q`, the 3 self-attention layers, and affinity with order 2.
- **No layer-name changes are expected.** SD 2.1-base's UNet has the same block layout as SD 1.5:
  - down levels 0-2 with 2 transformers each
  - a mid block
  - up levels 1-3 with 3 transformers each, 1 block each
  - spatial sizes 64/32/16/8 at 512 px

  generic-diffusion-feature names the layers from that layout, so the names are the same.
  Only the head counts differ (head_dim 64, so 5/10/20/20 heads). GoCA's head aggregation loops over
  `v.shape[0]`, so the head count makes no difference.

  Before scoring, the run asserts that all 16 cross, 16 `special-*` and 3 self-attention keys are present.
  If any are missing, the run stops and the mapping is amended here first.
- **Scheduler:** generic-diffusion-feature loads 2.1-base with EulerDiscreteScheduler (`models.py:30-39`).
  The scheduler stays as it is; the actual timestep is logged.
- **Pad-token fix:** the OpenCLIP tokenizer of SD 2.1 pads with `!` (id 0), not `<|endoftext|>`.
  Before alignment, positions past the attention mask, i.e. positions with the pad id, are mapped to `<pad>`.
  Those positions are then never assigned to an entity and never enter the rescaling set.
  This is the only adapter change for 2.1. It must leave SD 1.5 token ids unchanged, and this is checked.

## SDXL base 1.0: config_model_xl.py unchanged

- **Weights:** `stabilityai/stable-diffusion-xl-base-1.0`, cached snapshot 4621659, loaded as
  generic-diffusion-feature does (`variant="fp16"`, fp16 safetensors).
- **Config:** GoCA's shipped `goca/src-main/configs/config-model/config_model_xl.py`, unchanged:
  - `img_size 1024`, `feat_size 64`, `t=100`
  - head method `dot-product w/o clamp`, layer method `dot-product similarity`,
    rescaling `sum-1 rescaling + per-token renorm+`
  - `ref_layer` `up-level1-repeat0-vit-block0-cross-q`, 70 cross-attention layers
  - the 7 self-attention layers listed in the file, affinity with order 2
- **Scheduler:** whatever the SDXL repo ships (EulerDiscreteScheduler); the actual timestep is logged.
- **Input resolution:** the 512×512 benchmark images are upscaled to 1024×1024 by generic-diffusion-feature's
  own `_preprocess_basic` (`PIL.Image.resize((1024, 1024))` with Pillow's default resampling,
  bicubic for RGB images), followed by the pipeline's image processor.
  The 64×64 GoCA maps are then resized bilinearly to the 512×512 masks.
- **Tokenizer:** GoCA aligns tokens with `pipe.tokenizer`, the first CLIP tokenizer, which pads with
  `<|endoftext|>` as SD 1.5 does. The captioner states that the 1.5 and XL tokenizers are identical.
  No pad fix is needed for SDXL.
- **fp16-VAE NaN policy:** the SDXL VAE config has `force_upcast: true`, so diffusers encodes the image in fp32.
  If NaN maps still occur in the 2-image smoke or the full run, the affected backbone is rerun with
  the fp32 VAE from the same snapshot (`vae/diffusion_pytorch_model.safetensors`).
  That rerun is documented as a deviation in the amendments, and NaN counts are reported either way.
- **Order of work:** a 2-image smoke (first VOC-Sim item and first COCO-Cap item), then the full run on all
  1197 items. A100 node renyi is preferred.

## Amendments (dated, appended only)

### 2026-11-19: SD 2.1-base unblocked with a community mirror (Narmeen's approval)

- **Weights.** `sd2-community/stable-diffusion-2-1-base`, revision `4e63672c03103b6c636b8fb4119ba982469b2955`,
  diffusers format, fp32 safetensors only (no `.ckpt`, `.bin` or fp16 copies downloaded).
  The 768 px v-prediction model `sd2-community/stable-diffusion-2-1` was not used: the plan and the 512 px
  benchmark images need the 512 px epsilon-prediction base model.
  Local sha256 after download, equal to the mirror's LFS ids:

  | File | sha256 |
  |---|---|
  | `unet/diffusion_pytorch_model.safetensors` | `6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214` |
  | `text_encoder/model.safetensors` | `cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c` |
  | `vae/diffusion_pytorch_model.safetensors` | `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` |

  **Provenance check.** The same three hashes are listed for `kaseris/stable-diffusion-2-1-base`, an independent
  copy whose history says "Cloned from stabilityai/stable-diffusion-2-1-base". The mirror's
  `v2-1_512-ema-pruned.ckpt` has sha256 `88ecb782…0f36`, the id in the LFS pointer of Stability's original
  upload commit. The scheduler config is PNDM, `prediction_type: epsilon`; the UNet `sample_size` is 64.
  Side F should use the same revision for its #5 runs, so that every SD 2.1 row shares one set of weights.
- **Config.** `n_side/configs/config_model_21.py` (copy in `comparisons/goca/configs/`): `config_model_15.py`
  with only `name='21'` and `version='2-1'` changed, as planned.
- **Loading (deviation from "the only per-backbone input is `--config`").** generic-diffusion-feature hard-codes
  `stabilityai/stable-diffusion-2-1-base`, which no longer exists. `goca_full.py` gained
  `--model-path`, passed as `FeatureExtractor(offline_lora=<path>)` without a filename. `models.py` then uses
  it as `model_id` and loads no LoRA, so no vendored code changed. The SD 2.1 run passes the local snapshot dir;
  SD 1.5 and SDXL runs pass nothing and behave as before. The scheduler is the one models.py builds for '2-1'
  (EulerDiscreteScheduler from the snapshot's config); the timestep used is 99 (SD 1.5, PNDM: 101).
- **Pad-token fix, as planned.** `sd15_tokenization` maps every position with `attention_mask == 0` to `<pad>`.
  Checked on all 22 class names of both benchmarks: SD 1.5 tokens and prompt bookkeeping are identical
  to the old code; SD 2.1 has no `!` left and the same object, rescaling and background token ids as SD 1.5.
- **Layer check, as planned.** `goca_full.py` now checks, during the warm-up, that the extractor returns every
  requested key: 16 cross-map, 48 `special-*` (value, post-value, weight for each cross layer),
  16 self-map, 81 in total. All 81 were returned for SD 2.1 (and for SD 1.5 in the regression).
- **Regression of the edited adapter on SD 1.5** (7 items, same node): tokens and rescaling ids identical,
  IoU within 0.002 of the stored SD 1.5 run (fp16 non-determinism).
- **Run.** Job 253878, V100 16 GB (erdos), 5 min 38 s for 1197 items, 0 failures, 0 NaN or constant maps.

  | Backbone (fixed 0.5, no DenseCRF; sanity) | VOC-Sim | COCO-Cap | s / image | peak alloc. | GPU |
  |---|---|---|---|---|---|
  | SD 1.5 | 0.671 | 0.452 | 0.14 | 4.4 GB | A100 |
  | SD 2.1-base | 0.710 | 0.546 | 0.20 | 3.8 GB | V100 |
  | SDXL | 0.788 | 0.607 | 0.66 | 13.3 GB | A100 |

  SD 2.1 timings are on a V100, the others on an A100, so they are not directly comparable.
- **Official scoring** (`comparisons/official/`, `goca_sd21` added to `score_n.py` and `report.py`; the rows of the
  other methods are unchanged in the regenerated `result.md`). VOC-Sim / COCO-Cap:

  | GoCA SD 2.1-base | raw | DenseCRF R2 |
  |---|---|---|
  | VOC-sel. threshold (0.575 raw, 0.550 CRF; same in all 5 OOF folds) | 0.719 / 0.586 | 0.740 / 0.597 |
  | fixed 0.5 | 0.710 / 0.546 | 0.733 / 0.565 |
  | GoCA's own rule, VOC-sel. offset (+0.1) | 0.720 / 0.576 | 0.740 / 0.597 |
  | Paired Δ vs BRACE, both VOC-sel. | −0.098 [−0.117, −0.080] / −0.107 [−0.129, −0.084] | −0.091 [−0.110, −0.072] / −0.111 [−0.134, −0.086] |

  SD 2.1-base beats SD 1.5 only at the fixed 0.5 threshold (0.710 / 0.546 vs 0.671 / 0.452, raw). With the
  VOC-selected threshold SD 1.5 is as good or better (raw 0.738 / 0.610, DenseCRF 0.753 / 0.599; paired Δ vs BRACE
  −0.079 / −0.082 raw), because SD 1.5's best threshold is far from 0.5. SDXL is the best GoCA backbone under every
  rule. All three GoCA backbones are below BRACE, with intervals that exclude zero.
