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

(none)
