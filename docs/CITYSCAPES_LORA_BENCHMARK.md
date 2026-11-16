# Cityscapes LoRA image benchmark — 16 November 2026

Generated 500 candidates with the owner-supplied LoRA, visually reviewed all
500, and selected 50 clear images. Data live locally in
`datasets/cityscapes_lora_benchmark_20261116/`; this directory is ignored by Git.
The deliverable is `cityscapes_benchmark_50.zip` in that directory.

## Composition and selection

| Primary scene family | Selected |
|---|---:|
| Single pedestrian | 8 |
| Two pedestrians | 4 |
| Parked bicycle | 8 |
| Bicycle rider | 6 |
| Car | 6 |
| Bus | 4 |
| Truck | 4 |
| Train/tram | 4 |
| Motorcycle/scooter | 4 |
| Street/park scene | 2 |
| **Total** | **50** |

The intended major classes are road, sidewalk, building, vegetation, terrain,
sky, person, rider, car, truck, bus, train, motorcycle and bicycle. The final
set has visually checked examples of all 14; `coverage.json` identifies a
witness image for each. Presence tags are conservative and non-exhaustive.
Pole, traffic light, traffic sign, wall and fence are excluded as target
classes; incidental background objects can remain.

Selection prioritized complete nearby people, complete bicycles with both
wheels, coherent main subjects, clear backgrounds and class diversity. All
candidates were inspected in labeled contact sheets; selected candidates were
also inspected in larger detail sheets. Cropped vehicles, absent subjects,
malformed foregrounds and small distant people/bicycles were excluded.
`review/final_review.json` retains all 500 selection decisions and the earlier
review records remain available. Not every unselected image is defective:
others were omitted for composition and diversity.

This is a visually curated synthetic image set. Pixel masks have not been
created; segmentation scoring requires subsequent annotation. No model score
or scientific promotion is claimed. Fine text, spokes and faces can retain
ordinary diffusion artifacts; lettering is not an annotation source.

## Generation and prompt adaptation

Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`, cached revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.

LoRA: root `pytorch_lora_weights.safetensors`, SHA-256
`deec4bf94e3ed9e08c7dc7bfae08c0ab05f07d0e62e0d07887567a2c15f9f4d8`.
The loaded adapter has 128 UNet LoRA layers of rank 128; activation and nonzero
up-projections were checked before generation. The supplied adapter contains
no text-encoder LoRA weights, so Diffusers' corresponding warning is expected.

Inference used the A40, float16, PNDM, 30 steps and guidance 7.5. Images are
512×512 or 768×512. The first pilots used LoRA scales 1.0, 0.75 and 0.55.
Strong adaptation repeatedly produced dashboard/hood views or tiny subjects.
Foreground-first prompts and a lower scale improved clarity. The final 140
candidates used revised close-subject prompts, scale 0.40 for riders, 0.45 for
cars/buses/trucks, 0.50 for trams and 0.55 for other scenes. Selected images
retain their actual per-image settings, including earlier successful images.

Prompt plans and all revisions are in `plans/`. The v3 job was deliberately
interrupted to revise only ungenerated images; its partial log is retained.
The effective final plan is `plans/all_candidates.json`. No existing image
was overwritten or regenerated during curation. Generation ran in detached
Screen sessions; all jobs are complete and released the GPU.

## Files and reproducibility

- `selected/`: 50 original PNGs and matching provenance/selection JSON files.
- `manifest.csv`, `manifest.json`, `manifest.jsonl`: exact prompts, negative
  prompts, seeds, image IDs and inference settings for the selected images.
- `all_candidates.jsonl`: generation records and integrity hashes for all 500.
- `candidates/`: all 500 original PNGs and generation sidecars (local only,
  omitted from the compact selected-set ZIP).
- `review/`: selection rules, contact sheets and review decisions.
- `preview.jpg` and `preview_pages/`: labeled views of the selected 50.
- `coverage.json`: visual class witnesses; `validation.json`: integrity checks.
- `logs/`: generation logs, launch commands and runtime configuration.

Generation is resumable using `scripts/generate_cityscapes_benchmark.py`:

```bash
source scripts/brace_env.sh
export HF_HUB_OFFLINE=1
python scripts/generate_cityscapes_benchmark.py \
  --plan datasets/cityscapes_lora_benchmark_20261116/plans/all_candidates.json \
  --output datasets/cityscapes_lora_benchmark_20261116 --batch-size 4
```

This command skips the existing 500 images. For exact provenance, consult each
image's original batch IDs/position, runtime, prompt version, scheduler config,
model revision and weight hash. A different GPU/software stack or batch
composition need not reproduce pixels exactly. PNGs retain embedded generation
metadata, and selected copies are byte-identical to their source candidates.
The safety checker stayed enabled and flagged no candidate. All 500 images
decoded, matched their recorded resolution and final plan, and had unique
pixel hashes. The complete original candidate pool is retained.
