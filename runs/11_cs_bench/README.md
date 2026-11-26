# Small generated Cityscapes-style benchmark: generation and selection protocol

Written 2026-11-16, before generation. Code: `scripts/cs_bench_gen.py`; the prompts are in `prompts.json`.

## Generator

**Model:** SD1.5 with the Cityscapes LoRA at rank 128, training seed 2, final checkpoint (step 5000), from `outputs/lora/rank_128/seed_2`. Why this one:
- **Prompt match:** best BLIP ITM among ranks (0.690 mean over seeds).
- **Realism:** FID and KID are tied across ranks within seed noise. Seed 2 has the lowest FID of the 25 runs (28.3).
- **No bias:** FID and BLIP know nothing about segmentation, so choosing on them cannot favour BRACE.

**Settings:** BRACE's live generation settings exactly (`sg27_runtime` float32 pipe, global seeding, CUDA initial noise, 512 px, 30 PNDM steps, CFG 7.5). Images are generated on wald (V100).
- A later BRACE capture with the same LoRA on the same GPU type should regenerate the same image.
- Cross-GPU regeneration is not pixel-identical.

## Prompts

There are 100 prompts, cs000–cs099, with seeds 30000 + i. Every queried class is **named in the prompt**; unnamed classes are left for later.

**Construction:** each prompt names 2 "thing" classes and 2 "stuff" classes.
- Classes are drawn by balanced rotation over the 19 Cityscapes evaluation classes.
- The wording follows the training captions, e.g. "The image shows a busy intersection with poles and a tram, next to grass and buildings."
- **Query words:**
  - road, sidewalk, buildings, trees (vegetation), grass (terrain), sky, wall, fence;
  - cars, people, bus, truck, tram (train), motorcycle, bicycles, cyclist (rider), poles, traffic lights, traffic signs.

**Coverage:** each stuff class is named 25 times and each thing class 17–19 times.

## Selection (before any BRACE map is computed)

- **Who decides:** Claude reviews every image visually. No attention map or BRACE output has been looked at.
- **An image is kept if:**
  - it is a plausible street scene without severe artefacts;
  - at least 2 of its named classes are clearly visible and could be outlined unambiguously.
- **Per image, the queries are** the named classes that are clearly visible. A named class that is absent or too ambiguous to outline is dropped and the drop is recorded.
- **Target size:** about 50 images. The kept list, dropped queries and reasons are written to `selection.json` before annotation.

**Disclosure:** selection favours clear images. This makes the benchmark easier than random generations, and it is reported as a curated benchmark.
