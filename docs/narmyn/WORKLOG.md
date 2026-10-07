# Work log — LoRA rank sweep on Cityscapes

State as of **2026-09-15**. Everything below is on `renyi`/`erdos`/`neumann`
(PPKE cluster) under `~/TRDP`. All jobs were cancelled at this point; this file
records what exists so work can resume without re-deriving anything.

---

## 1. Pipeline, in order

| Script | Purpose | Status |
|---|---|---|
| `01_crop.py` | 2048×1024 → central 1024² crop | **done** |
| `02_caption.py` | Florence-2-large `<DETAILED_CAPTION>` | **done** |
| `03_generate_frozen.py` | frozen SD 1.5 baseline images | **done** |
| `04_train_lora.py` | LoRA fine-tune (vendored diffusers script) | **done, all ranks** |
| `05_sample_checkpoints.py` | per-checkpoint image grids | used for diagnosis |
| `06_generate_lora.py` | LoRA-adapted generation | **partial, see §4** |

---

## 2. Data

| Path | Contents |
|---|---|
| `data/cityscapes_dataset/leftImg8bit/` | originals, 2048×1024 |
| `data/cityscapes_cropped/` | **24,995** crops, all exactly 1024×1024 |
| `data/florence2_detailedcaptions/metadata.jsonl` | **24,995** captions (tracked in git) |
| `data/florence2_detailedcaptions_5k/` | earlier 5k Colab run, kept for reference |
| `data/lora_train_dataset/` | imagefolder tree: real dirs + per-file symlinks |

Cropping found **2 additional all-black frames** beyond the known troisdorf one:
`heidelberg_000000_000293` (mean 4.97) and `oberhausen_000000_000899` (mean 4.58).
Both skipped. 24,997 − 2 = 24,995.

**Index convention (critical).** `metadata.jsonl` line order *is* the index.
Image `{i:05d}.png` in every output dir corresponds to line `i`, and the
generation seed is `i`. Order is `test → train → train_extra`, cities and
filenames sorted. **Do not regenerate, re-sort or filter this file** — every
generated image would silently pair with the wrong prompt.
Current sha256 prefix: `eae76280f23a8d00`.

---

## 3. Training — complete

All five ranks, identical hyperparameters, `31000` steps, `lr 2e-4`,
`resolution 768`, `train_batch_size 2 × 2 GPUs × grad_accum 2` (total batch 8),
`snr_gamma 5.0`, `cosine` schedule, `warmup 1200`, `seed 42`.

| Rank | Job | Elapsed | Adapter | Checkpoints |
|---|---|---|---|---|
| 8 | 206208 | 4:38:12 | 6.1 MB | 3000…30000 (10) |
| 16 | 206215 | 4:38:55 | 12.2 MB | 10 |
| 32 | 206220 | 4:38:37 | 24.3 MB | 10 |
| 64 | 206335 | 4:40:28 | 48.6 MB | 10 |
| 128 | 206336 | 4:43:26 | 97.3 MB | 10 |

Weights in `output/lora_weights/rank{N}/`, each with
`pytorch_lora_weights.safetensors` plus `checkpoint-{3000..30000}/`.

Note the two weight formats: the final export uses diffusers keys
(`lora.down`/`lora.up`), checkpoints use peft keys (`lora_A`/`lora_B`).
`load_lora_weights` handles both.

---

## 4. Generation — partial (this is what to resume)

Resolution 768², 30 steps, guidance 7.5 (default), seed = prompt index.

| Set | Images | Missing | Contiguous blocks present |
|---|---|---|---|
| frozen | **24,995 / 24,995** | 0 | complete |
| rank 8 | 15,918 | 9,077 | 0-8201, 16000-19782, 20500-24420, 24983-24994 |
| rank 16 | **24,995 / 24,995** | 0 | complete |
| rank 32 | **24,995 / 24,995** | 0 | complete |
| rank 64 | 13,647 | 11,348 | 0-12348, 12500-13797 |
| rank 128 | 0 | 24,995 | — |

All of the above used **final** weights (step 31000), which §6 shows is the
worst checkpoint. Resuming is just re-running `06_generate_lora.py` with the
same `--rank`; existing images are skipped by per-index existence check.

---

## 5. Cluster facts worth not rediscovering

* **Only `renyi` can run the `trdp` env.** Its torch is `2.11.0+cu130`
  (CUDA 13.0, needs driver ≥ 580). Drivers: renyi 580.95.05 ✓,
  erdos 560.35.05 ✗, neumann 545.23.08 ✗. On the wrong node the job
  silently falls back to **CPU**.
* **`gen` env** = clone of `trdp` with `torch 2.4.1+cu121` → runs on erdos and
  neumann. Pass `CONDA_ENV=gen`.
* **`caption` env** = clone of `trdp` pinned to `transformers==4.44.2`.
  Florence-2's remote code breaks on transformers 5.x
  (`Florence2LanguageConfig has no attribute forced_bos_token_id`).
* **Throughput**, 768², 30 steps: a100 **0.44 img/s**; v100 **0.07–0.09 img/s**
  sustained (~5× slower — do not plan around the v100s).
* **QoS cap: 4 GPUs per user** (`normal` QoS, `gres/gpu=4`).
* `wald` has been `drain` ("Kill task failed") since ~Sept 8; `gpu_long` is
  therefore unusable.
* `gpu_lowpriority` (tier 1) is preempted by `gpu` (tier 2), `PreemptMode=REQUEUE`.
  Scheduler will not start a low-priority job whose time limit overruns the gap —
  request only the time that fits.
* `--export=ALL,VAR=x` — without `ALL` the job loses `PATH`/`HOME` and
  `source ~/.bashrc` fails.

---

## 6. Findings

**Over-training is the dominant image-quality problem.** Checkpoint grids for
rank 8 and rank 64 show domain adaptation is essentially complete at
**step 3000** (one epoch), after which quality degrades:

| Step | Appearance (rank 64) |
|---|---|
| 3000 | clean; Mercedes hood ornament sharp |
| 12000 | emblem turning cyan; magenta blotches on road |
| 21000 | emblem clearly corrupted |
| 30000 / final | cyan blob emblem, heavy colour patches, white speckles |

Degradation is **worse at higher rank**, consistent with `lora_alpha = rank`
(scale 1) meaning a fixed `lr 2e-4` produces larger effective updates as rank
grows. So the current final-weight image sets confound *rank* with
*over-training*. `31000` steps was ~10× more than this task needed.

**Adaptation saturates well below rank 16.** Effective update magnitude
`sum‖BA‖`: rank 8 = 867.8, rank 16 = 719.4 — doubling capacity produced a
*smaller* functional change, not a larger one.

**The model learned the ego-vehicle.** Cityscapes was shot from a Mercedes and
the hood + three-pointed star sit at the bottom-centre of every crop, so the
adapters reproduce it. It is a dataset artifact, not scene content.

**Mixed A100/V100 generation is safe.** Same seed on both gives the same image
to within float noise: mean |diff| **0.4–0.9 / 255**, versus **54 / 255** for
two genuinely different samples. No regeneration needed on hardware grounds.

**Caption duplication.** 18,595 distinct captions over 24,995 images; 1,819
captions are reused, the most common by 105 different images. A caption does
not identify an image, which is why evaluating on training prompts does not
reward memorisation.

**Caption length.** mean 43 CLIP tokens, 99th pct 66; only **15 of 24,995**
exceed CLIP's 77-token limit, so truncation is negligible and identical across
all arms.

---

## 7. Next step (decided, not yet run)

Screen **every checkpoint × every rank** on a modest image sample to pick a
single best checkpoint, then regenerate the evaluation sets from it.
Candidate default is step 9000; steps 3000–12000 are the plausible range.

Evaluation metrics: FID/KID use images only (real reference =
`data/cityscapes_cropped/`); CLIP/BLIP score need `prompt_mapping.json` to pair
each image with the caption that produced it.

---

## 8. Resume commands

```bash
cd ~/TRDP

# finish a partial generation set (skips what exists)
sbatch --job-name=genlora_r64 -p gpu -w renyi --gres=gpu:a100:1 --time=12:00:00 \
       --export=ALL,RANK=64 scripts/slurm/generate_lora.slurm --start 0 --end 12500

# same on a v100 node (erdos/neumann)
sbatch --job-name=genlora_r64_e -p gpu -w erdos --gres=gpu:v100:1 --time=12:00:00 \
       --export=ALL,RANK=64,CONDA_ENV=gen scripts/slurm/generate_lora.slurm --start 12500

# generate from a specific checkpoint (writes to rank{N}_ckpt{S}/)
... scripts/slurm/generate_lora.slurm --checkpoint 9000

# visual grid for one rank
sbatch --job-name=ckptgrid_r64 -p gpu -w erdos --gres=gpu:v100:1 --time=00:40:00 \
       --export=ALL,RANK=64,CONDA_ENV=gen scripts/slurm/sample_checkpoints.slurm

# useful squeue (default output hides GPU count and truncates names)
squeue -u $USER -o "%.10i %.30j %.16P %.10T %.8M %.11l %.15b %.16R"
```
