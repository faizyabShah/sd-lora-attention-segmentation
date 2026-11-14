# Setting up this repo on a new machine (for Claude Code)

> **Superseded on the A40 machine by `docs/NEW_MACHINE_SETUP.md` and `AGENTS.md`** (Codex's verified setup: `.venv` via `scripts/brace_env.sh`, cache in `BRACE_CACHE_DIR`). Use this file only for background: data sources, model ids and project state.

This repo is the BRACE project: training-free class-word segmentation from Stable Diffusion 1.5 attention. Everything needed to rebuild it from scratch is listed below. Attention maps and per-image arrays are **not** copied; they are regenerated.

Written 2026-11-14 from the original cluster setup (SLURM, V100/A100, CUDA 11.8).

## 1. Code

```bash
git clone git@github.com:faizyabShah/sd-lora-attention-segmentation.git
cd sd-lora-attention-segmentation
```

**About `ovam/`.** It is the official OVAM repo (`https://github.com/vpulab/ovam`, commit `12afd87`). It is not tracked here, and our code does not import it: `src/segmentation/ovam_baseline.py` is our own reimplementation. Clone it only for reference or for OVAM's token optimization:

```bash
git clone https://github.com/vpulab/ovam.git ovam
```

## 2. Python environment

**Versions on the cluster.** Conda env `trdp`, Python 3.10.20.

| package | version |
|---|---|
| torch | 2.6.0+cu118 |
| torchvision | 0.21.0+cu118 |
| diffusers | 0.36.0 |
| transformers | 4.44.2 (keep this; Florence-2 remote code was patched for it) |
| accelerate | 0.33.0 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| pillow | 12.1.1 |
| timm | 1.0.26 |
| einops | 0.8.2 |
| safetensors | 0.7.0 |
| tokenizers | 0.19.1 |

`requirements.txt` has the full pinned list, including LoRA-era extras: peft, bitsandbytes, clean-fid, lpips.

**Install:**

```bash
conda create -n trdp python=3.10 -y
conda activate trdp
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**DenseCRF (needed for every BRACE result).**
- The code imports `denseCRF` from SimpleCRF 0.2.1.1.
- If `import denseCRF` fails, `src/segmentation/ovam_full.py::_import_densecrf` appends `outputs/dependencies/simplecrf_py310` to `sys.path`, and the scripts pass that path (`st.CRF_SITE` in `scripts/sg_steps_test.py`).
- The simplest setup is to install SimpleCRF into exactly that folder:

```bash
pip install SimpleCRF==0.2.1.1 --target outputs/dependencies/simplecrf_py310
```

Check that the folder contains `denseCRF.cpython-310-x86_64-linux-gnu.so`. SimpleCRF compiles C++ on install, so it needs a compiler.

## 3. Model weights (Hugging Face)

The jobs run with `HF_HUB_OFFLINE=1`, so download the weights once with network access first:

| model | used for |
|---|---|
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | everything (BRACE, captures); the default `--model` in the scripts |
| `openai/clip-vit-base-patch16` | SCLIP baseline and the CLIP seed (`scripts/sg_clip_dense.py`, backbone `B16`) |
| `openai/clip-vit-large-patch14` | optional SCLIP ViT-L (`L14`) |
| `microsoft/Florence-2-large` | optional, caption experiments only. It needs `trust_remote_code`; the `flash_attn` import is patched out in `scripts/sg_caption.py` |

```bash
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5
```

```bash
huggingface-cli download openai/clip-vit-base-patch16
```

## 4. Datasets (all under `datasets/`, which is git-ignored)

| path | what | source |
|---|---|---|
| `datasets/ovam_experiment_with_dataset/` | **OVAM benchmark:** VOC-Sim (500 rows, 490 annotated) and COCO-Cap (707 annotated), with images, masks and CSV manifests | `http://www-vpu.eps.uam.es/publications/ovam/ovam_experiment_with_dataset.zip`, unzipped here |
| `datasets/voc2012/VOCdevkit/VOC2012/` | PASCAL VOC 2012 trainval (real-photo track: D/F sets and the val benchmark) | standard VOC2012 trainval tar from the PASCAL VOC site |
| `datasets/coco2017/` | optional, real COCO experiments only | `images.cocodataset.org` (val2017 + annotations_trainval2017) |

Paths the code expects (`scripts/sg_steps_test.py`, `scripts/sg_class_comp.py`):
- **VOC-Sim:**
  - CSV: `datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv`
  - images: `images/`
  - masks: `annotations/`
- **COCO-Cap:**
  - CSV: `datasets/ovam_experiment_with_dataset/evaluation/ovam/coco_captions/ovam_coco_cap.csv`
- **VOC2012:** `datasets/voc2012/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets}`
- **Real-photo set lists:** D, F and S live in `outputs/segmentation/sg_real_atlas/manifest.json`, which is tracked in git.

Not needed for BRACE: `cityscapes_*`, `leftImg8bit`, `real_cityscapes_512` and `outputs/lora/` (old LoRA direction, about 200 GB).

## 5. Hard-coded paths (important)

About 110 scripts and job files contain absolute paths. Easiest fix: use the same username and layout. Otherwise, symlink these or `sed` them.

| path | count | what |
|---|---:|---|
| `/home/shafa/sd-lora-attention-segmentation` | 93 | repo root |
| `/home/shafa/sg_comp` | 30 | experiment cache (stored maps and scores); regenerated, just create the folder |
| `/home/shafa/miniconda3` | 9 | conda env path in `jobs/*.slurm` |
| `/home/shafa/paper_cache`, `/home/shafa/sg_real_atlas`, `/home/shafa/real_coco` | a few | older caches; regenerated |

```bash
grep -rln "/home/shafa" scripts src jobs
```

## 6. Running jobs

- **SLURM.** `jobs/*.slurm` assume SLURM.
  - Partitions are `gpu` and `cpu`.
  - The original cluster excluded the node `wald`; that is irrelevant elsewhere.
  - Without SLURM, run the `python scripts/... "$@"` line inside each job file directly.
- **GPU.**
  - One SD1.5 capture is about 10–20 s per image on a V100/A100 (16 GB is enough).
  - CPU scoring with DenseCRF is about 1 min per image per worker for multi-variant runs.
- **Determinism.** Regenerated images are not bit-identical across GPUs or under shared-GPU load (mean |Δpixel| about 0.6/255), but aggregate scores reproduce (BRACE 0.830 / 0.696). Keep every compared variant inside one capture.

## 7. Where to start reading

- `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md`: experiment protocol and history.
- `paper/`: the draft. Supervisor feedback (JC, 2026-11) asks for:
  - a paired SOTA block (own OVAM reimplementation, GoCA, NERVE propagation);
  - BRACE with OVAM optimized tokens;
  - SD2.1 / SDXL backbones;
  - Table 1 Δ fixes;
  - ICPR 2027 as the target (registration 1 March 2027).
- `outputs/segmentation/sg_real_atlas/analysis/*.md`: the latest reports and pre-registrations.
- **Key scripts:**

  | script | role |
  |---|---|
  | `scripts/sg27_runtime.py` | frozen live pipeline: generation plus attention capture |
  | `scripts/sg_class_comp.py` | propagation, scoring, OOF τ, bootstrap |
  | `src/segmentation/sg_combined_v2.py` | capped hierarchy V0–V6; BRACE = V3 |
  | `scripts/sg_gen_clip.py` | SCLIP baselines on generated sets |
  | `scripts/sg_gen_capab.py` | cap ablation, late fusion, early fusion |
  | `scripts/sg_clip_dense.py` | SCLIP port |
  | `scripts/sg_voc21.py` | standard VOC val protocol |

## 8. Project state and working rules (carry over; these were in Claude's local memory)

**BRACE (frozen; do not change):**
- Pipeline: A3 cross-attention seed → capped 16→32→64 SA hierarchy (V3, one-sided cap at 16→32) → DenseCRF R2 at τ 0.55.
- Scores: VOC-Sim 0.830, COCO-Cap 0.696 (paper estimate; pick one capture and use it consistently).

**Results:**

| result | VOC-Sim | COCO-Cap |
|---|---:|---:|
| SCLIP + CRF | 0.752 | 0.608 |
| BRACE | 0.830 | 0.696 |
| **CLIP-seeded BRACE** (best claimable variant) | **0.845** | **0.717** |

- CLIP-seeded BRACE: SCLIP ViT-B/16 VOC21 probability as the seed.
- **Cap ablation with the SCLIP seed:** V3 beats no cap and the two-sided cap, and ties cap-at-both. Effects are about +0.01.
- **Late and early seed fusion:** both pre-registered tests failed against CLIP-seeded BRACE. Post hoc, geometric-mean or product seeds win on COCO-Cap (early geometric mean 0.741), but this is not claimable without a fresh annotated multi-object set.
- **Real photos:** the CLIP seed + SD SA beats SCLIP+CRF on fresh set F (0.733 vs 0.663, pre-registered). On standard VOC21 the pre-registered test fails (−2.00), because SA propagation spreads into the background.

**Rules:**
- Label-free at inference.
- Pre-register selection rules (commit before scoring).
- Report failures honestly.
- Delete files only on an explicit yes.
- Commit locally; push only when asked.
- .tex snippets go in chat, with no internal names (A3, V3, C16, SGxx) in the paper.
