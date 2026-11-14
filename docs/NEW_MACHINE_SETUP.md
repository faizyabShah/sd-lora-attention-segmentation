# BRACE: current context and local setup

Recorded 14 November 2026. The supplied
[Claude handoff](CLAUDE_MACHINE_HANDOFF_2026-11-14.md) is preserved verbatim.
[AGENTS.md](../AGENTS.md) carries the working rules into future sessions.

## Project map and evidence

The repository started with Cityscapes LoRA experiments; the active project is
training-free class-word segmentation from SD1.5 attention. `src/attention_capture/`
records attention, `src/segmentation/` implements the operators, `scripts/`
contains experiment entry points, `configs/` holds frozen selections/folds,
and `outputs/segmentation/` contains tracked reports and compact evidence.
Datasets, model weights, and large per-image caches are regenerated locally.
The optional upstream `ovam/` checkout is not imported by our implementation.

Read the shared [protocol](ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md),
especially Sections 126–137, then the newer
[real-atlas reports](../outputs/segmentation/sg_real_atlas/analysis/).
`docs/PAPER_REVISION_HANDOFF.md` and parts of `paper/README.md` describe older
drafts. The supplied November handoff names ICPR 2027 as the intended target;
the checked-in paper still uses a CVPR author kit. Conference dates in that
handoff have not been independently verified during machine setup.

BRACE remains the cross-attention seed → one-sided capped 16→32→64 hierarchy
→ fixed DenseCRF pipeline at threshold 0.55. Relevant implementations are
`scripts/sg27_runtime.py`, `src/segmentation/sg_combined_v2.py`, and
`src/segmentation/ovam_full.py`.
Protocol Section 135 explicitly adopts it by owner override after failure of
the registered worst-class guard; the registered selector retained the earlier
fine-only method. A setup handoff must not erase this distinction.

The most recent paired generated-set evidence is in
`gen_early_fusion.md` and `gen_cap_ablation.md` in the real-atlas reports:

| Fixed-threshold pipeline (+ DenseCRF) | VOC-Sim | COCO-Cap |
|---|---:|---:|
| BRACE | 0.830 | 0.696 |
| CLIP-seeded BRACE | 0.845 | 0.717 |

Do not confuse the CLIP-seeded COCO fixed-threshold score (0.717) with its
COCO-local OOF score (0.713). COCO-local OOF is diagnostic, not frozen transfer.
SCLIP binary + CRF scores 0.752 / 0.608 in `sclip_generated.md`.
The pre-registered late/early fusion tests did not establish a gain over
CLIP-seeded BRACE; the 0.741 early geometric-mean COCO result is post hoc.
The fresh real-photo F result is positive (0.733 vs 0.663, `clip_confirm_F.md`),
but uses the C16 path and a D-selected threshold, not the generated-set V3
protocol. Standard VOC21 validation failed its registered test by 2.00 mIoU
points (`voc21_val.md`). Preserve both outcomes.

## Environment

This checkout links `.venv` to `/home/fsh/.venvs/brace` on local disk, with
system Python 3.10.12. The initial NFS environment is preserved under
`outputs/setup/venv_nfs_incomplete`; small-file installation there was very slow.
Claude's original cluster used Conda `trdp` and Python 3.10.20. Both use the CPython 3.10
ABI; do not use the machine's default Conda base Python 3.14 for this stack.
The base and unrelated `mcmt_env` environments are not modified.

From the repository root:

```bash
BRACE_VENV_DIR="$HOME/.venvs/brace" bash scripts/setup_brace.sh
source scripts/brace_env.sh
python scripts/download_brace_models.py
python scripts/check_brace_setup.py
```

The setup script installs the pinned CUDA 11.8 PyTorch wheels,
`requirements.txt` with the legacy `bitsandbytes` extra omitted, and SimpleCRF
in the explicit fallback directory
`outputs/dependencies/simplecrf_py310`. A compiler is needed for SimpleCRF.
`brace_env.sh` selects the local HF cache and experiment cache. It does not
force offline mode; set `HF_HUB_OFFLINE=1` after downloading the weights.
The model downloader fetches the SD1.5 pipeline's standard safetensors and
CLIP B/16 PyTorch checkpoint, omitting duplicate formats and optional models.
Downloads stage under `/tmp/brace-hf-<uid>` because this NFS mount does not
support the required file locks; completed snapshots are copied to `HF_HOME`.
Override staging with `BRACE_DOWNLOAD_CACHE` on another local filesystem.

The full original requirements list installed and passed `pip check`, but its
`bitsandbytes==0.43.3` extra caused `StableDiffusionPipeline` to fail on import:
Triton raised `RuntimeError: 0 active drivers`. That package is only used by
old 8-bit LoRA training, not BRACE. Its installed package and metadata were
preserved at `/home/fsh/.venvs/brace-legacy-extras/`, outside the active import
path. All other handoff package pins are retained, including PEFT. The setup
script generates `outputs/setup/requirements-brace.txt` with just this one
extra omitted; the original tracked requirements remain unchanged. GPU LoRA
training with the legacy extra has not been validated here.

Verified on this machine: the readiness check and `pip check` pass; SD1.5
(float32, CPU, all 32 attention processors) and CLIP B/16 plus its processor
load fully offline. Nine existing A3/cap tests and a synthetic DenseCRF smoke
test pass. Both benchmark loaders return 490/707 annotated records. See the
local `outputs/setup/` logs and `STATUS.md` for the final check results.

This host has no `sbatch`, and `nvidia-smi` cannot communicate with the driver.
Installing CUDA-enabled PyTorch does not fix the host NVIDIA driver. CPU
imports/scoring and offline model loading can be checked here; live capture
requires a working GPU host. No production experiments are part of this setup.

## Datasets

The authoritative URLs from the handoff are reachable as follows:

```bash
mkdir -p datasets/downloads datasets/voc2012
curl -fL -C - -o datasets/downloads/ovam_experiment_with_dataset.zip \
  http://www-vpu.eps.uam.es/publications/ovam/ovam_experiment_with_dataset.zip
curl -fL -C - -o datasets/downloads/VOCtrainval_11-May-2012.tar \
  https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
python scripts/extract_brace_datasets.py
```

OVAM's archive is 14,319,487,053 bytes (13.34 GiB); VOC's is 1,999,639,040
bytes (1.86 GiB). The OVAM server's HTTPS endpoint failed, while its original
HTTP endpoint responded. Download archives are retained; do not delete them
without the user's explicit yes. COCO2017 and optional caption/CLIP-L models
are unnecessary for the basic generated-set setup.

The extractor keeps the original images, masks and manifests for both generated
benchmarks, and VOC's 2,913 segmentation trainval images, masks and split lists.
It skips released heatmaps, synthetic training sets and Mac metadata, and never
overwrites existing inputs. The handoff's `ovam_coco_cap.csv` does not exist in
the released archive: the actual loader joins `evaluation/coco_captions_sampled.csv`
with filenames in `evaluation/ovam/coco_captions/annotations/`.

## Paths and cluster compatibility

`scripts/sg_gen_clip.py`, `sg_gen_capab.py`, `sg_clip_dense.py`, and `sg_voc21.py`
now default to `outputs/sg_comp/`, overridable with `BRACE_CACHE_DIR`.
This changes storage paths only. Set `BRACE_CACHE_DIR=/home/shafa/sg_comp`
to reuse the original cluster caches there.

Historical scripts and jobs still contain original `/home/shafa` paths,
Conda environment names, SLURM partitions, and node exclusions. Setup found
113 files with such paths before the active-entry-point changes. The local
inventory is `outputs/setup/cluster_paths.txt`; regenerate it with:

```bash
rg -n /home/shafa scripts src jobs > outputs/setup/cluster_paths.txt
```

Use explicit CLI paths where available and adapt a historical job before
running it here. Do not globally rewrite historical experiment records or
create paths under another user's home. The capture runtime and dataset
loaders used by the active generated-set scripts already derive their paths
from the checkout.

Local installation/download logs and readiness output live in `outputs/setup/`.
The final verified machine status is recorded there in `STATUS.md`.
