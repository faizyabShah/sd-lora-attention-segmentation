#!/bin/bash
#SBATCH --job-name=lora_r64
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/narmyn_train_lora_%x_%j.log
#
# RANK and MAXSTEPS come from the environment, so one file serves every rank.
# %x in --output is the job name, so pass --job-name to keep logs separated.
#
#   smoke test:  sbatch --job-name=lora_r8  --export=ALL,RANK=8,MAXSTEPS=50    jobs/narmyn/train_lora.slurm
#   real run:    sbatch --job-name=lora_r64 --export=ALL,RANK=64,MAXSTEPS=31000 jobs/narmyn/train_lora.slurm
#
# The ALL in --export is REQUIRED. `--export=RANK=8` alone replaces the whole
# environment, so PATH/HOME vanish and `source ~/.bashrc` fails before conda
# is ever reached. ALL keeps the login environment and adds our two variables.
#
# 31000 steps at 768x768 will very likely exceed this 12h wall. That is handled:
# 04_train_lora.py always passes --resume_from_checkpoint=latest, so resubmitting
# the identical command picks up from the newest checkpoint-N in the rank's
# output dir. Expect 2-3 resubmissions per rank; no job dependency needed.
# (Alternative: --partition=gpu_lowpriority has no wall limit on renyi's a100s
# but is preemptible -- the same resume logic covers a preemption.)

set -eo pipefail

RANK="${RANK:-64}"
MAXSTEPS="${MAXSTEPS:-31000}"

source ~/.bashrc
conda activate trdp

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs" "$TRDP_ROOT/output/lora_weights/rank${RANK}" "$TRDP_ROOT/output/logs/tb_rank${RANK}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== rank=${RANK} max_train_steps=${MAXSTEPS} job=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/narmyn/04_train_lora.py \
  --rank "${RANK}" \
  --max-train-steps "${MAXSTEPS}" \
  --output-dir "$TRDP_ROOT/output/lora_weights/rank${RANK}" \
  --logging-dir "$TRDP_ROOT/output/logs/tb_rank${RANK}" \
  "$@"
