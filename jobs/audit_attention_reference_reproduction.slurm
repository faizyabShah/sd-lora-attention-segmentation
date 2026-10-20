#!/bin/bash
#SBATCH --job-name=attention-ref-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate trdp
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/audit_attention_reference_reproduction.py --output-dir "${AUDIT_OUT:?set AUDIT_OUT}"
