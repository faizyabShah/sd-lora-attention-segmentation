#!/bin/bash
#SBATCH --job-name=pre-sg18-sg21-merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/finalize_pre_sg18_sg21_diagnostics.py
MPLCONFIGDIR=outputs/segmentation/analysis/pre_sg18_sg21/plot_config \
    /home/shafa/miniconda3/envs/ovam_sdxl/bin/python \
    scripts/plot_pre_sg18_sg21_diagnostics.py \
    --output-dir outputs/segmentation/analysis/pre_sg18_sg21
