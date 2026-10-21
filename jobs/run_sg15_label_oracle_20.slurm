#!/bin/bash
#SBATCH --job-name=sg15-oracle-20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from repo root. No node constraint; one GPU assigned by Slurm.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
child_pid=""
requeue_at_timeout() {
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_at_timeout USR1
run_child() {
    "$@" &
    child_pid=$!
    wait "$child_pid"
    child_pid=""
}
run_child python tests/test_sg15_label_oracle_20.py
run_child python scripts/run_sg15_label_oracle_20.py \
    --max-classes "${MAX_CLASSES:-20}" \
    --output-dir "${OUTPUT_DIR:-outputs/segmentation/analysis/sg15_label_oracle_20}"
