#!/bin/bash
#SBATCH --job-name=pre-sg19-late
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
# Any available GPU. Generated images, QK, and attention matrices are discarded.
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
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then
    smoke_dir=outputs/segmentation/analysis/pre_sg19_late_support_smoke
    run_child python scripts/run_pre_sg19_late_support.py --smoke --output-dir "$smoke_dir"
    run_child python scripts/finalize_pre_sg19_late_support.py --smoke --output-dir "$smoke_dir"
    exit 0
fi
run_child python scripts/run_pre_sg19_late_support.py
run_child python scripts/finalize_pre_sg19_late_support.py
echo "Pre-SG19 diagnostic completed"
