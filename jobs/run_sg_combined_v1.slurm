#!/bin/bash
#SBATCH --job-name=sg-combined-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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
    "$@" & child_pid=$!
    wait "$child_pid"
    child_pid=""
}
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then
    smoke_dir="outputs/segmentation/controlled_attention_cv/sg_combined_v1_smoke${SMOKE_SUFFIX:-}"
    for dataset in voc coco; do
        run_child python scripts/run_sg_combined_v1.py --dataset "$dataset" --class-shard 0 \
            --max-samples 1 --verify-controls --output-dir "$smoke_dir"
    done
    run_child python scripts/finalize_sg_combined_v1.py --output-dir "$smoke_dir" --smoke
    exit 0
fi
: "${SLURM_ARRAY_TASK_ID:?submit with --array=0-39%N}"
if (( SLURM_ARRAY_TASK_ID < 20 )); then
    dataset=voc; shard="$SLURM_ARRAY_TASK_ID"
else
    dataset=coco; shard="$((SLURM_ARRAY_TASK_ID - 20))"
fi
run_child python scripts/run_sg_combined_v1.py --dataset "$dataset" --class-shard "$shard" --verify-controls
