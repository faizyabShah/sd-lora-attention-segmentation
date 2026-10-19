#!/bin/bash
#SBATCH --job-name=sg13-regions
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
# Submit from repo root. No node constraint: any available GPU may be assigned.
# One resumable job; independent numerical tests + live gate, VOC, COCO, report.
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
run_child python tests/test_target_region_inference.py
smoke_dir="outputs/segmentation/controlled_attention_cv/sg13_target_region_inference_smoke"
for dataset in voc coco; do
    run_child python scripts/run_sg13_target_region_inference.py \
        --dataset "$dataset" --class-shard 0 --max-samples 1 --output-dir "$smoke_dir"
done
run_child python scripts/finalize_sg13_target_region_inference.py --smoke --output-dir "$smoke_dir"
echo "Numerical checks and live VOC/COCO reference gates passed; starting full package"
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then exit 0; fi
for dataset in voc coco; do
    for shard in $(seq 0 19); do
        run_child python scripts/run_sg13_target_region_inference.py \
            --dataset "$dataset" --class-shard "$shard"
    done
done
run_child python scripts/finalize_sg13_target_region_inference.py
echo "SG13 region inference completed, including fold selection, region diagnostics, oracle comparison, visuals and COCO transfer"
