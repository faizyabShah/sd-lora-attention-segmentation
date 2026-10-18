#!/bin/bash
#SBATCH --job-name=sg5_renyi
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#
# SG5 (protocol Section 66) on one node class only (R7): renyi, 2x A100 40 GB, env gen.
# All 40 class shards (COCO-Cap first, the larger ones, then VOC-Sim) are pulled from
# one queue by WORKERS processes, alternating between the two GPUs (about 9.4 GB each);
# then finalize. Each shard is resumable. Run from the repository root:
#
#   CRF_SITE=/path/to/simplecrf LOG_DIR=/path/to/logs sbatch -o "$LOG_DIR/sg5_renyi_%j.log" jobs/narmyn/sg5/run_renyi.slurm
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
: "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}"
LOG_DIR="${LOG_DIR:-logs}"
WORKERS="${WORKERS:-6}"
mkdir -p "$LOG_DIR"
queue="$LOG_DIR/sg5_queue_${SLURM_JOB_ID}.txt"
failed="$LOG_DIR/sg5_failed_${SLURM_JOB_ID}.txt"
: > "$failed"
{ for s in $(seq 0 19); do echo "coco $s"; done; for s in $(seq 0 19); do echo "voc $s"; done; } > "$queue"
echo "node=$(hostname -s) env=${CONDA_ENV:-gen} workers=$WORKERS"

next_task() {
  flock "$queue.lock" bash -c 'line=$(head -n 1 "$1"); [ -n "$line" ] && sed -i 1d "$1"; echo "$line"' _ "$queue"
}

worker() {
  local gpu=$1 task
  while task=$(next_task) && [ -n "$task" ]; do
    set -- $task
    echo "$(date +%T) gpu=$gpu start $task"
    if CUDA_VISIBLE_DEVICES=$gpu python scripts/narmyn/attention/run_sg5_contrastive_seed.py --dataset "$1" --class-shard "$2" --crf-site "$CRF_SITE" > "$LOG_DIR/sg5_${1}_$(printf %02d "$2").log" 2>&1; then
      echo "$(date +%T) gpu=$gpu done $task"
    else
      echo "$task" >> "$failed"
      echo "$(date +%T) gpu=$gpu FAILED $task"
    fi
  done
}

for w in $(seq 0 $((WORKERS - 1))); do
  worker $((w % 2)) &
  sleep 20  # stagger model loading
done
wait
if [ -s "$failed" ]; then
  echo "FAILED shards:"; cat "$failed"; exit 1
fi
python scripts/narmyn/attention/finalize_sg5.py
