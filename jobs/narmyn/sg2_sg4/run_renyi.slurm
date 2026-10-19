#!/bin/bash
#SBATCH --job-name=sg24_renyi
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# SG2 and SG4 (protocol Section 81), standalone families, on one node class only (R7):
# renyi, 2x A100 40 GB, env gen. WORKERS processes pull the 80 class shards
# (SG2 then SG4; COCO-Cap then VOC-Sim) from one queue, alternating GPUs; then both
# finalizers; then the comparison pages (first VOC-Sim image per class, every cell)
# and their index. Every shard is resumable. Run from the repository root:
#
#   LOG_DIR=/path/to/logs sbatch -o "$LOG_DIR/sg24_renyi_%j.log" jobs/narmyn/sg2_sg4/run_renyi.slurm
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
LOG_DIR="${LOG_DIR:-logs}"
WORKERS="${WORKERS:-4}"
mkdir -p "$LOG_DIR"
queue="$LOG_DIR/sg24_queue_${SLURM_JOB_ID}.txt"
failed="$LOG_DIR/sg24_failed_${SLURM_JOB_ID}.txt"
: > "$failed"
make_queue() {  # $1 = "run" or "visual"
  : > "$queue"
  for fam in sg2 sg4; do
    if [ "$1" = run ]; then for s in $(seq 0 19); do echo "$fam coco $s"; done; fi
    for s in $(seq 0 19); do echo "$fam voc $s"; done
  done >> "$queue"
}
next_task() {
  flock "$queue.lock" bash -c 'line=$(head -n 1 "$1"); [ -n "$line" ] && sed -i 1d "$1"; echo "$line"' _ "$queue"
}
worker() {  # $1 = gpu, $2 = mode
  local gpu=$1 mode=$2 task extra
  while task=$(next_task) && [ -n "$task" ]; do
    set -- $task
    extra=(); [ "$mode" = visual ] && extra=(--visual-dir outputs/segmentation/sg_visual_audit_claude)
    if CUDA_VISIBLE_DEVICES=$gpu python scripts/narmyn/attention/run_claude_sg.py --family "$1" --dataset "$2" --class-shard "$3" "${extra[@]}" \
        > "$LOG_DIR/${mode}_${1}_${2}_$(printf %02d "$3").log" 2>&1; then
      echo "$(date +%T) gpu=$gpu done $mode $task"
    else
      echo "$mode $task" >> "$failed"; echo "$(date +%T) gpu=$gpu FAILED $mode $task"
    fi
  done
}
run_pool() {
  make_queue "$1"
  for w in $(seq 0 $((WORKERS - 1))); do worker $((w % 2)) "$1" & sleep 20; done
  wait
  if [ -s "$failed" ]; then echo "FAILED:"; cat "$failed"; exit 1; fi
}
echo "node=$(hostname -s) env=${CONDA_ENV:-gen} workers=$WORKERS"
run_pool run
for fam in sg2 sg4; do python scripts/narmyn/attention/finalize_claude_sg.py --family "$fam"; done
run_pool visual
python scripts/narmyn/attention/index_claude_visuals.py
