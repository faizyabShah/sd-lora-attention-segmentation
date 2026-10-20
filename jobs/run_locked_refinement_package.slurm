#!/bin/bash
#SBATCH --job-name=locked-refinements
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
# STAGE=sg12_crf, sg14 or sg15. Submit from repo root. No node constraint.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
: "${STAGE:?set STAGE=sg12_crf, sg14 or sg15}"
case "$STAGE" in sg12_crf|sg14|sg15) ;; *) exit 2 ;; esac
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
run_child python tests/test_locked_refinements.py
smoke_dir="outputs/segmentation/controlled_attention_cv/${STAGE}_smoke_v1"
for shard in 0 14; do
    run_child python scripts/run_locked_refinement_package.py --stage "$STAGE" \
        --dataset voc --class-shard "$shard" --max-samples 1 --output-dir "$smoke_dir"
done
run_child python scripts/run_locked_refinement_package.py --stage "$STAGE" \
    --dataset coco --class-shard 10 --max-samples 1 --output-dir "$smoke_dir"
run_child python scripts/finalize_locked_refinement_package.py --stage "$STAGE" \
    --output-dir "$smoke_dir" --smoke
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then exit 0; fi
for dataset in voc coco; do
    for shard in $(seq 0 19); do
        run_child python scripts/run_locked_refinement_package.py --stage "$STAGE" \
            --dataset "$dataset" --class-shard "$shard"
    done
done
run_child python scripts/finalize_locked_refinement_package.py --stage "$STAGE"
if [[ "$STAGE" == sg15 ]]; then
    run_child /home/shafa/miniconda3/envs/ovam_sdxl/bin/python \
        scripts/plot_locked_region_refinements.py \
        --output-dir outputs/segmentation/controlled_attention_cv/sg15
fi
echo "Completed locked stage $STAGE including VOC selection, frozen COCO transfer and diagnostics"
