#!/bin/bash
#SBATCH --job-name=sg-visuals
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-119%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -eo pipefail

repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4

families=(sg1_sg3 sg3_locked sg5 sg8 sg9 sg10)
if [[ "${SMOKE:-0}" == 1 ]]; then
  family="${families[$SLURM_ARRAY_TASK_ID]}"
  shard=0
  visual_dir="outputs/segmentation/sg_visual_audit_smoke"
else
  family="${families[$((SLURM_ARRAY_TASK_ID / 20))]}"
  shard=$((SLURM_ARRAY_TASK_ID % 20))
  visual_dir="outputs/segmentation/sg_visual_audit"
fi

common=(--dataset voc --class-shard "$shard" --visual-dir "$visual_dir")
case "$family" in
  sg1_sg3)   python scripts/run_sg1_sg3_graph_refinement.py "${common[@]}" ;;
  sg3_locked) python scripts/run_token_sg3_confirmation.py "${common[@]}" --crf-site outputs/dependencies/simplecrf_py310 ;;
  sg5)       python scripts/narmyn/attention/run_sg5_contrastive_seed.py "${common[@]}" --crf-site outputs/dependencies/simplecrf_py310 ;;
  sg8)       python scripts/run_sg8_observation_consensus.py "${common[@]}" ;;
  sg9)       python scripts/run_sg9_reciprocal_propagation.py "${common[@]}" ;;
  sg10)
    sg10_results="outputs/segmentation/controlled_attention_cv/sg10_seed_connected_bottleneck"
    if [[ ! -f "$sg10_results/summary.json" ]]; then
      sg10_results="/home/sidna/TRDP/sd-lora-attention-segmentation/outputs/segmentation/controlled_attention_cv/sg10_seed_connected_bottleneck"
    fi
    python scripts/narmyn/attention/run_sg10_bottleneck.py "${common[@]}" --threshold-results "$sg10_results"
    ;;
  *) echo "Unknown family: $family" >&2; exit 2 ;;
esac

