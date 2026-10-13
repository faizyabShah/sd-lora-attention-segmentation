#!/bin/bash
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs

evaluation_job="$(sbatch --parsable jobs/run_final_component_ablation.slurm)"
merge_job="$(sbatch --parsable --dependency="afterok:${evaluation_job}" jobs/merge_final_component_ablation.slurm)"

echo "VOC-Sim + COCO-Cap component ablation array: ${evaluation_job}"
echo "Dependent merge: ${merge_job}"
