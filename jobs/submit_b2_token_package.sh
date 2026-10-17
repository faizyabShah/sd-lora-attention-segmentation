#!/bin/bash
set -euo pipefail
smoke=$(sbatch --parsable jobs/smoke_b2_token_optimization.slurm)
b2a=$(sbatch --parsable --dependency="afterok:${smoke}" jobs/run_b2a_token_optimization.slurm)
b2b=$(sbatch --parsable --dependency="afterok:${smoke}" jobs/run_b2b_token_optimization.slurm)
voc=$(sbatch --parsable --dependency="afterok:${b2a}:${b2b}" jobs/run_b2_voc_evaluation.slurm)
voc_merge=$(sbatch --parsable --dependency="afterok:${voc}" jobs/merge_b2_voc_evaluation.slurm)
coco=$(sbatch --parsable --dependency="afterok:${voc_merge}" jobs/run_b2_coco_transfer.slurm)
final=$(sbatch --parsable --dependency="afterok:${coco}" jobs/merge_b2_token_package.slurm)
cat <<EOF
smoke=${smoke}
b2a=${b2a}
b2b=${b2b}
voc=${voc}
voc_merge=${voc_merge}
coco=${coco}
final=${final}
EOF

