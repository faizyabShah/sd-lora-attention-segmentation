#!/bin/bash
# Independent replication of the B2 token package (protocol Section 65). From the repository root:
#   CRF_SITE=<denseCRF build> bash jobs/narmyn/b2_replication/submit.sh
set -euo pipefail
: "${CRF_SITE:?set CRF_SITE}"
LOG=${LOG_DIR:-logs}; mkdir -p "$LOG"
o() { echo "--output=$LOG/$1_%A_%a.log"; }
b2a=$(sbatch --parsable --export=ALL "$(o b2rep_b2a)" jobs/narmyn/b2_replication/run_b2a.slurm)
b2b=$(sbatch --parsable --export=ALL "$(o b2rep_b2b)" jobs/narmyn/b2_replication/run_b2b.slurm)
voc=$(sbatch --parsable --export=ALL "$(o b2rep_voc)" --dependency="afterok:${b2a}:${b2b}" jobs/narmyn/b2_replication/run_voc_eval.slurm)
coco=$(sbatch --parsable --export=ALL "$(o b2rep_coco)" --dependency="afterok:${b2a}:${b2b}" jobs/narmyn/b2_replication/run_coco_eval.slurm)
merge=$(sbatch --parsable --export=ALL --output="$LOG/b2rep_merge_%j.log" --dependency="afterok:${voc}:${coco}" jobs/narmyn/b2_replication/merge.slurm)
echo "b2a=$b2a b2b=$b2b voc=$voc coco=$coco merge=$merge"
