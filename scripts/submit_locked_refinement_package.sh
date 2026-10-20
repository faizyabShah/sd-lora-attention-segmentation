#!/bin/bash
# Submit only after the three live smoke_validation.json files report passed.
# Run at repository root. Stages are independent and may run concurrently.
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
for stage in ('sg12_crf','sg14','sg15'):
    p=Path('outputs/segmentation/controlled_attention_cv')/(stage+'_smoke_v1')/'smoke_validation.json'
    if not json.loads(p.read_text()).get('passed'):
        raise SystemExit(f'{stage}: live smoke did not pass')
PY
first=$(sbatch --parsable --job-name=sg12-crf --export=ALL,STAGE=sg12_crf,SMOKE_ONLY=0 jobs/run_locked_refinement_package.slurm)
first=${first%%;*}
second=$(sbatch --parsable --job-name=sg14-consensus --export=ALL,STAGE=sg14,SMOKE_ONLY=0 jobs/run_locked_refinement_package.slurm)
second=${second%%;*}
third=$(sbatch --parsable --job-name=sg15-refinement --export=ALL,STAGE=sg15,SMOKE_ONLY=0 jobs/run_locked_refinement_package.slurm)
third=${third%%;*}
printf 'SG12+CRF job: %s\nSG14 job: %s\nSG15 job: %s\nIndependent jobs; Slurm may schedule them concurrently.\n' "$first" "$second" "$third"
