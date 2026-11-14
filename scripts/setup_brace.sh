#!/usr/bin/env bash
# Run from any directory; installs only into the repository.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PIP_CACHE_DIR="$repo_root/outputs/setup/cache/pip"
mkdir -p "$PIP_CACHE_DIR" outputs/dependencies outputs/sg_comp datasets
if [[ ! -x .venv/bin/python ]]; then
    env_dir="${BRACE_VENV_DIR:-$repo_root/.venv}"
    "${BRACE_PYTHON:-python3.10}" -m venv "$env_dir"
    if [[ "$env_dir" != "$repo_root/.venv" ]]; then
        ln -s "$env_dir" .venv
    fi
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "BRACE requires Python 3.10 for this setup"'
.venv/bin/python -m pip install --no-compile torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
# bitsandbytes is only for old 8-bit LoRA training. Its pinned release eagerly
# initializes Triton and breaks pipeline imports on a machine without a GPU.
.venv/bin/python - <<'PY'
from pathlib import Path
source = Path('requirements.txt').read_text().splitlines()
Path('outputs/setup/requirements-brace.txt').write_text(
    '\n'.join(line for line in source if not line.startswith('bitsandbytes==')) + '\n'
)
PY
.venv/bin/python -m pip install --no-compile -r outputs/setup/requirements-brace.txt
# Avoid a second NumPy installation in the fallback directory.
.venv/bin/python -m pip install --no-compile --no-deps SimpleCRF==0.2.1.1 --target outputs/dependencies/simplecrf_py310
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, torchvision, denseCRF; from diffusers import StableDiffusionPipeline; print("Core imports: OK")'
echo 'Dependencies installed. Source scripts/brace_env.sh; prepare datasets/models; then run python scripts/check_brace_setup.py.'
