# Source this file from Bash before working with BRACE.
brace_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$brace_repo_root/.venv/bin/activate"
export HF_HOME="$brace_repo_root/outputs/setup/huggingface"
export PIP_CACHE_DIR="$brace_repo_root/outputs/setup/cache/pip"
export BRACE_CACHE_DIR="${BRACE_CACHE_DIR:-$brace_repo_root/outputs/sg_comp}"
unset brace_repo_root
# Set HF_HUB_OFFLINE=1 only after downloading and verifying the model weights.
