#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_lora --config configs/train.yaml
