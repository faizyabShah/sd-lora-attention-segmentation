#!/bin/bash
set -e
source /home/shafa/miniconda3/etc/profile.d/conda.sh
conda create -y -q -n ovam_stack python=3.10
conda activate ovam_stack
pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.20.0 transformers==4.33.2 accelerate==0.23.0 "huggingface_hub==0.17.3" safetensors "numpy<2" pillow scipy
python -c "import torch,diffusers,transformers,huggingface_hub;print('OK',torch.__version__,diffusers.__version__,transformers.__version__,huggingface_hub.__version__)"
