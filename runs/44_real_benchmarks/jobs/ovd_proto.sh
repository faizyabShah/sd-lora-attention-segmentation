#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -c 8 --mem 64000 -t 12:00:00
# OVDiff README steps 2-4 (features and prototypes). Arg: key.
source ~/.bashrc; conda activate ovdiff; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
W=/home/shafa/sg_comp/jc_real44; cd $W/ovdiff_repo; K=$1; O=$W/ovdiff/out/$K; export RUN44_PHOTOS_$K=$W/ovdiff/photos_$K.json
set -e
python gen_vit_features.py --model_key clip_ViT-B/16 --layer -2 $K $O
python gen_vit_features.py --model_key dino_vitb8 $K $O
python gen_proto_vit.py $K $O --feature_path_prefix dino/dino_vitb8_8_0 dino_vitb8_cfbgv3_bpp_k32_n32_s43_off0
python gen_proto_vit.py $K $O --feature_path_prefix clip/clip_vit-b_16_16_-2_0 clipb16_-2_cfbgv3_bpp_k32_n32_s43_off0
python gen_proto_sd.py $K $O sd_k32_n32_s43_off0
