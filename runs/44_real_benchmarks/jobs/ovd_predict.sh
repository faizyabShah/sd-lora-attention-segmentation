#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -c 8 --mem 64000 -t 12:00:00
# OVDiff README step 5 (prediction), SLURM array over photos (so its built-in mmseg evaluation is skipped). Arg: key.
source ~/.bashrc; conda activate ovdiff; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
W=/home/shafa/sg_comp/jc_real44; cd $W/ovdiff_repo; K=$1; O=$W/ovdiff/out/$K; export RUN44_PHOTOS_$K=$W/ovdiff/photos_$K.json
python predict.py $K $W/ovdiff/pred/$K --nonlocal --no_overwrite --prots "$O/{dataset}_sd_k32_n32_s43_off0_0,6:13,15+_t200_proto.pt" \
  "$O/{dataset}_clipb16_-2_cfbgv3_bpp_k32_n32_s43_off0_proto.pt" "$O/{dataset}_dino_vitb8_cfbgv3_bpp_k32_n32_s43_off0_proto.pt"
# (positionals first: --prots takes nargs="+" and swallows anything after it, as in the README's own order)
