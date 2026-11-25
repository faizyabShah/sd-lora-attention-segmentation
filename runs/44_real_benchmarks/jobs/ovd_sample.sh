#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -c 8 --mem 48000 -t 12:00:00
# OVDiff README step 1 (support sets, 32 images per class), SLURM array over classes as the README shows. Arg: key.
source ~/.bashrc; conda activate ovdiff; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
W=/home/shafa/sg_comp/jc_real44; cd $W/ovdiff_repo; K=$1
export RUN44_PHOTOS_$K=$W/ovdiff/photos_$K.json
python sample_support_set.py $K $W/ovdiff/out/$K
