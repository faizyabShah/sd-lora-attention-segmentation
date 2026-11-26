#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -c 8 --mem 48000 -t 12:00:00
# OVDiff README step 1 (support sets, 32 images per class), SLURM array over classes as the README shows. Arg: key.
source ~/.bashrc; conda activate ovdiff; export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
W=/home/shafa/sg_comp/jc_real44; cd $W/ovdiff_repo; K=$1
export RUN44_PHOTOS_$K=$W/ovdiff/photos_$K.json
python sample_support_set.py --bs 2 $K $W/ovdiff/out/$K   # --bs 2 (release default 8): bs 8 needs a 4 GiB softmax block, out of memory on the 16 GB V100s; the smoke ran at batch 2
