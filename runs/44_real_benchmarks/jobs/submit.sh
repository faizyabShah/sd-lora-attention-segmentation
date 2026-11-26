#!/bin/bash
# Run 44 full chain (PREREG.md). SLURM's per-user GPU limit (4) queues the GPU jobs. Smoke outputs are moved, not deleted.
set -e
W=/home/shafa/sg_comp/jc_real44; R=/home/shafa/sd-lora-attention-segmentation; L=/home/shafa/sg_comp/tmp/r44/logs; mkdir -p $L
mkdir -p $W/_smoke
for d in clip_sclip clip_clearclip clip_naclip clip_proxyclip nerve iseg diffseg daam ovam scores_voc scores_context scores_coco; do
  [ -e $W/$d ] && mv $W/$d $W/_smoke/$d
done
[ -e $W/ours ] && mkdir -p $W/_smoke/ours && for d in dump44 fit44.pt cap_voc cap_context cap_coco; do [ -e $W/ours/$d ] && mv $W/ours/$d $W/_smoke/ours/$d; done
[ -e $W/_smoke/iseg ] && mkdir -p $W/iseg && cp $W/_smoke/iseg/list_*.txt $W/iseg/ 2>/dev/null || true
ENV="source ~/.bashrc && export BRACE_CACHE_DIR=/home/shafa/sg_comp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 && cd $R"
G="-p gpu --gres=gpu:1 -c 4 --mem 48000 -t 12:00:00"; C="-p cpu -c 16 --mem 64000 -t 12:00:00"
sub() { sbatch --parsable -J "$1" -o "$L/$1.%j.log" "${@:2}"; }
# ours
D0=$(sub e44-dump0 $G --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py dump44 --shard 0 --num-shards 2")
D1=$(sub e44-dump1 $G --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py dump44 --shard 1 --num-shards 2")
FIT=$(sub e44-fit $C --dependency=afterok:$D0:$D1 --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py fit44")
OURS=""
OURS+=":$(sub e44-cap-voc $G --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py capture --set voc")"
for k in 0 1; do OURS+=":$(sub e44-cap-coco$k $G --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py capture --set coco --shard $k --num-shards 2")"; done
for k in 0 1 2; do OURS+=":$(sub e44-cap-ctx$k $G --dependency=afterok:$FIT --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py capture --set context --shard $k --num-shards 3")"; done
# CLIP methods, NERVE, iSeg
B=""
for r in sclip clearclip naclip proxyclip; do
  B+=":$(sub e44-$r $G --wrap "$ENV && conda activate clipseg && for s in voc context coco; do python scripts/jc_clip44.py --repo $r --set \$s; done")"
done
B+=":$(sub e44-nerve $G --wrap "$ENV && conda activate /home/sidna/miniconda3/envs/nerve && for s in voc context coco; do python scripts/jc_nerve44.py --set \$s; done")"
B+=":$(sub e44-iseg $G --wrap "$ENV && conda activate iseg && for s in voc context coco; do python scripts/jc_iseg44.py labels --set \$s && python scripts/jc_iseg44.py seg --set \$s && python scripts/jc_iseg44.py seg --set \$s --given; done")"
# DiffSegmenter, DAAM, OVAM (item-level)
for k in 0 1 2; do B+=":$(sub e44-dseg$k $G --wrap "$ENV && conda activate /home/sidna/miniconda3/envs/diffseg && for s in voc context coco; do python scripts/jc_diffseg44.py --set \$s --shard $k --num-shards 3; done")"; done
for k in 0 1; do B+=":$(sub e44-daam$k $G --wrap "$ENV && conda activate daam44 && for s in voc context coco; do python scripts/jc_daamovam44.py --method daam --set \$s --shard $k --num-shards 2; done")"; done
for k in 0 1 2; do B+=":$(sub e44-ovam$k $G --wrap "$ENV && conda activate ovam44 && for s in voc context coco; do python scripts/jc_daamovam44.py --method ovam --set \$s --shard $k --num-shards 3; done")"; done
echo "ours$OURS base$B fit $FIT" > /home/shafa/sg_comp/tmp/r44/jobs.txt; cat /home/shafa/sg_comp/tmp/r44/jobs.txt
