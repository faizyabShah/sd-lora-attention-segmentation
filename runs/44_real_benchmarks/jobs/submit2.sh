#!/bin/bash
# Run 44, second part: fit44 + Context capture (behind dump0/dump1), OVDiff full (behind its smoke), then score + report
# behind every method job.
set -e
R=/home/shafa/sd-lora-attention-segmentation; L=/home/shafa/sg_comp/tmp/r44/logs; T=/home/shafa/sg_comp/tmp/r44
ENV="source ~/.bashrc && export BRACE_CACHE_DIR=/home/shafa/sg_comp HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 && cd $R"
G="-p gpu --gres=gpu:1 -c 4 --mem 48000 -t 12:00:00"; C="-p cpu -c 16 --mem 64000 -t 12:00:00"
sub() { sbatch --parsable -J "$1" -o "$L/$1.%j.log" "${@:2}"; }
FIT=$(sub e44-fit $C --dependency=afterok:263488:263489 --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py fit44")
ALL=":263491:263492:263493:263488:263489:$(squeue -u shafa -h -o '%i %j' | awk '$2 ~ /e44-(sclip|clearclip|naclip|proxyclip|nerve|iseg|dseg|daam|ovam)/ {print $1}' | paste -sd:)"
for k in 0 1 2; do ALL+=":$(sub e44-cap-ctx$k $G --dependency=afterok:$FIT --wrap "$ENV && conda activate trdp && python scripts/jc_ours44.py capture --set context --shard $k --num-shards 3")"; done
OV=""
for K in voc44 context44 coco44; do
  S=$(sbatch --parsable -J e44-ovs-$K -o "$L/e44-ovs-$K.%A_%a.log" --array=0-3 --dependency=afterok:263512 $T/ovd_sample.sh $K)
  P=$(sbatch --parsable -J e44-ovp-$K -o "$L/e44-ovp-$K.%j.log" --dependency=afterok:$S $T/ovd_proto.sh $K)
  Q=$(sbatch --parsable -J e44-ovq-$K -o "$L/e44-ovq-$K.%A_%a.log" --array=0-1 --dependency=afterok:$P $T/ovd_predict.sh $K)
  OV+=":$Q"
done
SC=""
for s in voc context coco; do
  SC+=":$(sub e44-score-$s $C --dependency=afterany${ALL}${OV} --wrap "$ENV && conda activate trdp && python scripts/jc_score44.py score --set $s --workers 16")"
done
REP=$(sub e44-report -p cpu -c 4 --mem 64000 -t 04:00:00 --dependency=afterok$SC --wrap "$ENV && conda activate trdp && python scripts/jc_score44.py report")
echo "fit $FIT ovdiff$OV score$SC report $REP all$ALL" | tee $T/jobs2.txt
