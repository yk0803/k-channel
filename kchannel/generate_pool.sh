#!/bin/bash
FA=${KCHANNEL_ROOT:-$HOME/k-channel-data}/third_party/FALCO; FPY=${FALCO_PYTHON:-python}; LOG=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/e3_gen.log
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1; cd $FA
for s in $(seq 1 10); do
  N=$((10000 + s)); D=$(ls -d $FA/datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-${N}* 2>/dev/null | head -1)
  if [ -n "$D" ] && [ "$(find "$D" -name image.jpg | wc -l)" -ge "$N" ]; then echo "$(date +%T) [e3] shard $s ($N) complete, skipping" | tee -a $LOG; continue; fi
  [ -n "$D" ] && rm -rf "$D" && echo "$(date +%T) [e3] shard $s partial, regenerating" | tee -a $LOG
  echo "$(date +%T) [e3] shard $s/10 start ($N faces)" | tee -a $LOG
  $FPY create_fake_dataset.py --num-samples $N --no-clip --no-farl --no-dino --no-arcface --cuda 2>&1 | tr '\r' '\n' | grep --line-buffered -E '[0-9]+/[0-9]+|Error|Traceback|error' | awk 'NR % 50 == 1 || /Error|Traceback|error/' | tee -a $LOG
  EX=${PIPESTATUS[0]}; D=$(ls -d $FA/datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-${N}* 2>/dev/null | head -1)
  echo "$(date +%T) [e3] shard $s exit $EX | images $(find "$D" -name image.jpg | wc -l)/$N | $D" | tee -a $LOG
  [ "$EX" -ne 0 ] && { echo "$(date +%T) [e3] STOPPED on shard $s" | tee -a $LOG; exit 1; }
done
echo "$(date +%T) [e3] ALL SHARDS DONE" | tee -a $LOG
