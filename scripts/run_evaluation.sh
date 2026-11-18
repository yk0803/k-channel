#!/bin/bash
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work; cd ${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; PY=${PYTHON:-python}
export PYTHONPATH=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3
export CS="paths.celeba_root=${KCHANNEL_ROOT:-$HOME/k-channel-data}/CelebA experiments.e12.min_images_per_identity=20 experiments.e12.max_identities=500"
F='find model|Applied providers|set det-size|model ignore|input mean'; LOG=$L/v7e.log; DONE=$L/v7e.done; touch $DONE
log() { echo "$(date '+%F %T') [v7e] $1" | tee -a $LOG; }
pick() { while true; do
    G=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2 >= 20000 && $3 < 30 {print $1; exit}')
    [ -n "$G" ] && return; log "waiting for a GPU with >=20 GB free and <30% load: $(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 60; done; }
job() { local N=$1; shift; grep -qx "$N" $DONE && { log "$N done earlier, skipping"; return; }
  pick; log "$N start on GPU $G"; CUDA_VISIBLE_DEVICES=$G "$@" 2>&1 | grep --line-buffered -vE "$F" | tee -a $LOG
  if [ ${PIPESTATUS[0]} -eq 0 ]; then echo "$N" >> $DONE; log "$N finished"; else log "$N FAILED (will be retried on the next start)"; fi; }
DC=$(ls -d $W/anon/celeba/genrelease/*_microagg_v7/seed0/test | tr '\n' ' '); DM=$(ls -d $W/anon/mtf/genrelease/*_microagg_v7/seed0/test | tr '\n' ' ')
log "releases: celeba $(echo $DC | wc -w), mtf $(echo $DM | wc -w)"
job celeba_score $PY -m anonbench.score --dataset celeba --anon-dir $DC --set $CS alignment=detect
job mtf_score    $PY -m anonbench.score --dataset mtf --anon-dir $DM --set alignment=detect
job celeba_tsim  $PY -m anonbench.tsim --dataset celeba --anon-dir $DC --set $CS
job mtf_tsim     $PY -m anonbench.tsim --dataset mtf --anon-dir $DM
job celeba_h3    $PY -m anonbench.h3 --dataset celeba --anon-dir $DC --set $CS
job mtf_h3       $PY -m anonbench.h3 --dataset mtf --anon-dir $DM
job celeba_h8    $PY -m anonbench.h8 --dataset celeba --anon-dir $DC --iters 4000 --set $CS
job mtf_h8       $PY -m anonbench.h8 --dataset mtf --anon-dir $DM --iters 4000
job celeba_h16   $PY -m anonbench.h16_strong_linkage --dataset celeba --anon-dir $DC --set $CS
job mtf_h16      $PY -m anonbench.h16_strong_linkage --dataset mtf --anon-dir $DM
job attr         $PY $L/v7_evalpatch.py attr
job u40          $PY $L/v7_evalpatch.py u40
for d in $DC; do job "sota_celeba_$(basename $(dirname $(dirname $d)))" $PY -m anonbench.sota_utility --dataset celeba --anon-dir $d --set $CS; done
for d in $DM; do job "sota_mtf_$(basename $(dirname $(dirname $d)))" $PY -m anonbench.sota_utility --dataset mtf --anon-dir $d; done
job race $PY $L/race_v7.py
log "ALL DONE ($(wc -l < $DONE) jobs finished)"
