#!/bin/bash
# CelebA replication: EDI-Anon + FAMS d=1.25 on the CelebA test split (3036 imgs, 500 ids).
# session celeba, auto-kills when done. Resumable. Progress pane reads the shard logs.
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; RUNNER=$L/celeba_runner.sh; PROG=$L/celeba_prog.sh

cat > "$RUNNER" <<'RUNEOF'
#!/bin/bash
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work; AB=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; cd $AB
PY=${PYTHON:-python}; PYF=${FAMS_PYTHON:-python}
Q="TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
LOG=$L/celeba_driver.log; SUM=$L/PRINT_FOR_ME_celeba.txt
CS="paths.celeba_root=${KCHANNEL_ROOT:-$HOME/k-channel-data}/CelebA experiments.e12.min_images_per_identity=20 experiments.e12.max_identities=500"
FS="anonymizers.fams.faithful={anonymization_degree: 1.25, guidance_scale: 4.0, num_inference_steps: 200}"
log() { echo "$(date '+%F %T') [celeba] $1" | tee -a $LOG; }

log "EDI-Anon start (CelebA test)"
env CUDA_VISIBLE_DEVICES=0 $Q $PY -m anonbench.anonymize --dataset celeba --method edi_anon --split test \
    --set $CS > $L/celeba_edianon.log 2>&1
grep -q '^\[done\]' $L/celeba_edianon.log && log "EDI-Anon done" || log "EDI-Anon FAILED (see celeba_edianon.log)"
ED=$(ls -d $W/anon/celeba/edi_anon/*/seed0/test 2>/dev/null | head -1)

log "FAMS launching 6 shards"
for i in 0 1 2 3 4 5; do
  env CUDA_VISIBLE_DEVICES=$((i%2)) $Q $PYF -m anonbench.anonymize --dataset celeba --method fams --split test \
      --shard $i/6 --set $CS "$FS" > $L/celeba_fams_s$i.log 2>&1 &
  sleep 90
done
wait
n=$(grep -l '^\[done\]' $L/celeba_fams_s*.log | wc -l); log "FAMS shards done $n/6"
FD=""
if [ "$n" -eq 6 ]; then
  $PY -m anonbench.anonymize --dataset celeba --method fams --split test --merge-shards 6 --set $CS "$FS" >> $LOG 2>&1
  FD=$(ls -d $W/anon/celeba/fams/*/seed0/test 2>/dev/null | head -1)
  log "FAMS merged -> $FD"
else
  log "FAMS SHARDS FAILED, skipping FAMS eval"
fi

DIRS=""
for d in $ED $FD; do [ -n "$d" ] && [ -f "$d/manifest.csv" ] && DIRS="$DIRS $d"; done
if [ -n "$DIRS" ]; then
  log "scoring:$DIRS"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.score  --dataset celeba --anon-dir $DIRS --set $CS >> $LOG 2>&1
  log "T-sim"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim   --dataset celeba --anon-dir $DIRS --set $CS >> $LOG 2>&1
  log "H8 learned linkage"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.h8     --dataset celeba --anon-dir $DIRS --iters 4000 --set $CS >> $LOG 2>&1
  log "H3 clustering"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.h3     --dataset celeba --anon-dir $DIRS --set $CS >> $LOG 2>&1
  log "H6 by gender"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.h6     --dataset celeba --anon-dir $DIRS \
      --groups Male Female --boot 2000 --set $CS >> $LOG 2>&1
fi
log "done"
{
  echo '***** PRINT FOR ME *****'
  echo '=== CelebA replication (test split, 3036 imgs, 500 ids; T-sim gallery 12690, chance 7.9e-05)'
  grep -E '\[celeba\]|FAILED|Traceback|Error' $LOG | tail -30
  for f in scores/scores.csv Tsim/tsim.csv H8/h8.csv H3/h3.csv H6/h6_gaps.csv; do
    echo "=== $f"
    $PY -c "
import pandas as pd,sys
pd.set_option('display.width',250)
try:
    d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/celeba/$f')
    print(d.to_string(index=False)[:3000])
except Exception as e: print('(none:',e,')')" 2>&1 | head -40
  done
  echo '*****'
} > $SUM
tmux kill-session -t celeba
RUNEOF
chmod +x "$RUNNER"

cat > "$PROG" <<'PEOF'
#!/bin/bash
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs
while true; do
  clear
  echo "=== CelebA replication   $(date '+%F %T')"
  echo "--- driver"; tail -6 $L/celeba_driver.log 2>/dev/null
  echo "--- EDI-Anon"; tail -2 $L/celeba_edianon.log 2>/dev/null | cut -c1-110
  if ls $L/celeba_fams_s*.log >/dev/null 2>&1; then
    echo "--- FAMS shards"; tot=0; dn=0
    for f in $L/celeba_fams_s*.log; do
      s=$(basename $f .log | sed 's/.*_s//')
      line=$(grep -E '^\[fams\] [0-9]+/' $f 2>/dev/null | tail -1)
      if grep -q '^\[done\]' $f 2>/dev/null; then echo "  shard $s  DONE"
        n=$(echo "$line"|sed -E 's#^\[fams\] [0-9]+/([0-9]+).*#\1#'); dn=$((dn+${n:-0})); tot=$((tot+${n:-0}))
      elif [ -n "$line" ]; then echo "  shard $s  $line"
        a=$(echo "$line"|sed -E 's#^\[fams\] ([0-9]+)/.*#\1#'); b=$(echo "$line"|sed -E 's#^\[fams\] [0-9]+/([0-9]+).*#\1#')
        dn=$((dn+${a:-0})); tot=$((tot+${b:-0}))
      else echo "  shard $s  starting"; tail -1 $f 2>/dev/null | cut -c1-100 | sed 's/^/           /'; fi
    done
    [ "$tot" -gt 0 ] && echo "  TOTAL: $dn/$tot ($((100*dn/tot))%)"
  fi
  echo "--- errors"; grep -hE 'Traceback|Error|CUDA out of memory' $L/celeba_*.log 2>/dev/null | tail -3 | sed 's/^/  /'
  sleep 20
done
PEOF
chmod +x "$PROG"
tmux kill-session -t celeba 2>/dev/null
tmux new-session -d -s celeba -x 200 -y 50 "$RUNNER"
sleep 3
if tmux has-session -t celeba 2>/dev/null; then
  tmux split-window -h -t celeba "watch -n5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; free -g | head -2'"
  tmux split-window -v -t celeba.0 "$PROG"
  tmux resize-pane -t celeba.0 -y 6
  echo "launched"
else
  echo "SESSION DIED"; tail -20 $L/celeba_driver.log
fi
