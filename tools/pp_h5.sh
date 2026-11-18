#!/bin/bash
# H5: FAMS d sweep 1.2 and 1.4 on MTF test. 6 shards per d (both GPUs), merge, score, closedset, tsim, h3, h8.
# session h5, auto-kills when done. Resumable: shards skip images already generated.
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work
SUM=$L/PRINT_FOR_ME_h5.txt; LOG=$L/h5_driver.log; RUNNER=$L/h5_runner.sh

cat > "$RUNNER" <<'RUNEOF'
#!/bin/bash
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work; AB=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; cd $AB
PY=${PYTHON:-python}; PYF=${FAMS_PYTHON:-python}
Q="TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
LOG=$L/h5_driver.log; SUM=$L/PRINT_FOR_ME_h5.txt
log() { echo "$(date '+%F %T') [h5] $1" | tee -a $LOG; }
DIRS=""
for D in 1.2 1.4; do
  FS="anonymizers.fams.faithful={anonymization_degree: $D, guidance_scale: 4.0, num_inference_steps: 200}"
  OD=$W/anon/mtf/fams/anonymization_degree-${D}_guidance_scale-4.0_num_inference_steps-200/seed0/test
  log "d=$D launching 6 shards"
  for i in 0 1 2 3 4 5; do
    env CUDA_VISIBLE_DEVICES=$((i%2)) $Q $PYF -m anonbench.anonymize --method fams --split test --shard $i/6 --set "$FS" > $L/h5_d${D}_s$i.log 2>&1 &
    sleep 90
  done
  wait
  n=$(grep -l '^\[done\]' $L/h5_d${D}_s*.log | wc -l)
  log "d=$D shards done $n/6"
  if [ "$n" -ne 6 ]; then log "d=$D SHARDS FAILED, see $L/h5_d${D}_s*.log"; continue; fi
  $PY -m anonbench.anonymize --method fams --split test --merge-shards 6 --set "$FS" >> $LOG 2>&1
  $PY - >> $LOG 2>&1 <<PYCHK
import pandas as pd
from PIL import Image
m = pd.read_csv("$OD/manifest.csv"); bad = 0
for p in m.anon_path:
    try:
        with Image.open(p) as im: im.load()
    except Exception: bad += 1
print(f"[h5] d=$D png check: {len(m)} rows, unreadable {bad}")
PYCHK
  log "d=$D merged, scoring"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.score --anon-dir $OD >> $LOG 2>&1
  env CUDA_VISIBLE_DEVICES=0 $Q $PY -m anonbench.closedset_eval --anon-dir $OD >> $LOG 2>&1
  DIRS="$DIRS $OD"
done
if [ -n "$DIRS" ]; then
  log "tsim + h3 + h8 on:$DIRS"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim --anon-dir $DIRS >> $LOG 2>&1
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.h3  --anon-dir $DIRS >> $LOG 2>&1
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.h8 --iters 4000 --anon-dir $DIRS >> $LOG 2>&1
fi
log "done"
{
  echo '***** PRINT FOR ME *****'
  echo '=== H5 FAMS d sweep (1.2 / 1.4 vs existing 1.25)'
  grep -E '\[h5\]|png check|SHARDS FAILED|Traceback|Error' $LOG | tail -40
  echo '=== open-set (all FAMS d)'
  $PY -c "
import pandas as pd
pd.set_option('display.width',250)
d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/scores/scores.csv')
d=d[d.run.str.contains('fams')]
d['d']=d.run.str.extract(r'degree-([0-9.]+)')
print(d[['d','embedder','rank1','rank1_ci_low','rank1_ci_high','tar@0.001','median_src_psnr']].sort_values(['d','embedder']).to_string(index=False))" 2>&1 | tail -30
  echo '=== closed-set'
  $PY -c "
import pandas as pd
d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/closedset/closedset.csv')
d=d[d.run.str.contains('fams')]
d['d']=d.run.str.extract(r'degree-([0-9.]+)')
print(d[['d','model','acc']].sort_values(['d','model']).to_string(index=False))" 2>&1 | tail -30
  echo '=== T-sim'
  $PY -c "
import pandas as pd
d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/Tsim/tsim.csv')
d=d[d.run.str.contains('fams')]
d['d']=d.run.str.extract(r'degree-([0-9.]+)')
print(d[['d','feature','mode','source_r1','identity_r1','identity_r1_clean']].sort_values(['d','mode','feature']).to_string(index=False))" 2>&1 | tail -30
  echo '=== H3 / H8'
  $PY -c "
import pandas as pd
for f,c in (('H3/h3.csv',['feature','mode','anon_anon_r1','clean_clean_r1']),('H8/h8.csv',['train_fold','test_fold','source_r1','identity_r1'])):
    d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/'+f); d=d[d.run.str.contains('fams')]
    d['d']=d.run.str.extract(r'degree-([0-9.]+)')
    print(f); print(d[['d']+c].sort_values('d').to_string(index=False))" 2>&1 | tail -40
  echo '*****'
} > $SUM
tmux kill-session -t h5
RUNEOF
chmod +x "$RUNNER"
tmux kill-session -t h5 2>/dev/null
tmux new-session -d -s h5 -x 200 -y 50 "$RUNNER"
sleep 3
tmux has-session -t h5 2>/dev/null && { tmux split-window -h -t h5 "watch -n5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; free -g | head -2; tail -3 ${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/h5_d1.2_s0.log 2>/dev/null'"; tmux select-pane -t h5.0; echo "launched"; } || { echo "SESSION DIED"; tail -20 "$LOG"; }
