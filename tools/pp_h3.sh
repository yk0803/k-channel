#!/bin/bash
# H3 anon<->anon linkage. GPU2 (CUDA 1). session h3, auto-kills when done.
LOG=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/h3.log; SUM=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/PRINT_FOR_ME_h3.txt; RUNNER=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/h3_runner.sh
W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/anon/mtf
DIRS=""
for g in "$W/falco"/*/seed0/test "$W/fams"/*/seed0/test "$W/blendpose"/*/seed0/test \
         "$W/edials"/*/seed0/test "$W/edi_anon"/*/seed0/train; do
  for d in $g; do
    case "$d" in *__random/*|*__probe__*) continue;; esac
    [ -f "$d/manifest.csv" ] && DIRS="$DIRS $d"
  done
done
[ -z "$DIRS" ] && { echo "NO RUN DIRS FOUND"; exit 1; }
{ echo "=== H3 runs:"; for d in $DIRS; do echo "  $d"; done; } | tee "$LOG"

cat > "$RUNNER" <<RUNEOF
#!/bin/bash
cd ${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench
PY=\$(conda info --base)/envs/popets/bin/python
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
CUDA_VISIBLE_DEVICES=1 \$PY -m anonbench.h3 --anon-dir$DIRS 2>&1 | tee -a $LOG
{
  echo '***** PRINT FOR ME *****'
  echo '=== H3 anon<->anon linkage'
  grep -E 'RESULT|dropping|Error|Traceback|No such' $LOG
  echo '=== csv'
  \$PY -c "import pandas as pd; d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/H3/h3.csv'); print(d[['run','feature','mode','n_identities','anon_anon_r1','clean_clean_r1','chance']].to_string(index=False))" 2>&1 | tail -70
  echo '*****'
} > $SUM
tmux kill-session -t h3
RUNEOF
chmod +x "$RUNNER"
tmux kill-session -t h3 2>/dev/null
tmux new-session -d -s h3 -x 200 -y 50 "$RUNNER"
sleep 3
tmux has-session -t h3 2>/dev/null && { tmux split-window -h -t h3 "watch -n5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; free -g | head -2'"; tmux select-pane -t h3.0; echo "launched"; } || { echo "SESSION DIED, log tail:"; tail -20 "$LOG"; }
