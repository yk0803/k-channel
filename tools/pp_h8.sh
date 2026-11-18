#!/bin/bash
# H8 learned linkage attacker. GPU2 (CUDA 1). session h8, auto-kills when done.
LOG=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/h8.log; SUM=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/PRINT_FOR_ME_h8.txt; RUNNER=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs/h8_runner.sh
W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/anon/mtf
DIRS=""
for g in "$W/falco"/*/seed0/test "$W/fams"/*/seed0/test "$W/blendpose"/diffusion-True*/seed0/test \
         "$W/edials"/*/seed0/test "$W/edi_anon"/blur_kernel-40_max_iters-200000_perturbation-blur_q-90_rir_threshold-1.0/seed0/train; do
  for d in $g; do
    case "$d" in *__random/*|*__probe__*|*nodiff*|*donor_only*|*text_only*) continue;; esac
    [ -f "$d/manifest.csv" ] && DIRS="$DIRS $d"
  done
done
[ -z "$DIRS" ] && { echo "NO RUN DIRS FOUND"; exit 1; }
{ echo "=== H8 runs:"; for d in $DIRS; do echo "  $d"; done; } | tee "$LOG"

cat > "$RUNNER" <<RUNEOF
#!/bin/bash
cd ${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench
PY=\$(conda info --base)/envs/popets/bin/python
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
echo "=== smoke" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \$PY -m anonbench.h8 --smoke --anon-dir$DIRS 2>&1 | tail -30 | tee -a $LOG
if [ \${PIPESTATUS[0]} -ne 0 ]; then
  { echo '***** PRINT FOR ME *****'; echo 'H8 SMOKE FAILED'; tail -40 $LOG; echo '*****'; } > $SUM
  tmux kill-session -t h8; exit 1
fi
echo "=== full" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \$PY -m anonbench.h8 --iters 4000 --anon-dir$DIRS 2>&1 | tee -a $LOG
{
  echo '***** PRINT FOR ME *****'
  echo '=== H8 learned linkage (identity-disjoint folds)'
  grep -E 'RESULT|skip|Error|Traceback' $LOG
  echo '=== csv'
  \$PY -c "import pandas as pd; d=pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf/H8/h8.csv'); print(d[['run','train_fold','test_fold','n_test','source_r1','source_chance','identity_r1','identity_chance']].to_string(index=False))" 2>&1 | tail -40
  echo '*****'
} > $SUM
tmux kill-session -t h8
RUNEOF
chmod +x "$RUNNER"
tmux kill-session -t h8 2>/dev/null
tmux new-session -d -s h8 -x 200 -y 50 "$RUNNER"
sleep 3
tmux has-session -t h8 2>/dev/null && { tmux split-window -h -t h8 "watch -n5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; free -g | head -2'"; tmux select-pane -t h8.0; echo "launched"; } || { echo "SESSION DIED, log tail:"; tail -20 "$LOG"; }
