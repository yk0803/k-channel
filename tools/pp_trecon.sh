#!/usr/bin/env bash
# T-recon pipeline: select runs -> smoke -> restore (2-fold) -> panel score -> closed-set -> summary.
# usage: GPU=1 bash tools/pp_trecon.sh launch      (tmux session "trecon": left progress, right GPU/RAM)
R=${KCHANNEL_ROOT:-$HOME/k-channel-data}; A=$R/anonbench; L=$R/logs; W=$R/work
PY=${PY:-${PYTHON:-python}}
GPU=${GPU:-1}
LOG=$L/trecon.log; DRV=$L/trecon_driver.log; RUNS=$L/trecon_runs.tsv; SUM=$L/PRINT_FOR_ME_trecon.txt
QE="TQDM_DISABLE=1 PYTHONUNBUFFERED=1 GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8"
ts() { date '+%F %T'; }
d() { echo "$(ts) $*" | tee -a "$DRV"; }
# show progress lines; once a Traceback appears show everything after it
filt() { awk '{ if ($0 ~ /Traceback/) t=1; if (t || $0 ~ /^\[trecon\]|RESULT|ERROR|WARNING|done ->|\[census\]/) { print; fflush() } }'; }

select_runs() {
  "$PY" - "$W" > "$RUNS" <<'PYEOF'
import glob, re, sys
W = sys.argv[1]
c = sorted(p[:-len("/manifest.csv")] for p in glob.glob(f"{W}/anon/mtf/*/*/seed0/*/manifest.csv"))
bad = r"random|probe|__n\d|jpeg|reenc"
S = [("edi_faithful", rf"/edi_anon/(?!.*({bad}|nodecay)).*/seed0/train$"),
     ("edi_nodecay",  rf"/edi_anon/(?=.*nodecay)(?!.*({bad})).*/seed0/train$"),
     ("edials_train", r"/edials/alpha-0\.5_max_iterations-100000_num_samples-240/seed0/train$"),
     ("edials_test",  r"/edials/alpha-0\.5_max_iterations-100000_num_samples-240/seed0/test$"),
     ("fams",         rf"/fams/(?!.*({bad})).*/seed0/test$")]
for lab, pat in S:
    h = [p for p in c if re.search(pat, p)]
    if len(h) == 1:
        print(f"{lab}\t{h[0]}")
    else:
        print(f"SKIP {lab}: {len(h)} matches {h}", file=sys.stderr)
PYEOF
}

run() {
  : > "$LOG"; d "[trecon] start GPU=$GPU PY=$PY"
  cd "$A" || { d "[trecon] FAILED cd $A"; return 1; }
  select_runs 2>&1 | tee -a "$DRV"
  [ -s "$RUNS" ] || { d "[trecon] FAILED no runs selected"; return 1; }
  awk -F'\t' '{print "  run: "$1"  "$2}' "$RUNS" | tee -a "$DRV"
  first=$(head -1 "$RUNS" | cut -f2)
  d "[trecon] smoke test on $first"
  env $QE "$PY" -m anonbench.trecon --anon-dir "$first" --smoke --force 2>&1 | tee -a "$LOG" | filt
  [ "${PIPESTATUS[0]}" = 0 ] || { d "[trecon] FAILED smoke"; return 1; }
  OUTS=()
  while IFS=$'\t' read -r lab dir; do
    d "[trecon] $lab start"
    env $QE "$PY" -m anonbench.trecon --anon-dir "$dir" 2>&1 | tee -a "$LOG" | filt
    if [ "${PIPESTATUS[0]}" = 0 ]; then
      o=$(tac "$LOG" | grep -m1 '^\[trecon\] OUT ' | cut -d' ' -f3-); OUTS+=("$o"); d "[trecon] $lab OK -> $o"
    else d "[trecon] $lab FAILED"; fi
  done < "$RUNS"
  [ ${#OUTS[@]} -gt 0 ] || { d "[trecon] FAILED nothing restored"; return 1; }
  d "[trecon] panel scoring ${#OUTS[@]} runs"
  env $QE "$PY" -m anonbench.score --anon-dir "${OUTS[@]}" 2>&1 | tee -a "$LOG" | filt
  d "[trecon] closed-set (guides + held-out)"
  env $QE "$PY" -m anonbench.closedset_eval --anon-dir "${OUTS[@]}" 2>&1 | tee -a "$LOG" | filt
  d "[trecon] done"
}

summary() {
  { echo '***** PRINT FOR ME *****'
    echo '===== driver'; cat "$DRV"
    echo '===== restoration (median over images)'; grep -E '^\[trecon\] RESULT' "$LOG"
    echo '===== panel / closed-set on restored'; grep -E 'RESULT|ERROR' "$LOG" | grep -v '^\[trecon\]'
    echo '===== errors'; grep -nE 'Traceback|Error|error:' "$LOG" | tail -15
    echo '===== source vs restored'
    cd "$A" && "$PY" - "$W" <<'PYEOF'
import sys, pandas as pd
from pathlib import Path
W = Path(sys.argv[1]); R = W / "results/mtf"
def set_tag(_, d):  # same rule as anonbench.embed.set_tag
    return "anon__" + "__".join(Path(d).resolve().relative_to((W / "anon").resolve()).parts[1:])
cfg = None
t = pd.read_csv(R / "Trecon/trecon.csv")
s = pd.read_csv(R / "scores/scores.csv"); c = pd.read_csv(R / "closedset/closedset.csv")
s = s[s.alignment == "transfer"] if "alignment" in s else s
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
for r in t.itertuples():
    a, b = set_tag(cfg, Path(r.source)), set_tag(cfg, Path(r.out))
    print(f"--- {r.run}  PSNR {r.psnr_anon:.2f}->{r.psnr_rec:.2f}  SSIM {r.ssim_anon:.3f}->{r.ssim_rec:.3f}")
    x = s[s.run.isin([a, b])].assign(v=lambda d: d.run.map({a: "anon", b: "restored"}))
    if len(x):
        print(x.pivot_table(index="embedder", columns="v", values=["rank1", "tar@0.001"]).round(3).to_string())
    y = c[c.run.isin([a, b])].assign(v=lambda d: d.run.map({a: "anon", b: "restored"}))
    if len(y):
        print((y.pivot_table(index="model", columns="v", values="acc") * 100).round(2).to_string())
PYEOF
    echo '*****'; } > "$SUM" 2>&1
}

gpu() {
  while true; do
    clear; echo "GPU$((GPU+1)) (CUDA $GPU)  $(date +%T)"
    nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo; echo "trecon/score procs on this GPU (pid, MiB):"
    P=$(pgrep -d'|' -f 'anonbench\.(trecon|score|closedset_eval)')
    nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader | grep -E "^(${P:-x}),"
    echo; free -g | head -2; echo; ps -o pid,rss,etime,args -p "${P//|/,}" 2>/dev/null | cut -c1-120
    sleep 5
  done
}

launch() {
  tmux kill-session -t trecon 2>/dev/null
  tmux new-session -d -s trecon -x 220 -y 50 "export PY=$PY GPU=$GPU; bash $A/tools/pp_trecon.sh run; bash $A/tools/pp_trecon.sh summary; sleep 3; tmux kill-session -t trecon"
  tmux split-window -h -t trecon "export PY=$PY GPU=$GPU; bash $A/tools/pp_trecon.sh gpu"
  tmux select-pane -t trecon:0.0
}

export PY GPU
"${1:-launch}"
