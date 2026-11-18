#!/bin/bash
# PoPETs job driver: pipelines (fams | e4 | tsim) run detached; `monitor` + `gpu` are the tmux viewer panes.
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; AB=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work; cd $AB
PY=${PYTHON:-python}; PYF=${FAMS_PYTHON:-python}
Q="TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
FS='anonymizers.fams.faithful={anonymization_degree: 1.25, guidance_scale: 4.0, num_inference_steps: 200}'
FD=$W/anon/mtf/fams/anonymization_degree-1.25_guidance_scale-4.0_num_inference_steps-200/seed0/test
TD=$W/anon/mtf/edials/alpha-0.5_max_iterations-100000_num_samples-240/seed0/train
ED=$W/anon/mtf/edials/alpha-0.5_max_iterations-100000_num_samples-240__random/seed0/train
OUT=$L/PRINT_FOR_ME.txt
log() { echo "$(date '+%F %T') [$1] $2" >> $L/pp_driver.log; }
st() { [ -f $L/pp_$1.done ] && cat $L/pp_$1.done || echo running; }
wait_tsim() { while [ ! -f $L/pp_tsim.done ]; do sleep 60; done; }

fams() {
  log fams "launch 6 shards (resume on)"
  for i in 0 1 2 3 4 5; do env CUDA_VISIBLE_DEVICES=$((i%2)) $Q $PYF -m anonbench.anonymize --method fams --split test --shard $i/6 --set "$FS" > $L/fams_s$i.log 2>&1 & sleep 90; done
  wait
  n=$(grep -l '^\[done\]' $L/fams_s*.log | wc -l); log fams "shards done $n/6"
  [ "$n" -ne 6 ] && { echo "FAILED shards $n/6" > $L/pp_fams.done; return; }
  $PY -m anonbench.anonymize --method fams --split test --merge-shards 6 --set "$FS" > $L/fams_merge.log 2>&1
  $PY - > $L/fams_pngcheck.txt 2>&1 <<PY
import pandas as pd; from PIL import Image
m = pd.read_csv("$FD/manifest.csv"); bad = 0
for p in m.anon_path:
    try:
        with Image.open(p) as im: im.load()
    except Exception: bad += 1
print(f"FAMS png check: {len(m)} rows, unreadable {bad}")
PY
  log fams "merged; scoring"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.score --anon-dir $FD > $L/fams_score.log 2>&1
  wait_tsim; env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim --anon-dir $FD > $L/fams_tsim.log 2>&1
  log fams "done"; echo OK > $L/pp_fams.done
}

e4() {
  while pgrep -f 'select random' >/dev/null; do sleep 60; done
  n=$(grep -l '^\[done\]' $L/edials_rand_s*.log | wc -l); log e4 "shards done $n/12"
  [ "$n" -ne 12 ] && { echo "FAILED shards $n/12" > $L/pp_e4.done; return; }
  $PY -m anonbench.anonymize --method edials --split train --select random --budget-from $TD --merge-shards 12 > $L/e4_merge.log 2>&1
  env CUDA_VISIBLE_DEVICES=0 $Q $PY -m anonbench.closedset_eval --anon-dir $ED > $L/e4_closedset.log 2>&1 &
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.score --anon-dir $ED > $L/e4_score.log 2>&1 &
  wait
  wait_tsim; env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim --anon-dir $ED > $L/e4_tsim.log 2>&1
  log e4 "done"; echo OK > $L/pp_e4.done
}

tsim() {
  RUNS=$(ls -d $W/anon/mtf/{edi_anon,edials,blendpose,simple}/*/seed0/* 2>/dev/null | grep -vE "__n[0-9]+|shard")
  log tsim "start $(echo $RUNS | wc -w) runs"
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim --anon-dir $RUNS > $L/tsim.log 2>&1
  grep -q "done ->" $L/tsim.log && echo OK > $L/pp_tsim.done || echo "FAILED (see tsim.log)" > $L/pp_tsim.done
  log tsim "$(cat $L/pp_tsim.done)"
}

summary() {
{
echo "*****"; echo "PRINT FOR ME"; echo "*****"
echo "pipelines: fams=$(st fams) | e4=$(st e4) | tsim=$(st tsim)"
echo "--- FAMS merge: $(grep merge $L/fams_merge.log 2>/dev/null | tail -1)"; cat $L/fams_pngcheck.txt 2>/dev/null
echo "--- E4 merge: $(grep merge $L/e4_merge.log 2>/dev/null | tail -1)"
$PY - <<'PY'
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
R = "${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf"
def short(r): return (r.replace("anon__", "").replace("__seed0", "").replace("alpha-0.5_max_iterations-100000_num_samples-240", "a0.5")
                       .replace("anonymization_degree-1.25_guidance_scale-4.0_num_inference_steps-200", "d1.25"))
try:
    s = pd.read_csv(f"{R}/scores/scores.csv")
    s = s[s.run.str.contains("fams__anonymization|edials__alpha-0.5_max_iterations-100000_num_samples-240") & ~s.run.str.contains("probe")]
    s["run"] = s.run.map(short)
    print("--- OPEN-SET rank1 / TAR@1e-3"); print(s.pivot_table(index="embedder", columns="run", values=["rank1", "tar@0.001"]).round(3).to_string())
except Exception as e: print("scores:", e)
try:
    c = pd.read_csv(f"{R}/closedset/closedset.csv"); c = c[c.run.str.contains("edials__alpha-0.5_max_iterations-100000_num_samples-240")]
    c["run"] = c.run.map(short)
    print("--- CLOSED-SET acc % (EDIALS targeted vs random)"); print(c.pivot_table(index="model", columns="run", values="acc").mul(100).round(2).to_string())
except Exception as e: print("closedset:", e)
try:
    t = pd.read_csv(f"{R}/Tsim/tsim.csv"); t["run"] = t.run.map(short)
    for v in ("source_r1", "identity_r1"):
        print(f"--- T-sim {v} (chance src {t.source_chance.iloc[0]:.5f}, id {t.identity_chance.iloc[0]:.4f})")
        print(t.pivot_table(index="run", columns=["mode", "feature"], values=v).round(3).to_string())
    print("--- T-sim clean identity_r1 reference:", t.groupby(["mode", "feature"]).identity_r1_clean.first().round(3).to_dict())
except Exception as e: print("tsim:", e)
PY
echo "--- FALCO real: $(tr '\r' '\n' < $L/falco_real.log | grep -vE '^\s*$' | tail -1 | cut -c1-120)"
echo "--- FALCO fake: $(tr '\r' '\n' < $L/falco_fake.log | grep -vE '^\s*$' | tail -1 | cut -c1-120)"
for f in $(grep -l Traceback $L/fams_s*.log $L/fams_*.log $L/e4_*.log $L/tsim.log $L/edials_rand_s*.log $L/falco_*.log 2>/dev/null); do echo "== TRACEBACK in $f"; grep -A6 Traceback $f | tail -7; done
echo "--- driver log:"; tail -12 $L/pp_driver.log
echo "*****"
} > $OUT 2>&1
}

monitor() {
  while true; do
    clear
    echo "=== PoPETs $(date +%T) | RAM free $(free -g | awk '/Mem/{print $7}') GB | pipelines: fams=$(st fams) e4=$(st e4) tsim=$(st tsim)"
    echo "--- FAMS test d1.25 (6 shards)"
    for i in 0 1 2 3 4 5; do printf " s%s %s\n" $i "$(grep -hE '^\[fams\] [0-9]|^\[done\]' $L/fams_s$i.log 2>/dev/null | tail -1 | cut -c1-60)"; done
    echo "--- E4 EDIALS random: done $(grep -l '^\[done\]' $L/edials_rand_s*.log | wc -l)/12"
    for i in $(seq 0 11); do grep -q '^\[done\]' $L/edials_rand_s$i.log || printf " s%-2s %s\n" $i "$(grep -h 'random\]' $L/edials_rand_s$i.log | tail -1 | grep -oE '[0-9]+/[0-9]+|[0-9.]+ s/img|ETA [0-9]+ min' | tr '\n' ' ')"; done
    echo "--- T-sim: $(grep -c RESULT $L/tsim.log 2>/dev/null) runs done | $(grep -E '\[tsim\]|RESULT' $L/tsim.log 2>/dev/null | tail -1 | cut -c1-80)"
    echo "--- FALCO real: $(tr '\r' '\n' < $L/falco_real.log 2>/dev/null | grep -vE '^\s*$' | tail -1 | cut -c1-80)"
    echo "--- FALCO fake: $(tr '\r' '\n' < $L/falco_fake.log 2>/dev/null | grep -vE '^\s*$' | tail -1 | cut -c1-80)"
    echo "--- logs with Traceback: $(grep -l Traceback $L/fams_s*.log $L/edials_rand_s*.log $L/tsim.log $L/falco_*.log $L/fams_*.log $L/e4_*.log 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
    [ -f $L/pp_fams.done ] && [ -f $L/pp_e4.done ] && [ -f $L/pp_tsim.done ] && break
    sleep 60
  done
  summary; touch $L/pp_all.done; tmux kill-session -t pp
}

gpu() {
  while true; do
    clear; echo "=== GPUs $(date +%T)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | awk -F', ' '{printf "GPU%d (cuda%d) util %s | %s / %s | %s C\n",$1+1,$1,$2,$3,$4,$5}'
    echo
    declare -A IDX; while IFS=', ' read i u; do IDX[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=', ' read u p m mu; do
      a=$(ps -o args= -p $p 2>/dev/null)
      case "$a" in
        *"method fams"*) n="FAMS $(echo "$a" | grep -oE 'shard [0-9]/6')";;
        *"select random"*) n="E4 EDIALS $(echo "$a" | grep -oE 'shard [0-9]+/12')";;
        *anonbench.tsim*) n="T-sim";; *anonbench.score*) n="score";; *closedset_eval*) n="closedset";;
        *create_fake*) n="FALCO fake";; *invert.py*) n="FALCO invert";; *extract_features*) n="FALCO features";;
        *PRISMA*|*fawkes*|*lowkey*|*protect.py*|*train_fr*) n="(other project)";;
        *) n=$(echo "$a" | cut -c1-28);;
      esac
      echo "GPU$(( ${IDX[$u]} + 1 )) pid $p ${m}${mu}  $n"
    done | sort
    sleep 10
  done
}
"$@"
