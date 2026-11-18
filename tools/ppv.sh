#!/bin/bash
# Viewer panes per job: `prog <job>` (left) and `res <job>` (right). job = fams | e4 | tsim | falco
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs
pat() { case $1 in fams) echo 'anonymize --method fams|anonbench.score --anon-dir .*fams|merge-shards 6';;
  e4) echo 'select random|closedset_eval --anon-dir .*random|score --anon-dir .*random';; tsim) echo 'anonbench.tsim';;
  falco) echo 'create_fake_dataset|invert.py|extract_features|pair_nn|FALCO.*anonymize';; esac; }
prog() {
  while true; do
    clear; echo "=== $1 | progress | $(date +%T)"
    case $1 in
      fams) for i in 0 1 2 3 4 5; do printf " s%s %s\n" $i "$(grep -hE '^\[fams\] [0-9]|^\[done\]' $L/fams_s$i.log 2>/dev/null | tail -1 | cut -c1-70)"; done
            for f in fams_merge fams_score fams_tsim; do [ -f $L/$f.log ] && echo " $f: $(grep -hE 'merge\]|RESULT|done ->' $L/$f.log | tail -1 | cut -c1-110)"; done
            [ -f $L/fams_pngcheck.txt ] && echo " $(cat $L/fams_pngcheck.txt)";;
      e4)   echo " done $(grep -l '^\[done\]' $L/edials_rand_s*.log | wc -l)/12"
            for i in $(seq 0 11); do printf " s%-2s %s\n" $i "$(grep -hE 'random\]|^\[done\]' $L/edials_rand_s$i.log | tail -1 | cut -c1-85)"; done
            for f in e4_merge e4_closedset e4_score e4_tsim; do [ -f $L/$f.log ] && echo " $f: $(grep -hE 'merge\]|RESULT|done ->' $L/$f.log | tail -1 | cut -c1-110)"; done;;
      tsim) grep -hE '\[tsim\]|RESULT|done ->|Error' $L/tsim.log 2>/dev/null | tail -30 | cut -c1-230;;
      falco) for f in falco_real falco_fake; do echo " $f: $(tr '\r' '\n' < $L/$f.log 2>/dev/null | grep -vE '^\s*$' | tail -1 | cut -c1-110)"; done;;
    esac
    echo; [ "$1" = falco ] && echo " state: $(pgrep -fc "$(pat falco)") process(es) alive" || echo " pipeline state: $(cat $L/pp_$1.done 2>/dev/null || echo running)"
    t=$(grep -l Traceback $L/${1/e4/edials_rand}*.log $L/${1}_*.log 2>/dev/null | grep -v fams_dl | xargs -r -n1 basename | tr '\n' ' '); [ -n "$t" ] && echo " TRACEBACK in: $t"
    sleep 30
  done
}
res() {
  P=$(pat $1)
  while true; do
    clear; echo "=== $1 | resources | $(date +%T)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "GPU%d  util %s | %s / %s\n",$1+1,$2,$3,$4}'
    unset IDX VR GI; declare -A IDX VR GI
    while IFS=', ' read i u; do IDX[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    while IFS=', ' read u p m mu; do VR[$p]="$m$mu"; GI[$p]=GPU$(( ${IDX[$u]} + 1 )); done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
    echo; printf "%-8s %-5s %-9s %-6s %-7s %-12s %s\n" PID GPU VRAM CPU% RAM_GB ELAPSED WHAT
    n=0
    for p in $(pgrep -f "$P"); do
      a=$(ps -o args= -p $p 2>/dev/null); case "$a" in ""|*bash*|*grep*) continue;; esac
      read c r e < <(ps -o %cpu=,rss=,etime= -p $p)
      w=$(echo "$a" | grep -oE 'shard [0-9]+/[0-9]+|merge-shards|create_fake|invert|extract_features|pair_nn|anonbench\.[a-z_]+' | head -1)
      printf "%-8s %-5s %-9s %-6s %-7s %-12s %s\n" $p "${GI[$p]:--}" "${VR[$p]:--}" $c $(awk "BEGIN{printf \"%.1f\", $r/1048576}") $e "$w"; n=$((n+1))
    done
    [ $n -eq 0 ] && echo " (no live process for $1 right now)"
    echo; free -g | awk '/Mem/{print "host RAM free " $7 " / " $2 " GB"}'
    sleep 5
  done
}
"$@"
