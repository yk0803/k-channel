#!/bin/bash
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work; cd ${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; E=${CONDA_ENVS:-$HOME/miniconda3/envs}; G=1
export PYTHONPATH=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TQDM_DISABLE=1 GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=4
export CS="paths.celeba_root=${KCHANNEL_ROOT:-$HOME/k-channel-data}/CelebA experiments.e12.min_images_per_identity=20 experiments.e12.max_identities=500"
LOG=$L/bench.log; OUT=$W/results/celeba/timing.csv
[ -f $OUT ] || echo "method,images,seconds_per_image_reported,wall_seconds,exit,peak_gpu_mib,other_gpu_util_at_start" > $OUT
echo "$(date +%T) [bench] start (resume: $(($(wc -l < $OUT)-1)) rows already in timing.csv)" | tee -a $LOG
wait_free() {
  while true; do
    F=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits); R=$(free -g | awk '/Mem/{print $7}')
    [ "$F" -ge 30000 ] && [ "$R" -ge 24 ] && return
    echo "$(date +%T) [bench] waiting: GPU$G free ${F} MiB, host RAM available ${R} GB" | tee -a $LOG; sleep 60
  done; }
done_ok() { awk -F, -v m="$1" '$1==m && $5==0' $OUT | grep -q .; }
measure() {  # NAME IMAGES CMD... ; times CMD, polls peak GPU memory, appends a row
  NAME=$1; N=$2; shift 2
  if done_ok "$NAME"; then echo "$(date +%T) [bench] $NAME already measured, skipping" | tee -a $LOG; return; fi
  wait_free; OU=$(nvidia-smi -i $((1-G)) --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  PK=$L/bench_peak_$$.txt; echo 0 > $PK
  ( while true; do U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -gt "$(cat $PK)" ] && echo $U > $PK; sleep 2; done ) & POLL=$!
  echo "$(date +%T) [bench] $NAME start (other GPU at ${OU}%)" | tee -a $LOG; T0=$(date +%s.%N)
  "$@" 2>&1 | tee $L/bench_${NAME// /_}.log | grep -E '\] [0-9]+/|\[done\]|RESULT-V6B|Traceback|Error' | tee -a $LOG
  EX=${PIPESTATUS[0]}; T1=$(date +%s.%N); kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
  WS=$(echo "$T1 - $T0" | bc); S=$(grep -oE '= [0-9.]+ s/image' $L/bench_${NAME// /_}.log | tail -1 | grep -oE '[0-9.]+')
  [ -z "$S" ] && S=$(echo "scale=4; $WS / $N" | bc)
  echo "$NAME,$N,$S,$WS,$EX,$(cat $PK),$OU" | tee -a $OUT | tee -a $LOG; rm -f $PK; }
anon() {  # NAME METHOD_DIR PYTHON args...
  NAME=$1; MD=$2; PY=$3; shift 3
  done_ok "$NAME" || rm -rf $W/anon/celeba/$MD/*__bench__n100
  measure "$NAME" 100 $PY -m anonbench.anonymize --dataset celeba --split test --limit 100 --tag-suffix bench "$@"; }
anon EDI-Anon  edi_anon  $E/popets/bin/python   --method edi_anon  --set $CS
anon EDIALS    edials    $E/popets/bin/python   --method edials    --set $CS "anonymizers.edials.faithful={alpha: 0.5, max_iterations: 100000, num_samples: 240}"
anon BlendPose blendpose $E/popets/bin/python   --method blendpose --set $CS "anonymizers.blendpose.faithful={diffusion: True, img_weight: 0.3, k: 4, num_steps: 500}"
anon FAMS      fams      $E/fams/bin/python     --method fams      --set $CS "anonymizers.fams.faithful={anonymization_degree: 1.25, guidance_scale: 4.0, num_inference_steps: 200}"
anon NullFace  nullface  $E/nullface/bin/python --method nullface  --set $CS "anonymizers.nullface.faithful={guidance_scale: 10.0, num_diffusion_steps: 100, eta: 1.0, skip: 70, ip_adapter_scale: 1.0, id_emb_scale: 1.0, det_thresh: 0.1, det_size: 640, mask_delay_steps: 10}"
anon CIAGAN    ciagan    $E/nullface/bin/python --method ciagan    --set $CS "anonymizers.ciagan.faithful={num_classes: 1200, img_size: 128, landmarks: face_alignment}"
done_ok "k-Channel v6 k5" || { rm -f $W/feats/celeba_test_src_bench.npy; rm -rf $W/anon/celeba/genrelease/k-5_microagg_v6__bench; }
measure "k-Channel v6 k5" 3036 $E/popets/bin/python $L/bench_kc.py celeba
measure "k-Channel pose estimation" 300 $E/popets/bin/python -c "
import time, cv2, pandas as pd
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='${KCHANNEL_ROOT:-$HOME/k-channel-data}/weights/insightface', allowed_modules=['detection', 'landmark_3d_68'], providers=['CUDAExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))
P = pd.read_csv('${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/index/celeba.csv'); P = P[P.split == 'test'].path.tolist()[:300]; app.get(cv2.imread(P[0])); t = time.time()
for p in P: app.get(cv2.imread(p))
print(f'[done] {len(P)} images = {(time.time() - t) / len(P):.4f} s/image')"
{ echo "*****"; echo "Print for me (timing)"; echo "*****"; cat $OUT; grep -E 'waiting|Traceback|Error' $LOG | tail -n 8
  echo "--- FALCO (CelebA run logs, 4 shards on 2 GPUs)"; grep -E '\[fc\] (extract|pair_nn|invert|3/5|anonymize finished)' $L/fc_driver.log
  for f in $L/fc_anon_s*.log; do echo "$(basename $f): $(tr '\r' '\n' < $f | grep -E 'Anonymize celebahq images' | tail -1 | cut -c1-150)"; done
  echo "--- full-release wall times"; grep -hE 'launching|shards done' $L/bpc_driver.log $L/edc_driver.log
  echo "--- StyleGAN2 rendering rate (one-time pool cost)"; grep -E 'rendered [0-9]+/' $L/pse.log | tail -1
  grep -E 'Traceback' -A5 $L/bench_*.log | tail -n 12; echo "*****"; } > $L/PRINT_FOR_ME_bench.txt
tmux kill-session -t bench
