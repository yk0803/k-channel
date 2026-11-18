#!/bin/bash
# FALCO on MTF test: run (2 shard views) -> collect -> score (alignment=detect) -> T-sim. Viewer: prog. Session 'falco'.
L=${KCHANNEL_ROOT:-$HOME/k-channel-data}/logs; F=${KCHANNEL_ROOT:-$HOME/k-channel-data}/third_party/FALCO; AB=${KCHANNEL_ROOT:-$HOME/k-channel-data}/anonbench; W=${KCHANNEL_ROOT:-$HOME/k-channel-data}/work
PYF=${FALCO_PYTHON:-python}; PY=${PYTHON:-python}
FK=datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-60000-CLIP-FaRL-DINO-ArcFace
NS=2; OUT=$W/anon/mtf/falco/m-0.0_epochs-50_nn-farl-euclidean/seed0/test; SUM=$L/PRINT_FOR_ME_falco.txt
Q="PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TQDM_DISABLE=1"
run() {
  M=$W/falco_mtf_test
  for k in $(seq 0 $((NS-1))); do
    R=$W/falco_mtf_test_s$k; mkdir -p $R/annotations; ln -sfn $M/data $R/data
    cp $M/annotations/list_eval_partition.txt $M/annotations/CelebAMask-HQ-attribute-anno.txt $R/annotations/
    awk -v k=$k -v n=$NS 'NR==1 || ($1 % n == k)' $M/annotations/CelebA-HQ-to-CelebA-mapping.txt > $R/annotations/CelebA-HQ-to-CelebA-mapping.txt
  done
  cd $F
  for k in $(seq 0 $((NS-1))); do
    env CUDA_VISIBLE_DEVICES=$((k%2)) PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore $PYF $F/anonymize.py -v --dataset=celebahq --dataset-root=$W/falco_mtf_test_s$k \
      --fake-nn-map=$FK/farl_auto_euclidean_nn_map_celebahq.json --latent-space=W+ -m 0.0 --epochs=50 --cuda > $L/falco_anon_s$k.log 2>&1 &
    sleep 60
  done
  wait
  cd $AB
  $PY - > $L/falco_collect.log 2>&1 <<PY
import re, glob, os, numpy as np, pandas as pd
from PIL import Image
from anonbench.manifest import ManifestWriter
mp = pd.read_csv("$W/falco_mtf_test/mtf_map.csv"); idx = pd.read_csv("$W/index/mtf.csv").set_index("image_id")
dirs = []
for f in sorted(glob.glob("$L/falco_anon_s*.log")):
    m = re.search(r"Create dir for storing.*?\n\s*\\\\__(\S+)", open(f, errors="ignore").read())
    if m: dirs.append(os.path.join("$F", m.group(1)))
dirs = sorted(set(dirs)); print("FALCO out dirs:", dirs)
Wm = ManifestWriter("$OUT", meta=dict(method="falco", params=dict(m=0.0, epochs=50, latent="W+", nn="farl_auto_euclidean"), seed=0, falco_dirs=dirs), geometry="ffhq_aligned_crop")
ok = miss = 0
for _, q in mp.iterrows():
    r = dict(idx.loc[q.image_id]); r["image_id"] = q.image_id
    p = next((os.path.join(d, "data", f"{q.hq_idx}.jpg") for d in dirs if os.path.exists(os.path.join(d, "data", f"{q.hq_idx}.jpg"))), None)
    if p is None:
        Wm.add(r, None, "failed", 0); miss += 1; continue
    with Image.open(p) as im: Wm.save_image(r, np.asarray(im.convert("RGB")), "modified", 50, falco_file=p)
    ok += 1
Wm.close(); print(f"FALCO collected {ok}/{len(mp)} (missing {miss}) -> $OUT")
PY
  grep -q "missing 0" $L/falco_collect.log || { echo "FAILED collect (see falco_collect.log)" > $L/pp_falco.done; return; }
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.score --anon-dir $OUT --set alignment=detect > $L/falco_score.log 2>&1
  env CUDA_VISIBLE_DEVICES=1 $Q $PY -m anonbench.tsim --anon-dir $OUT > $L/falco_tsim.log 2>&1
  echo OK > $L/pp_falco.done
}
summary() {
{
echo "*****"; echo "PRINT FOR ME"; echo "*****"
echo "FALCO pipeline: $(cat $L/pp_falco.done 2>/dev/null)"
for f in $L/falco_anon_s*.log; do echo "$(basename $f): $(tr '\r' '\n' < $f | grep -E 'Anonymize celebahq images' | tail -1 | cut -c1-110)"; done
tail -2 $L/falco_collect.log
$PY - <<'PY'
import pandas as pd; pd.set_option("display.width", 250)
R = "${KCHANNEL_ROOT:-$HOME/k-channel-data}/work/results/mtf"
try:
    s = pd.read_csv(f"{R}/scores/scores.csv"); s = s[s.run.str.contains("falco")]
    print("--- FALCO open-set (alignment=detect)"); print(s[["embedder", "rank1", "rank1_ci_low", "rank1_ci_high", "rank1_clean_same_probes", "tar@0.001", "median_src_psnr", "alignment"]].round(3).to_string(index=False))
except Exception as e: print("scores:", e)
try:
    t = pd.read_csv(f"{R}/Tsim/tsim.csv"); t = t[t.run.str.contains("falco")]
    print("--- FALCO T-sim"); print(t[["mode", "feature", "source_r1", "identity_r1", "identity_r1_clean"]].round(3).to_string(index=False))
except Exception as e: print("tsim:", e)
PY
for f in $(grep -l Traceback $L/falco_anon_s*.log $L/falco_collect.log $L/falco_score.log $L/falco_tsim.log 2>/dev/null); do echo "== TRACEBACK in $f"; grep -A8 Traceback $f | tail -9; done
echo "*****"
} > $SUM 2>&1
}
prog() {
  while true; do
    clear; echo "=== FALCO m=0 on MTF test (1252) | $(date +%T) | RAM free $(free -g | awk '/Mem/{print $7}') GB | state: $(cat $L/pp_falco.done 2>/dev/null || echo running)"
    for f in $L/falco_anon_s*.log; do printf " %-14s %s\n" $(basename $f .log) "$(tr '\r' '\n' < $f | grep -E 'Anonymize celebahq images|Traceback|Error' | tail -1 | cut -c1-110)"; done
    for f in falco_collect falco_score falco_tsim; do [ -f $L/$f.log ] && echo " $f: $(grep -hE 'collected|RESULT|done ->|Error' $L/$f.log | tail -1 | cut -c1-110)"; done
    [ -f $L/pp_falco.done ] && break
    sleep 30
  done
  summary; touch $L/pp_falco.all; tmux kill-session -t falco
}
"$@"
