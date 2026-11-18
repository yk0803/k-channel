"""E8 - k-anonymity audit of BlendPose blends (AUDIT F5).

Each output lists its blend members in the manifest (`extra.members`, `extra.donors`, `extra.fallback`).
For every output and every frozen evaluator: cosine similarity of the output to the CLEAN embedding of the
target and of each donor. If the blend were k-anonymous, the target would be the most similar member
with probability 1/k. We report the target-first rate, a one-sided exact binomial test against 1/k
(rows with a full donor set only), the mean rank of the target, and the fallback census.

Control `include_target: false` (donor-only blend): the target is NOT in the blend, so its target-first
rate among {target + donors} measures what pose/demographic similarity alone gives.

    python -m anonbench.e08_kanon --anon-dir <blend run dir> [...]
      -> work/results/mtf/E8/kanon.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import Paths, load_config
from .embed import embed_set, set_tag
from .manifest import read_manifest
from .models.embedders import get_embedder


def audit(anon: dict, clean: dict, man: pd.DataFrame) -> dict:
    cpos = {k: i for i, k in enumerate(clean["image_id"])}
    apos = {k: i for i, k in enumerate(anon["image_id"])}
    first, rank, kk, fb = [], [], [], []
    for r in man.itertuples(index=False):
        if r.image_id not in apos:
            continue
        donors = list(r.extra.get("donors", []))
        cand = [r.image_id] + [d for d in donors if d != r.image_id]
        if len(cand) < 2:
            continue
        e = anon["emb"][apos[r.image_id]]
        s = np.array([float(e @ clean["emb"][cpos[c]]) for c in cand])
        rk = int((s > s[0]).sum()) + 1                      # rank of the target; ties count against the defender
        first.append(rk == 1)
        rank.append(rk)
        kk.append(len(cand))
        fb.append(r.extra.get("fallback", "none"))
    first, rank, kk, fb = map(np.asarray, (first, rank, kk, fb))
    full = fb == "none"
    out = dict(n=int(len(first)), n_full=int(full.sum()),
               n_fewer_donors=int((fb == "fewer_donors").sum()), n_same_person=int((fb == "same_person").sum()),
               target_first_rate=float(first.mean()) if len(first) else float("nan"),
               expected_rate=float(np.mean(1.0 / kk)) if len(kk) else float("nan"),
               mean_target_rank=float(rank.mean()) if len(rank) else float("nan"))
    if full.any():
        k = int(np.median(kk[full]))
        sel = full & (kk == k)
        bt = binomtest(int(first[sel].sum()), int(sel.sum()), 1.0 / k, alternative="greater")
        out.update(k_full=k, target_first_rate_full=float(first[sel].mean()), chance_full=1.0 / k,
                   binom_p_full=float(bt.pvalue), n_full_k=int(sel.sum()))
    return out


def _save(out: Path, row: dict):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        if "alignment" not in old:
            old["alignment"] = "transfer"
        old = old[~((old.run == row["run"]) & (old.embedder == row["embedder"]) & (old.alignment == row["alignment"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--embedders", nargs="*", default=None, help="default: the frozen panel")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    names = a.embedders or cfg["evaluators"]["embedders"]
    runs = [Path(d) for d in a.anon_dir]
    out = P.results("E8") / "kanon.csv"
    jobs = [(n, d) for n in names for d in runs]
    for i, (n, d) in enumerate(jobs, 1):
        t0 = time.time()
        if i == 1 or jobs[i - 2][0] != n:
            emb = get_embedder(n, cfg)
        print(f"[{i}/{len(jobs)}] E8 {n} on {set_tag(cfg, d)} ...", flush=True)
        try:
            man = read_manifest(d)
            meta = json.loads((d / "run_meta.json").read_text())
            clean = embed_set(cfg, a.dataset, n, None, embedder=emb)
            anon = embed_set(cfg, a.dataset, n, d, embedder=emb)
            r = audit(anon, clean, man)
        except Exception as e:  # keep going
            print(f"[{i}/{len(jobs)}] ERROR {n} {d}: {e!r}", flush=True)
            continue
        row = dict(run=set_tag(cfg, d), embedder=n, alignment=cfg.get("alignment", "transfer"), k=meta.get("k"), include_target=meta.get("include_target"),
                   diffusion=meta.get("diffusion"), img_weight=meta.get("img_weight"), **r)
        _save(out, row)
        print(f"[{i}/{len(jobs)}] RESULT {n:26s} target-first {r['target_first_rate']:.3f} vs chance {r['expected_rate']:.3f} "
              f"| full-k: {r.get('target_first_rate_full', float('nan')):.3f} vs {r.get('chance_full', float('nan')):.3f} "
              f"p={r.get('binom_p_full', float('nan')):.2e} n={r.get('n_full_k', 0)} | fallbacks fewer={r['n_fewer_donors']} "
              f"same_person={r['n_same_person']} ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
