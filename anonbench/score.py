"""Score anonymizer runs against the FROZEN evaluator panel (open-set embedders).

For each run directory (manifest) x each embedder:
  * gallery identification: probe = anonymized image, gallery = ALL clean images except the probe's own source
    -> rank-1 / rank-5, with an identity-clustered 95% bootstrap CI on rank-1
  * the same metric for the CLEAN versions of the same probes (the no-anonymization reference)
  * verification: thresholds fixed on clean genuine/impostor pairs at FAR 1e-3 / 1e-4, applied to
    anonymized-vs-clean pairs -> TAR (re-identification) and FAR
  * manifest census: claimed status counts + MEASURED identity to the source (src_linf == 0)
Rows are upserted into work/results/<dataset>/scores/scores.csv after every (run, embedder).

    python -m anonbench.score --anon-dir <run dir> [<run dir> ...] [--embedders ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, load_config
from .data import load_index
from .embed import embed_set, set_tag
from .manifest import read_manifest
from .metrics.bootstrap import cluster_bootstrap
from .metrics.gallery import identify
from .metrics.verification import far_at, pair_scores, tar_at, threshold_at_far
from .models.embedders import get_embedder

FARS = (1e-3, 1e-4)


def census(anon_dir: Path) -> dict:
    m = read_manifest(anon_dir)
    st = m.status.value_counts().to_dict()
    ok = m[m.status != "failed"]
    return dict(n_rows=len(m), n_modified=st.get("modified", 0), n_unmodified_claimed=st.get("unmodified", 0),
                n_failed=st.get("failed", 0),
                n_identical_measured=int((ok.src_linf == 0).sum()) if "src_linf" in ok else -1,
                median_src_mae=float(ok.src_mae.median()) if "src_mae" in ok and len(ok) else float("nan"),
                median_src_psnr=float(ok.src_psnr.replace(np.inf, 99.0).median()) if "src_psnr" in ok and len(ok) else float("nan"))


def clean_thresholds(cache: dict, key: str, E: np.ndarray, ids, src) -> dict:
    if key not in cache:
        gen, imp, _ = pair_scores(E, ids, src, E, ids, src, max_impostors=3_000_000)
        cache[key] = {f: threshold_at_far(imp, f) for f in FARS}
    return cache[key]


def score_run(cfg, dataset, anon_dir: Path, emb_name: str, embedder, idx: pd.DataFrame, thr_cache: dict) -> dict:
    clean = embed_set(cfg, dataset, emb_name, None, embedder=embedder)
    anon = embed_set(cfg, dataset, emb_name, anon_dir, embedder=embedder)
    I = idx.set_index("image_id")
    c_ids = I.loc[clean["image_id"], "identity"].to_numpy()
    c_src = np.asarray(clean["image_id"])
    a_ids = I.loc[anon["image_id"], "identity"].to_numpy()
    a_src = np.asarray(anon["image_id"])
    thr = clean_thresholds(thr_cache, emb_name, clean["emb"], c_ids, c_src)

    g = identify(anon["emb"], a_ids, a_src, clean["emb"], c_ids, c_src, exclude_same_source=True)
    pos = {k: i for i, k in enumerate(c_src)}
    probe_clean = clean["emb"][[pos[s] for s in a_src]]
    g0 = identify(probe_clean, a_ids, a_src, clean["emb"], c_ids, c_src, exclude_same_source=True)
    v = g["_valid"]
    ci = cluster_bootstrap(g["_per_probe_correct1"][v].astype(float), a_ids[v], n_boot=1000)

    gen, imp, _ = pair_scores(anon["emb"], a_ids, a_src, clean["emb"], c_ids, c_src, max_impostors=2_000_000)
    gen0, _, _ = pair_scores(probe_clean, a_ids, a_src, clean["emb"], c_ids, c_src, max_impostors=1)
    r = dict(rank1=g["rank1"], rank1_ci_low=ci["ci_low"], rank1_ci_high=ci["ci_high"], rank5=g["rank5"],
             rank1_clean_same_probes=g0["rank1"], chance_rank1=g["chance_rank1"], n_probes=int(v.sum()),
             n_identities=int(len(np.unique(a_ids))), alignment=cfg.get("alignment", "transfer"))
    for f in FARS:
        r[f"tar@{f:g}"] = tar_at(gen, thr[f])
        r[f"tar@{f:g}_clean_same_probes"] = tar_at(gen0, thr[f])
        r[f"far@{f:g}"] = far_at(imp, thr[f])
    return r


def _save(out: Path, row: dict):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
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
    idx = load_index(cfg, a.dataset)
    names = a.embedders or cfg["evaluators"]["embedders"]
    runs = [Path(d) for d in a.anon_dir]
    out = P.results("scores") / "scores.csv"
    for d in runs:
        c = census(d)
        print(f"[census] {set_tag(cfg, d)}: rows {c['n_rows']} modified {c['n_modified']} "
              f"unmodified(claimed) {c['n_unmodified_claimed']} identical(measured) {c['n_identical_measured']} "
              f"failed {c['n_failed']} median MAE {c['median_src_mae']:.2f} PSNR {c['median_src_psnr']:.1f}", flush=True)
    thr_cache: dict = {}
    jobs = [(n, d) for n in names for d in runs]
    for i, (n, d) in enumerate(jobs, 1):
        t0 = time.time()
        if i == 1 or jobs[i - 2][0] != n:
            embedder = get_embedder(n, cfg)            # load each model once
        print(f"[{i}/{len(jobs)}] {n} on {set_tag(cfg, d)} ...", flush=True)
        try:
            r = score_run(cfg, a.dataset, d, n, embedder, idx, thr_cache)
        except Exception as e:  # keep going
            print(f"[{i}/{len(jobs)}] ERROR {n} {d}: {e!r}", flush=True)
            continue
        meta_p = d / "run_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        row = dict(run=set_tag(cfg, d), method=meta.get("method", "?"), embedder=n, **census(d), **r)
        _save(out, row)
        print(f"[{i}/{len(jobs)}] RESULT {n:26s} rank1 {r['rank1']:.3f} [{r['rank1_ci_low']:.3f},{r['rank1_ci_high']:.3f}] "
              f"(clean {r['rank1_clean_same_probes']:.3f}, chance {r['chance_rank1']:.4f})  "
              f"TAR@1e-3 {r['tar@0.001']:.3f} (clean {r['tar@0.001_clean_same_probes']:.3f})  ({time.time() - t0:.0f}s, saved)",
              flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
