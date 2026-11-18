"""E0 - unmodified-release census (AUDIT F1).

For each run: split the manifest by the method's own status (unmodified = never perturbed by the method,
modified = perturbed at least once). For each subset report its size, the MEASURED distance to the source
(median MAE / PSNR, share with PSNR >= 40 dB), and re-identification by every frozen evaluator
(gallery = all clean images except the probe's own source).

    python -m anonbench.e00_unmodified --anon-dir <run dir> [...]
      -> work/results/<dataset>/E0/e0.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, load_config
from .data import load_index
from .embed import embed_set, set_tag
from .manifest import read_manifest
from .metrics.gallery import identify
from .models.embedders import get_embedder


def _save(out: Path, rows: list[dict]):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        key = lambda d: d.run + "|" + d.embedder + "|" + d.subset  # noqa: E731
        old = old[~key(old).isin(key(new))]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--embedders", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    I = load_index(cfg, a.dataset).set_index("image_id")
    names = a.embedders or cfg["evaluators"]["embedders"]
    runs = [Path(d) for d in a.anon_dir]
    out = P.results("E0") / "e0.csv"
    for d in runs:
        m = read_manifest(d)
        m = m[m.status != "failed"]
        for st in ("unmodified", "modified"):
            s = m[m.status == st]
            if len(s):
                print(f"[census] {set_tag(cfg, d)[:70]} {st:10s} n={len(s):5d} ({len(s) / len(m) * 100:5.1f}%)  "
                      f"median MAE {s.src_mae.median():6.2f}  median PSNR {s.src_psnr.replace(np.inf, 99).median():5.1f} dB  "
                      f"PSNR>=40dB {(s.src_psnr >= 40).mean() * 100:5.1f}%", flush=True)
    jobs = [(n, d) for n in names for d in runs]
    for i, (n, d) in enumerate(jobs, 1):
        t0 = time.time()
        if i == 1 or jobs[i - 2][0] != n:
            emb = get_embedder(n, cfg)
        try:
            clean = embed_set(cfg, a.dataset, n, None, embedder=emb)
            anon = embed_set(cfg, a.dataset, n, d, embedder=emb)
            m = read_manifest(d).set_index("image_id")
            st = m.loc[anon["image_id"], "status"].to_numpy()
            a_ids = I.loc[anon["image_id"], "identity"].to_numpy()
            c_ids = I.loc[clean["image_id"], "identity"].to_numpy()
            rows, line = [], []
            for sub in ("unmodified", "modified", "all"):
                sel = np.ones(len(st), bool) if sub == "all" else (st == sub)
                if not sel.any():
                    continue
                g = identify(anon["emb"][sel], a_ids[sel], np.asarray(anon["image_id"])[sel],
                             clean["emb"], c_ids, np.asarray(clean["image_id"]))
                rows.append(dict(run=set_tag(cfg, d), embedder=n, subset=sub, n=int(sel.sum()),
                                 share=float(sel.mean()), rank1=g["rank1"], rank5=g["rank5"]))
                line.append(f"{sub} r1 {g['rank1']:.3f} (n={int(sel.sum())})")
            _save(out, rows)
            print(f"[{i}/{len(jobs)}] RESULT {n:26s} " + " | ".join(line) + f"  ({time.time() - t0:.0f}s, saved)", flush=True)
        except Exception as e:  # keep going
            print(f"[{i}/{len(jobs)}] ERROR {n} {d}: {e!r}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
