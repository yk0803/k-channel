"""H3: anon<->anon same-identity linkage (dataset-release threat).

The attacker holds only the released set. Can images of the same person be
grouped together without any original? Gallery = the released images of the run
itself; for each probe its own image is excluded, similarity per identity is the
max over that identity's other released images, rank-1 over identities.
Clean reference: the same computation on the source images of the same run.

    python -m anonbench.h3 --anon-dir <run dir> [...] [--features ...] [--modes ...]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .embed import set_tag
from .manifest import read_manifest
from .tsim import Feat, load_many, masked, rank_identity


def _save(out, rows):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        key = lambda d: d.run + "|" + d.feature + "|" + d["mode"]  # noqa: E731
        old = old[~key(old).isin(key(new))]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--features", nargs="*", default=["pixel", "phash", "dinov2", "clip"])
    ap.add_argument("--modes", nargs="*", default=["full", "context"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    lm = load_landmarks(cfg, a.dataset)
    Kg = np.mean(np.stack(list(lm.values())), 0)  # ONE fixed face box (per-image boxes leak the source)
    gpos = {k: i for i, k in enumerate(idx.image_id.to_numpy())}
    all_ids = idx.identity.to_numpy()
    out = P.results("H3") / "h3.csv"
    runs = [Path(d) for d in a.anon_dir if (Path(d) / "manifest.csv").exists()]
    print(f"[h3] {len(runs)} runs, device {dev}", flush=True)
    feats = {}
    for ri, d in enumerate(runs, 1):
        t0 = time.time()
        m = read_manifest(d)
        m = m[m.status != "failed"].reset_index(drop=True)
        tag = set_tag(cfg, d)
        p_ids = all_ids[np.array([gpos[i] for i in m.image_id])]
        keep = pd.Series(p_ids).groupby(p_ids).transform("size").to_numpy() >= 2
        if keep.sum() < len(m):
            print(f"[h3] {tag[:60]}: dropping {len(m) - int(keep.sum())} singleton-identity probes", flush=True)
        m, p_ids = m[keep].reset_index(drop=True), p_ids[keep]
        n, nid = len(m), len(set(p_ids))
        A_raw = load_many(m.anon_path.tolist(), f"anon ({tag[:50]})")
        C_raw = load_many(m.src_path.tolist(), f"src ({tag[:50]})")
        self_col = np.arange(n)
        rows, line = [], []
        for mode in a.modes:
            XA = A_raw if mode == "full" else masked(A_raw, [Kg] * n)
            XC = C_raw if mode == "full" else masked(C_raw, [Kg] * n)
            for f in a.features:
                feats.setdefault(f, Feat(f, dev))
                Ae, Ce = feats[f](XA), feats[f](XC)
                ra, _ = rank_identity(Ae @ Ae.T, self_col, p_ids, p_ids)
                rc, _ = rank_identity(Ce @ Ce.T, self_col, p_ids, p_ids)
                rows.append(dict(run=tag, feature=f, mode=mode, n=n, n_identities=nid,
                                 anon_anon_r1=float((ra == 1).mean()), anon_anon_r5=float((ra <= 5).mean()),
                                 clean_clean_r1=float((rc == 1).mean()), clean_clean_r5=float((rc <= 5).mean()),
                                 chance=1.0 / nid))
                line.append(f"{f}/{mode} anon {rows[-1]['anon_anon_r1']:.3f} clean {rows[-1]['clean_clean_r1']:.3f}")
                print(f"    [h3] {tag[:45]} {f}/{mode} done", flush=True)
        _save(out, rows)
        print(f"[{ri}/{len(runs)}] RESULT {tag[:65]} n={n} ids={nid} chance={1.0 / nid:.4f} | "
              + " | ".join(line) + f" ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
