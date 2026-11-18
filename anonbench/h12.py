"""H12: the privacy/utility dial for the proposed conditioning code.

For a target k, merge attribute classes upward until every equivalence class contains >= k
IDENTITIES (not images), then report what survives:
  privacy  - min/median class size, attacker rank-1 from the code, bits leaked
  utility  - how much of each attribute the code still distinguishes, as the share of images whose
             attribute value is still recoverable from the code (= the plurality value of its class),
             which upper-bounds what any downstream attribute classifier can get from a generated image.

Merge order is data-driven: repeatedly collapse the two smallest classes that share the most
attribute values, so the coarsening is reported rather than hand-picked.

    python -m anonbench.h12 --dataset mtf [--ks 2 5 10 25 50] [--pose-bins 3]
      -> work/results/<ds>/H12/kmerge.csv
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .h11 import geom_features, quantize


def class_stats(df, col, nid_total):
    g = df.groupby(col)
    ids = g.identity.nunique()
    sizes = g.size()
    best = g.identity.agg(lambda s: Counter(s).most_common(1)[0][1])
    h = 0.0
    for _, grp in g:
        p = grp.identity.value_counts(normalize=True).to_numpy()
        h += (len(grp) / len(df)) * float(-(p * np.log2(p)).sum())
    pid = df.identity.value_counts(normalize=True).to_numpy()
    h0 = float(-(pid * np.log2(pid)).sum())
    return dict(n_codes=int(sizes.size), min_ids=int(ids.min()), median_ids=float(ids.median()),
                min_imgs=int(sizes.min()), attacker_rank1=float(best.sum() / len(df)),
                bits_leaked=h0 - h, chance=1.0 / nid_total)


def _eff_ids(s):
    """Effective number of identities in a class: 2^H(identity|class). Equals the count when the
    class is balanced, and drops toward 1 when one identity dominates."""
    p = s.value_counts(normalize=True).to_numpy()
    return float(2.0 ** (-(p * np.log2(p)).sum()))


def merge_to_k(df, k, criterion="count", weighted=False):
    """Collapse classes below the threshold into the nearest class (most shared attribute values).
    criterion 'count': >= k distinct identities.  'entropy': >= k EFFECTIVE identities."""
    df = df.copy()
    df["cls"] = df.code
    def score(g):
        if criterion == "count":
            return g.nunique()
        if criterion == "entropy":
            return _eff_ids(g)
        return 1.0 / g.value_counts(normalize=True).max()   # plurality: 1/p_max >= k  <=>  p_max <= 1/k
    while True:
        ids = df.groupby("cls").identity.agg(score)
        small = ids[ids < k]
        if small.empty or len(ids) == 1:
            break
        victim = small.idxmin()
        others = [c for c in ids.index if c != victim]
        vparts = str(victim).split("|")
        if weighted:   # merge across pose first, then age, then race, gender last
            an = [c for c in ("race", "gender", "age") if c in df.columns]
            wts = [{"gender": 1000, "race": 100, "age": 10}[c] for c in an] + [1] * (len(vparts) - len(an))
            sim = [sum(w * (a == b) for w, a, b in zip(wts, vparts, str(o).split("|"))) for o in others]
        else:
            sim = [sum(a == b for a, b in zip(vparts, str(o).split("|"))) for o in others]
        order = np.lexsort((ids[others].to_numpy(), -np.asarray(sim)))
        target = others[order[0]]
        df.loc[df.cls == victim, "cls"] = target
    return df


def utility(df, attr_cols):
    """Share of images whose attribute value equals the plurality value of its class."""
    out = {}
    for a in attr_cols:
        keep = df[a].notna()
        d = df[keep]
        plur = d.groupby("cls")[a].transform(lambda s: s.value_counts().idxmax())
        out[f"util_{a}"] = float((d[a] == plur).mean())
        out[f"base_{a}"] = float(d[a].value_counts(normalize=True).max())  # majority-class baseline
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--split", default="all")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 5, 10, 25, 50])
    ap.add_argument("--pose-bins", type=int, default=3)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset)
    if a.split != "all":
        idx = idx[idx.split == a.split]
    idx = idx.reset_index(drop=True)
    nid = idx.identity.nunique()
    lm = load_landmarks(cfg, a.dataset)
    lm_key = "path" if next(iter(lm)) in set(idx.path) else "image_id"
    G = geom_features(lm, idx[lm_key], idx.image_id)
    pose = np.stack([quantize(G[c].to_numpy(), a.pose_bins) for c in ("yaw", "roll")], 1)
    attr_cols = [c for c in ("race", "gender", "age") if c in idx.columns and idx[c].notna().any()]
    base = idx[attr_cols].astype(str)
    df = pd.DataFrame(dict(identity=idx.identity.to_numpy()))
    for c in attr_cols:
        df[c] = base[c].to_numpy()
    df["code"] = ["|".join(list(r) + [str(p[0]), str(p[1])]) for r, p in zip(base.to_numpy(), pose)]
    print(f"[h12] {len(df)} images, {nid} identities, {df.code.nunique()} raw codes "
          f"(attrs {attr_cols} + pose-{a.pose_bins})", flush=True)
    rows = []
    for crit in ("count", "entropy", "plurality"):
        for k in a.ks:
            m = merge_to_k(df, k, crit)
            r = dict(k=k, criterion=crit)
            r.update(class_stats(m, "cls", nid))
            r["min_eff_ids"] = float(m.groupby("cls").identity.agg(_eff_ids).min())
            r["max_identity_share"] = float(m.groupby("cls").identity.agg(lambda g: g.value_counts(normalize=True).max()).max())
            r["bound_1_over_k"] = 1.0 / k
            r.update(utility(m, attr_cols))
            rows.append(r)
            print(f"  [h12] {crit:7s} k={k:3d}  classes {r['n_codes']:4d}  min_ids {r['min_ids']:3d}  "
                  f"min_eff {r['min_eff_ids']:5.1f}  attacker {r['attacker_rank1']:.3f} "
                  f"(bound {1.0 / k:.3f})  bits {r['bits_leaked']:.2f}  "
                  + "  ".join(f"{c} {r['util_' + c]:.2f}" for c in attr_cols), flush=True)
    out = P.results("H12") / "kmerge.csv"
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(d.to_string(index=False), flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
