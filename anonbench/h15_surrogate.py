"""H15: simulate the proposed method with real images (no generator needed).

The proposal: release an image generated from an identity-level, plurality-merged, count-capped
class code and nothing else. A perfect class-conditional generator would emit a sample from the
class distribution. We upper-bound that by emitting a REAL image drawn from the same class and a
DIFFERENT identity, chosen uniformly at random (never nearest-neighbour: that is the BlendPose
donor-only failure). Any attack that cannot beat chance on this release cannot beat it on a
generator's output either, because the generator strictly destroys information relative to this.

Modes:
  unlinkable    surrogate drawn independently per image
  pseudonymous  one surrogate identity fixed per real identity (release groupable by design)

Writes ordinary run dirs so score / tsim / h3 / h8 work unchanged.

    python -m anonbench.h15_surrogate --dataset mtf --split test --ks 5 10 25
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .h11 import geom_features, quantize
from .h12 import merge_to_k
from .manifest import ManifestWriter


def build_classes(cfg, dataset, idx, k, pose_bins, cap_quantile=0.10):
    """Identity-level code -> plurality merge to k -> cap images per identity."""
    lm = load_landmarks(cfg, dataset)
    lm_key = "path" if next(iter(lm)) in set(idx.path) else "image_id"
    G = geom_features(lm, idx[lm_key], idx.image_id)
    pose = np.stack([quantize(G[c].to_numpy(), pose_bins) for c in ("yaw", "roll")], 1)
    attr_cols = [c for c in ("race", "gender", "age") if c in idx.columns and idx[c].notna().any()]
    base = idx[attr_cols].astype(str)
    df = pd.DataFrame(dict(identity=idx.identity.to_numpy(), image_id=idx.image_id.to_numpy()))
    for c in attr_cols:
        df[c] = base[c].to_numpy()
    df["code"] = ["|".join(list(r) + [str(p[0]), str(p[1])]) for r, p in zip(base.to_numpy(), pose)]
    idcode = df.groupby("identity").code.agg(lambda s: s.value_counts().idxmax())
    df["code"] = df.identity.map(idcode)                       # identity-level: kills per-image channel
    m = merge_to_k(df, k, "plurality", weighted=True)                         # kills the which-class channel
    cap = max(int(m.groupby("identity").size().quantile(cap_quantile)), 1)
    m = m.groupby("identity", group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=0))
    return m.reset_index(drop=True), cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ks", nargs="*", type=int, default=[5, 10, 25])
    ap.add_argument("--pose-bins", type=int, default=3)
    ap.add_argument("--modes", nargs="*", default=["unlinkable", "pseudonymous"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    full = load_index(cfg, a.dataset).reset_index(drop=True)
    rows = full[full.split == a.split].reset_index(drop=True) if a.split != "all" else full
    path_of = dict(zip(full.image_id, full.path))
    print(f"[h15] {a.dataset} {a.split}: {len(rows)} images, {rows.identity.nunique()} identities", flush=True)

    for k in a.ks:
        m, cap = build_classes(cfg, a.dataset, full, k, a.pose_bins)
        m = m[m.image_id.isin(set(rows.image_id))].reset_index(drop=True)
        cls_of = dict(zip(m.image_id, m.cls))
        pool = m.groupby("cls").apply(lambda g: g[["image_id", "identity"]].to_dict("records")).to_dict()
        sizes = {c: len(v) for c, v in pool.items()}
        print(f"[h15] k={k}: {len(m)} released images, {m.cls.nunique()} classes, cap {cap}/identity, "
              f"class sizes min {min(sizes.values())} median {int(np.median(list(sizes.values())))}", flush=True)
        for mode in a.modes:
            rng = np.random.default_rng(a.seed)
            fixed = {}
            if mode == "pseudonymous":
                for ident, g in m.groupby("identity"):
                    c = g.cls.iloc[0]
                    others = [r for r in pool[c] if r["identity"] != ident]
                    if others:
                        fixed[ident] = others[rng.integers(len(others))]["identity"]
            out = P.anon_dir("surrogate", f"k-{k}_pose-{a.pose_bins}_{mode}", a.split, a.seed)
            W = ManifestWriter(out, meta=dict(method="surrogate",
                                              params=dict(k=k, mode=mode, pose_bins=a.pose_bins, cap=cap),
                                              seed=a.seed))
            t0, skipped = time.time(), 0
            byid = {}
            for c, recs in pool.items():
                for r in recs:
                    byid.setdefault((c, r["identity"]), []).append(r["image_id"])
            for n, r in enumerate(rows.itertuples(index=False), 1):
                if r.image_id not in cls_of:
                    continue
                c = cls_of[r.image_id]
                if mode == "pseudonymous" and r.identity in fixed:
                    choices = byid.get((c, fixed[r.identity]), [])
                else:
                    choices = [x["image_id"] for x in pool[c] if x["identity"] != r.identity]
                if not choices:
                    skipped += 1
                    continue
                src = choices[rng.integers(len(choices))]
                with Image.open(path_of[src]) as im:
                    img = im.convert("RGB").resize((a.size, a.size), Image.BILINEAR)
                W.save_image(r._asdict(), img, "modified", 1, surrogate_of=src, cls=str(c), mode=mode)
                if n % 200 == 0 or n == len(rows):
                    el = time.time() - t0
                    print(f"  [h15] k={k} {mode} {n}/{len(rows)} ({el:.0f}s, ETA {el / n * (len(rows) - n):.0f}s)",
                          flush=True)
            W.close()
            print(f"[h15] k={k} {mode} -> {out}  (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
