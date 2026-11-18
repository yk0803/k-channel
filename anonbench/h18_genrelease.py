"""H18: build a k-Channel release from GENERATED faces.

Same construction as h15 (identity-level code -> plurality merge to k -> count cap), but each
released image is drawn from the StyleGAN2 pool, binned into the same class, instead of from a real
image. Nothing of the input survives and the output matches no image in any gallery.

    python -m anonbench.h18_genrelease --dataset mtf --split test --ks 5 10 25
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

from .config import Paths, load_config
from .data import load_index
from .h11 import quantize
from .h15_surrogate import build_classes
from .manifest import ManifestWriter


def pool_codes(pool, attr_cols, pose_bins, ref):
    """Quantize the pool with the SAME bin edges the real data used (edges from `ref`)."""
    out = []
    for c in ("yaw", "roll"):
        qs = np.quantile(ref[c].to_numpy(float), np.linspace(0, 1, pose_bins + 1)[1:-1]) \
            if pose_bins > 1 else np.array([])
        out.append(np.digitize(pool[c].to_numpy(float), qs))
    pose = np.stack(out, 1)
    base = pool[attr_cols].astype(str).to_numpy()
    return ["|".join(list(r) + [str(p[0]), str(p[1])]) for r, p in zip(base, pose)]


def _pick(g, st, cols, rng, topm=20):
    """v3: within the class, match the source image's per-image state, then sample among the topm nearest in pose."""
    if "ybin" in st.index:   # v4: output depends only on the cell; uniform among matching pool faces
        sub = g
        for col in ["Smiling", "Mouth_Slightly_Open", "Eyeglasses", "Wearing_Hat", "ybin", "pbin"]:
            if col in sub.columns and st.get(col, -1) != -1:
                s2 = sub[sub[col] == st[col]]
                if len(s2):
                    sub = s2
        return sub.path.iloc[int(rng.integers(len(sub)))]
    sub = g
    for col in cols:
        if col in sub.columns:
            s2 = sub[sub[col] == st[col]]
            if len(s2) >= topm:
                sub = s2
    d = (sub["ia_yaw"] - st["yaw"]) ** 2 + (sub["ia_pitch"] - st["pitch"]) ** 2
    cand = sub.loc[d.nsmallest(topm).index]
    return cand.path.iloc[int(rng.integers(len(cand)))]


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
    ap.add_argument("--pool", default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--state-csv", default=None)
    ap.add_argument("--state-cols", nargs="*", default=["Eyeglasses", "Wearing_Hat", "Smiling", "Mouth_Slightly_Open"])
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    full = load_index(cfg, a.dataset).reset_index(drop=True)
    rows = full[full.split == a.split].reset_index(drop=True) if a.split != "all" else full
    pool = pd.read_csv(a.pool) if a.pool else pd.read_csv(P.work / "index" / "genpool.csv")
    attr_cols = [c for c in ("race", "gender", "age")
                 if c in full.columns and full[c].notna().any() and c in pool.columns]
    print(f"[h18] {a.dataset} {a.split}: {len(rows)} images; pool {len(pool)}; attrs {attr_cols}", flush=True)

    from .align import load_landmarks
    from .h11 import geom_features
    lm = load_landmarks(cfg, a.dataset)
    lm_key = "path" if next(iter(lm)) in set(full.path) else "image_id"
    ref = geom_features(lm, full[lm_key], full.image_id)
    pool["code"] = pool_codes(pool, attr_cols, a.pose_bins, ref)

    for k in a.ks:
        m, cap = build_classes(cfg, a.dataset, full, k, a.pose_bins)
        raw2cls = dict(zip(m.code, m.cls)) if "code" in m else {}
        m = m[m.image_id.isin(set(rows.image_id))].reset_index(drop=True)
        cls_of = dict(zip(m.image_id, m.cls))
        pool_cls = pool.code.map(raw2cls)
        bank = {c: g.path.tolist() for c, g in pool.assign(cls=pool_cls).dropna(subset=["cls"]).groupby("cls")}
        bank_df, _st = {}, None
        if a.state_csv:
            _st = pd.read_csv(a.state_csv).set_index("image_id")
            bank_df = {c: g for c, g in pool.assign(cls=pool_cls).dropna(subset=["cls"]).groupby("cls")}
        covered = sum(len(v) for v in bank.values())
        print(f"[h18] k={k}: {len(m)} released, {m.cls.nunique()} classes, cap {cap}; "
              f"pool usable {covered}/{len(pool)}, classes with faces {len(bank)}", flush=True)
        missing = [c for c in m.cls.unique() if c not in bank]
        if missing:
            print(f"[h18] WARNING {len(missing)} class(es) have no generated face: {missing[:5]}", flush=True)
        for mode in a.modes:
            rng = np.random.default_rng(a.seed)
            fixed = {}
            if mode == "pseudonymous":
                for ident, g in m.groupby("identity"):
                    c = g.cls.iloc[0]
                    if c in bank:
                        fixed[ident] = bank[c][rng.integers(len(bank[c]))]
            out = P.anon_dir("genrelease", f"k-{k}_pose-{a.pose_bins}_{mode}{a.tag_suffix}", a.split, a.seed)
            W = ManifestWriter(out, meta=dict(method="genrelease",
                                              params=dict(k=k, mode=mode, pose_bins=a.pose_bins, cap=cap,
                                                          pool="stylegan2_ffhq1024"), seed=a.seed))
            t0, skipped = time.time(), 0
            for n, r in enumerate(rows.itertuples(index=False), 1):
                if r.image_id not in cls_of:
                    continue
                c = cls_of[r.image_id]
                if c not in bank:
                    skipped += 1
                    continue
                src = fixed.get(r.identity) if mode == "pseudonymous" else None
                if src is None and bank_df and r.image_id in _st.index:
                    src = _pick(bank_df[c], _st.loc[r.image_id], a.state_cols, rng, a.topm)
                if src is None:
                    src = bank[c][rng.integers(len(bank[c]))]
                with Image.open(src) as im:
                    img = im.convert("RGB").resize((a.size, a.size), Image.LANCZOS)
                W.save_image(r._asdict(), img, "modified", 1, gen_face=src, cls=str(c), mode=mode)
                if n % 200 == 0 or n == len(rows):
                    el = time.time() - t0
                    print(f"  [h18] k={k} {mode} {n}/{len(rows)} ({el:.0f}s, "
                          f"ETA {el / n * (len(rows) - n):.0f}s)", flush=True)
            W.close()
            print(f"[h18] k={k} {mode} -> {out}  (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
