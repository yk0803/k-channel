"""H11: how much identity does a candidate conditioning code carry?

The remedy conditions a generator on a short code C(x) and nothing else. Privacy then rests on one
claim: knowing C(x) must not narrow down which image x was. This measures that directly, with no
generator involved.

For each candidate code we report
  * distinct codes, and the equivalence-class size distribution (min / median / share of images in
    a class smaller than k) for k = 2, 5, 10, 25        -> can the code be released at all?
  * identity purity: share of codes whose images are all one person; and, for an attacker who sees
    the code, rank-1 identification of the source identity by picking the most frequent identity in
    the class (chance 1/#identities)                     -> does the code leak who it is?
  * mutual-information-style summary: H(identity | code) in bits vs H(identity).

Codes tested (all quantized, discrete):
  attrs        race x gender x age from the index
  landmarks-N  face geometry (inter-ocular, eye-mouth, yaw/roll proxies) quantized to N bins each
  pose-N       yaw/roll/scale only
  colour-N     coarse skin/hair/background colour quantized to N bins
  combos of the above

    python -m anonbench.h11 --dataset mtf [--split test] [--bins 2 3 4]
      -> work/results/<ds>/H11/codes.csv
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index


def quantize(v, bins):
    """Equal-frequency quantization -> integer codes."""
    v = np.asarray(v, float)
    qs = np.quantile(v[~np.isnan(v)], np.linspace(0, 1, bins + 1)[1:-1]) if bins > 1 else np.array([])
    return np.digitize(v, qs)


def geom_features(lm, keys, image_ids):
    """Crude but deterministic geometry from the 5 landmarks: eye distance, eye-mouth distance,
    roll (eye-line angle), yaw proxy (nose offset between the eyes), mouth width."""
    rows = []
    missing = 0
    for i in keys:
        if i not in lm:
            missing += 1
            rows.append((np.nan,) * 5)
            continue
        k = np.asarray(lm[i], float)            # 5x2 normalised
        le, re, no, ml, mr = k[0], k[1], k[2], k[3], k[4]
        eye = re - le
        iod = np.linalg.norm(eye)
        roll = np.degrees(np.arctan2(eye[1], eye[0]))
        mid = (le + re) / 2
        yaw = float((no[0] - mid[0]) / max(iod, 1e-6))
        emd = float(np.linalg.norm(no - (ml + mr) / 2) / max(iod, 1e-6))
        mw = float(np.linalg.norm(mr - ml) / max(iod, 1e-6))
        rows.append((iod, roll, yaw, emd, mw))
    if missing:
        print(f"      [h11] WARNING {missing}/{len(rows)} images have no landmarks", flush=True)
    return pd.DataFrame(rows, columns=["iod", "roll", "yaw", "eye_mouth", "mouth_w"], index=list(image_ids))


def colour_features(paths, image_ids, size=16):
    rows = []
    t0 = time.time()
    for n, p in enumerate(paths, 1):
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), float)
        cx, cy = size // 2, size // 2
        face = a[cy - 3:cy + 3, cx - 3:cx + 3].reshape(-1, 3).mean(0)   # centre patch ~ skin
        top = a[:3].reshape(-1, 3).mean(0)                              # top ~ hair/background
        edge = np.concatenate([a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)]).mean(0)
        rows.append(tuple(face) + tuple(top) + tuple(edge))
        if n % 1000 == 0 or n == len(paths):
            print(f"      [h11] colour {n}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=[f"c{i}" for i in range(9)], index=list(image_ids))


def analyse(codes, ids, name, ks=(2, 5, 10, 25)):
    codes = ["|".join(map(str, c)) if isinstance(c, (tuple, list, np.ndarray)) else str(c) for c in codes]
    ids = np.asarray(ids)
    df = pd.DataFrame({"code": pd.Series(codes, dtype=object), "identity": pd.Series(ids)})
    g = df.groupby("code")
    sizes = g.size()
    nid = len(np.unique(ids))
    pure = g.identity.nunique()
    # attacker: given the code, guess the class's most frequent identity
    best = g.identity.agg(lambda s: Counter(s).most_common(1)[0][1])
    attacker_r1 = float(best.sum() / len(df))
    # H(identity | code) in bits
    hcond = 0.0
    for c, grp in g:
        p = grp.identity.value_counts(normalize=True).to_numpy()
        hcond += (len(grp) / len(df)) * float(-(p * np.log2(p)).sum())
    pid = df.identity.value_counts(normalize=True).to_numpy()
    h0 = float(-(pid * np.log2(pid)).sum())
    row = dict(code=name, n_images=len(df), n_codes=int(sizes.size), n_identities=nid,
               min_class=int(sizes.min()), median_class=float(sizes.median()),
               pure_code_share=float((pure == 1).sum() / sizes.size),
               attacker_rank1=attacker_r1, chance=1.0 / nid,
               H_identity=h0, H_identity_given_code=hcond, bits_leaked=h0 - hcond)
    for k in ks:
        small = sizes[sizes < k]
        row[f"share_in_class_lt_{k}"] = float(df.code.isin(small.index).sum() / len(df))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bins", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset)
    if a.split != "all":
        idx = idx[idx.split == a.split]
    idx = idx.reset_index(drop=True)
    ids = idx.identity.to_numpy()
    print(f"[h11] {len(idx)} images, {len(np.unique(ids))} identities, split={a.split}", flush=True)

    lm = load_landmarks(cfg, a.dataset)
    lm_key = "path" if next(iter(lm)) in set(idx.path) else "image_id"
    print(f"      [h11] landmarks keyed by {lm_key}", flush=True)
    G = geom_features(lm, idx[lm_key], idx.image_id)
    print("      [h11] geometry done", flush=True)
    C = colour_features(idx.path.tolist(), idx.image_id)

    def pack(cols_df, cols, bins):
        q = np.stack([quantize(cols_df[c].to_numpy(), bins) for c in cols], 1)
        return [tuple(r) for r in q]

    rows = []
    # 1. index attributes
    attr_cols = [c for c in ("race", "gender", "age") if c in idx.columns and idx[c].notna().any()]
    rows.append(analyse([tuple(r) for r in idx[attr_cols].astype(str).to_numpy()], ids, "attrs(" + "+".join(attr_cols) + ")"))
    for b in a.bins:
        rows.append(analyse(pack(G, ["yaw", "roll"], b), ids, f"pose-{b}"))
        rows.append(analyse(pack(G, ["iod", "roll", "yaw", "eye_mouth", "mouth_w"], b), ids, f"landmarks-{b}"))
        rows.append(analyse(pack(C, [f"c{i}" for i in range(9)], b), ids, f"colour-{b}"))
        att = [tuple(r) for r in idx[attr_cols].astype(str).to_numpy()]
        pos = pack(G, ["yaw", "roll"], b)
        rows.append(analyse([x + y for x, y in zip(att, pos)], ids, f"attrs+pose-{b}"))
        col = pack(C, [f"c{i}" for i in range(9)], b)
        rows.append(analyse([x + y + z for x, y, z in zip(att, pos, col)], ids, f"attrs+pose+colour-{b}"))
        print(f"      [h11] bins={b} done", flush=True)

    out = P.results("H11") / "codes.csv"
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(d[["code", "n_codes", "min_class", "median_class", "pure_code_share", "attacker_rank1",
             "chance", "bits_leaked", "share_in_class_lt_5", "share_in_class_lt_25"]].to_string(index=False), flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
