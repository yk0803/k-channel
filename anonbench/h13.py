"""H13: composition. Does per-image k-anonymity survive many images per person?

An identity with n images contributes n coded records, so even when every class is k-anonymous
per image, the identity's SIGNATURE (which classes they appear in, and how often) may be unique.
This is the documented composition failure of k-anonymity under multiple records per respondent,
applied to image release. Also measures within-class attribute homogeneity (l-diversity analogue).

Merge rules: count / entropy / plurality (per image, from h12) and `multiset` (new: merge until
every identity's class-support signature is shared by at least k identities).

    python -m anonbench.h13 --dataset mtf [--ks ...] [--pose-bins N] [--rules ...]
      -> work/results/<ds>/H13/composition.csv
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter

import numpy as np
import pandas as pd

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .h11 import geom_features, quantize
from .h12 import merge_to_k


def signature(codes, ordered):
    if ordered:
        return "|".join(f"{k}x{v}" for k, v in sorted(Counter(codes).items()))
    return "|".join(sorted(set(map(str, codes))))


def sig_table(m, ordered):
    s = m.groupby("identity").cls.agg(lambda x: signature(x.tolist(), ordered))
    counts = s.value_counts()
    return s, counts


def merge_to_k_multiset(df, k, max_steps=400):
    """Merge classes until every identity's class-support signature is shared by >= k identities."""
    df = df.copy()
    df["cls"] = df.code
    for step in range(max_steps):
        s, counts = sig_table(df, ordered=False)
        bad = counts[counts < k]
        if bad.empty:
            break
        classes = sorted(df.cls.unique())
        if len(classes) <= 1:
            break
        exposed = set(s[s.isin(bad.index)].index)
        # candidate merges: pairs of classes that exposed identities actually appear in
        cand = sorted({c for c in df[df.identity.isin(exposed)].cls.unique()})
        best, best_score = None, None
        for a, b in itertools.combinations(cand if len(cand) > 1 else classes, 2):
            t = df.cls.where(df.cls != b, a)
            tmp = df.assign(cls=t)
            _, c2 = sig_table(tmp, ordered=False)
            score = int((c2 < k).sum())
            if best_score is None or score < best_score:
                best, best_score = (a, b), score
            if score == 0:
                break
        if best is None:
            break
        df.loc[df.cls == best[1], "cls"] = best[0]
        if step % 10 == 0:
            print(f"      [h13] multiset merge step {step}: {df.cls.nunique()} classes, "
                  f"{best_score} identities still exposed", flush=True)
    return df


def analyse(m, nid, attr_cols):
    r = {}
    for name, ordered in (("multiset", True), ("support", False)):
        s, counts = sig_table(m, ordered)
        shared = counts.reindex(s.values).to_numpy().astype(float)
        r[f"{name}_unique"] = float((shared == 1).sum() / len(s))
        r[f"{name}_attacker"] = float(np.mean(1.0 / shared))
        r[f"{name}_n_signatures"] = int(counts.size)
    # count channel: identity signature = (class support, number of images), isolating how much
    # the per-identity IMAGE COUNT leaks over and above which classes the identity appears in.
    sup = m.groupby("identity").cls.agg(lambda x: signature(x.tolist(), False))
    cnt = m.groupby("identity").size()
    for nm, sg in (("countonly", cnt.astype(str)), ("supcount", sup + "#" + cnt.astype(str))):
        c = sg.value_counts().reindex(sg.values).to_numpy().astype(float)
        r[f"{nm}_unique"] = float((c == 1).sum() / len(sg))
        r[f"{nm}_attacker"] = float(np.mean(1.0 / c))
    sizes = m.groupby("cls").size()
    ids_per = m.groupby("cls").identity.nunique()
    g = m.groupby("cls")
    best = g.identity.agg(lambda s: Counter(s).most_common(1)[0][1])
    r["n_classes"] = int(sizes.size)
    r["min_imgs"] = int(sizes.min())
    r["min_ids"] = int(ids_per.min())
    r["per_image_attacker"] = float(best.sum() / len(m))
    r["source_linkage_bound"] = float(1.0 / sizes.min())
    r["identity_chance"] = 1.0 / nid
    for a in attr_cols:
        if a in m.columns:
            pure = g[a].agg(lambda s: s.nunique() == 1)
            r[f"homog_{a}"] = float(m.cls.isin(pure[pure].index).sum() / len(m))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 5, 10, 25])
    ap.add_argument("--pose-bins", nargs="*", type=int, default=[3])
    ap.add_argument("--rules", nargs="*", default=["count", "entropy", "plurality", "multiset"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    nid = idx.identity.nunique()
    lm = load_landmarks(cfg, a.dataset)
    lm_key = "path" if next(iter(lm)) in set(idx.path) else "image_id"
    G = geom_features(lm, idx[lm_key], idx.image_id)
    attr_cols = [c for c in ("race", "gender", "age") if c in idx.columns and idx[c].notna().any()]
    ipi = idx.groupby("identity").size()
    print(f"[h13] {len(idx)} images, {nid} identities, images/identity "
          f"min {ipi.min()} median {ipi.median():.0f} max {ipi.max()}", flush=True)
    rows = []
    for pb in a.pose_bins:
        pose = np.stack([quantize(G[c].to_numpy(), pb) for c in ("yaw", "roll")], 1)
        base = idx[attr_cols].astype(str)
        df = pd.DataFrame(dict(identity=idx.identity.to_numpy()))
        for c in attr_cols:
            df[c] = base[c].to_numpy()
        df["code"] = ["|".join(list(r) + [str(p[0]), str(p[1])]) for r, p in zip(base.to_numpy(), pose)]
        # identity-level code: every image of a person carries that person's code, so the
        # per-identity multiset is {code x n} and adds nothing over the code itself.
        idcode = df.groupby("identity").code.agg(lambda s: s.value_counts().idxmax())
        dfi = df.copy(); dfi["code"] = dfi.identity.map(idcode)
        rawi = dfi.copy(); rawi["cls"] = rawi.code
        r = dict(rule="identity_code", k=1, pose_bins=pb)
        r.update(analyse(rawi, nid, attr_cols)); rows.append(r)
        print(f"  [h13] pose-{pb} IDENTITY-LEVEL code: {r['n_classes']} classes  "
              f"multiset_unique {r['multiset_unique']:.3f}  multiset_attacker {r['multiset_attacker']:.3f}  "
              f"per-image {r['per_image_attacker']:.3f}", flush=True)
        for rule in ("count", "plurality", "plurality_capped"):
            for k in a.ks:
                if rule == "plurality_capped":
                    # also equalise released images per identity within a class (count channel)
                    m0 = merge_to_k(dfi, k, "plurality")
                    cap = int(m0.groupby("identity").size().quantile(0.10))
                    cap = max(cap, 1)
                    m = (m0.groupby("identity", group_keys=False)
                           .apply(lambda g: g.sample(min(len(g), cap), random_state=0)))
                else:
                    m = merge_to_k(dfi, k, rule)
                r = dict(rule=f"identity_{rule}", k=k, pose_bins=pb)
                r.update(analyse(m, nid, attr_cols)); rows.append(r)
                print(f"  [h13] pose-{pb} identity_{rule:17s} k={k:3d}  classes {r['n_classes']:4d}  "
                      f"support {r['support_attacker']:.3f}  count {r['countonly_attacker']:.3f}  "
                      f"sup+count {r['supcount_attacker']:.3f}  multiset {r['multiset_attacker']:.3f}  "
                      f"per-image {r['per_image_attacker']:.3f}  min_imgs {r['min_imgs']:4d}", flush=True)
        raw = df.copy(); raw["cls"] = raw.code
        r = dict(rule="none", k=1, pose_bins=pb); r.update(analyse(raw, nid, attr_cols)); rows.append(r)
        print(f"  [h13] pose-{pb} no merge: {r['n_classes']} classes  "
              f"multiset_unique {r['multiset_unique']:.3f}  multiset_attacker {r['multiset_attacker']:.3f}  "
              f"support_unique {r['support_unique']:.3f}  per-image {r['per_image_attacker']:.3f}", flush=True)
        for rule in a.rules:
            for k in a.ks:
                t0 = time.time()
                m = merge_to_k_multiset(df, k) if rule == "multiset" else merge_to_k(df, k, rule)
                r = dict(rule=rule, k=k, pose_bins=pb)
                r.update(analyse(m, nid, attr_cols))
                rows.append(r)
                print(f"  [h13] pose-{pb} {rule:9s} k={k:3d}  classes {r['n_classes']:4d}  "
                      f"multiset_unique {r['multiset_unique']:.3f}  multiset_attacker {r['multiset_attacker']:.3f}  "
                      f"support_unique {r['support_unique']:.3f}  per-image {r['per_image_attacker']:.3f}  "
                      f"min_imgs {r['min_imgs']:4d}  ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(P.results("H13") / "composition.csv", index=False)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    cols = ["rule", "k", "pose_bins", "n_classes", "multiset_unique", "multiset_attacker",
            "support_unique", "support_attacker", "per_image_attacker", "min_imgs", "identity_chance"]
    print(d[[c for c in cols if c in d.columns]].to_string(index=False), flush=True)
    print(f"done -> {P.results('H13') / 'composition.csv'}", flush=True)


if __name__ == "__main__":
    main()
