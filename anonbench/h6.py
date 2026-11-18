"""H6: per-group re-identification gap, with an identity-clustered bootstrap.

Per race: rank-1 of the anonymized probes, and of the same probes clean.
Within a run:  gap = r1(group_a) - r1(group_b)
Between runs:  diff-in-diff = gap(run_1) - gap(run_2)   (is the FAMS gap larger than BlendPose's?)
The bootstrap resamples IDENTITIES with replacement (images of one person are not independent),
the same identities in both runs for the diff-in-diff, so the two runs stay paired.
Optional --cap limits the gallery to that many images per identity (mean over --draws draws),
which removes gallery-size differences between groups.

    python -m anonbench.h6 --anon-dir <run> [<run> ...] [--cap 5] [--draws 5] [--boot 2000]
"""
from __future__ import annotations

import argparse
import itertools
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


def per_probe(cfg, dataset, anon_dir, emb_name, embedder, idx, cap, draws, rng):
    clean = embed_set(cfg, dataset, emb_name, None, embedder=embedder)
    anon = embed_set(cfg, dataset, emb_name, anon_dir, embedder=embedder)
    I = idx.set_index("image_id")
    c_ids = I.loc[clean["image_id"], "identity"].to_numpy()
    c_src = np.asarray(clean["image_id"])
    a_ids = I.loc[anon["image_id"], "identity"].to_numpy()
    a_src = np.asarray(anon["image_id"])
    pos = {k: i for i, k in enumerate(c_src)}
    probe_clean = clean["emb"][[pos[s] for s in a_src]]

    def run_once(keep):
        g = identify(anon["emb"], a_ids, a_src, clean["emb"][keep], c_ids[keep], c_src[keep], exclude_same_source=True)
        g0 = identify(probe_clean, a_ids, a_src, clean["emb"][keep], c_ids[keep], c_src[keep], exclude_same_source=True)
        return g["_per_probe_correct1"].astype(float), g0["_per_probe_correct1"].astype(float), g["_valid"]

    if cap is None:
        hit, hit0, valid = run_once(np.ones(len(c_src), bool))
    else:
        H, H0 = [], []
        for d in range(draws):
            keep = np.zeros(len(c_src), bool)
            for ident in np.unique(c_ids):
                w = np.nonzero(c_ids == ident)[0]
                keep[rng.choice(w, min(cap, len(w)), replace=False)] = True
            h, h0, valid = run_once(keep)
            H.append(h); H0.append(h0)
            print(f"      [h6] {emb_name} cap draw {d + 1}/{draws}", flush=True)
        hit, hit0 = np.mean(H, 0), np.mean(H0, 0)
    return pd.DataFrame(dict(image_id=a_src, identity=a_ids,
                             race=I.loc[a_src, "race"].to_numpy(), hit=hit, hit_clean=hit0, valid=valid))


def boot_gaps(tabs, ga, gb, n_boot, rng):
    """tabs: {run: per-probe df}. Resample identities once, apply to every run."""
    ids = sorted(set.intersection(*[set(t.identity) for t in tabs.values()]))
    byid = {r: {i: g for i, g in t[t.valid].groupby("identity")} for r, t in tabs.items()}
    first = tabs[list(tabs)[0]]
    race = first.drop_duplicates("identity").set_index("identity").race.to_dict()
    ids_a = [i for i in ids if race.get(i) == ga and all(i in byid[r] for r in tabs)]
    ids_b = [i for i in ids if race.get(i) == gb and all(i in byid[r] for r in tabs)]
    print(f"      [h6] bootstrap over {len(ids_a)} {ga} ids vs {len(ids_b)} {gb} ids", flush=True)
    assert ids_a and ids_b, f"empty group: {len(ids_a)} / {len(ids_b)}"
    out = {r: [] for r in tabs}
    for _ in range(n_boot):
        sa = rng.choice(ids_a, len(ids_a), replace=True)
        sb = rng.choice(ids_b, len(ids_b), replace=True)
        for r in tabs:
            A = np.concatenate([byid[r][i].hit.to_numpy() for i in sa])
            B = np.concatenate([byid[r][i].hit.to_numpy() for i in sb])
            out[r].append(A.mean() - B.mean())
    return {r: np.array(v) for r, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--embedders", nargs="*", default=["adaface_vit_base_webface4m", "arcface_r50_webface600k"])
    ap.add_argument("--groups", nargs=2, default=["White", "Asian_chinese_korean"])
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    runs = [Path(d) for d in a.anon_dir if (Path(d) / "manifest.csv").exists()]
    ga, gb = a.groups
    rows, dd_rows = [], []
    print(f"[h6] {len(runs)} runs, groups {ga} vs {gb}, cap={a.cap}, boot={a.boot}", flush=True)
    for emb_name in a.embedders:
        embedder = get_embedder(emb_name, cfg)
        rng = np.random.default_rng(a.seed)
        tabs = {}
        for d in runs:
            t0 = time.time()
            tag = set_tag(cfg, d)
            tabs[tag] = per_probe(cfg, a.dataset, d, emb_name, embedder, idx, a.cap, a.draws, rng)
            t = tabs[tag]
            for race, g in t[t.valid].groupby("race"):
                rows.append(dict(embedder=emb_name, run=tag, race=race, n=len(g),
                                 n_ids=g.identity.nunique(), cap=a.cap,
                                 rank1=g.hit.mean(), rank1_clean=g.hit_clean.mean()))
            print(f"  [h6] {emb_name} {tag[:55]} done ({time.time() - t0:.0f}s)", flush=True)
        rng = np.random.default_rng(a.seed + 1)
        B = boot_gaps(tabs, ga, gb, a.boot, rng)
        for r, v in B.items():
            t = tabs[r]
            obs = t[t.valid & (t.race == ga)].hit.mean() - t[t.valid & (t.race == gb)].hit.mean()
            dd_rows.append(dict(embedder=emb_name, run=r, comparison=f"{ga}-{gb}", cap=a.cap,
                                gap=obs, ci_low=np.nanpercentile(v, 2.5), ci_high=np.nanpercentile(v, 97.5),
                                p_gap_le_0=float(np.nanmean(v <= 0))))
            print(f"  [h6] GAP {emb_name} {r[:50]} {obs:+.3f} [{np.nanpercentile(v, 2.5):+.3f},{np.nanpercentile(v, 97.5):+.3f}] p(<=0)={np.nanmean(v <= 0):.3f}", flush=True)
        for r1, r2 in itertools.combinations(B, 2):
            dd = B[r1] - B[r2]
            t1, t2 = tabs[r1], tabs[r2]
            o1 = t1[t1.valid & (t1.race == ga)].hit.mean() - t1[t1.valid & (t1.race == gb)].hit.mean()
            o2 = t2[t2.valid & (t2.race == ga)].hit.mean() - t2[t2.valid & (t2.race == gb)].hit.mean()
            dd_rows.append(dict(embedder=emb_name, run=f"DID {r1[:30]} vs {r2[:30]}", comparison=f"{ga}-{gb}",
                                cap=a.cap, gap=o1 - o2, ci_low=np.nanpercentile(dd, 2.5),
                                ci_high=np.nanpercentile(dd, 97.5), p_gap_le_0=float(np.nanmean(dd <= 0))))
            print(f"  [h6] DID {emb_name} {r1[:35]} vs {r2[:35]}: {o1 - o2:+.3f} "
                  f"[{np.nanpercentile(dd, 2.5):+.3f},{np.nanpercentile(dd, 97.5):+.3f}] p(<=0)={np.nanmean(dd <= 0):.3f}", flush=True)
    suf = "" if a.cap is None else f"_cap{a.cap}"
    pd.DataFrame(rows).to_csv(P.results("H6") / f"h6_per_race{suf}.csv", index=False)
    pd.DataFrame(dd_rows).to_csv(P.results("H6") / f"h6_gaps{suf}.csv", index=False)
    print(f"done -> {P.results('H6')}/h6_gaps{suf}.csv", flush=True)


if __name__ == "__main__":
    main()
