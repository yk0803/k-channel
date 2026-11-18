"""Run any anonymizer over a split and write a manifest. Resume-by-default (skips finished runs).

    python -m anonbench.anonymize --method simple   --which sweep     --split test
    python -m anonbench.anonymize --method blendpose --which faithful controls --split test
    python -m anonbench.anonymize --method edi_anon --split train --limit 20 --tag-suffix probe   # timing probe
    python -m anonbench.anonymize --method edials   --split train --select random --budget-from <targeted run dir>

Output: work/anon/<dataset>/<method>/<config tag>[__<suffix>]/seed<k>/<split>/{manifest.csv, run_meta.json, images/}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import Paths, config_tag, load_config, method_configs
from .data import load_index
from .manifest import is_done, merge_shards, read_manifest


def select_rows(idx: pd.DataFrame, split: str, fold: str | None, limit: int | None, seed: int) -> pd.DataFrame:
    rows = idx[idx.split == split] if split != "all" else idx
    if fold:
        rows = rows[rows.fold == fold]
    if limit:
        # stratified-by-identity sample so a probe is not 20 images of one person
        rows = (rows.groupby("identity", group_keys=False).apply(lambda g: g.sample(1, random_state=seed))
                .sample(frac=1, random_state=seed).head(limit))
        if len(rows) < limit:
            rest = idx[(idx.split == split) & ~idx.image_id.isin(rows.image_id)] if split != "all" else idx
            rows = pd.concat([rows, rest.sample(min(limit - len(rows), len(rest)), random_state=seed)])
    return rows.sort_values("image_id").reset_index(drop=True)


def run_one(cfg, dataset, method, params, rows, out, seed, a):
    P = Paths(cfg, dataset)
    guide_name = cfg["guides"].get(method)
    guide_dir = P.model_dir(guide_name, a.guide_seed) if guide_name else None
    if method == "ciagan":
        from .anonymizers import external
        return external.run_ciagan(cfg, rows, params, out, seed=seed)
    if method == "nullface":
        from .anonymizers import external
        return external.run_nullface(cfg, rows, params, out, seed=seed)
    if method == "simple":
        from .anonymizers import simple
        return simple.run(cfg, rows, params, out, seed=seed)
    if method == "edi_anon":
        from .anonymizers import edi_anon
        if a.select == "random":
            b = read_manifest(a.budget_from)
            return edi_anon.run(cfg, rows, params, out, seed=seed, random_mask=True,
                                budget=dict(zip(b.image_id, b.n_steps)))
        return edi_anon.run(cfg, rows, params, out, guide_dir=guide_dir, seed=seed)
    if method == "edials":
        from .anonymizers import edials
        aae = a.aae or (P.model_dir("edials_aae", 0) / "aae.pt")
        budget = read_manifest(a.budget_from) if a.select == "random" else None
        return edials.run(cfg, rows, params, out, guide_dir=guide_dir, aae_path=aae, seed=seed,
                          select=a.select, budget=budget)
    if method == "blendpose":
        from .anonymizers import blendpose
        return blendpose.run(cfg, rows, params, out, dataset=dataset, seed=seed, ddpm_path=a.ddpm)
    if method == "deepprivacy2":
        from .anonymizers.external import run_deepprivacy2
        return run_deepprivacy2(cfg, rows, params, out, seed=seed)
    if method == "fams":
        from .anonymizers.external import run_fams
        return run_fams(cfg, rows, params, out, seed=seed)
    raise KeyError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--method", required=True,
                    choices=["simple", "edi_anon", "edials", "blendpose", "deepprivacy2", "fams", "nullface", "ciagan"])
    ap.add_argument("--which", nargs="+", default=["faithful"], choices=["faithful", "sweep", "controls", "all"])
    ap.add_argument("--split", default=None, help="train|val|test|all (default: the paper's own split)")
    ap.add_argument("--fold", default=None, choices=[None, "A", "B"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--limit", type=int, default=None, help="timing probe / equivalence test subset")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--guide-seed", type=int, default=0)
    ap.add_argument("--select", default="rules", choices=["rules", "random"], help="E4 random control")
    ap.add_argument("--budget-from", default=None, help="targeted run dir whose per-image budget E4 reuses")
    ap.add_argument("--aae", default=None)
    ap.add_argument("--ddpm", default=None)
    ap.add_argument("--names", nargs="*", default=None, help="only run configs whose `name` is in this list (controls)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: process rows i, i+n, ... into <out>/shard<i>of<n>")
    ap.add_argument("--merge-shards", type=int, default=None, help="n: merge <out>/shard*of<n> into <out>")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.select == "random" and not a.budget_from:
        ap.error("--select random needs --budget-from")
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset)
    split = a.split or cfg["anonymizers"][a.method].get("paper_eval_split", "test")
    params_list = []
    for w in a.which:
        params_list += method_configs(cfg, a.method, w)
    if a.names:
        params_list = [p for p in params_list if p.get("name") in set(a.names)]
        if not params_list:
            ap.error(f"--names {a.names} matched no config")
    for params in params_list:
        tag = config_tag(params) + (f"__{a.tag_suffix}" if a.tag_suffix else "") + ("__random" if a.select == "random" else "")
        if a.limit:
            tag += f"__n{a.limit}"
        for seed in a.seeds:
            out = P.anon_dir(a.method, tag, split if not a.fold else f"{split}_fold{a.fold}", seed)
            if a.merge_shards:
                merge_shards(out, a.merge_shards)
                continue
            if a.shard:
                si, sn = map(int, a.shard.split("/"))
                out = out / f"shard{si}of{sn}"
            if is_done(out) and not a.force:
                print(f"[skip] {out}")
                continue
            rows = select_rows(idx, split, a.fold, a.limit, seed)
            if a.shard:
                rows = rows.iloc[si::sn].reset_index(drop=True)
            print(f"[run] {a.method} {tag} seed{seed} split={split} n={len(rows)} -> {out}")
            t0 = time.time()
            run_one(cfg, a.dataset, a.method, dict(params), rows, out, seed, a)
            dt = time.time() - t0
            print(f"[done] {len(rows)} images in {dt / 60:.1f} min = {dt / max(len(rows), 1):.2f} s/image")


if __name__ == "__main__":
    main()
