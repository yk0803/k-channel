"""Train every model the experiments need on ORIGINAL images.

  python -m anonbench.train --what guides       # EDI-Anon + EDIALS guide (seed 0)
  python -m anonbench.train --what replicas     # E3: same recipes, other seeds
  python -m anonbench.train --what heldout      # held-out closed-set FR adversaries + BlendPose FaceNet classifier
  python -m anonbench.train --what utility      # age / gender / race ConvNeXt (train-clean)
  python -m anonbench.train --what aae          # EDIALS adversarial autoencoder (all train identities)
  python -m anonbench.train --what aae_foldA    # E6: EDIALS AAE on fold-A identities only
  python -m anonbench.train --what all
"""
from __future__ import annotations

import argparse

from .config import Paths, load_config
from .data import load_index
from .models.classifiers import recipe_for, train_classifier

UTILITY_TASKS = ("gender", "age", "race")


def train_fr(cfg, dataset, name, seed, train_df=None, val_df=None, suffix=""):
    idx = load_index(cfg, dataset)
    tr = train_df if train_df is not None else idx[idx.split == "train"]
    va = val_df if val_df is not None else idx[idx.split == "val"]
    C = idx.label.max() + 1
    out = Paths(cfg, dataset).model_dir(name + suffix, seed)
    return train_classifier(cfg, recipe_for(cfg, name), tr.path.tolist(), tr.label.tolist(), va.path.tolist(),
                            va.label.tolist(), C, out, seed=seed, tag=f"{name}{suffix}_s{seed}")


def utility_labels(df, task):
    classes = sorted(df[task].dropna().unique())
    m = {c: i for i, c in enumerate(classes)}
    keep = df[task].notna()
    return df[keep], df[keep][task].map(m).tolist(), classes


def train_utility(cfg, dataset, task, seed=0, train_paths=None, train_df=None, suffix="clean"):
    """Utility classifier. `train_paths` lets E10 train on anonymized versions of train_df."""
    idx = load_index(cfg, dataset)
    tr = train_df if train_df is not None else idx[idx.split == "train"]
    va = idx[idx.split == "val"]
    tr, ytr, classes = utility_labels(tr, task)
    va, yva, _ = utility_labels(va, task)
    paths = train_paths if train_paths is not None else tr.path.tolist()
    out = Paths(cfg, dataset).model_dir(f"utility_{task}_{suffix}", seed)
    return train_classifier(cfg, recipe_for(cfg, "utility"), paths, ytr, va.path.tolist(), yva, len(classes),
                            out, seed=seed, class_names=classes, tag=f"utility_{task}_{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--what", nargs="+", default=["all"],
                    choices=["all", "guides", "replicas", "heldout", "utility", "aae", "aae_foldA"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    w = set(a.what)
    allw = "all" in w
    guides = [g for g in ("convnext_base_ediAnon", "convnext_base_edials")]
    if allw or "guides" in w:
        for g in guides:
            train_fr(cfg, a.dataset, g, 0)
    if allw or "replicas" in w:
        for g in guides:
            for s in cfg["seeds"]["guide_replicas"]:
                train_fr(cfg, a.dataset, g, s)
    if allw or "heldout" in w:
        for name in cfg["evaluators"]["classifiers"]:
            if name not in guides:
                train_fr(cfg, a.dataset, name, 0)
    if allw or "utility" in w:
        for t in UTILITY_TASKS:
            idx = load_index(cfg, a.dataset)
            if idx[t].notna().any():
                train_utility(cfg, a.dataset, t)
    if allw or "aae" in w or "aae_foldA" in w:
        from .anonymizers.edials import train_aae
        idx = load_index(cfg, a.dataset)
        if allw or "aae" in w:
            train_aae(cfg, a.dataset, idx[idx.split == "train"], idx[idx.split == "val"], name="aae")
        if allw or "aae_foldA" in w:
            tr = idx[(idx.split == "train") & (idx.fold == "A")]
            va = idx[(idx.split == "val") & (idx.fold == "A")]
            train_aae(cfg, a.dataset, tr, va, name="aae_foldA")


if __name__ == "__main__":
    main()
