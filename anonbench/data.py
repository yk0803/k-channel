"""Dataset indexing.

Produces one canonical CSV per dataset with columns
    image_id, path, identity, label, race, gender, age, split, fold
where
    image_id  unique, filesystem-safe id ("<identity>/<stem>") used everywhere else
    label     integer identity label (0..C-1, sorted identity names)
    split     train | val | test   (the dataset's own IMAGE-LEVEL split, as used by the papers)
    fold      A | B                (our IDENTITY-DISJOINT folds; stratified by race x gender x age)
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RES = Path(__file__).resolve().parent / "resources"
RACES = {"asian_chinese_korean": "Asian_chinese_korean", "asian_indian": "Asian_indian",
         "black": "Black", "white": "White"}
GENDERS = {"male": "Male", "males": "Male", "female": "Female", "females": "Female"}
AGES = {"young": "Young", "old": "Old"}
SPLITS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
          "test": "test", "testing": "test"}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s).strip("_")


# ----------------------------------------------------------------------------- MTF
def build_mtf_index(root: Path, seed: int = 0) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MTF root {root} does not exist - set paths.mtf_root in config.yaml")
    # Prefer the face-recognition subtree of the task-specific release (avoids duplicates
    # across race/gender/age task folders).
    fr = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in ("face_recognition", "face recognition"))
    if len(fr) > 1:  # B9: never pick one silently
        raise RuntimeError(f"several face_recognition dirs under {root}: {[str(p) for p in fr]} - "
                           f"point paths.mtf_root at exactly one MTF release")
    scan_root = fr[0] if fr else root
    print(f"[data] scanning {scan_root} ({'task-specific face_recognition release' if fr else 'curated/flat layout'})")
    labels = pd.read_csv(RES / "mtf_identity_labels.csv").set_index("identity")

    rows = []
    for p in sorted(scan_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXT):
            continue
        rel = p.relative_to(scan_root).parts
        if any("non" in x.lower() and "curat" in x.lower() for x in rel[:-1]):
            continue  # the non-curated MTF set is not used by any of the papers
        identity = p.parent.name
        split = next((SPLITS[x.lower()] for x in rel[:-1] if x.lower() in SPLITS), None)
        race = next((RACES[x.lower()] for x in rel[:-1] if x.lower() in RACES), None)
        gender = next((GENDERS[x.lower()] for x in rel[:-1] if x.lower() in GENDERS), None)
        age = next((AGES[x.lower()] for x in rel[:-1] if x.lower() in AGES), None)
        if identity in labels.index:
            race = race or labels.at[identity, "race"]
            gender = gender or labels.at[identity, "gender"]
            age = age or labels.at[identity, "age"]
        rows.append(dict(path=str(p), identity=identity, race=race, gender=gender, age=age, split=split))
    if not rows:
        raise RuntimeError(f"No images found under {scan_root}")
    df = pd.DataFrame(rows)
    unknown = df[df[["race", "gender", "age"]].isna().any(axis=1)].identity.unique()
    if len(unknown):
        print(f"[warn] {len(unknown)} identities without demographic labels (e.g. {list(unknown)[:5]}); "
              f"they are kept for FR but ignored by utility tasks")
    if df.split.isna().all():
        print("[info] no train/val/test folders found: creating MTF-like image-level split 70/6/24 (seeded)")
        df["split"] = _image_level_split(df, seed)
    elif df.split.isna().any():
        raise RuntimeError("Some images are inside a split folder and some are not; point mtf_root at one release.")
    return _finalize(df)


def _image_level_split(df: pd.DataFrame, seed: int) -> list[str]:
    """MTF-style: every identity appears in every split (>=3 train images, >=1 val/test)."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("identity"):
        idx = rng.permutation(g.index.values)
        n = len(idx)
        n_test = max(1, int(round(0.24 * n))) if n >= 5 else (1 if n >= 3 else 0)
        n_val = max(1, int(round(0.06 * n))) if n >= 6 else 0
        out[idx[:n_test]] = "test"
        out[idx[n_test:n_test + n_val]] = "val"
        out[idx[n_test + n_val:]] = "train"
    return out.tolist()


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ids = sorted(df.identity.unique())
    df["label"] = df.identity.map({k: i for i, k in enumerate(ids)}).astype(int)
    stem = df.path.map(lambda p: Path(p).stem)
    df["image_id"] = (df.identity.map(_safe) + "/" + df.split + "_" + stem.map(_safe))
    dup = df.image_id.duplicated(keep=False)
    if dup.any():  # same stem twice in one identity/split -> disambiguate
        df.loc[dup, "image_id"] = df.loc[dup, "image_id"] + "_" + df.loc[dup].groupby("image_id").cumcount().astype(str)
    cols = ["image_id", "path", "identity", "label", "race", "gender", "age", "split"]
    return df[cols].sort_values("image_id").reset_index(drop=True)


def assign_identity_folds(df: pd.DataFrame, frac_b: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Identity-disjoint folds A/B, stratified by race x gender x age (identity level)."""
    rng = np.random.default_rng(seed)
    idt = df.groupby("identity")[["race", "gender", "age"]].first().fillna("NA")
    fold = {}
    for _, g in idt.groupby(["race", "gender", "age"]):
        names = rng.permutation(g.index.values)
        n_b = int(round(frac_b * len(names)))
        if len(names) >= 2:
            n_b = min(max(n_b, 1), len(names) - 1)
        for i, n in enumerate(names):
            fold[n] = "B" if i < n_b else "A"
    df = df.copy()
    df["fold"] = df.identity.map(fold)
    return df


# ----------------------------------------------------------------------------- CelebA (E12)
def build_celeba_index(root: Path, min_images: int = 20, max_identities: int = 500, seed: int = 0) -> pd.DataFrame:
    root = Path(root)
    ident = pd.read_csv(root / "identity_CelebA.txt", sep=r"\s+", header=None, names=["file", "identity"])
    attr = pd.read_csv(root / "list_attr_celeba.txt", sep=r"\s+", skiprows=1)
    attr.index.name = "file"
    attr = attr.reset_index() if "file" not in attr.columns else attr
    part_f = root / "list_eval_partition.txt"
    imgdir = root / "img_align_celeba"
    if not imgdir.exists():
        cand = list(root.rglob("img_align_celeba"))
        imgdir = cand[0] if cand else imgdir
    counts = ident.identity.value_counts()
    keep = counts[counts >= min_images].index.values
    rng = np.random.default_rng(seed)
    if len(keep) > max_identities:
        keep = rng.choice(keep, max_identities, replace=False)
    df = ident[ident.identity.isin(set(keep))].merge(attr[["file", "Male", "Young"]], on="file", how="left")
    df["identity"] = "celeba_" + df.identity.astype(str)
    df["path"] = df.file.map(lambda f: str(imgdir / f))
    df["gender"] = np.where(df.Male > 0, "Male", "Female")
    df["age"] = np.where(df.Young > 0, "Young", "Old")
    df["race"] = None  # filled later with pseudo-labels from the MTF race classifier (see e12)
    # identity-level majority vote for gender/age (attributes are per-image in CelebA)
    for c in ("gender", "age"):
        df[c] = df.groupby("identity")[c].transform(lambda s: s.mode().iat[0])
    df["split"] = _image_level_split(df, seed)
    del part_f
    return _finalize(df[["path", "identity", "race", "gender", "age", "split"]])


# ----------------------------------------------------------------------------- I/O helpers
def load_index(cfg: dict, dataset: str = "mtf") -> pd.DataFrame:
    p = Paths(cfg, dataset).index_csv()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m anonbench.data --dataset {dataset}` first")
    return pd.read_csv(p)


def prompt_of(row) -> str:
    """BlendPose prompt format, identical to the files shipped in their repo: 'Race, Gender, Age'."""
    return f"{row['race']}, {row['gender']}, {row['age']}"


def leakage_census(df: pd.DataFrame) -> pd.DataFrame:
    """E6(a): identity overlap between the dataset's own train and test splits."""
    rows = []
    tr = set(df[df.split == "train"].identity)
    for sp in ("val", "test"):
        sub = df[df.split == sp]
        ids = set(sub.identity)
        rows.append(dict(split=sp, n_images=len(sub), n_identities=len(ids),
                         identities_also_in_train=len(ids & tr),
                         images_whose_identity_in_train=int(sub.identity.isin(tr).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build the canonical dataset index (+ identity folds).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf", choices=["mtf", "celeba"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    if a.dataset == "mtf":
        df = build_mtf_index(cfg["paths"]["mtf_root"])
    else:
        e = cfg["experiments"]["e12"]
        df = build_celeba_index(cfg["paths"]["celeba_root"], e["min_images_per_identity"], e["max_identities"])
    df = assign_identity_folds(df, cfg["splits"]["identity_fold_B_fraction"], cfg["splits"]["identity_split_seed"])
    out = P.index_csv()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} images, {df.identity.nunique()} identities")
    print(df.groupby("split").agg(images=("image_id", "size"), identities=("identity", "nunique")))
    print(df.groupby("fold").agg(images=("image_id", "size"), identities=("identity", "nunique")))
    lc = leakage_census(df)
    lc.to_csv(out.parent / f"{a.dataset}_leakage_census.csv", index=False)
    print("Identity overlap between the dataset's own splits (E6 census):\n", lc.to_string(index=False))


if __name__ == "__main__":
    main()
