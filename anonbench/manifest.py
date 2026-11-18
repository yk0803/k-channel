"""Manifests: the contract between anonymizers and evaluators.

Every anonymizer run writes `<anon_dir>/manifest.csv` with one row per source image:
    image_id, src_path, anon_path, identity, label, split, status, n_steps, extra(json)
status:
    modified            the output differs from the input
    unmodified          the method deliberately released the input as-is
                        (e.g. guide already misclassified it / no counterfactual rule)
    failed              the method crashed/skipped this image (anon_path may be empty)
n_steps: number of perturbation steps applied (iterative methods), used by E4 budget matching.

Measured distance to the source (B3) - `status` is what the adapter CLAIMS; these columns are what
was MEASURED, and E0 uses them:
    ref_kind   'exact'  = compared with the exact array the method started from (adapter passed `ref`)
               'resize' = compared with the source file resized (bilinear) to the output size (approximate)
    src_linf   max |anon - ref| over pixels/channels, 0..255
    src_mae    mean |anon - ref|, 0..255
    src_psnr   PSNR in dB (inf if identical)
At close(), rows whose claimed status disagrees with the measurement are counted and written to
run_meta.json (`status_disagreements`) and printed.

Geometry (B11): run_meta.json carries `geometry`. 'resize' means every output is a (possibly
anisotropic) resize of the FULL source frame, with no crop - the precondition for transfer alignment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = ["image_id", "src_path", "anon_path", "identity", "label", "split", "status", "n_steps",
           "ref_kind", "src_linf", "src_mae", "src_psnr", "extra"]

# thresholds used ONLY to flag status/measurement disagreements (E0 itself reports the raw numbers)
UNMOD_MAE = 2.0     # a claimed-unmodified output further than this from its reference is suspicious
MOD_LINF = 0        # a claimed-modified output with L-inf 0 is pixel-identical to its reference


def distance(anon_u8: np.ndarray, ref_u8: np.ndarray) -> dict:
    a = anon_u8.astype(np.float32)
    r = ref_u8.astype(np.float32)
    d = np.abs(a - r)
    mse = float((d ** 2).mean())
    return dict(src_linf=float(d.max()), src_mae=float(d.mean()),
                src_psnr=(math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)))


def _ref_from_source(src_path: str, size_wh: tuple[int, int]) -> np.ndarray:
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        if im.size != size_wh:
            im = im.resize(size_wh, Image.BILINEAR)
        return np.asarray(im)


class ManifestWriter:
    def __init__(self, out_dir: Path, meta: dict | None = None, geometry: str = "resize"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.meta = {"geometry": geometry, **(meta or {})}

    def image_path(self, image_id: str, ext: str = ".png") -> Path:
        p = self.out_dir / "images" / f"{image_id}{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add(self, row, anon_path: Path | str | None, status: str, n_steps: int = 0, dist: dict | None = None,
            ref_kind: str = "", **extra):
        dist = dist or {}
        self.rows.append(dict(image_id=row["image_id"], src_path=row["path"],
                              anon_path=str(anon_path) if anon_path else "",
                              identity=row["identity"], label=int(row["label"]), split=row["split"],
                              status=status, n_steps=int(n_steps), ref_kind=ref_kind,
                              src_linf=dist.get("src_linf", np.nan), src_mae=dist.get("src_mae", np.nan),
                              src_psnr=dist.get("src_psnr", np.nan),
                              extra=json.dumps(extra, default=_json_default)))

    def save_image(self, row, img: np.ndarray | Image.Image, status: str, n_steps: int = 0,
                   ref: np.ndarray | None = None, **extra) -> Path:
        """`ref`: the exact HxWx3 array (uint8 or float in [0,1]) the method started from, if the
        adapter has it. Otherwise the source file is resized to the output size (approximate)."""
        p = self.image_path(row["image_id"])
        arr = _to_uint8(np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else img)
        Image.fromarray(arr).save(p)
        if ref is not None:
            ref_u8, kind = _to_uint8(np.asarray(ref)), "exact"
            if ref_u8.shape != arr.shape:
                raise ValueError(f"{row['image_id']}: ref shape {ref_u8.shape} != output shape {arr.shape}")
        else:
            ref_u8, kind = _ref_from_source(row["path"], (arr.shape[1], arr.shape[0])), "resize"
        self.add(row, p, status, n_steps, dist=distance(arr, ref_u8), ref_kind=kind, **extra)
        return p

    def close(self) -> Path:
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        p = self.out_dir / "manifest.csv"
        df.to_csv(p, index=False)
        ex = df[df.ref_kind == "exact"]
        dis = dict(
            claimed_unmodified_but_far=int(((ex.status == "unmodified") & (ex.src_mae > UNMOD_MAE)).sum()),
            claimed_modified_but_identical=int(((ex.status == "modified") & (ex.src_linf <= MOD_LINF)).sum()),
        )
        self.meta["status_counts"] = df.status.value_counts().to_dict()
        self.meta["status_disagreements"] = dis
        if any(dis.values()):
            print(f"[manifest] WARNING status/measurement disagreements in {self.out_dir}: {dis}")
        with open(self.out_dir / "run_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=_json_default)
        return p


def _to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    a = np.asarray(a, dtype=np.float32)
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_manifest(anon_dir: Path) -> pd.DataFrame:
    p = Path(anon_dir) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the anonymizer first (python -m anonbench.anonymize ...)")
    df = pd.read_csv(p, keep_default_na=False)
    df["extra"] = df["extra"].map(lambda s: json.loads(s) if s else {})
    for c in ("src_linf", "src_mae", "src_psnr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def is_done(anon_dir: Path) -> bool:
    return (Path(anon_dir) / "manifest.csv").exists()


def merge_shards(out: Path, n: int) -> None:
    """Merge <out>/shard<i>of<n> (written by `anonymize --shard i/n`) into <out>; removes the shard dirs."""
    import shutil
    out = Path(out)
    parts = [out / f"shard{i}of{n}" for i in range(n)]
    miss = [p.name for p in parts if not is_done(p)]
    if miss:
        print(f"[merge] {out}: not finished: {miss}")
        return
    (out / "images").mkdir(parents=True, exist_ok=True)
    dfs, metas = [], []
    for p in parts:
        d = pd.read_csv(p / "manifest.csv", keep_default_na=False)

        def _mv(s, p=p):
            if not s:
                return s
            src = Path(s)
            dst = out / "images" / src.relative_to(p / "images")
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
            elif not dst.exists():
                raise FileNotFoundError(f"{src} missing and not already at {dst}")
            return str(dst)
        d["anon_path"] = d.anon_path.map(_mv)
        dfs.append(d)
        metas.append(json.load(open(p / "run_meta.json")))
    df = pd.concat(dfs, ignore_index=True).sort_values("image_id")
    assert df.image_id.is_unique, "duplicate image ids across shards"
    m = dict(metas[0])
    m["shards"] = n
    m["minutes"] = max(x.get("minutes", 0) for x in metas)
    if "stats" in m:
        st = dict(total=0, unmodified=0, misclassified=0, unmodified_reasons={})
        for x in metas:
            for k in ("total", "unmodified", "misclassified"):
                st[k] += x["stats"].get(k, 0)
            for k, v in x["stats"].get("unmodified_reasons", {}).items():
                st["unmodified_reasons"][k] = st["unmodified_reasons"].get(k, 0) + v
        m["stats"] = st
    m["status_counts"] = df.status.value_counts().to_dict()
    m["status_disagreements"] = {k: sum(x.get("status_disagreements", {}).get(k, 0) for x in metas)
                                 for k in metas[0].get("status_disagreements", {})}
    df.to_csv(out / "manifest.csv", index=False)
    with open(out / "run_meta.json", "w") as f:
        json.dump(m, f, indent=2, default=_json_default)
    for p in parts:
        shutil.rmtree(p)
    print(f"[merge] {out}: {len(df)} rows from {n} shards, status {m['status_counts']}, "
          f"{m.get('stats', {}).get('unmodified_reasons', '')}, wall {m['minutes']:.0f} min")
