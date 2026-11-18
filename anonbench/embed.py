"""P1.1 - aligned, cached embeddings for one image set x one embedder.

Image sets:
  clean           every original image in the index (aligned with its own landmarks)
  <anon run dir>  every row of that run's manifest (status != failed)

Alignment (config `alignment`, D2):
  transfer  landmarks detected on the ORIGINAL source are reused on the anonymized image (default)
  detect    landmarks are re-detected on the anonymized image; detection failures fall back to the
            centred template and are counted (sensitivity analysis)

Cache: work/embeddings/<dataset>/<set_tag>/<embedder>__<alignment>.npz  {image_id, emb, det_ok}
Re-running is free: a cache whose image_ids match is reused.

    python -m anonbench.embed --clean
    python -m anonbench.embed --anon-dir work/anon/mtf/simple/<tag>/seed0/test
    python -m anonbench.embed --clean --embedders arcface_r50_webface600k facenet_casia
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .align import Detector, fallback_kps_norm, load_landmarks, load_rgb, warp
from .config import Paths, load_config
from .models.embedders import embedder_meta, get_embedder


def set_tag(cfg: dict, anon_dir: Path | None) -> str:
    if anon_dir is None:
        return "clean"
    anon_root = Paths(cfg).work / "anon"
    try:
        rel = Path(anon_dir).resolve().relative_to(anon_root.resolve())
    except ValueError:
        rel = Path(Path(anon_dir).name)
    return "anon__" + "__".join(rel.parts[1:] if len(rel.parts) > 1 else rel.parts)  # drop dataset


def items_clean(cfg: dict, dataset: str) -> pd.DataFrame:
    from .data import load_index
    idx = load_index(cfg, dataset)
    return pd.DataFrame(dict(image_id=idx.image_id, img_path=idx.path, src_path=idx.path))


def items_anon(anon_dir: Path) -> tuple[pd.DataFrame, dict]:
    from .manifest import read_manifest
    m = read_manifest(anon_dir)
    m = m[m.status != "failed"]
    meta_p = Path(anon_dir) / "run_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return pd.DataFrame(dict(image_id=m.image_id, img_path=m.anon_path, src_path=m.src_path)), meta


def cache_path(cfg, dataset, tag, emb_name, alignment) -> Path:
    return Paths(cfg, dataset).emb_dir() / tag / f"{emb_name}__{alignment}.npz"


def load_cached(p: Path, image_ids) -> dict | None:
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if list(z["image_id"]) != list(image_ids):
        return None
    return dict(image_id=z["image_id"], emb=z["emb"], det_ok=z["det_ok"])


def compute(cfg, items: pd.DataFrame, embedder, alignment: str, landmarks: dict | None,
            detector: Detector | None = None, bs: int | None = None) -> dict:
    bs = bs or cfg["hardware"]["eval_batch_size"]
    embs, det_ok, buf = [], [], []

    def flush():
        if buf:
            embs.append(embedder.embed(buf))
            buf.clear()

    for r in tqdm(items.itertuples(index=False), total=len(items), desc=f"embed {embedder.name}", leave=False):
        img = load_rgb(r.img_path)
        if alignment == "transfer":
            k = landmarks.get(r.src_path) if landmarks is not None else None
            ok = k is not None
            k = k if ok else fallback_kps_norm()
        elif alignment == "detect":
            k, ok = detector.kps_norm(img)
        else:
            raise ValueError(alignment)
        det_ok.append(ok)
        buf.append(warp(img, k, embedder.size, embedder.margin))
        if len(buf) >= bs:
            flush()
    flush()
    E = np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, 1), np.float32)
    return dict(image_id=items.image_id.to_numpy().astype(str), emb=E, det_ok=np.asarray(det_ok, bool))


def embed_set(cfg, dataset: str, emb_name: str, anon_dir: Path | None = None, alignment: str | None = None,
              embedder=None, landmarks=None, detector=None, force: bool = False) -> dict:
    alignment = alignment or cfg.get("alignment", "transfer")
    if anon_dir is None:
        items, meta = items_clean(cfg, dataset), {}
        align_mode = "transfer"            # originals: their own detected landmarks, by definition
    else:
        items, meta = items_anon(anon_dir)
        align_mode = alignment
        if align_mode == "transfer" and meta.get("geometry", "resize") != "resize":
            raise ValueError(f"{anon_dir}: geometry={meta.get('geometry')} - transfer alignment needs "
                             f"outputs that are full-frame resizes of the source (B11)")
    tag = set_tag(cfg, anon_dir)
    p = cache_path(cfg, dataset, tag, emb_name, alignment if anon_dir is not None else "transfer")
    if not force:
        hit = load_cached(p, items.image_id.astype(str))
        if hit is not None:
            return hit
    if landmarks is None and align_mode == "transfer" and not emb_name.startswith("fake"):
        landmarks = load_landmarks(cfg, dataset)
    if align_mode == "detect" and detector is None:
        detector = Detector(cfg["paths"]["weights_dir"], cfg["hardware"]["device"])
    embedder = embedder or get_embedder(emb_name, cfg)
    out = compute(cfg, items, embedder, align_mode, landmarks, detector)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    with open(p.with_suffix(".json"), "w") as f:
        json.dump(dict(embedder=emb_name, **embedder_meta(emb_name), alignment=align_mode, set=tag,
                       n=int(len(out["emb"])), det_rate=float(out["det_ok"].mean()) if len(out["det_ok"]) else None,
                       anon_dir=str(anon_dir) if anon_dir else None), f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute + cache aligned embeddings.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--anon-dir", nargs="*", default=[])
    ap.add_argument("--embedders", nargs="*", default=None, help="default: config evaluators.embedders")
    ap.add_argument("--alignment", default=None, choices=[None, "transfer", "detect"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    names = a.embedders or cfg["evaluators"]["embedders"]
    sets = ([None] if a.clean else []) + [Path(d) for d in a.anon_dir]
    if not sets:
        ap.error("nothing to do: pass --clean and/or --anon-dir")
    lm = None
    for n in names:
        emb = get_embedder(n, cfg)                       # load each model once
        for s in sets:
            if lm is None and not n.startswith("fake"):
                lm = load_landmarks(cfg, a.dataset)
            out = embed_set(cfg, a.dataset, n, s, a.alignment, embedder=emb, landmarks=lm, force=a.force)
            print(f"{n:30s} {set_tag(cfg, s):60s} n={len(out['emb'])} det_rate={out['det_ok'].mean():.3f}")


if __name__ == "__main__":
    main()
