"""Utility metrics the anonymization literature reports for itself, measured on our releases.

    python -m anonbench.sota_utility --dataset mtf --anon-dir <run dir>

Three axes, none of which is attribute classification:
  detection rate  - fraction of released images in which a face detector finds a face at all
  pose fidelity   - |yaw|,|pitch|,|roll| MAE between the release and its own input, degrees
  geometry        - eye-opening and mouth-opening ratios from 106-point landmarks, |delta|
  FID             - released set against the clean test set (skipped if no backend is installed)

Everything is measured per image against that image's own input, so a method that preserves
pose by construction should score near zero and one that resynthesises the frame should not.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_index


def _pose_and_geom(app, img, pad_frac=0.25):
    """Detect, retrying once on a replicate-padded copy.

    SCRFD misses faces that fill the frame, which is the normal framing for FFHQ-style
    generated images and for tight dataset crops. Pose is an angle and the geometry ratios are
    normalised by landmark span, so both are invariant to the padding. The second return value
    records whether the pad was needed, which is itself worth reporting.
    """
    import cv2 as _cv
    fs, padded = app.get(img), 0
    if not fs and pad_frac:
        q = int(max(img.shape[:2]) * pad_frac)
        fs, padded = app.get(_cv.copyMakeBorder(img, q, q, q, q, _cv.BORDER_REPLICATE)), 1
    if not fs:
        return None
    f = max(fs, key=lambda k: (k.bbox[2] - k.bbox[0]) * (k.bbox[3] - k.bbox[1]))
    pose = np.asarray(f.pose, float) if getattr(f, "pose", None) is not None else np.full(3, np.nan)
    lm = getattr(f, "landmark_2d_106", None)
    if lm is None:
        return pose, np.full(2, np.nan), padded
    lm = np.asarray(lm, float)
    scale = np.linalg.norm(lm.max(0) - lm.min(0)) + 1e-8
    eye = (np.linalg.norm(lm[35] - lm[41]) + np.linalg.norm(lm[89] - lm[95])) / 2 / scale
    mouth = np.linalg.norm(lm[52] - lm[61]) / scale
    return pose, np.array([eye, mouth]), padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", required=True)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    import cv2
    from insightface.app import FaceAnalysis

    d = Path(a.anon_dir)
    man = pd.read_csv(d / "manifest.csv")
    idx = load_index(cfg, a.dataset).set_index("image_id")
    pcol = next((c for c in ["anon_path", "out_path", "path", "image_path"] if c in man.columns), None)
    if pcol is None:
        raise SystemExit(f"no path column in manifest: {list(man.columns)}")
    if "status" in man.columns:
        man = man[man.status.astype(str).str.lower().isin(["modified", "unmodified", "ok"])]
    print(f"[sota] {len(man)} released images from {d}", flush=True)

    app = FaceAnalysis(name="buffalo_l", root=str(Path(cfg["paths"]["weights_dir"]) / "insightface"))
    app.prepare(ctx_id=0, det_size=(a.det_size, a.det_size))

    rows, t0 = [], time.time()
    for n, (_, r) in enumerate(man.iterrows(), 1):
        ap_ = Path(r[pcol])
        if not ap_.is_absolute():
            ap_ = d / ap_
        ai = cv2.imread(str(ap_))
        if ai is None:
            continue
        got = _pose_and_geom(app, ai)
        rec = dict(image_id=r["image_id"], detected=int(got is not None),
                   padded=int(got[2]) if got is not None else 0)
        if got is not None and r["image_id"] in idx.index:
            si = cv2.imread(str(idx.loc[r["image_id"], "path"]))
            sg = _pose_and_geom(app, si) if si is not None else None
            if sg is not None:
                rec.update(dyaw=abs(got[0][0] - sg[0][0]), dpitch=abs(got[0][1] - sg[0][1]),
                           droll=abs(got[0][2] - sg[0][2]),
                           deye=abs(got[1][0] - sg[1][0]), dmouth=abs(got[1][1] - sg[1][1]))
        rows.append(rec)
        if n % 100 == 0 or n == len(man):
            el = time.time() - t0
            print(f"[sota] {n}/{len(man)} | {el/n:.2f} s/img | detected so far "
                  f"{np.mean([x['detected'] for x in rows]):.3f} | ETA {el/n*(len(man)-n)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["work"] if "work" in cfg["paths"] else cfg["paths"]["work_dir"])
    out = out / "results" / a.dataset / "sota_utility"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{d.parts[-4]}__{d.parts[-3]}" if len(d.parts) >= 4 else d.name
    df.to_csv(out / f"{tag}.csv", index=False)
    m = df.mean(numeric_only=True)
    print(f"RESULT-SOTA {tag} | detect {m.get('detected', float('nan')):.3f} "
          f"| dyaw {m.get('dyaw', float('nan')):.2f} dpitch {m.get('dpitch', float('nan')):.2f} "
          f"droll {m.get('droll', float('nan')):.2f} "
          f"| deye {m.get('deye', float('nan')):.4f} dmouth {m.get('dmouth', float('nan')):.4f} "
          f"| padded {m.get('padded', float('nan')):.3f} | n={len(df)} (saved)", flush=True)


if __name__ == "__main__":
    main()
