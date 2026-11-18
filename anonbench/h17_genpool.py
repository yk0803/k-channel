"""H17: label the StyleGAN2 fake pool so generated faces can be binned into code classes.

The pool is the 60k FFHQ-StyleGAN2 sample generated for FALCO. None of these people exist and no
image is in any gallery, so a release drawn from it is not a copy of any real image - the flaw of
the surrogate simulation in h15.

Pose comes from the same detector used on the real images (insightface SCRFD); race/gender/age from
the utility_*_clean heads trained on real MTF.

    python -m anonbench.h17_genpool [--limit N] [--chunk 2000]
      -> work/index/genpool.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .align import Detector, detect_landmarks
from .config import Paths, load_config
from .models.classifiers import load_classifier, predict_logits

POOL = ("third_party/FALCO/datasets/fake/"
        "fake_dataset_stylegan2_ffhq1024-0.7-60000-CLIP-FaRL-DINO-ArcFace")


def geom_from_kps(df):
    k = df[[f"k{i}{c}" for i in range(5) for c in "xy"]].to_numpy(float).reshape(-1, 5, 2)
    le, re, no = k[:, 0], k[:, 1], k[:, 2]
    eye = re - le
    iod = np.maximum(np.linalg.norm(eye, axis=1), 1e-6)
    roll = np.degrees(np.arctan2(eye[:, 1], eye[:, 0]))
    yaw = (no[:, 0] - (le[:, 0] + re[:, 0]) / 2) / iod
    return yaw, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    root = Path(a.pool) if a.pool else Path.home() / "PoPETs" / POOL
    paths = sorted(str(p) for p in root.glob("*/image.jpg"))
    if a.limit:
        paths = paths[:a.limit]
    print(f"[h17] pool {root}\n[h17] {len(paths)} generated images", flush=True)

    det = Detector(cfg["paths"]["weights_dir"], cfg["hardware"]["device"])
    parts, t0 = [], time.time()
    for i in range(0, len(paths), a.chunk):
        parts.append(detect_landmarks(paths[i:i + a.chunk], det))
        done = min(i + a.chunk, len(paths))
        el = time.time() - t0
        print(f"  [h17] landmarks {done}/{len(paths)} ({el:.0f}s, "
              f"ETA {el / done * (len(paths) - done) / 60:.1f} min)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df = df[df.detected == 1].reset_index(drop=True)
    print(f"[h17] {len(df)}/{n0} with a detected face", flush=True)
    df["yaw"], df["roll"] = geom_from_kps(df)

    for task in ("race", "gender", "age"):
        d = P.model_dir(f"utility_{task}_clean", 0)
        if not (d / "meta.json").exists():
            print(f"  [h17] no utility_{task}_clean head, skipping", flush=True)
            continue
        t1 = time.time()
        m, tf, meta = load_classifier(d, cfg)
        names = meta.get("class_names") or [str(i) for i in range(meta["num_classes"])]
        lg = predict_logits(m, tf, df.path.tolist(), cfg)
        e = np.exp(lg - lg.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        df[task] = [names[j] for j in pr.argmax(1)]
        df[f"{task}_conf"] = pr.max(1)
        print(f"  [h17] {task}: {df[task].value_counts().to_dict()} ({time.time() - t1:.0f}s)", flush=True)
        del m

    out = P.work / "index" / "genpool.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"done -> {out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
