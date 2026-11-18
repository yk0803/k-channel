"""H14: which regions of an image carry identity?

Writes region-ablated copies of a split as ordinary anonbench run dirs, so the EXISTING attack
suite (score / tsim / h3 / h8) can be pointed at them unchanged. Each variant keeps some part of
the image and fills the rest with flat grey, using ONE dataset-mean landmark box for every image
(a per-image box would fingerprint the source, bug B16).

Variants:
  face_only     keep the face box
  head_only     keep the face box enlarged (face + hair)
  context_only  grey the face box (the FAMS condition)
  no_head       grey the enlarged box
  border_only   keep the outer 20% frame
  lower_only    keep the bottom third

    python -m anonbench.h14_regions --dataset mtf --split test [--variants ...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .manifest import ManifestWriter

GREY = 128


def mean_box(lm, H, W, grow_x=0.8, grow_y_top=1.0, grow_y_bot=0.9):
    k = np.mean(np.stack(list(lm.values())), 0)
    x0, y0 = k.min(0)
    x1, y1 = k.max(0)
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    X0 = int(max(0, (x0 - grow_x * bw) * W)); X1 = int(min(W, (x1 + grow_x * bw) * W))
    Y0 = int(max(0, (y0 - grow_y_top * bh) * H)); Y1 = int(min(H, (y1 + grow_y_bot * bh) * H))
    return X0, Y0, X1, Y1


def apply_variant(a, variant, face, head):
    H, W = a.shape[:2]
    out = np.full_like(a, GREY)
    fx0, fy0, fx1, fy1 = face
    hx0, hy0, hx1, hy1 = head
    if variant == "face_only":
        out[fy0:fy1, fx0:fx1] = a[fy0:fy1, fx0:fx1]
    elif variant == "head_only":
        out[hy0:hy1, hx0:hx1] = a[hy0:hy1, hx0:hx1]
    elif variant == "context_only":
        out = a.copy(); out[fy0:fy1, fx0:fx1] = GREY
    elif variant == "no_head":
        out = a.copy(); out[hy0:hy1, hx0:hx1] = GREY
    elif variant == "border_only":
        out = a.copy()
        m = int(0.2 * min(H, W))
        out[m:H - m, m:W - m] = GREY
    elif variant == "lower_only":
        out[int(2 * H / 3):] = a[int(2 * H / 3):]
    else:
        raise ValueError(variant)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", nargs="*",
                    default=["face_only", "head_only", "context_only", "no_head", "border_only", "lower_only"])
    ap.add_argument("--size", type=int, default=512, help="output size; keeps runtime sane")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset)
    rows = idx[idx.split == a.split].reset_index(drop=True) if a.split != "all" else idx.reset_index(drop=True)
    lm = load_landmarks(cfg, a.dataset)
    S = a.size
    face = mean_box(lm, S, S)
    head = mean_box(lm, S, S, grow_x=1.1, grow_y_top=1.6, grow_y_bot=1.1)
    print(f"[h14] {len(rows)} images, size {S}, face box {face}, head box {head}", flush=True)
    for v in a.variants:
        out = P.anon_dir("regions", v, a.split, 0)
        W = ManifestWriter(out, meta=dict(method="regions", params=dict(variant=v, size=S), seed=0))
        t0 = time.time()
        for n, r in enumerate(rows.itertuples(index=False), 1):
            d = r._asdict()
            with Image.open(r.path) as im:
                arr = np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR))
            img = Image.fromarray(apply_variant(arr, v, face, head))
            W.save_image(d, img, "modified", 1, variant=v)
            if n % 200 == 0 or n == len(rows):
                el = time.time() - t0
                print(f"  [h14] {v} {n}/{len(rows)} ({el:.0f}s, ETA {el / n * (len(rows) - n):.0f}s)", flush=True)
        W.close()
        print(f"[h14] {v} -> {out}", flush=True)


if __name__ == "__main__":
    main()
