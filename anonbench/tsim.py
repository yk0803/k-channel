"""T-sim (Q14): linkage of anonymized images WITHOUT face recognition.

Features (all mapped to L2-normalised vectors, similarity = cosine):
  pixel   grey 224 -> 32x32 mean-pool, zero-mean (normalised cross-correlation)
  phash   grey 32x32 DCT, 8x8 low band, median threshold -> 64 bits as +-1 (cosine = 1 - 2*Hamming/64)
  dinov2  facebook/dinov2-small pooled CLS
  clip    openai/clip-vit-large-patch14 image embedding
Modes: full frame, or `context` = face box (from the SOURCE landmarks, transfer) filled with grey on probe and gallery.
Settings vs the gallery of ALL clean images of the dataset:
  source linkage    rank of the exact original among all clean images        (chance 1/|gallery|)
  identity linkage  own source excluded, max-similarity per identity, rank-1 (chance 1/#identities)
Clean reference: identity linkage of the clean source images themselves.

    python -m anonbench.tsim --anon-dir <run dir> [...] [--features ...] [--modes ...]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .embed import set_tag
from .manifest import read_manifest

S = 224


def _load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR))


def load_many(paths, desc):
    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        out = list(ex.map(_load, paths))
    print(f"[tsim] loaded {len(out)} {desc} images in {time.time() - t0:.0f}s", flush=True)
    return np.stack(out)


def mask_face(a, kps):
    k = kps * np.array([S, S], np.float32)
    (x0, y0), (x1, y1) = k.min(0), k.max(0)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    X0, X1 = int(max(0, x0 - 0.8 * bw)), int(min(S, x1 + 0.8 * bw))
    Y0, Y1 = int(max(0, y0 - 1.0 * bh)), int(min(S, y1 + 0.9 * bh))
    a = a.copy()
    a[Y0:Y1, X0:X1] = 128
    return a


def _l2(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Feat:
    def __init__(self, name, dev):
        self.name, self.dev = name, dev
        if name == "dinov2":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
            self.mu, self.sd = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif name == "clip":
            from transformers import CLIPVisionModelWithProjection
            self.m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
            self.mu, self.sd = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    @torch.no_grad()
    def __call__(self, X):  # X: N x 224 x 224 x 3 uint8
        if self.name in ("pixel", "phash"):
            g = X.astype(np.float32).mean(3).reshape(len(X), 32, 7, 32, 7).mean((2, 4))
            if self.name == "pixel":
                f = g.reshape(len(X), -1)
                return _l2(f - f.mean(1, keepdims=True))
            d = dctn(g, axes=(1, 2), norm="ortho")[:, :8, :8].reshape(len(X), -1)
            return _l2(np.where(d > np.median(d, axis=1, keepdims=True), 1.0, -1.0))
        out = []
        mu = torch.tensor(self.mu, device=self.dev).view(1, 3, 1, 1)
        sd = torch.tensor(self.sd, device=self.dev).view(1, 3, 1, 1)
        for i in range(0, len(X), 128):
            x = torch.from_numpy(X[i:i + 128]).to(self.dev).permute(0, 3, 1, 2).float().div(255)
            x = (x - mu) / sd
            o = self.m(pixel_values=x)
            out.append((o.pooler_output if self.name == "dinov2" else o.image_embeds).float().cpu().numpy())
        return _l2(np.concatenate(out))


def masked(X, kps_list):
    return np.stack([mask_face(a, k) for a, k in zip(X, kps_list)])


def rank_source(Sm, src_col):
    t = Sm[np.arange(len(Sm)), src_col]
    return (Sm > t[:, None]).sum(1) + 1


def rank_identity(Sm, src_col, p_ids, g_ids, exclude=True):
    Sm = Sm.copy()
    if exclude:
        Sm[np.arange(len(Sm)), src_col] = -np.inf
    order = np.argsort(g_ids, kind="stable")
    gs = g_ids[order]
    starts = np.r_[0, np.nonzero(gs[1:] != gs[:-1])[0] + 1]
    ids = gs[starts]
    M = np.maximum.reduceat(Sm[:, order], starts, axis=1)
    col = {k: i for i, k in enumerate(ids)}
    true = M[np.arange(len(M)), [col[p] for p in p_ids]]
    return (M > true[:, None]).sum(1) + 1, len(ids)


def _save(out, rows):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        key = lambda d: d.run + "|" + d.feature + "|" + d["mode"]  # noqa: E731
        old = old[~key(old).isin(key(new))]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--features", nargs="*", default=["pixel", "phash", "dinov2", "clip"])
    ap.add_argument("--modes", nargs="*", default=["full", "context"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    lm = load_landmarks(cfg, a.dataset)
    Kg = np.mean(np.stack(list(lm.values())), 0)  # ONE fixed face box for every image (context mode; per-image boxes leak the source)
    g_ids, g_img = idx.identity.to_numpy(), idx.image_id.to_numpy()
    gpos = {k: i for i, k in enumerate(g_img)}
    cache = P.work / "tsim_cache" / a.dataset
    cache.mkdir(parents=True, exist_ok=True)
    out = P.results("Tsim") / "tsim.csv"
    runs = [Path(d) for d in a.anon_dir if (Path(d) / "manifest.csv").exists()]
    print(f"[tsim] {len(runs)} runs, gallery {len(idx)} clean images / {len(set(g_ids))} ids, device {dev}", flush=True)
    G_raw = None
    feats = {}
    G = {}
    for mode in a.modes:
        for f in a.features:
            cp = cache / f"clean__{f}__{mode}.npz"
            if cp.exists() and list(np.load(cp, allow_pickle=True)["image_id"]) == list(g_img):
                G[(f, mode)] = np.load(cp, allow_pickle=True)["emb"]
                continue
            if G_raw is None:
                G_raw = load_many(idx.path.tolist(), "clean")
            feats.setdefault(f, Feat(f, dev))
            X = G_raw if mode == "full" else masked(G_raw, [Kg] * len(G_raw))
            G[(f, mode)] = feats[f](X)
            tmp = cp.with_name(cp.stem + '.tmp.npz'); np.savez(tmp, image_id=g_img.astype(str), emb=G[(f, mode)]); os.replace(tmp, cp)
            print(f"[tsim] gallery {f}/{mode} done", flush=True)
    del G_raw
    for ri, d in enumerate(runs, 1):
        t0 = time.time()
        m = read_manifest(d)
        m = m[m.status != "failed"].reset_index(drop=True)
        tag = set_tag(cfg, d)
        A_raw = load_many(m.anon_path.tolist(), f"anon ({tag[:50]})")
        src_col = np.array([gpos[i] for i in m.image_id])
        p_ids = g_ids[src_col]
        kps = [lm[p] for p in m.src_path]
        rows, line = [], []
        for mode in a.modes:
            X = A_raw if mode == "full" else masked(A_raw, [Kg] * len(A_raw))
            for f in a.features:
                feats.setdefault(f, Feat(f, dev))
                Ae = feats[f](X)
                Gm = G[(f, mode)]
                Sm = Ae @ Gm.T
                rs = rank_source(Sm, src_col)
                ri_, nid = rank_identity(Sm, src_col, p_ids, g_ids)
                rc, _ = rank_identity(Gm[src_col] @ Gm.T, src_col, p_ids, g_ids)
                rows.append(dict(run=tag, feature=f, mode=mode, n=len(m),
                                 source_r1=float((rs == 1).mean()), source_r5=float((rs <= 5).mean()),
                                 source_chance=1.0 / len(g_img), identity_r1=float((ri_ == 1).mean()),
                                 identity_r5=float((ri_ <= 5).mean()), identity_chance=1.0 / nid,
                                 identity_r1_clean=float((rc == 1).mean())))
                line.append(f"{f}/{mode} src {rows[-1]['source_r1']:.3f} id {rows[-1]['identity_r1']:.3f}")
        _save(out, rows)
        print(f"[{ri}/{len(runs)}] RESULT {tag[:70]} | " + " | ".join(line) + f" ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
