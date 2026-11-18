"""H8: learned linkage attacker (no face recognition model).

Trains a small two-branch embedder on (original, anonymized) pairs of fold-A
identities with InfoNCE, then evaluates on fold B: for every fold-B anonymized
image, rank its own source among all fold-B clean images (source linkage), and
rank its identity with the own source excluded (identity linkage).
Identity-disjoint: no fold-B person is ever seen in training.

    python -m anonbench.h8 --anon-dir <run dir> [...] [--iters N] [--smoke]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import Paths, load_config
from .data import load_index
from .embed import set_tag
from .manifest import read_manifest

R = 160


def _read(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((R, R), Image.BILINEAR))


class Pairs(Dataset):
    def __init__(self, df, train):
        self.src = df.src_path.tolist()
        self.anon = df.anon_path.tolist()
        self.train = train

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        a, b = _read(self.src[i]), _read(self.anon[i])
        if self.train and np.random.rand() < 0.5:
            a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)  # noqa: E731
        return t(a), t(b)


def block(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Enc(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.f = nn.Sequential(block(3, 32, 2), block(32, 64, 2), block(64, 128, 2),
                               block(128, 256, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


@torch.no_grad()
def embed_all(enc, paths, dev, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        X = np.stack([_read(p) for p in paths[i:i + bs]])
        x = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)
        out.append(enc(x).float().cpu().numpy())
    return np.concatenate(out)


def rank_identity(Sm, self_col, p_ids, g_ids, exclude=True):
    Sm = Sm.copy()
    if exclude:
        Sm[np.arange(len(Sm)), self_col] = -np.inf
    order = np.argsort(g_ids, kind="stable")
    gs = g_ids[order]
    starts = np.r_[0, np.nonzero(gs[1:] != gs[:-1])[0] + 1]
    ids = gs[starts]
    M = np.maximum.reduceat(Sm[:, order], starts, axis=1)
    col = {k: i for i, k in enumerate(ids)}
    true = M[np.arange(len(M)), [col[p] for p in p_ids]]
    return (M > true[:, None]).sum(1) + 1, len(ids)


def train_fold(df, dev, iters, log_every=200):
    enc_a, enc_b = Enc().to(dev), Enc().to(dev)
    opt = torch.optim.AdamW(list(enc_a.parameters()) + list(enc_b.parameters()), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    tau = 0.07
    dl = DataLoader(Pairs(df, True), batch_size=48, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    it, t0 = 0, time.time()
    while it < iters:
        for a, b in dl:
            if it >= iters:
                break
            a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
            za, zb = enc_a(a), enc_b(b)
            logits = zb @ za.T / tau
            lab = torch.arange(len(za), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % log_every == 0 or it == iters:
                print(f"      [h8] iter {it}/{iters} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return enc_a.eval(), enc_b.eval()


def _save(out, rows):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.run.isin(new.run)]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.smoke:
        a.iters = 30
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    assert "fold" in idx.columns, "index has no fold column"
    fold = dict(zip(idx.image_id, idx.fold))
    ident = dict(zip(idx.image_id, idx.identity))
    out = P.results("H8") / "h8.csv"
    runs = [Path(d) for d in a.anon_dir if (Path(d) / "manifest.csv").exists()]
    print(f"[h8] {len(runs)} runs, iters={a.iters}, device {dev}", flush=True)
    for ri, d in enumerate(runs, 1):
        t0 = time.time()
        m = read_manifest(d)
        m = m[m.status != "failed"].reset_index(drop=True)
        m["fold"] = [fold[i] for i in m.image_id]
        m["identity"] = [ident[i] for i in m.image_id]
        tag = set_tag(cfg, d)
        rows, line = [], []
        for tr, te in (("A", "B"), ("B", "A")):
            dtr, dte = m[m.fold == tr].reset_index(drop=True), m[m.fold == te].reset_index(drop=True)
            if len(dtr) < 50 or len(dte) < 20:
                print(f"[h8] {tag[:50]} fold {tr}->{te}: too few images ({len(dtr)}/{len(dte)}), skip", flush=True)
                continue
            print(f"    [h8] {tag[:45]} train fold {tr} (n={len(dtr)}) -> test fold {te} (n={len(dte)})", flush=True)
            ea, eb = train_fold(dtr, dev, a.iters)
            Ge = embed_all(ea, dte.src_path.tolist(), dev)
            Ae = embed_all(eb, dte.anon_path.tolist(), dev)
            Sm = Ae @ Ge.T
            self_col = np.arange(len(dte))
            t = Sm[self_col, self_col]
            src_r1 = float(((Sm > t[:, None]).sum(1) + 1 == 1).mean())
            p_ids = dte.identity.to_numpy()
            keep = pd.Series(p_ids).groupby(p_ids).transform("size").to_numpy() >= 2
            if keep.sum() >= 20:
                rk, nid = rank_identity(Sm[np.ix_(keep, keep)], np.arange(int(keep.sum())), p_ids[keep], p_ids[keep])
                id_r1, chance = float((rk == 1).mean()), 1.0 / nid
            else:
                id_r1, chance, nid = float("nan"), float("nan"), 0
            rows.append(dict(run=tag, train_fold=tr, test_fold=te, n_test=len(dte), n_identities=nid,
                             iters=a.iters, source_r1=src_r1, source_chance=1.0 / len(dte),
                             identity_r1=id_r1, identity_chance=chance))
            line.append(f"{tr}->{te} src {src_r1:.3f} (chance {1.0 / len(dte):.4f}) id {id_r1:.3f}")
        if rows:
            _save(out, rows)
        print(f"[{ri}/{len(runs)}] RESULT {tag[:60]} | " + " | ".join(line) + f" ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
