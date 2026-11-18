"""H16: adversary scaling for the linkage attack.

H8 used two 1.5M-parameter CNNs trained from scratch. A real attacker would start from a strong
pretrained backbone. This repeats the H8 protocol with a frozen DINOv2 backbone plus a trained
projection head on each branch, so the attacker inherits general visual features and only learns
the clean<->released correspondence. Identity-disjoint folds, same metric, so numbers are directly
comparable to h8.csv.

If a stronger adversary beats a bound that the weak one respected, the bound is wrong.

    python -m anonbench.h16_strong_linkage --dataset mtf --anon-dir D [D ...] [--iters 3000]
      -> work/results/<ds>/H16/strong_linkage.csv
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
from .h8 import rank_identity
from .manifest import read_manifest

R = 224
MU = (0.485, 0.456, 0.406)
SD = (0.229, 0.224, 0.225)


def _read(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize((R, R), Image.BILINEAR))


class Pairs(Dataset):
    def __init__(self, df, train):
        self.src, self.anon, self.train = df.src_path.tolist(), df.anon_path.tolist(), train

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        a, b = _read(self.src[i]), _read(self.anon[i])
        if self.train and np.random.rand() < 0.5:
            a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
        t = lambda x: ((torch.from_numpy(x).permute(2, 0, 1).float() / 255.0)
                       - torch.tensor(MU).view(3, 1, 1)) / torch.tensor(SD).view(3, 1, 1)
        return t(a), t(b)


class Head(nn.Module):
    def __init__(self, backbone, d_in, d=256):
        super().__init__()
        self.b = backbone
        self.p = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d))

    def forward(self, x):
        with torch.no_grad():
            f = self.b(pixel_values=x).pooler_output
        return F.normalize(self.p(f.float()), dim=1)


def make_pair(dev):
    from transformers import AutoModel
    b = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    for p in b.parameters():
        p.requires_grad = False
    d = b.config.hidden_size
    return Head(b, d).to(dev), Head(b, d).to(dev)


@torch.no_grad()
def embed_all(head, paths, dev, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        X = np.stack([_read(p) for p in paths[i:i + bs]])
        x = ((torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float() / 255.0)
             - torch.tensor(MU, device=dev).view(1, 3, 1, 1)) / torch.tensor(SD, device=dev).view(1, 3, 1, 1)
        out.append(head(x).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(df, dev, iters, log_every=200):
    ea, eb = make_pair(dev)
    opt = torch.optim.AdamW(list(ea.p.parameters()) + list(eb.p.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    dl = DataLoader(Pairs(df, True), batch_size=32, shuffle=True, num_workers=8, drop_last=True,
                    persistent_workers=True)
    it, t0 = 0, time.time()
    while it < iters:
        for a, b in dl:
            if it >= iters:
                break
            a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
            za, zb = ea(a), eb(b)
            logits = zb @ za.T / 0.07
            lab = torch.arange(len(za), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % log_every == 0 or it == iters:
                print(f"      [h16] iter {it}/{iters} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return ea.eval(), eb.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.smoke:
        a.iters = 20
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = load_index(cfg, a.dataset).reset_index(drop=True)
    fold = dict(zip(idx.image_id, idx.fold))
    ident = dict(zip(idx.image_id, idx.identity))
    out = P.results("H16") / "strong_linkage.csv"
    runs = [Path(d) for d in a.anon_dir if (Path(d) / "manifest.csv").exists()]
    print(f"[h16] {len(runs)} runs, DINOv2-small frozen + trained head, iters={a.iters}, {dev}", flush=True)
    rows = []
    for ri, d in enumerate(runs, 1):
        t0 = time.time()
        m = read_manifest(d)
        m = m[m.status != "failed"].reset_index(drop=True)
        m["fold"] = [fold[i] for i in m.image_id]
        m["identity"] = [ident[i] for i in m.image_id]
        tag = set_tag(cfg, d)
        line = []
        for tr, te in (("A", "B"), ("B", "A")):
            dtr, dte = m[m.fold == tr].reset_index(drop=True), m[m.fold == te].reset_index(drop=True)
            if len(dtr) < 50 or len(dte) < 20:
                print(f"[h16] {tag[:50]} {tr}->{te}: too few ({len(dtr)}/{len(dte)}), skip", flush=True)
                continue
            print(f"    [h16] {tag[:45]} train {tr} (n={len(dtr)}) -> test {te} (n={len(dte)})", flush=True)
            ea, eb = train_fold(dtr, dev, a.iters)
            Ge = embed_all(ea, dte.src_path.tolist(), dev)
            Ae = embed_all(eb, dte.anon_path.tolist(), dev)
            S = Ae @ Ge.T
            self_col = np.arange(len(dte))
            t = S[self_col, self_col]
            src_r1 = float(((S > t[:, None]).sum(1) + 1 == 1).mean())
            p_ids = dte.identity.to_numpy()
            keep = pd.Series(p_ids).groupby(p_ids).transform("size").to_numpy() >= 2
            if keep.sum() >= 20:
                rk, nid = rank_identity(S[np.ix_(keep, keep)], np.arange(int(keep.sum())),
                                        p_ids[keep], p_ids[keep])
                id_r1, chance = float((rk == 1).mean()), 1.0 / nid
            else:
                id_r1, chance = float("nan"), float("nan")
            rows.append(dict(run=tag, backbone="dinov2_small", train_fold=tr, test_fold=te,
                             n_test=len(dte), iters=a.iters, source_r1=src_r1,
                             source_chance=1.0 / len(dte), identity_r1=id_r1, identity_chance=chance))
            line.append(f"{tr}->{te} src {src_r1:.3f} (chance {1.0 / len(dte):.4f}) id {id_r1:.3f}")
            old = pd.read_csv(out) if out.exists() else pd.DataFrame()
            pd.concat([old[~old.run.isin([tag])] if len(old) else old, pd.DataFrame(rows)],
                      ignore_index=True).drop_duplicates(
                          subset=["run", "train_fold", "test_fold"], keep="last").to_csv(out, index=False)
        print(f"[{ri}/{len(runs)}] RESULT-H16 {tag[:60]} | " + " | ".join(line)
              + f" ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
