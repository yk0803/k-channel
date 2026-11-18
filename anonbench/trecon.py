"""T-recon: learned restoration attack on released anonymized images.

Threat model: the attacker holds (original, anonymized) pairs for some identities (e.g. a public
dataset he anonymized himself with the released code) and trains a restorer anon -> original.
He then applies it to released images of OTHER people. Pairs come from one identity-disjoint fold,
the restorer is applied to the other fold (A->B and B->A), so every image is restored by a model
that never saw its identity. The restorer gets no face-recognition loss (no panel model is used).

Output (2-fold union = every non-failed row of the source run):
    work/anon/<ds>/trecon/<method>__<tag>/seed<k>/<split>/{images/,manifest.csv,run_meta.json}
    -> score it with anonbench.score / anonbench.closedset_eval like any anonymizer run.
    results/<ds>/Trecon/trecon.csv (+ <run>_perimage.csv): PSNR/SSIM to the original, anon vs restored.

    python -m anonbench.trecon --anon-dir work/anon/mtf/edials/<tag>/seed0/train [--iters 6000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from .config import Paths, load_config
from .data import load_index
from .manifest import ManifestWriter, _ref_from_source, read_manifest


def log(*a):
    print("[trecon]", *a, flush=True)


# ----------------------------------------------------------------------------- data
def _load_pair(args):
    anon_p, src_p, wh = args
    with Image.open(anon_p) as im:
        im = im.convert("RGB")
        resized = im.size != wh
        if resized:
            im = im.resize(wh, Image.BILINEAR)
        a = np.asarray(im).copy()
    o = _ref_from_source(src_p, wh)          # same reference as ManifestWriter ('resize' kind)
    return a, np.ascontiguousarray(o), resized


def _to_t(a: np.ndarray, dev) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


# ----------------------------------------------------------------------------- model
def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    """Residual U-Net, 4 down-samplings (input side must be a multiple of 16). Starts as identity."""

    def __init__(self, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.inc = _blk(3, ch[0])
        self.downs = nn.ModuleList([_blk(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(len(ch) - 1))])
        self.decs = nn.ModuleList([_blk(2 * ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.out = nn.Conv2d(ch[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = [self.inc(x)]
        for d in self.downs:
            h.append(d(F.max_pool2d(h[-1], 2)))
        y = h.pop()
        for up, dec in zip(self.ups, self.decs):
            y = dec(torch.cat([up(y), h.pop()], 1))
        return x + self.out(y)


def ssim_t(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    g = torch.exp(-(torch.arange(11, device=x.device, dtype=x.dtype) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def f(z):
        return F.conv2d(z, w, groups=3)

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


@torch.no_grad()
def predict(net, A: np.ndarray, dev, bs=8) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(A), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            y = net(_to_t(A[i:i + bs], dev))
        out.append((y.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0,) + A.shape[1:], np.uint8)


def train(A, O, vA, vO, a, dev, label) -> tuple[nn.Module, dict]:
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(200, a.iters // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.iters) / a.iters)))
    N, H, W, _ = A.shape
    crop = min(a.crop or (192 if H <= 256 else 256), H, W) // 16 * 16
    base = float(np.mean([psnr_u8(x, y) for x, y in zip(vA, vO)])) if len(vA) else float("nan")
    best, best_state, t0, run_loss = -1.0, None, time.time(), 0.0
    log(f"{label}: train {N} pairs, val {len(vA)}, {H}x{W}, crop {crop}, iters {a.iters}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M, val PSNR before {base:.2f} dB")
    for it in range(1, a.iters + 1):
        idx = rng.integers(0, N, a.bs)
        ys, xs = rng.integers(0, H - crop + 1, a.bs), rng.integers(0, W - crop + 1, a.bs)
        ba = np.stack([A[i, y:y + crop, x:x + crop] for i, y, x in zip(idx, ys, xs)])
        bo = np.stack([O[i, y:y + crop, x:x + crop] for i, y, x in zip(idx, ys, xs)])
        fl = rng.random(a.bs) < 0.5
        ba[fl], bo[fl] = ba[fl][:, :, ::-1], bo[fl][:, :, ::-1]
        ta, to = _to_t(ba, dev), _to_t(bo, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(ta)
        out = out.float()
        loss = F.l1_loss(out, to) + 0.2 * (1 - ssim_t(out.clamp(0, 1), to))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss)
        if it % a.eval_every == 0 or it == a.iters:
            msg = ""
            if len(vA):
                v = float(np.mean([psnr_u8(x, y) for x, y in zip(predict(net, vA, dev), vO)]))
                if v > best:
                    best, best_state = v, {k: t.detach().clone().cpu() for k, t in net.state_dict().items()}
                msg = f"val PSNR {v:.2f} (best {best:.2f}, before {base:.2f})"
            el = time.time() - t0
            log(f"{label}: it {it}/{a.iters} loss {run_loss / a.eval_every:.4f} {msg} "
                f"| {el / 60:.1f} min, ETA {el / it * (a.iters - it) / 60:.1f} min")
            run_loss = 0.0
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, dict(val_psnr_before=base, val_psnr_best=best, crop=crop, train_min=(time.time() - t0) / 60)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", required=True)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=0, help="0 = 192 (<=256 px images) / 256 (larger)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.1, help="share of training-fold identities for model selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="32 images per fold, 30 iters, output to work/trecon_smoke")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.eval_every = 30, 10
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = Path(a.anon_dir).resolve()
    rel = src.relative_to((P.work / "anon" / a.dataset).resolve()).parts   # method, tag, seedK, split
    assert len(rel) == 4, f"unexpected run dir layout {rel}"
    method, tag, seed_dir, split = rel
    name = f"{method}__{tag}"
    root = P.work / ("trecon_smoke" if a.smoke else "anon") / a.dataset / "trecon"
    out = root / name / seed_dir / split
    if (out / "manifest.csv").exists() and not a.force:
        log(f"exists, skip (use --force): {out}")
        log(f"OUT {out}")
        return

    man = read_manifest(src)
    man = man[man.status != "failed"].reset_index(drop=True)
    idx = load_index(cfg, a.dataset)[["image_id", "fold"]]
    m = man.merge(idx, on="image_id", how="left")
    assert m.fold.notna().all(), "manifest rows missing from index"
    if a.smoke:
        m = m.groupby("fold", group_keys=False).head(32).reset_index(drop=True)
    meta_p = src / "run_meta.json"
    smeta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

    with Image.open(m.anon_path.iloc[0]) as im:
        wh = im.size
    wh = (wh[0] // 16 * 16, wh[1] // 16 * 16)
    log(f"{name}/{split}: loading {len(m)} pairs at {wh[0]}x{wh[1]} (folds {m.fold.value_counts().to_dict()})")
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        pairs = list(ex.map(_load_pair, [(r.anon_path, r.src_path, wh) for r in m.itertuples()]))
    A = np.stack([p[0] for p in pairs])
    O = np.stack([p[1] for p in pairs])
    n_rs = sum(p[2] for p in pairs)
    del pairs
    log(f"loaded in {time.time() - t0:.0f}s, {A.nbytes * 2 / 2**30:.2f} GB, resized-to-common-size {n_rs}")

    R = np.zeros_like(A)
    info = {}
    rng = np.random.default_rng(a.seed)
    for tr, te in (("A", "B"), ("B", "A")):
        trm = (m.fold == tr).to_numpy()
        tem = (m.fold == te).to_numpy()
        ids = np.array(sorted(m.identity[trm].unique()))
        n_val = max(2, int(round(a.val_frac * len(ids)))) if len(ids) >= 10 else 0
        val_ids = set(rng.choice(ids, n_val, replace=False)) if n_val else set()
        vm = trm & m.identity.isin(val_ids).to_numpy()
        fm = trm & ~vm
        net, inf = train(A[fm], O[fm], A[vm], O[vm], a, dev, f"{name}/{split} fold {tr}->{te}")
        R[tem] = predict(net, A[tem], dev)
        info[f"{tr}->{te}"] = dict(n_train=int(fm.sum()), n_val=int(vm.sum()), n_test=int(tem.sum()), **inf)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    log("metrics + writing restored images ...")
    with ThreadPoolExecutor(16) as ex:
        ps_a = list(ex.map(lambda i: psnr_u8(A[i], O[i]), range(len(m))))
        ps_r = list(ex.map(lambda i: psnr_u8(R[i], O[i]), range(len(m))))
        ss_a = list(ex.map(lambda i: ssim_u8(A[i], O[i]), range(len(m))))
        ss_r = list(ex.map(lambda i: ssim_u8(R[i], O[i]), range(len(m))))
    W = ManifestWriter(out, meta=dict(method="trecon", source_run=str(src), source_method=method, iters=a.iters,
                                      folds=info, smoke=a.smoke, source_meta=smeta),
                       geometry=smeta.get("geometry", "resize"))
    for i, r in enumerate(m.itertuples()):
        row = dict(image_id=r.image_id, path=r.src_path, identity=r.identity, label=r.label, split=r.split)
        W.save_image(row, R[i], status="modified", ref=O[i], fold=r.fold, src_status=r.status,
                     psnr_anon=ps_a[i], psnr_rec=ps_r[i])
    W.close()

    per = pd.DataFrame(dict(image_id=m.image_id, identity=m.identity, fold=m.fold, src_status=m.status,
                            psnr_anon=ps_a, psnr_rec=ps_r, ssim_anon=ss_a, ssim_rec=ss_r))
    res = P.work / "trecon_smoke" if a.smoke else P.results("Trecon")
    res.mkdir(parents=True, exist_ok=True)
    per.to_csv(res / f"{name}__{split}_perimage.csv", index=False)
    mod = per[per.src_status == "modified"]
    row = dict(run=f"{name}__{split}", source=str(src), out=str(out), n=len(per), n_modified=len(mod), size=f"{wh[0]}x{wh[1]}",
               psnr_anon=per.psnr_anon.median(), psnr_rec=per.psnr_rec.median(),
               ssim_anon=per.ssim_anon.median(), ssim_rec=per.ssim_rec.median(),
               psnr_anon_mod=mod.psnr_anon.median(), psnr_rec_mod=mod.psnr_rec.median(),
               frac_improved=float((per.psnr_rec > per.psnr_anon).mean()),
               train_min=sum(v["train_min"] for v in info.values()), iters=a.iters)
    csv = res / "trecon.csv"
    old = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    if len(old):
        old = old[old.run != row["run"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(csv, index=False)
    log(f"RESULT {row['run'][:70]} n={row['n']} median PSNR anon {row['psnr_anon']:.2f} -> restored {row['psnr_rec']:.2f} dB, "
        f"SSIM {row['ssim_anon']:.3f} -> {row['ssim_rec']:.3f}, improved {row['frac_improved']:.1%} "
        f"(modified only: {row['psnr_anon_mod']:.2f} -> {row['psnr_rec_mod']:.2f})")
    log(f"OUT {out}")


if __name__ == "__main__":
    main()
