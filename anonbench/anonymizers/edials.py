"""EDIALS (Khan et al., CODASPY'26) - adapter around the original code.

* `train_aae`: port of 'Train AAE.ipynb' (ConvNeXt-Base encoder, 1x1 bottleneck, the repo's
  ConvNextDecoder / Discriminator, hinge + gradient-penalty latent adversary). Saved as the
  same {'encoder','bottleneck','decoder','discriminator'} dict that run_ilore_iterative.py loads.
* `run`: re-implementation of utils/generate_iterative.generate_adversarial_with_ilore that
  uses the ORIGINAL ILOREM explainer from the repo, with three bookkeeping changes:
  outputs keep the source file name (the original saves 'adversarial_<c>_to_<c'>_<uuid>.png'),
  the status of every image is logged (incl. originals released unmodified), and a
  `select='random'` control (E4) that distorts the same NUMBER of latent features for the
  same NUMBER of iterations, chosen uniformly at random, with no black box in the loop.
"""
from __future__ import annotations

import gc
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from ..config import Paths, add_third_party_to_path
from ..manifest import ManifestWriter
from ..models.classifiers import PathDataset, load_classifier, set_seed

IMG_SIZE = 256


def _tf():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


def _repo(cfg):
    p = add_third_party_to_path(cfg, "EDIALS")
    import utils.decoder as dec  # noqa: F401  (their package; needs repo root on sys.path)
    return p


# ----------------------------------------------------------------------------- AAE training
def train_aae(cfg, dataset, train_df: pd.DataFrame, val_df: pd.DataFrame, name="aae", seed=0) -> Path:
    _repo(cfg)
    from utils.decoder import ConvNextDecoder, Discriminator, extract_encoder_features
    out = Paths(cfg, dataset).model_dir(f"edials_{name}", seed)
    if (out / "aae.pt").exists():
        print(f"[skip] {out}/aae.pt exists")
        return out / "aae.pt"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    r = cfg["training"]["edials_aae"]
    smoke = cfg.get("smoke", False)
    cb = models.convnext_base(weights=None if smoke else models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(cb.children())[:-2]).to(dev)
    bottleneck = nn.Conv2d(1024, 1024, kernel_size=1).to(dev)
    decoder = ConvNextDecoder().to(dev)
    disc = Discriminator().to(dev)
    tf = _tf()
    dl = DataLoader(PathDataset(train_df.path.tolist(), [0] * len(train_df), tf), batch_size=r["batch_size"],
                    shuffle=True, num_workers=cfg["hardware"]["num_workers"], drop_last=True)
    dlv = DataLoader(PathDataset(val_df.path.tolist(), [0] * len(val_df), tf), batch_size=r["batch_size"],
                     shuffle=False, num_workers=cfg["hardware"]["num_workers"])
    oe = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    od = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    odisc = torch.optim.Adam(disc.parameters(), lr=5e-5)
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(odisc, T_max=50)
    se = torch.optim.lr_scheduler.CosineAnnealingLR(oe, T_max=50)
    adv_w0, gp_w, n_critic = 0.0005, 5, 5

    def gp(real, fake):
        a = torch.rand(real.size(0), 1, 1, 1, device=dev)
        it = (a * real + (1 - a) * fake).requires_grad_(True)
        d = disc(it)
        g = torch.autograd.grad(d, it, torch.ones_like(d), create_graph=True, retain_graph=True)[0]
        return ((g.reshape(g.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    def hinge(rp, fp):
        return F.relu(1.0 - rp).mean() + F.relu(1.0 + fp).mean()

    best, bad = float("inf"), 0
    E = r["epochs"] if not smoke else 1
    t0 = time.time()
    print(f"[AAE {name}] train={len(train_df)} val={len(val_df)} epochs<={E} patience={r['patience']} device={dev}", flush=True)
    for ep in range(E):
        te = time.time()
        encoder.train(); decoder.train(); disc.train()
        rec_tot = 0.0
        for bi, (x, _) in enumerate(tqdm(dl, desc=f"AAE ep{ep}", leave=False)):
            if ep == 0 and bi % 50 == 0:
                print(f"[AAE {name}] epoch 1 batch {bi}/{len(dl)} ({time.time() - te:.0f}s)", flush=True)
            x = x.to(dev)
            for _ in range(n_critic):
                disc.zero_grad()
                with torch.no_grad():
                    _, z = extract_encoder_features(encoder, x)
                    z = bottleneck(z)
                real = torch.randn_like(z)
                nz = 0.01 * torch.randn_like(z)
                z, real = z + nz, real + nz
                dl_ = hinge(disc(real), disc(z.detach())) + gp_w * gp(real, z.detach())
                dl_.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.5)
                odisc.step()
            encoder.zero_grad(); decoder.zero_grad()
            feats, z = extract_encoder_features(encoder, x)
            z = bottleneck(z)
            rec = decoder(z, feats)
            rl = F.mse_loss(rec, x)
            loss = rl + adv_w0 * (ep / max(E, 1)) * (-disc(z).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            oe.step(); od.step()
            rec_tot += rl.item()
        sd.step(); se.step()
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            vl = [F.mse_loss(decoder(bottleneck(extract_encoder_features(encoder, x.to(dev))[1]),
                                     extract_encoder_features(encoder, x.to(dev))[0]), x.to(dev)).item()
                  for x, _ in dlv]
        v = float(np.mean(vl)) if vl else rec_tot
        print(f"[AAE {name}] epoch {ep + 1}/{E} train_rec {rec_tot / max(len(dl), 1):.5f} val_rec {v:.5f} "
              f"(best {min(best, v):.5f}, {time.time() - te:.0f}s/epoch, {(time.time() - t0) / 60:.1f} min total)", flush=True)
        if v < best:
            best, bad = v, 0
            torch.save(dict(encoder=encoder.cpu(), bottleneck=bottleneck.cpu(), decoder=decoder.cpu(),
                            discriminator=disc.cpu()), out / "aae.pt")
            encoder.to(dev); bottleneck.to(dev); decoder.to(dev); disc.to(dev)
        else:
            bad += 1
            if bad >= r["patience"]:
                print(f"[AAE {name}] early stop at epoch {ep + 1}", flush=True)
                break
    print(f"[AAE {name}] DONE best val_rec {best:.5f} in {(time.time() - t0) / 60:.1f} min -> {out / 'aae.pt'}", flush=True)
    return out / "aae.pt"


# ----------------------------------------------------------------------------- anonymization
def _load(cfg, aae_path, guide_dir):
    _repo(cfg)
    from utils.decoder import AutoencoderWrapper
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    d = torch.load(aae_path, map_location=dev, weights_only=False)
    wrapper = AutoencoderWrapper(encoder=d["encoder"].to(dev).eval(), bottleneck=d["bottleneck"].to(dev).eval(),
                                 decoder=d["decoder"].to(dev).eval(), discriminator=d["discriminator"].to(dev).eval(),
                                 latent_dim=1024 * 8 * 8)
    bb = None
    if guide_dir is not None:
        bb, _, _ = load_classifier(guide_dir, cfg)
    return wrapper, bb, dev


def _one_image(image, label, bb, wrapper, params, dev, select, budget_row, rng):
    """Returns (final_image[3,H,W] in [0,1], status, n_iterations, extra)."""
    from utils.decoder import extract_encoder_features
    from utils.ilorem import ILOREM
    alpha = params.get("alpha", 0.5)
    max_it = params.get("max_iterations", 100000)

    def bb_predict(images):
        if isinstance(images, np.ndarray):
            if images.shape[-1] == 3:
                images = np.transpose(images, (0, 3, 1, 2))
            images = torch.tensor(images, dtype=torch.float32).to(dev)
        with torch.no_grad():
            return bb(images).argmax(1).cpu().numpy()

    original_class = int(label)
    if select == "rules":
        if bb_predict(image.unsqueeze(0))[0] != original_class:
            return image, "unmodified", 0, dict(reason="guide_already_misclassifies")
        explainer = ILOREM(bb_predict=bb_predict, class_name="class", class_values=list(range(params["_n_classes"])),
                           neigh_type="rnd", autoencoder=wrapper, use_rgb=True, verbose=False)
        exp = explainer.explain_instance(image.permute(1, 2, 0).cpu().numpy(), num_samples=params.get("num_samples", 240))
        if not exp.crules:
            return image, "unmodified", 0, dict(reason="no_counterfactual_rules")
        feats = []
        for delta in exp.deltas:
            for c in delta:
                fi = int(c.att.split("_")[1])
                if not any(f[0] == fi for f in feats):
                    feats.append((fi, c.op))
        if not feats:  # B2: original 'No features to distort in rules. Saving original image.' (returns False)
            return image, "unmodified", 0, dict(reason="no_features")
    else:  # random control: same number of features, random dims and directions
        if budget_row is None or budget_row["status"] == "unmodified":
            return image, "unmodified", 0, dict(reason="budget_unmodified")
        n_feat = int(budget_row["extra"].get("n_features", 0))
        feats = [(int(fi), ">" if rng.random() < 0.5 else "<=")
                 for fi in rng.choice(1024 * 8 * 8, size=max(n_feat, 1), replace=False)]
        max_it = int(budget_row["n_steps"])

    latent = wrapper.encode(image.unsqueeze(0).cpu().numpy())[0]
    cur = latent.copy()
    with torch.no_grad():
        enc_feats, _ = extract_encoder_features(wrapper.encoder, image.unsqueeze(0).to(dev))
    ob = torch.mean(image).item()
    on = np.linalg.norm(latent)
    feats = sorted(feats, key=lambda f: abs(latent[f[0]]), reverse=True)
    levels = {f[0]: 0 for f in feats}
    nF = len(feats)
    it_done, misclassified, img_out = 0, False, None
    for it in range(max_it + (1 if select == "random" else 0)):
        n = np.linalg.norm(cur)
        if n > 0:
            cur = cur * (on / n)
        z = torch.tensor(cur.reshape(1, 1024, 8, 8), dtype=torch.float32, device=dev)
        with torch.no_grad():
            img = wrapper.decoder(z, enc_feats)
            cb = torch.mean(img.squeeze(0)).item()
            if cb > 0:
                img = torch.clamp(img * max(min(ob / cb, 2.0), 0.5), 0, 1)
        img_out = img
        if select == "rules":
            if bb_predict(img)[0] != original_class:
                misclassified = True
                break
        elif it >= max_it:
            break
        fi = feats[it][0] if it < nF else feats[(it - nF) % nF][0]
        levels[fi] = 1 if it < nF else levels[fi] + 1
        for f, op in feats:
            if levels[f] > 0:
                cur[f] += alpha if op == ">" else -alpha
        it_done += 1
        if it % 5 == 0:
            gc.collect()
    extra = dict(n_features=nF, misclassified_by_guide=bool(misclassified) if select == "rules" else None,
                 reason=("reconstruction_only" if it_done == 0 else "latent_distortion"))
    return img_out.squeeze(0).cpu(), "modified", it_done, extra


def run(cfg, rows: pd.DataFrame, params: dict, out_dir, guide_dir, aae_path, seed=0, select="rules",
        budget: pd.DataFrame | None = None):
    wrapper, bb, dev = _load(cfg, aae_path, guide_dir if select == "rules" else None)
    W = ManifestWriter(out_dir, meta=dict(method="edials", params=params, seed=seed, select=select,
                                          guide=str(guide_dir), aae=str(aae_path)))
    bud = budget.set_index("image_id") if budget is not None else None
    if bb is not None:
        with torch.no_grad():
            params = {**params, "_n_classes": int(bb(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=dev)).shape[1])}
    tf = _tf()
    t0 = time.time()
    stats = dict(total=0, unmodified=0, misclassified=0, unmodified_reasons={})
    for k, (_, r) in enumerate(tqdm(rows.iterrows(), total=len(rows), desc=f"edials[{select}]")):
        s = int.from_bytes(hashlib.sha256(f"{r['image_id']}|{seed}".encode()).digest()[:4], "little")  # per-image, shard-independent
        random.seed(s); np.random.seed(s); torch.manual_seed(s)  # the repo seeds 42 globally once
        rng = np.random.default_rng(s)
        with Image.open(r["path"]) as im:
            image = tf(im.convert("RGB")).to(dev)
        brow = bud.loc[r["image_id"]] if (bud is not None and r["image_id"] in bud.index) else None
        try:
            out, status, n_it, extra = _one_image(image, r["label"], bb, wrapper, params, dev, select, brow, rng)
        except Exception as e:  # keep going, but record it
            print(f"[edials] {r['image_id']} failed: {e!r}")
            W.add(r, None, "failed", 0, error=repr(e))
            continue
        arr = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        ref = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        W.save_image(r, arr, status, n_it, ref=ref, **extra)
        stats["total"] += 1
        stats["unmodified"] += status == "unmodified"
        if status == "unmodified":
            rs = extra.get("reason", "?")
            stats["unmodified_reasons"][rs] = stats["unmodified_reasons"].get(rs, 0) + 1
        stats["misclassified"] += bool(extra.get("misclassified_by_guide"))
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            el = time.time() - t0
            print(f"[edials {select}] {k + 1}/{len(rows)} | unmodified {stats['unmodified']} {stats['unmodified_reasons']} "
                  f"| last n_iter {n_it} | {el / (k + 1):.1f} s/img, ETA {el / (k + 1) * (len(rows) - k - 1) / 60:.0f} min", flush=True)
        if dev.type == "cuda" and k % 20 == 0:
            torch.cuda.empty_cache()
    W.meta.update(stats=stats, minutes=(time.time() - t0) / 60)
    return W.close()
