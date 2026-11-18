"""Closed-set classifiers: guide replicas, held-out FR adversaries, FaceNet classifier, utility models.

Recipes mirror the original notebooks (see AUDIT.md):
  convnext_base_ediAnon : EDI-Anon  Utils/MTF_classification_five_models.ipynb  (Resize 224, no norm, Adam 1e-5,
                          StepLR(20, 0.001), early stop on TRAIN loss, patience 5)
  convnext_base_edials  : EDIALS   'Train black box.ipynb'                       (Resize 256, no norm, Adam 1e-4, patience 3)
  facenet_vggface2_clf  : BlendPose 'FaceNet FR adversary' (InceptionResNetV1-VGGFace2 fine-tuned on MTF)
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- data
class PathDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(self.labels[i])


class _FaceNetNorm:
    def __call__(self, t):
        return (t * 255.0 - 127.5) / 128.0


def build_transform(img_size: int, normalize, augment: bool = False):
    t = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    if augment:
        t = [transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
             transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1)]
    t.append(transforms.ToTensor())
    if normalize == "imagenet":
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize == "facenet":
        t.append(_FaceNetNorm())
    return transforms.Compose(t)


# ----------------------------------------------------------------------------- architectures
def build_arch(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    W = (lambda w: w if pretrained else None)
    if arch == "convnext_base":
        m = models.convnext_base(weights=W(models.ConvNeXt_Base_Weights.IMAGENET1K_V1))
        m.classifier[2] = nn.Linear(1024, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=W(models.ResNet50_Weights.IMAGENET1K_V2))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=W(models.ResNet18_Weights.IMAGENET1K_V1))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=W(models.ViT_B_16_Weights.IMAGENET1K_V1))
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif arch == "swin_t":
        m = models.swin_t(weights=W(models.Swin_T_Weights.IMAGENET1K_V1))
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=W(models.EfficientNet_V2_S_Weights.IMAGENET1K_V1))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "facenet_vggface2":
        from facenet_pytorch import InceptionResnetV1
        m = InceptionResnetV1(pretrained="vggface2" if pretrained else None, classify=True, num_classes=num_classes)
    elif arch == "tiny_cnn":  # smoke tests only
        m = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
    else:
        raise KeyError(arch)
    return m


def recipe_for(cfg: dict, model_name: str) -> dict:
    """Map an evaluator/model name to its training recipe dict."""
    tr = cfg["training"]
    if model_name in tr:
        r = dict(tr[model_name])
    elif model_name.endswith("_heldout"):
        r = dict(tr["heldout"], arch=model_name[: -len("_heldout")])
    elif model_name.startswith("utility"):
        r = dict(tr["utility"])
    else:
        raise KeyError(f"no training recipe for {model_name}")
    r.setdefault("optimizer", "adam")
    r.setdefault("schedule", "step")
    r.setdefault("pretrained", cfg.get("training_pretrained", True))
    if cfg.get("smoke"):
        r.update(arch="tiny_cnn", pretrained=False, max_epochs=min(r.get("max_epochs", 2), 2),
                 img_size=min(r.get("img_size", 64), 64))
    return r


# ----------------------------------------------------------------------------- train / load / predict
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(cfg):
    d = cfg["hardware"]["device"]
    return torch.device("cuda" if d == "cuda" and torch.cuda.is_available() else "cpu")


def train_classifier(cfg: dict, recipe: dict, train_paths, train_labels, val_paths, val_labels,
                     num_classes: int, out_dir: Path, seed: int = 0, class_names=None, tag: str = "") -> Path:
    """Train and save `model.pt` + `meta.json` in out_dir. Returns out_dir. Skips if already trained."""
    out_dir = Path(out_dir)
    if (out_dir / "meta.json").exists():
        print(f"[skip] {out_dir} already trained")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = _device(cfg)
    print(f"[train] START {tag}: arch={recipe['arch']} img={recipe['img_size']} train={len(train_paths)} "
          f"val={len(val_paths)} classes={num_classes} max_epochs={recipe['max_epochs']} device={dev}", flush=True)
    m = build_arch(recipe["arch"], num_classes, recipe.get("pretrained", True)).to(dev)
    tf_tr = build_transform(recipe["img_size"], recipe["normalize"], recipe.get("augment", False))
    tf_ev = build_transform(recipe["img_size"], recipe["normalize"], False)
    nw = cfg["hardware"]["num_workers"]
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(PathDataset(train_paths, train_labels, tf_tr), batch_size=recipe["batch_size"],
                       shuffle=True, num_workers=nw, pin_memory=True, generator=g, drop_last=len(train_paths) > recipe["batch_size"])
    dl_va = DataLoader(PathDataset(val_paths, val_labels, tf_ev), batch_size=cfg["hardware"]["eval_batch_size"],
                       shuffle=False, num_workers=nw) if len(val_paths) else None
    if recipe["optimizer"] == "adamw":
        opt = torch.optim.AdamW(m.parameters(), lr=recipe["lr"], weight_decay=0.05)
    else:
        opt = torch.optim.Adam(m.parameters(), lr=recipe["lr"])
    E = recipe["max_epochs"]
    if recipe["schedule"] == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    else:
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=recipe.get("step_size", 20), gamma=recipe.get("gamma", 0.1))
    crit = nn.CrossEntropyLoss()
    use_amp = cfg["hardware"].get("amp", False) and dev.type == "cuda" and recipe["optimizer"] == "adamw"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad, prev_loss, hist = -1.0, None, 0, math.inf, []
    # faithful recipes (EDI-Anon/EDIALS notebooks) early-stop on TRAINING loss; held-out ones on val acc
    # B10: explicit key; the old inference from the optimizer name is kept only as the default
    stop_on_train = recipe.get("early_stop", "train_loss" if recipe["optimizer"] == "adam" else "val_acc") == "train_loss"
    t0 = time.time()
    for ep in range(E):
        te = time.time()
        m.train()
        tot, n = 0.0, 0
        for x, y in tqdm(dl_tr, desc=f"{tag} ep{ep}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = crit(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        sch.step()
        tr_loss = tot / max(n, 1)
        va = evaluate_accuracy(m, dl_va, dev) if dl_va is not None else float("nan")
        hist.append(dict(epoch=ep, train_loss=tr_loss, val_acc=va))
        el = time.time() - t0
        print(f"[{tag}] epoch {ep + 1}/{E} train_loss {tr_loss:.4f} val_acc {va:.4f} "
              f"({time.time() - te:.0f}s/epoch, {el / 60:.1f} min elapsed)", flush=True)
        if stop_on_train:
            if tr_loss >= prev_loss:
                bad += 1
            else:
                bad, prev_loss = 0, tr_loss
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            best = va
        else:
            if va > best:
                best, bad = va, 0
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
        if bad >= recipe.get("patience", 5):
            print(f"[{tag}] early stop at epoch {ep + 1}", flush=True)
            break
    torch.save(best_state, out_dir / "model.pt")
    # training-set accuracy of the saved weights (eval transform): the F3 'memorised guide' premise
    m.load_state_dict(best_state)
    dl_tr_eval = DataLoader(PathDataset(train_paths, train_labels, tf_ev), batch_size=cfg["hardware"]["eval_batch_size"],
                            shuffle=False, num_workers=nw)
    train_acc = evaluate_accuracy(m, dl_tr_eval, dev)
    meta = dict(recipe=recipe, num_classes=int(num_classes), seed=int(seed), val_acc=float(best), train_acc=float(train_acc),
                history=hist,
                class_names=list(class_names) if class_names is not None else None,
                train_size=len(train_paths), minutes=(time.time() - t0) / 60, tag=tag)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    print(f"[train] DONE {tag}: train_acc {train_acc:.4f} val_acc {best:.4f} in {meta['minutes']:.1f} min -> {out_dir}",
          flush=True)
    return out_dir


def _jsonable(o):
    """numpy scalars/arrays -> plain Python for json.dump (the int64 crash of 10-27)."""
    return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))


@torch.no_grad()
def evaluate_accuracy(m, dl, dev):
    m.eval()
    c = n = 0
    for x, y in dl:
        p = m(x.to(dev)).argmax(1).cpu()
        c += (p == y).sum().item()
        n += len(y)
    return c / max(n, 1)


def load_classifier(model_dir: Path, cfg: dict):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    r = meta["recipe"]
    m = build_arch(r["arch"], meta["num_classes"], pretrained=False)
    m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    m = m.to(_device(cfg)).eval()
    return m, build_transform(r["img_size"], r["normalize"], False), meta


@torch.no_grad()
def predict_logits(m, tf, paths, cfg, bs=None) -> np.ndarray:
    dev = next(m.parameters()).device
    dl = DataLoader(PathDataset(paths, [0] * len(paths), tf), batch_size=bs or cfg["hardware"]["eval_batch_size"],
                    shuffle=False, num_workers=cfg["hardware"]["num_workers"])
    out = []
    for x, _ in dl:
        out.append(m(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
