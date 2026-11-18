"""EDI-Anon (Haffar et al., TDP 2025) - faithful port of Iterative/EDI_Anon.ipynb.

Differences from the notebook are limited to bookkeeping:
  * outputs keep the source file name (the notebook saves '<label>-<i>-K..-q...jpg', losing the mapping);
  * the per-iteration JPEG write/read is done in memory (byte-identical to imwrite/imread);
  * per-image step counts and 'released unmodified' status are logged in the manifest.

The perturbation functions are imported from the original repository. Behaviours that are
reproduced on purpose (configurable): JPEG round trip each iteration, the RGB/BGR channel
mismatch between the noise and the base image (`replicate_channel_swap`), min-max
re-normalisation of images the guide already misclassifies.

Random-mask control (E4): `random_mask=True, budget={image_id: n_steps}` applies the same
perturbation the same number of times per image, but on a uniformly random mask of the same
size (IDPS fraction), with no gradient and no guide at all.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from ..manifest import ManifestWriter
from ..models.classifiers import load_classifier
from .common import edi_methods, jpeg_roundtrip, to_u8


def _load_224(path, size):
    from PIL import Image
    from torchvision import transforms
    with Image.open(path) as im:
        im = transforms.Resize(size)(im.convert("RGB"))
        return np.asarray(im).copy()


def _perturb(x_t: torch.Tensor, p: dict, M) -> np.ndarray:
    """x_t: [1,3,H,W] in [0,1] -> HWC float in [0,1] (the original functions)."""
    k = p["perturbation"]
    if k == "blur":
        return M.create_blurred_image(x_t, p["blur_kernel"])
    if k == "pixelate":
        return M.create_pixelated_image(x_t, p["pixel_size"])
    if k == "dp_pix":
        return M.create_dp_pixelated_image(x_t, p["b"], p["m"], p["eps"])
    if k == "dp_blur":
        return M.create_dp_blurred_image(x_t, p["b0"], p["m"], p["eps"], p["k"], p["sigma"])
    raise KeyError(k)


def _mask_from_grad(grad: np.ndarray, q: float) -> np.ndarray:
    """Utils.optimize_gradients: binarise the SIGNED gradient at its q-th percentile -> HWC mask."""
    thr = np.percentile(grad, [q])
    m = (grad >= thr).astype(np.float32)
    return np.transpose(m, (1, 2, 0))


def _random_mask(shape_chw, q, rng) -> np.ndarray:
    n = int(np.prod(shape_chw))
    k = int(round(n * (100 - q) / 100.0))
    flat = np.zeros(n, np.float32)
    flat[rng.choice(n, k, replace=False)] = 1.0
    return np.transpose(flat.reshape(shape_chw), (1, 2, 0))


def _apply(state_u8: np.ndarray, mask_hwc: np.ndarray, p: dict, M, replicate_swap: bool) -> np.ndarray:
    x = torch.from_numpy(state_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    perturbed = _perturb(x, p, M)                                 # HWC, RGB order
    img1 = state_u8.astype(np.float32) / 255.0                    # calculate_noise: x - perturbed (RGB)
    noise = img1 - perturbed
    upd = noise * mask_hwc
    base = img1[:, :, ::-1] if replicate_swap else img1           # notebook: cvtColor(BGR2RGB) of an RGB array
    anon = (base - upd) * 255.0                                   # -> BGR base minus RGB-ordered noise
    anon = to_u8(anon)
    rgb = anon[:, :, ::-1] if replicate_swap else anon            # imwrite treats it as BGR
    return jpeg_roundtrip(np.ascontiguousarray(rgb))


def _minmax(state_u8: np.ndarray) -> np.ndarray:
    a = state_u8.astype(np.float32)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6) * 255.0
    return jpeg_roundtrip(to_u8(a))


@torch.no_grad()
def _predict(model, states, dev, bs=32):
    preds, logits = [], []
    for i in range(0, len(states), bs):
        x = torch.from_numpy(np.stack(states[i:i + bs])).permute(0, 3, 1, 2).float().div(255).to(dev)
        lo = model(x)
        logits.append(lo.float().cpu())
        preds.append(lo.argmax(1).cpu())
    return torch.cat(preds).numpy(), torch.cat(logits)


def run(cfg, rows: pd.DataFrame, params: dict, out_dir, guide_dir=None, seed=0, random_mask=False,
        budget: dict | None = None):
    M = edi_methods(cfg)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # DP perturbations use np.random.laplace
    q = params.get("q", 90)
    replicate_swap = params.get("replicate_channel_swap", True)
    size = params.get("img_size", 224)
    rows = rows.reset_index(drop=True)
    labels = rows.label.values
    states = [_load_224(p, size) for p in tqdm(rows.path, desc="load")]
    originals = [s.copy() for s in states]          # B3: exact reference for the measured distance
    shapes = {s.shape for s in states}
    if len(shapes) > 1:
        raise ValueError(f"EDI-Anon batching needs square crops; got shapes {list(shapes)[:3]}")
    steps = np.zeros(len(rows), int)
    ever = np.zeros(len(rows), bool)
    W = ManifestWriter(out_dir, meta=dict(method="edi_anon", params=params, seed=seed, random_mask=random_mask,
                                          guide=str(guide_dir)))
    t0 = time.time()
    history = []

    if random_mask:  # ---------------- E4 control: no guide, same per-image budget
        assert budget is not None, "random_mask needs the per-image budget of the targeted run"
        for i in tqdm(range(len(rows)), desc="edi_anon[random]"):
            n = int(budget.get(rows.image_id[i], 0))
            for _ in range(n):
                mask = _random_mask((3,) + states[i].shape[:2], q, rng)
                states[i] = _apply(states[i], mask, params, M, replicate_swap)
            if n == 0:
                states[i] = _minmax(states[i])
            steps[i] = n
            if (i + 1) % 250 == 0 or i + 1 == len(rows):
                print(f"[edi_anon random] {i + 1}/{len(rows)} images, {(time.time() - t0) / 60:.1f} min", flush=True)
        for i, r in rows.iterrows():
            W.save_image(r, states[i], "modified" if steps[i] > 0 else "unmodified", steps[i], ref=originals[i])
        W.meta["minutes"] = (time.time() - t0) / 60
        return W.close()

    model, _, _ = load_classifier(guide_dir, cfg)
    dev = next(model.parameters()).device
    crit = nn.CrossEntropyLoss(reduction="sum")
    pred, logits = _predict(model, states, dev)
    rir = float((pred == labels).mean() * 100)
    history.append(dict(iteration=-1, rir=rir))
    print(f"[edi_anon] {len(rows)} images, guide {guide_dir}, initial guide RIR = {rir:.2f}%", flush=True)
    bs = 16
    # B4: the notebook's cap is range(0, 200000); the paper stops on the RIR threshold. Keep the cap
    # that high by default and record whether it was hit instead of silently stopping early.
    max_iters = int(params.get("max_iters", 200000))
    cap_hit = False
    t_it = time.time()
    for it in range(max_iters):
        # notebook: every image is visited; perturb only if still correctly classified,
        # else save it "as it is" (min-max normalised)
        correct = np.where(pred == labels)[0]
        wrong = np.where(pred != labels)[0]
        for i in wrong:  # the notebook re-normalises every misclassified image at every iteration
            states[i] = _minmax(states[i])
        for b in tqdm(range(0, len(correct), bs), desc=f"edi_anon it{it}", leave=False):
            ids = correct[b:b + bs]
            x = torch.from_numpy(np.stack([states[i] for i in ids])).permute(0, 3, 1, 2).float().div(255).to(dev)
            x.requires_grad_(True)
            lo = model(x)
            y2 = torch.topk(logits[ids], 2, dim=1).indices[:, 1].to(dev)   # get_second_largest
            model.zero_grad(set_to_none=True)
            crit(lo, y2).backward()                                        # per-sample grads (no BatchNorm)
            g = x.grad.detach().cpu().numpy()
            for j, i in enumerate(ids):
                mask = _mask_from_grad(g[j], q)
                states[i] = _apply(states[i], mask, params, M, replicate_swap)
                steps[i] += 1
                ever[i] = True
        pred, logits = _predict(model, states, dev)
        rir = float((pred == labels).mean() * 100)
        history.append(dict(iteration=it, rir=rir, n_perturbed=len(correct)))
        print(f"[edi_anon] iteration {it + 1}: perturbed {len(correct)} images, guide RIR = {rir:.2f}% "
              f"(target < {params['rir_threshold']}%) | {time.time() - t_it:.0f}s, {(time.time() - t0) / 60:.1f} min total", flush=True)
        t_it = time.time()
        if rir < params["rir_threshold"]:
            break
    else:
        cap_hit = True
        print(f"[edi_anon] WARNING: iteration cap {max_iters} hit before guide RIR < {params['rir_threshold']}%")
    for i, r in rows.iterrows():
        W.save_image(r, states[i], "modified" if ever[i] else "unmodified", steps[i], ref=originals[i],
                     guide_correct_at_end=bool(pred[i] == labels[i]))
    W.meta.update(history=history, final_guide_rir=rir, iterations=len(history) - 1, cap_hit=cap_hit,
                  minutes=(time.time() - t0) / 60)
    return W.close()
