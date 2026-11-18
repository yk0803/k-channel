"""Conventional whole-image baselines exactly as in EDI-Anon/'Simple Codes' notebooks.

The notebooks work on cv2.imread() BGR uint8 images at ORIGINAL resolution and save JPEG.
Note: DP-Pix on uint8 arrays wraps around on negative/over-255 noise (numpy cast) - this is
reproduced on purpose (see AUDIT.md) unless `fix_uint8_wrap: true`.
"""
from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from ..manifest import ManifestWriter
from .common import edi_methods, jpeg_roundtrip


def perturb_bgr(img: np.ndarray, p: dict, M) -> np.ndarray:
    kind = p["perturbation"]
    if p.get("fix_uint8_wrap"):
        img = img.astype(np.float32)
    if kind == "blur":
        return cv2.blur(img, (p["blur_kernel"], p["blur_kernel"]))
    if kind == "pixelate":
        h, w = img.shape[:2]
        n = p["pixel_size"]
        small = cv2.resize(img, (max(1, w // n), max(1, h // n)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if kind == "dp_pix":
        return M.DP_Pix(M.Pix(img, p["b"]), p["b"], p["m"], p["eps"])
    if kind == "dp_blur":
        x = M.DP_Pix(M.Pix(img, p["b0"]), p["b0"], p["m"], p["eps"])
        return M.Gauss_Blur2(x, p["k"], p["sigma"])
    raise KeyError(kind)


def run(cfg, rows, params, out_dir, seed=0):
    M = edi_methods(cfg)
    np.random.seed(seed)
    W = ManifestWriter(out_dir, meta=dict(method="simple", params=params, seed=seed))
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"simple:{params['perturbation']}"):
        bgr = cv2.imread(r["path"])
        out = perturb_bgr(bgr, params, M)
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8) if out.dtype != np.uint8 else out
        rgb = jpeg_roundtrip(out[:, :, ::-1].copy())
        W.save_image(r, rgb, "modified", 1, ref=bgr[:, :, ::-1])
    return W.close()
