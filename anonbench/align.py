"""Face landmarks + alignment for the open-set FR evaluators.

Landmarks are detected ONCE on the original images (insightface SCRFD from buffalo_l) and
stored in normalized coordinates. By default (`alignment: transfer`) the same landmarks are
reused for the anonymized version of each image, so a method gets no "privacy" credit just
for breaking the face detector. `alignment: detect` re-detects on the anonymized image
(reported as a sensitivity analysis together with the detection-failure rate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import Paths, load_config

ARCFACE_T = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def ort_preload():
    """onnxruntime-gpu >= 1.21: load the CUDA/cuDNN libs shipped as nvidia-* pip packages (installed with
    torch) so the CUDA provider is usable without a system CUDA toolkit."""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as e:  # noqa: BLE001
        print(f"[align] onnxruntime preload skipped: {e!r}")


def template(size: int, margin: float = 0.0) -> np.ndarray:
    t = (ARCFACE_T / 112.0 - 0.5) / (1.0 + margin) + 0.5
    return (t * size).astype(np.float32)


def fallback_kps_norm() -> np.ndarray:
    """Used when no face is detected: assume a centred face crop (MTF images are Haar crops)."""
    return (0.1 + 0.8 * ARCFACE_T / 112.0).astype(np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def warp(img_rgb: np.ndarray, kps_norm: np.ndarray, size: int, margin: float = 0.0) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    src = kps_norm.reshape(5, 2) * np.array([w, h], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, template(size, margin), method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(fallback_kps_norm() * np.array([w, h]), template(size, margin))
    return cv2.warpAffine(img_rgb, M, (size, size), borderValue=0.0)


class Detector:
    def __init__(self, weights_dir: Path, device: str = "cuda"):
        ort_preload()
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=str(Path(weights_dir) / "insightface"),
                                allowed_modules=["detection"], providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640), det_thresh=0.3)
        used = self.app.models["detection"].session.get_providers()
        print(f"[align] detector running on {used[0]}")
        if device == "cuda" and used[0] != "CUDAExecutionProvider":
            print("[align] WARNING: CUDA requested but onnxruntime fell back to CPU (correct, only slower)")

    def kps_norm(self, img_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        h, w = img_rgb.shape[:2]
        bgr = img_rgb[:, :, ::-1].copy()
        for pad in (0, int(0.25 * max(h, w))):  # tight crops: retry with padding
            im = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else bgr
            faces = self.app.get(im)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) * f.det_score)
                k = f.kps.astype(np.float32) - pad
                return k / np.array([w, h], dtype=np.float32), True
        return fallback_kps_norm(), False


def detect_landmarks(paths: list[str], detector: Detector) -> pd.DataFrame:
    rows = []
    for p in tqdm(paths, desc="landmarks"):
        k, ok = detector.kps_norm(load_rgb(p))
        rows.append([p, int(ok)] + k.reshape(-1).tolist())
    cols = ["path", "detected"] + [f"k{i}{c}" for i in range(5) for c in "xy"]
    return pd.DataFrame(rows, columns=cols)


def load_landmarks(cfg: dict, dataset: str) -> dict[str, np.ndarray]:
    p = Paths(cfg, dataset).landmarks_csv()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m anonbench.align --dataset {dataset}`")
    df = pd.read_csv(p)
    kcols = [f"k{i}{c}" for i in range(5) for c in "xy"]
    return {r.path: np.asarray([getattr(r, c) for c in kcols], np.float32).reshape(5, 2) for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser(description="Detect 5-point landmarks on all original images.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    from .data import load_index
    idx = load_index(cfg, a.dataset)
    det = Detector(cfg["paths"]["weights_dir"], cfg["hardware"]["device"])
    df = detect_landmarks(idx.path.tolist(), det)
    out = Paths(cfg, a.dataset).landmarks_csv()
    df.to_csv(out, index=False)
    print(f"wrote {out}; detection rate on originals = {df.detected.mean():.3f}")


if __name__ == "__main__":
    main()
