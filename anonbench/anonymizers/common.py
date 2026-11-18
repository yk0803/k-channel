"""Small helpers shared by the anonymizer adapters."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def jpeg_roundtrip(img_rgb_u8: np.ndarray, quality: int = 95) -> np.ndarray:
    """Exactly what cv2.imwrite(.jpg) followed by a reload does in the original notebooks."""
    ok, buf = cv2.imencode(".jpg", img_rgb_u8[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def to_u8(a: np.ndarray) -> np.ndarray:
    """cv2.imwrite on a float array saturate-casts to uint8 (round + clip)."""
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def import_file(name: str, path: Path):
    """Import a module from an explicit file path (avoids name clashes between the cloned repos)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def edi_methods(cfg):
    """The perturbation functions shipped with EDI-Anon (Iterative/anonymization_methods.py)."""
    p = Path(cfg["paths"]["third_party"]) / "EDI-Anon" / "Iterative" / "anonymization_methods.py"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run ./setup.sh")
    return import_file("edi_anon_methods", p)
