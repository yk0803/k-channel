"""Registry of pretrained open-set face-recognition embedders (held-out adversaries).

None of these models is trained on MTF and none is used by any anonymizer, except
`facenet_vggface2`, which is the FaceNet used by BlendPose (kept so the E2 matrix has
BlendPose's own evaluator on its diagonal).

All embedders expose:  .name  .size  .margin  .embed(list[np.uint8 HxWx3 RGB aligned]) -> (N,D) L2-normed
torch is imported lazily so the fake (smoke-test) embedder works without it.
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import numpy as np

EMBEDDER_INFO = {
    # name: (family, source, size, margin, architecture, training data)
    "arcface_r50_webface600k": ("insightface", "buffalo_l/w600k_r50.onnx", 112, 0.0, "ResNet-50", "WebFace600K"),
    "arcface_r100_glint360k": ("insightface", "antelopev2/glintr100.onnx", 112, 0.0, "ResNet-100", "Glint360K"),
    "adaface_ir101_webface12m": ("cvlface", "minchul/cvlface_adaface_ir101_webface12m", 112, 0.0, "IR-101", "WebFace12M"),
    "adaface_vit_base_webface4m": ("cvlface", "minchul/cvlface_adaface_vit_base_webface4m", 112, 0.0, "ViT-B", "WebFace4M"),
    "arcface_ir101_webface4m": ("cvlface", "minchul/cvlface_arcface_ir101_webface4m", 112, 0.0, "IR-101", "WebFace4M"),
    "facenet_casia": ("facenet", "casia-webface", 160, 0.25, "InceptionResNetV1", "CASIA-WebFace"),
    "facenet_vggface2": ("facenet", "vggface2", 160, 0.25, "InceptionResNetV1", "VGGFace2"),
}


class Embedder:
    name: str
    size: int
    margin: float

    def embed(self, imgs: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _ensure_insightface_pack(root: Path, pack: str) -> None:
    """Download an insightface model pack if missing; flatten the nested folder some zips create."""
    d = root / "models" / pack
    if not list(d.glob("*.onnx")):
        from insightface.utils import storage
        storage.ensure_available("models", pack, root=str(root))
    nested = d / pack
    if nested.is_dir():
        for f in glob.glob(str(nested / "*")):
            shutil.move(f, str(d))
        nested.rmdir()


class InsightFaceEmbedder(Embedder):
    def __init__(self, name, onnx_rel, weights_dir, device, flip=True):
        from ..align import ort_preload
        ort_preload()
        from insightface.model_zoo import get_model
        root = Path(weights_dir) / "insightface"
        _ensure_insightface_pack(root, onnx_rel.split("/")[0])
        p = root / "models" / onnx_rel
        if not p.exists():
            raise FileNotFoundError(f"{p} missing after download attempt")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.m = get_model(str(p), providers=prov)
        self.m.prepare(ctx_id=0 if device == "cuda" else -1)
        print(f"[embed] {name} running on {self.m.session.get_providers()[0]}")
        self.name, self.size, self.margin, self.flip = name, 112, 0.0, flip

    def embed(self, imgs):
        bgr = [im[:, :, ::-1].copy() for im in imgs]
        f = self.m.get_feat(bgr)
        if self.flip:
            f = f + self.m.get_feat([b[:, ::-1].copy() for b in bgr])
        return _l2(np.asarray(f))


class CVLFaceEmbedder(Embedder):
    """AdaFace / ArcFace models from the CVLface model zoo on Hugging Face."""

    def __init__(self, name, repo_id, weights_dir, device, flip=True, channel_order="rgb"):
        # B6: CVLface wrapper input order is unverified; anonbench.sanity measures both and the
        # winner is pinned in config.yaml `embedder_options` BEFORE the evaluator freeze.
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel
        self.channel_order = channel_order
        local = Path(weights_dir) / "cvlface" / repo_id.split("/")[-1]
        local.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN")
        if not (local / "files.txt").exists():
            hf_hub_download(repo_id, "files.txt", token=token, local_dir=local)
        files = [f for f in (local / "files.txt").read_text().split("\n") if f]
        for f in files + ["config.json", "wrapper.py", "model.safetensors"]:
            if not (local / f).exists():
                hf_hub_download(repo_id, f, token=token, local_dir=local)
        import sys
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]  # each CVLface repo ships its own top-level `models` package
        cwd = os.getcwd()
        os.chdir(local)
        sys.path.insert(0, str(local))
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # wrapper's model.pt copy into meta params is a no-op; weights come from model.safetensors
                self.m = AutoModel.from_pretrained(str(local), trust_remote_code=True, token=token)
            if any(t.is_meta for t in list(self.m.parameters()) + list(self.m.buffers())):
                raise RuntimeError(f"{name}: weights left on the meta device (not loaded)")
        finally:
            os.chdir(cwd)
            sys.path.remove(str(local))
        self.dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.m = self.m.to(self.dev).eval()
        self.name, self.size, self.margin, self.flip = name, 112, 0.0, flip

    def _fwd(self, x):
        import torch
        with torch.no_grad():
            out = self.m(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return out.float()

    def embed(self, imgs):
        import torch
        arr = np.stack(imgs)
        if self.channel_order == "bgr":
            arr = arr[..., ::-1].copy()
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div(255.0)
        x = ((x - 0.5) / 0.5).to(self.dev)
        f = self._fwd(x)
        if self.flip:
            f = f + self._fwd(torch.flip(x, dims=[3]))
        return _l2(f.cpu().numpy())


class FaceNetEmbedder(Embedder):
    def __init__(self, name, pretrained, device, flip=True, margin=0.25):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.m = InceptionResnetV1(pretrained=pretrained).eval().to(self.dev)
        print(f"[embed] {name} running on {self.dev} (crop margin {margin})")
        self.name, self.size, self.margin, self.flip = name, 160, margin, flip

    def embed(self, imgs):
        import torch
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
        x = ((x - 127.5) / 128.0).to(self.dev)
        with torch.no_grad():
            f = self.m(x)
            if self.flip:
                f = f + self.m(torch.flip(x, dims=[3]))
        return _l2(f.cpu().numpy())


def build_embedder(name: str, cfg: dict, **override) -> Embedder:
    if name not in EMBEDDER_INFO:
        raise KeyError(f"unknown embedder {name}; known: {sorted(EMBEDDER_INFO)}")
    fam, src, *_ = EMBEDDER_INFO[name]
    wd, dev = cfg["paths"]["weights_dir"], cfg["hardware"]["device"]
    opts = {**((cfg.get("embedder_options") or {}).get(name) or {}), **override}
    if fam == "insightface":
        return InsightFaceEmbedder(name, src, wd, dev)
    if fam == "cvlface":
        return CVLFaceEmbedder(name, src, wd, dev, channel_order=opts.get("channel_order", "rgb"))
    if fam == "facenet":
        return FaceNetEmbedder(name, src, dev, margin=opts.get("margin", EMBEDDER_INFO[name][3]))
    raise ValueError(fam)


class FakeEmbedder(Embedder):
    """Pixel random-projection 'embedder' used ONLY by tests (names start with 'fake')."""

    def __init__(self, name):
        import zlib
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        self.P = rng.standard_normal((24 * 24 * 3, 64)).astype(np.float32)
        self.name, self.size, self.margin = name, 112, 0.0

    def embed(self, imgs):
        import cv2
        x = np.stack([cv2.resize(i, (24, 24), interpolation=cv2.INTER_AREA) for i in imgs]).reshape(len(imgs), -1)
        x = x.astype(np.float32) / 255.0
        x = x - x.mean(axis=1, keepdims=True)
        return _l2(x @ self.P)


def get_embedder(name: str, cfg: dict, **override) -> Embedder:
    if name.startswith("fake"):
        return FakeEmbedder(name)
    return build_embedder(name, cfg, **override)


def embedder_meta(name: str) -> dict:
    if name.startswith("fake"):
        return dict(family="fake", architecture="random-projection", train_data="none")
    fam, src, size, margin, arch, data = EMBEDDER_INFO[name]
    return dict(family=fam, source=src, architecture=arch, train_data=data)
