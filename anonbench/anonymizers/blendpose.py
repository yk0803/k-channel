"""Generative BlendPose-LDM (Ali, Haffar, Sanchez, MLWA 2026) - adapter around the original code.

Faithful to resampling.img_generation / Inference_generative.sample_with_img:
  * pool = images of the same split; class key = prompt "Race, Gender, Age" (lower-cased)
  * pose = MediaPipe Euler angles (their pose.get_face_pose_euler_angles), L1 pose distance
  * k-anonymous blend = uniform mean of the k-1 nearest-pose other-identity images + the target
    (paper k = k_blending + 1), including the repo's two fallback branches
  * 500-step DDPM with text (CLIP-L/14 text encoder) and image (CLIP vision last hidden state of the
    blended image) conditioning, mixed by text_weight = 1 - img_weight, VAE decode
Additions: every output logs its blend members (target + donors) so E8 can audit k-anonymity,
and the E8 controls `include_target=False` (donor-only) and `diffusion=False` (k-NoDiff_Pose).
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ..config import Paths, add_third_party_to_path
from ..data import prompt_of
from ..manifest import ManifestWriter

IMG = 128


def _tf():
    return transforms.Compose([transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.LANCZOS),
                               transforms.ToTensor(), transforms.Lambda(lambda t: t * 2 - 1)])


# ----------------------------------------------------------------------------- poses (cached)
def poses_for(cfg, dataset, rows: pd.DataFrame) -> dict:
    add_third_party_to_path(cfg, "Generative-BlendPose-LDM")
    p = Paths(cfg, dataset).work / "index" / f"{dataset}_poses.csv"
    cache = pd.read_csv(p).set_index("image_id") if p.exists() else pd.DataFrame(columns=["yaw", "pitch", "roll", "ok"])
    todo = rows[~rows.image_id.isin(cache.index)]
    if len(todo):
        if cfg.get("smoke"):
            get = lambda img: None  # noqa: E731  (no mediapipe in the smoke test)
        else:
            from pose import get_face_pose_euler_angles as get
        tf, new = _tf(), []
        for _, r in tqdm(todo.iterrows(), total=len(todo), desc="poses (MediaPipe)"):
            with Image.open(r["path"]) as im:
                pz = get(tf(im.convert("RGB")))
            new.append(dict(image_id=r["image_id"], yaw=(pz or [0.0] * 3)[0], pitch=(pz or [0.0] * 3)[1],
                            roll=(pz or [0.0] * 3)[2], ok=pz is not None))
        cache = pd.concat([cache, pd.DataFrame(new).set_index("image_id")])
        p.parent.mkdir(parents=True, exist_ok=True)
        cache.reset_index().rename(columns={"index": "image_id"}).to_csv(p, index=False)
    return {i: [cache.at[i, "yaw"], cache.at[i, "pitch"], cache.at[i, "roll"]] for i in rows.image_id}


def image_seed(image_id: str, seed: int, attempt: int = 0) -> int:
    """Stable 63-bit seed from the image id (NOT its position in a batch or in the dataframe)."""
    h = hashlib.sha256(f"{image_id}|{seed}|{attempt}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def image_noise(image_id: str, seed: int, shape, attempt: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(image_seed(image_id, seed, attempt))
    return torch.randn(*shape, generator=g)


def _pose_distance(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))


# ----------------------------------------------------------------------------- blending
def make_blends(rows: pd.DataFrame, poses: dict, k: int, include_target: bool, seed: int):
    """Returns {image_id: (member_ids, donor_ids, fallback)} - faithful to the repo's three branches."""
    rng = random.Random(seed)
    kb = k - 1
    key = rows.apply(lambda r: prompt_of(r).lower(), axis=1)
    groups = {c: g for c, g in rows.assign(_key=key).groupby("_key")}
    out = {}
    for _, r in rows.assign(_key=key).iterrows():
        g = groups[r["_key"]]
        others = g[g.identity != r["identity"]]
        cur = poses[r["image_id"]]
        fb = "none"
        if kb == 0:
            donors = []
        elif len(others) >= kb:
            donors = sorted(others.image_id, key=lambda i: _pose_distance(poses[i], cur))[:kb]
        elif len(others) >= 1:
            donors = sorted(others.image_id, key=lambda i: _pose_distance(poses[i], cur))
            fb = "fewer_donors"
        else:
            same = g[g.identity == r["identity"]].image_id.tolist()
            donors = rng.sample(same, min(kb, len(same)))
            fb = "same_person"
        members = donors + ([r["image_id"]] if include_target or kb == 0 else [])
        out[r["image_id"]] = (members, donors, fb)
    return out


# ----------------------------------------------------------------------------- models
class _SmokeText(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None, **kw):
        class O:  # noqa: D401
            pass
        o = O()
        g = torch.Generator().manual_seed(int(input_ids.sum()) % 997)
        o.last_hidden_state = torch.randn(input_ids.shape[0], 77, 768, generator=g).to(input_ids.device)
        return o


class _SmokeVision(torch.nn.Module):
    def forward(self, pixel_values=None, output_hidden_states=True):
        class O:
            pass
        o = O()
        x = F.adaptive_avg_pool2d(pixel_values, 32).flatten(1)[:, :1024]   # 3*32*32 >= 1024
        o.hidden_states = [x[:, None, :].repeat(1, 257, 1)]
        return o


def _smoke_tokenizer(prompts, **kw):
    ids = torch.tensor([[sum(map(ord, p)) % 1000] * 77 for p in prompts])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def load_models(cfg, dev, ddpm_path=None):
    add_third_party_to_path(cfg, "Generative-BlendPose-LDM")
    from linear_noise_scheduler import LinearNoiseScheduler
    from unet_cond import Unet
    from vae import VAE
    vae = VAE(im_channels=3).to(dev).eval()
    unet = Unet(im_channels=4).to(dev).eval()
    sched = LinearNoiseScheduler(num_timesteps=500, beta_start=1e-4, beta_end=0.02, ldm_scheduler=True)
    if cfg.get("smoke"):
        return vae, unet, sched, _smoke_tokenizer, _SmokeText().to(dev), _SmokeVision().to(dev)
    vp = Path(cfg["paths"]["blendpose_vae"])
    dp = Path(ddpm_path or cfg["paths"]["blendpose_ddpm"])
    for p in (vp, dp):
        if not p.exists() or p.stat().st_size < 10_000:
            raise FileNotFoundError(f"{p} missing or a git-lfs pointer - run ./setup.sh")
    vae.load_state_dict(torch.load(vp, map_location=dev))
    ck = torch.load(dp, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    unet.load_state_dict(OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()))
    from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    tok = CLIPTokenizer.from_pretrained(name)
    te = CLIPTextModel.from_pretrained(name).to(dev).eval()
    clip = CLIPModel.from_pretrained(name, output_hidden_states=True).to(dev).eval()
    return vae, unet, sched, tok, te, clip.vision_model


def _ref(t: torch.Tensor) -> np.ndarray:
    """Target image tensor in [-1,1] (3xHxW, the 128px input) -> uint8 HWC reference."""
    return ((t.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def run(cfg, rows: pd.DataFrame, params: dict, out_dir, dataset="mtf", seed=0, ddpm_path=None):
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    rows = rows.reset_index(drop=True)
    k = int(params.get("k", 4))
    imgw = float(params.get("img_weight", 0.30))
    diffusion = bool(params.get("diffusion", True))
    include_target = bool(params.get("include_target", True))
    steps = int(params.get("num_steps", 500)) if not cfg.get("smoke") else 3
    bs = int(params.get("batch_size", 32))   # outputs are batch-independent (B1 per-image noise)
    poses = poses_for(cfg, dataset, rows)
    blends = make_blends(rows, poses, k, include_target, seed)
    tf = _tf()
    imgs = {}
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="load"):
        with Image.open(r["path"]) as im:
            imgs[r["image_id"]] = tf(im.convert("RGB"))
    W = ManifestWriter(out_dir, meta=dict(method="blendpose", params=params, seed=seed, k=k, img_weight=imgw,
                                          diffusion=diffusion, include_target=include_target,
                                          ddpm=str(ddpm_path or cfg["paths"]["blendpose_ddpm"])))
    t0 = time.time()
    merged = {i: torch.stack([imgs[m] for m in blends[i][0]]).mean(0).clamp(-1, 1) for i in rows.image_id}
    if not diffusion:
        for _, r in rows.iterrows():
            mem, don, fb = blends[r["image_id"]]
            W.save_image(r, ((merged[r["image_id"]] + 1) / 2).permute(1, 2, 0).numpy(), "modified", 0,
                         ref=_ref(imgs[r["image_id"]]),
                         members=mem, donors=don, fallback=fb, k=k, include_target=include_target)
        return W.close()

    vae, unet, sched, tok, te, vision = load_models(cfg, dev, ddpm_path)
    nb = (len(rows) + bs - 1) // bs
    t1 = time.time()
    print(f"[blendpose] k={k} img_weight={imgw} include_target={include_target} steps={steps} "
          f"images={len(rows)} batch={bs} -> {nb} batches", flush=True)
    for bi, b in enumerate(range(0, len(rows), bs), 1):
        sub = rows.iloc[b:b + bs]
        im = torch.stack([merged[i] for i in sub.image_id]).to(dev)
        prompts = [prompt_of(r) for _, r in sub.iterrows()]
        t = tok(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        temb = te(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
        lat = ((im + 1) / 2).clamp(0, 1)
        pv = F.interpolate(lat, size=(224, 224), mode="bilinear", align_corners=False)
        ihs = vision(pixel_values=pv, output_hidden_states=True).hidden_states[-1]
        # B1: initial noise is drawn PER IMAGE from a stable hash of (image_id, seed), so an image's
        # output does not depend on batch size or on which other images share its batch.
        xt = torch.stack([image_noise(i, seed, (4, IMG // 4, IMG // 4)) for i in sub.image_id]).to(dev)
        for i in torch.linspace(499, 0, steps, dtype=torch.long).tolist():
            ti = torch.as_tensor(i).to(dev)
            npred = unet(xt, ti.unsqueeze(0), temb, ihs, 1.0 - imgw, imgw)
            xt, _ = sched.sample_prev_timestep(xt, npred, ti)
        ims = vae.decode(xt.float())
        ims = ((torch.clamp(ims, -1, 1).cpu() + 1) / 2 * 255).to(torch.uint8)
        for j, (_, r) in enumerate(sub.iterrows()):
            mem, don, fb = blends[r["image_id"]]
            W.save_image(r, ims[j].permute(1, 2, 0).numpy(), "modified", steps, ref=_ref(imgs[r["image_id"]]),
                         members=mem, donors=don, fallback=fb, k=k, include_target=include_target,
                         noise_seed=image_seed(r["image_id"], seed))
        done, el = min(b + bs, len(rows)), time.time() - t1
        print(f"[blendpose] batch {bi}/{nb}: {done}/{len(rows)} images, {el / done:.2f} s/img, "
              f"ETA {(len(rows) - done) * el / done / 60:.1f} min", flush=True)
    W.meta["minutes"] = (time.time() - t0) / 60
    return W.close()
