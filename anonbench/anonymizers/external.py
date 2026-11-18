"""External baselines. FAMS (Face Anonymization Made Simple, Kung et al., WACV 2025), run through its own repo code.

Deviations from the repo README (documented): SD2.1 VAE/scheduler from the community mirror
`sd2-community/stable-diffusion-2-1` (official repo withdrawn by Stability); torch 2.11/cu128 instead of 2.1
(Blackwell); no xformers; per-image noise seed from a stable hash of image_id (shard/order independent).
Input: the source image resized to 512x512 (full frame), as in the README example; output 512x512.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..manifest import ManifestWriter

SD_MIRROR = "sd2-community/stable-diffusion-2-1"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _image_seed(image_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{image_id}|{seed}".encode()).digest()[:4], "little")


def run_fams(cfg, rows, params, out_dir, seed=0):
    repo = Path(cfg["paths"]["third_party"]) / "face_anon_simple"
    if not repo.exists():
        raise FileNotFoundError(f"FAMS not found at {repo}")
    sys.path.insert(0, str(repo))
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
    from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
    from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline
    fid, cid = "hkung/face-anon-simple", "openai/clip-vit-large-patch14"
    pipe = StableDiffusionReferenceNetPipeline(
        unet=UNet2DConditionModel.from_pretrained(fid, subfolder="unet", use_safetensors=True),
        referencenet=ReferenceNetModel.from_pretrained(fid, subfolder="referencenet", use_safetensors=True),
        conditioning_referencenet=ReferenceNetModel.from_pretrained(fid, subfolder="conditioning_referencenet",
                                                                    use_safetensors=True),
        vae=AutoencoderKL.from_pretrained(SD_MIRROR, subfolder="vae", use_safetensors=True),
        feature_extractor=CLIPImageProcessor.from_pretrained(cid, use_safetensors=True),
        image_encoder=CLIPVisionModel.from_pretrained(cid, use_safetensors=True),
        scheduler=DDPMScheduler.from_pretrained(SD_MIRROR, subfolder="scheduler", use_safetensors=True),
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    W = ManifestWriter(out_dir, meta=dict(method="fams", params=params, seed=seed, sd_vae=SD_MIRROR, input="resize512"))
    t0 = time.time()
    rows = rows.reset_index(drop=True)
    print(f"[fams] d={params['anonymization_degree']} steps={params['num_inference_steps']} "
          f"cfg={params['guidance_scale']} images={len(rows)}", flush=True)
    for k, r in rows.iterrows():
        prev = W.image_path(r["image_id"])
        if prev.exists():  # resume: reuse an image already generated (a truncated one is regenerated)
            try:
                with Image.open(prev) as im:
                    done_img = im.convert("RGB")
                W.save_image(r, done_img, "modified", 1, noise_seed=_image_seed(r["image_id"], seed), resumed=True)
                continue
            except Exception:
                pass
        with Image.open(r["path"]) as im:
            src = im.convert("RGB").resize((512, 512), Image.LANCZOS)
        g = torch.Generator(device="cpu").manual_seed(_image_seed(r["image_id"], seed))
        out = pipe(source_image=src, conditioning_image=src, num_inference_steps=params["num_inference_steps"],
                   guidance_scale=params["guidance_scale"], generator=g,
                   anonymization_degree=params["anonymization_degree"], width=512, height=512).images[0]
        W.save_image(r, out, "modified", 1, noise_seed=_image_seed(r["image_id"], seed))
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            el = time.time() - t0
            print(f"[fams] {k + 1}/{len(rows)} | {el / (k + 1):.1f} s/img, ETA {el / (k + 1) * (len(rows) - k - 1) / 60:.0f} min",
                  flush=True)
    W.meta["minutes"] = (time.time() - t0) / 60
    return W.close()


def run_nullface(cfg, rows, params, out_dir, seed=0):
    """NullFace (Kung et al., IEEE FG 2026), training-free localized face anonymization.

    Deviations from the repo README (documented): torch 2.11/cu128 and torchvision 0.26 instead of the
    authors' upper pins (Blackwell sm_120); stock `stable-diffusion-v1-5/stable-diffusion-v1-5` and
    `h94/IP-Adapter-FaceID` from HuggingFace, no custom checkpoints; per-image seed from a stable hash
    of image_id. No mask file is supplied, so their code builds a white mask at the correct size, which
    is the full-face setting used for their baseline comparisons. Their detector fails on tight crops
    such as MTF's, so an image whose first call returns None is retried on a replicate-padded copy and
    the border is cropped back out of the output; the manifest records which images needed this.
    """
    import cv2
    repo = Path(cfg["paths"]["third_party"]) / "nullface"
    if not repo.exists():
        raise FileNotFoundError(f"NullFace not found at {repo}")
    sys.path.insert(0, str(repo))
    from anonymize_face import anonymize_face

    W = ManifestWriter(out_dir, meta=dict(method="nullface", params=params, seed=seed,
                                          sd_model=params.get("sd_model_path", SD15),
                                          input="source_resolution"))
    mask_p = Path(out_dir) / "__full_face_white_mask__"   # deliberately absent -> white mask
    log_f = str(Path(out_dir) / "nullface_log.txt")
    tmp_p = Path(out_dir) / "_padded.png"

    def call(path, sd):
        return anonymize_face(
            image_path=str(path), mask_image_path=str(mask_p),
            sd_model_path=params.get("sd_model_path", SD15),
            insightface_model_path=params.get("insightface_model_path", "~/.insightface"),
            device_num=int(params.get("device_num", 0)),
            guidance_scale=float(params["guidance_scale"]),
            num_diffusion_steps=int(params["num_diffusion_steps"]),
            eta=float(params["eta"]), skip=int(params["skip"]),
            ip_adapter_scale=float(params["ip_adapter_scale"]),
            id_emb_scale=float(params["id_emb_scale"]),
            output_log_file=log_f,
            det_thresh=float(params["det_thresh"]), det_size=int(params["det_size"]),
            seed=sd, mask_delay_steps=int(params["mask_delay_steps"]))

    t0, n_pad, n_fail = time.time(), 0, 0
    rows = rows.reset_index(drop=True)
    print(f"[nullface] steps={params['num_diffusion_steps']} skip={params['skip']} "
          f"cfg={params['guidance_scale']} ip={params['ip_adapter_scale']} images={len(rows)}", flush=True)
    for k, r in rows.iterrows():
        sd = _image_seed(r["image_id"], seed) % (2 ** 31)
        prev = W.image_path(r["image_id"])
        if prev.exists():                                   # resume
            try:
                with Image.open(prev) as im:
                    W.save_image(r, im.convert("RGB"), "modified", 1, noise_seed=sd, resumed=True)
                continue
            except Exception:
                pass
        out, padded = call(r["path"], sd), False
        if out is None:
            a = np.asarray(Image.open(r["path"]).convert("RGB"))
            h, w = a.shape[:2]
            pad = int(0.25 * max(h, w))
            Image.fromarray(cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_REPLICATE)).save(tmp_p)
            out = call(tmp_p, sd)
            if out is not None:
                Wo, Ho = out.size
                x0 = int(round(Wo * pad / (w + 2 * pad))); x1 = int(round(Wo * (pad + w) / (w + 2 * pad)))
                y0 = int(round(Ho * pad / (h + 2 * pad))); y1 = int(round(Ho * (pad + h) / (h + 2 * pad)))
                out = out.crop((x0, y0, x1, y1)).resize((Wo, Ho), Image.LANCZOS)
                padded, n_pad = True, n_pad + 1
        if out is None:
            n_fail += 1
            W.add(r, None, "failed", 0, reason="no_face_detected")
            print(f"[nullface] NO FACE: {r['image_id']}", flush=True)
            continue
        W.save_image(r, out, "modified", 1, noise_seed=sd, padded=padded)
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            el = time.time() - t0
            print(f"[nullface] {k + 1}/{len(rows)} | {el / (k + 1):.1f} s/img, "
                  f"padded {n_pad}, failed {n_fail}, "
                  f"ETA {el / (k + 1) * (len(rows) - k - 1) / 60:.0f} min", flush=True)
    W.meta.update(minutes=(time.time() - t0) / 60, n_padded=n_pad, n_failed=n_fail)
    print(f"[nullface] done: {n_pad} needed padding, {n_fail} had no detectable face", flush=True)
    return W.close()


CIAGAN_W, CIAGAN_H, CIAGAN_SZ, CIAGAN_NCLS = 178, 218, 128, 1200


def _fa_landmarks(fa, rgb):
    """68-point iBUG landmarks via face_alignment (drop-in for the dlib predictor CIAGAN uses)."""
    pr = fa.get_landmarks(rgb)
    return None if not pr else max(pr, key=lambda k: (k[:, 0].max() - k[:, 0].min()) * (k[:, 1].max() - k[:, 1].min()))


def _ciagan_inputs(rgb, lm):
    """Port of CIAGAN process_data.get_lndm: crop, landmark drawing, face mask. RGB throughout."""
    import cv2
    c_x = int((lm[42][0] + lm[39][0]) / 2)
    c_y = int((lm[42][1] + lm[39][1]) / 2)
    h_r = int((lm[42][0] - lm[39][0]) * 5)
    w_r = int(h_r / CIAGAN_H * CIAGAN_W)
    if h_r <= 0 or w_r <= 0:
        return None
    w, h = w_r * 2, h_r * 2
    pd = int(w)
    img_p = np.stack([np.pad(rgb[:, :, c], pd, "edge") for c in range(3)], 2)
    crop = img_p[c_y - h_r + pd:c_y + h_r + pd, c_x - w_r + pd:c_x + w_r + pd]
    if crop.size == 0:
        return None
    clr = cv2.resize(crop, (CIAGAN_W, CIAGAN_H), interpolation=cv2.INTER_CUBIC)
    t_x, t_y = c_x - w_r, c_y - h_r
    rw, rh = CIAGAN_W / w, CIAGAN_H / h
    P = lambda i: (int((lm[i][0] - t_x) * rw), int((lm[i][1] - t_y) * rh))   # noqa: E731

    lndm = np.ones((CIAGAN_H, CIAGAN_W, 3), np.uint8) * 255
    for a, b in ([(i, i + 1) for i in range(16)] + [(27 + i, 28 + i) for i in range(3)]
                 + [(60 + i, 61 + i) for i in range(4)] + [(64 + i, 65 + i) for i in range(3)] + [(67, 60)]):
        cv2.line(lndm, P(a), P(b), (255, 0, 0), 1)                           # red in RGB

    msk = np.ones((CIAGAN_H, CIAGAN_W, 3), np.uint8) * 255
    pts = [[int((lm[0][0] - t_x) * rw), int((lm[19][1] - t_y) * rh)]]
    pts += [[int((lm[p][0] - t_x) * rw), int((lm[p][1] - t_y) * rh)] for p in range(17)]
    pts += [[int((lm[16][0] - t_x) * rw), int((lm[24][1] - t_y) * rh)]]
    cv2.fillPoly(msk, pts=[np.array(pts, int)], color=(0, 0, 0))
    return clr, lndm, msk


def _ciagan_load(arr):
    """Port of util_data.ImageDataset.load_img, including the JPEG round trip their data had."""
    import io
    from torchvision import transforms
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im = im.resize([int(CIAGAN_SZ * 1.125)] * 2, resample=Image.LANCZOS)
    w, h = im.size
    c = CIAGAN_SZ / 2
    im = im.crop((int(w / 2 - c), int(h / 2 - c), int(w / 2 + c), int(h / 2 + c)))
    return transforms.ToTensor()(im.resize((CIAGAN_SZ, CIAGAN_SZ), resample=Image.LANCZOS))


def run_ciagan(cfg, rows, params, out_dir, seed=0):
    """CIAGAN (Maximov et al., CVPR 2020), conditional-GAN face anonymization.

    Deviations from the repo (documented): 68-point landmarks from face_alignment rather than dlib
    (identical iBUG scheme, avoids a dlib build); the target identity one-hot is a stable hash of the
    source identity modulo the model's 1200 classes, so every image of a person maps to the same
    invented identity, instead of the repo's "next folder index" which depends on directory order.
    Output is 128x128, the resolution the released generator produces, and the background inside the
    crop is preserved by the method's own compositing (gen*mask + input*(1-mask)).
    """
    import torch
    repo = Path(cfg["paths"]["third_party"]) / "ciagan" / "source"
    if not repo.exists():
        raise FileNotFoundError(f"CIAGAN not found at {repo}")
    sys.path.insert(0, str(repo))
    import arch.arch_unet_flex as arch_gen
    import face_alignment

    wpath = params.get("model_path") or str(Path(cfg["paths"]["weights_dir"]) / "ciagan" / "modelG.pth")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = arch_gen.Generator(num_classes=CIAGAN_NCLS, img_size=CIAGAN_SZ)
    G.load_state_dict(torch.load(wpath, map_location="cpu"))
    G = G.to(dev).eval()
    try:
        lt = face_alignment.LandmarksType.TWO_D
    except AttributeError:
        lt = face_alignment.LandmarksType._2D
    fa = face_alignment.FaceAlignment(lt, device=str(dev), flip_input=False)

    W = ManifestWriter(out_dir, meta=dict(method="ciagan", params=params, seed=seed,
                                          model=wpath, input="ciagan_crop128"))
    t0, n_fail = time.time(), 0
    rows = rows.reset_index(drop=True)
    print(f"[ciagan] {len(rows)} images, {CIAGAN_NCLS} classes, out {CIAGAN_SZ}px", flush=True)
    for k, r in rows.iterrows():
        prev = W.image_path(r["image_id"])
        if prev.exists():
            try:
                with Image.open(prev) as im:
                    W.save_image(r, im.convert("RGB"), "modified", 1, resumed=True)
                continue
            except Exception:
                pass
        rgb = np.asarray(Image.open(r["path"]).convert("RGB"))
        lm = _fa_landmarks(fa, rgb)
        tri = _ciagan_inputs(rgb, lm) if lm is not None else None
        if tri is None:
            n_fail += 1
            W.add(r, None, "failed", 0, reason="no_landmarks")
            print(f"[ciagan] NO LANDMARKS: {r['image_id']}", flush=True)
            continue
        clr, lndm, msk = (_ciagan_load(a).unsqueeze(0).to(dev) for a in tri)
        msk = ((1 - msk) > 0.2).float()
        tgt = _image_seed(str(r["identity"]), seed) % CIAGAN_NCLS      # same fake id for one person
        oh = torch.zeros(1, CIAGAN_NCLS, device=dev); oh[0, tgt] = 1.0
        with torch.no_grad():
            gen = G(torch.cat((lndm, clr * (1 - msk)), 1), onehot=oh)
            out = torch.clamp(gen * msk + clr * (1 - msk), 0, 1)
        img = Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        W.save_image(r, img, "modified", 1, target_class=int(tgt))
        if (k + 1) % 50 == 0 or k + 1 == len(rows):
            el = time.time() - t0
            print(f"[ciagan] {k + 1}/{len(rows)} | {el / (k + 1):.2f} s/img, failed {n_fail}, "
                  f"ETA {el / (k + 1) * (len(rows) - k - 1) / 60:.0f} min", flush=True)
    W.meta.update(minutes=(time.time() - t0) / 60, n_failed=n_fail)
    print(f"[ciagan] done: {n_fail} without landmarks", flush=True)
    return W.close()
