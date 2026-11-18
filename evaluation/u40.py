import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, glob, time, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy import linalg
from transformers import CLIPVisionModelWithProjection
W = Path(ROOT + "/work"); C = Path(ROOT + "/CelebA")
OUT = W / "results" / "celeba" / "U40"; OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"
L = open(C / "list_attr_celeba.txt").read().splitlines(); names = L[1].split()
A = {p[0]: np.array([int(v) > 0 for v in p[1:]], np.float32) for p in (l.split() for l in L[2:]) if len(p) == len(names) + 1}
print(f"[u40] {len(names)} attributes, {len(A)} labelled images", flush=True)
idx = pd.read_csv(W / "index" / "celeba.csv"); test = idx[idx.split == "test"].reset_index(drop=True)
key = os.path.basename; src = dict(zip(idx.image_id, idx.path))
ids = dict(l.split() for l in open(C / "identity_CelebA.txt") if l.strip())
used = {ids[key(p)] for p in idx.path if key(p) in ids}
cand = sorted(f for f, i in ids.items() if i not in used and f in A)
real = list(np.random.default_rng(0).choice(cand, 5000, replace=False))
real_paths = [str(C / "img_align_celeba" / f) for f in real]
assert os.path.exists(real_paths[0]), real_paths[0]
print(f"[u40] held-out real: 5000 images from {len({ids[f] for f in real})} identities outside the {len(used)} indexed", flush=True)
RUNS = [("EDI-Anon", "edi_anon/blur_kernel-40*"), ("EDIALS", "edials/alpha-0.5*"), ("FAMS", "fams/anonymization_degree-1.25*"),
        ("NullFace", "nullface/*"), ("CIAGAN", "ciagan/*"), ("FALCO", "falco/m-0.0*"), ("BlendPose", "blendpose/diffusion-True*"),
        ("k-Channel v2 k5", "genrelease/k-5_pose-3_unlinkable_v2"), ("k-Channel v2 k10", "genrelease/k-10_pose-3_unlinkable_v2"),
        ("k-Channel v2 k25", "genrelease/k-25_pose-3_unlinkable_v2")]
def release(d):
    m = pd.read_csv(d / "manifest.csv"); P, Y = [], []
    for r in m.itertuples():
        if not isinstance(r.anon_path, str) or not r.anon_path: continue
        a = Path(r.anon_path); a = a if a.is_absolute() else d / a
        if not a.exists(): a = d / "images" / Path(r.anon_path).name
        k = key(src[r.image_id])
        if a.exists() and k in A: P.append(str(a)); Y.append(A[k])
    return P, np.stack(Y)
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)
ex = ThreadPoolExecutor(8)
def feats(paths, tag, bs=128):
    out, t0 = [], time.time()
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack(list(ex.map(prep, paths[i:i + bs])))).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): e = model(pixel_values=((x - MEAN) / STD).half()).image_embeds.float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        if (i // bs) % 8 == 0: print(f"[u40] feats {tag} {min(i + bs, len(paths))}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)
def probe(Xtr, Ytr, Xte, Yte):
    Xtr, Ytr, Xte = (torch.tensor(v, device=dev) for v in (Xtr, Ytr, Xte))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    pos = Ytr.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    lin = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500):
        opt.zero_grad(); lf(lin(Xtr), Ytr).backward(); opt.step()
    with torch.no_grad(): P = (lin(Xte) > 0).cpu().numpy()
    Y = Yte.astype(bool)
    return ((P & Y).sum(0) / np.maximum(Y.sum(0), 1) + (~P & ~Y).sum(0) / np.maximum((~Y).sum(0), 1)) / 2
def fd(a, b):
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False); c, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * c.real))
Xre = feats(real_paths, "real"); Yre = np.stack([A[f] for f in real])
sets = [("Clean source", test.path.tolist(), np.stack([A[key(p)] for p in test.path]))]
for name, pat in RUNS:
    ds = [d for d in glob.glob(str(W / "anon" / "celeba" / pat / "seed0" / "test")) if "smoke" not in d and "__n" not in d]
    if len(ds) != 1: print("[u40] SKIP", name, ds, flush=True); continue
    P, Y = release(Path(ds[0])); sets.append((name, P, Y))
rows, folders = [], {"real": real_paths}
for name, P, Y in sets:
    X = feats(P, name); ba = probe(X, Y, Xre, Yre); folders[name] = P
    r = dict(method=name, n=len(P), mean_ba=ba.mean(), fd_clip=fd(X, Xre), **{n: b for n, b in zip(names, ba)}); rows.append(r)
    print(f"RESULT-U40 {name:18s} n={len(P)} | mean BA {ba.mean():.3f} | Male {r['Male']:.3f} Young {r['Young']:.3f} Smiling {r['Smiling']:.3f} "
          f"Eyeglasses {r['Eyeglasses']:.3f} MouthOpen {r['Mouth_Slightly_Open']:.3f} | FD_CLIP {r['fd_clip']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "u40.csv", index=False); print("[u40] saved", OUT / "u40.csv", flush=True)
try:
    from cleanfid import fid
    F = W / "fid"
    for name, P in folders.items():
        d = F / name.replace(" ", "_"); d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(P):
            q = d / f"{i:06d}{Path(p).suffix}"
            if not q.exists(): os.symlink(p, q)
    for name in [n for n in folders if n != "real"]:
        v = fid.compute_fid(str(F / name.replace(" ", "_")), str(F / "real"), mode="clean", num_workers=8, device=torch.device(dev), verbose=False)
        print(f"RESULT-FID {name:18s} {v:.2f}", flush=True)
except Exception as e:
    print(f"[u40] Inception FID skipped: {type(e).__name__}: {e}", flush=True)
