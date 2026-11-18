import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import glob, time, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPVisionModelWithProjection
W = Path(ROOT + "/work"); FE = W / "feats"; dev = "cuda"; ex = ThreadPoolExecutor(8)
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2; return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)
def feats(paths, bs=128):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack(list(ex.map(prep, paths[i:i + bs])))).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): out.append(torch.nn.functional.normalize(model(pixel_values=((x - MEAN) / STD).half()).image_embeds.float(), dim=1).cpu())
    return torch.cat(out).to(dev)
Xa = torch.tensor(np.load(FE / "fairface125_train.npy"), device=dev); ya = torch.tensor(np.load(FE / "fairface125_train_labels.npz")["race"], device=dev).long()
mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; lin = torch.nn.Linear(768, 7).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=1e-3)
for _ in range(1500): opt.zero_grad(); torch.nn.functional.cross_entropy(lin((Xa - mu) / sd), ya).backward(); opt.step()
race = lambda X: lin((X - mu) / sd).argmax(1).cpu().numpy(); G4 = np.array([0, 1, 2, 3, 3, 3, 0])
RUNS = [("k-Channel v7 k5", "genrelease/k-5_microagg_v7"), ("k-Channel v7 k10", "genrelease/k-10_microagg_v7"), ("k-Channel v7 k25", "genrelease/k-25_microagg_v7")]
idx = pd.read_csv(W / "index" / "celeba.csv").set_index("image_id"); test_ids = idx[idx.split == "test"].index
with torch.no_grad(): rs = pd.Series(race(feats([idx.at[i, "path"] for i in test_ids])), index=test_ids)
for name, pat in RUNS:
    ds = [d for d in glob.glob(str(W / "anon" / "celeba" / pat / "seed0" / "test")) if "smoke" not in d and "__n" not in d]
    if len(ds) != 1: print("[race] SKIP", name, ds, flush=True); continue
    d = Path(ds[0]); m = pd.read_csv(d / "manifest.csv"); P, I = [], []
    for r in m.itertuples():
        if not isinstance(r.anon_path, str) or not r.anon_path: continue
        a = Path(r.anon_path); a = a if a.is_absolute() else d / a
        if not a.exists(): a = d / "images" / Path(r.anon_path).name
        if a.exists(): P.append(str(a)); I.append(r.image_id)
    t0 = time.time()
    with torch.no_grad(): rp = race(feats(P))
    s = rs.loc[I].to_numpy()
    print(f"RESULT-RACE {name:18s} n={len(P)} | race agreement with source: 4 groups {(G4[s] == G4[rp]).mean():.3f}, 7 classes {(s == rp).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
