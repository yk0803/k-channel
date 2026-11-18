"""E3b: describe the 100k new pool faces exactly as the existing pool (insightface labels + CLIP ViT-L/14 features).
New files only: work/index/genpool_v7_new_ia.csv, work/feats/pool_new_v7_part*.npy, work/feats/pool_new_v7.npy. Resumable."""
import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, glob, time, numpy as np, pandas as pd, torch, cv2
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
W = Path(ROOT + "/work"); FE = W / "feats"; O = W / "index" / "genpool_v7_new_ia.csv"; dev = "cuda"
paths = []
for N in range(10001, 10011):
    d = f"{ROOT}/third_party/FALCO/datasets/fake/fake_dataset_stylegan2_ffhq1024-0.7-{N}"
    ps = sorted(glob.glob(f"{d}/*/image.jpg")); assert len(ps) == N, (d, len(ps)); paths += ps
print(f"[e3d] {len(paths)} new pool images in 10 shards", flush=True)
done = pd.read_csv(O) if O.exists() else pd.DataFrame(columns=["path"]); dset = set(done.path); todo = [p for p in paths if p not in dset]
print(f"[e3d] labels: done {len(done)} | todo {len(todo)}", flush=True)
if todo:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=ROOT + "/weights/insightface", allowed_modules=["detection", "genderage", "landmark_3d_68"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"]); app.prepare(ctx_id=0, det_size=(640, 640))
    rows, t0 = [], time.time()
    def flush():
        global done, rows
        if rows: done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); tmp = str(O) + ".tmp"; done.to_csv(tmp, index=False); os.replace(tmp, O); rows = []
    for n, p in enumerate(todo, 1):
        im = cv2.imread(p); rec = dict(path=p, ia_det=0, ia_pad=0, ia_male=np.nan, ia_age=np.nan, ia_pitch=np.nan, ia_yaw=np.nan, ia_roll=np.nan)
        if im is not None:
            for pad in (0, 0.25):
                x = cv2.copyMakeBorder(im, *[int(pad * im.shape[0])] * 2, *[int(pad * im.shape[1])] * 2, cv2.BORDER_REPLICATE) if pad else im
                f = app.get(x)
                if f:
                    f = max(f, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1])); pp = getattr(f, "pose", None)
                    rec.update(ia_det=1, ia_pad=int(pad > 0), ia_male=int(f.sex == "M"), ia_age=float(f.age),
                               ia_pitch=float(pp[0]) if pp is not None else np.nan, ia_yaw=float(pp[1]) if pp is not None else np.nan, ia_roll=float(pp[2]) if pp is not None else np.nan)
                    break
        rows.append(rec)
        if n % 1000 == 0: r = n / (time.time() - t0); print(f"[e3d] labels {n}/{len(todo)} | {r:.1f} img/s | ETA {(len(todo) - n) / r / 60:.1f} min", flush=True)
        if n % 2000 == 0: flush()
    flush(); del app
lab = pd.read_csv(O).set_index("path").loc[paths].reset_index(); assert len(lab) == len(paths)
det = lab[lab.ia_det == 1].path.tolist(); print(f"[e3d] labels complete: detected {len(det)}/{len(paths)}", flush=True)
from transformers import CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half(); ex = ThreadPoolExecutor(8)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2; return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)
parts = []
for j, i0 in enumerate(range(0, len(det), 10000)):
    f = FE / f"pool_new_v7_part{j:02d}.npy"; chunk = det[i0:i0 + 10000]; parts.append(f)
    if f.exists() and len(np.load(f, mmap_mode="r")) == len(chunk): print(f"[e3d] feats part {j} cached", flush=True); continue
    out, t0 = [], time.time()
    for i in range(0, len(chunk), 128):
        x = torch.from_numpy(np.stack(list(ex.map(prep, chunk[i:i + 128])))).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): e = model(pixel_values=((x - MEAN) / STD).half()).image_embeds.float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        if (i // 128) % 10 == 0: print(f"[e3d] feats part {j} {min(i + 128, len(chunk))}/{len(chunk)} {time.time() - t0:.0f}s", flush=True)
    np.save(f, np.concatenate(out))
F = np.concatenate([np.load(f) for f in parts]); assert len(F) == len(det); np.save(FE / "pool_new_v7.npy", F)
pd.DataFrame(dict(path=det)).to_csv(W / "index" / "genpool_v7_new_feat_order.csv", index=False)
d = lab[lab.ia_det == 1]; print(f"[e3d] DONE | detected {len(d)} | adults (insightface age>=18) {(d.ia_age >= 18).sum()} | male share {d.ia_male.mean():.3f} | features {F.shape}", flush=True)
torch.manual_seed(0)
Lb = np.load(FE / "fairface125_train_labels.npz"); Xa = torch.tensor(np.load(FE / "fairface125_train.npy"), device=dev); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
lin = torch.nn.Linear(Xa.shape[1], int(Lb["race"].max()) + 1).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=1e-3); Y = torch.tensor(Lb["race"], device=dev).long()
for _ in range(1500): opt.zero_grad(); torch.nn.functional.cross_entropy(lin((Xa - mu) / sd), Y).backward(); opt.step()
rc = lambda Z: lin((torch.tensor(Z, device=dev) - mu) / sd).argmax(1).cpu().numpy()
old, new = np.bincount(rc(np.load(FE / "pool_all.npy")), minlength=7), np.bincount(rc(F), minlength=7)
print("[e3d] race-class counts (FairFace label index 0-6) | old pool:", old.tolist(), "| new faces:", new.tolist(), "| combined:", (old + new).tolist(), flush=True)
