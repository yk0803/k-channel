import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, io, time, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy import linalg
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import list_repo_files, hf_hub_download
W = Path(ROOT + "/work"); C = Path(ROOT + "/CelebA"); FE = W / "feats"; R = W / "results" / "celeba" / "v6"
FE.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True); dev = "cuda"; ex = ThreadPoolExecutor(8)
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
def prep_im(im):
    w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2; return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)
open_path = lambda p: Image.open(p).convert("RGB"); open_bytes = lambda b: Image.open(io.BytesIO(b)).convert("RGB")
def feats(items, tag, load, bs=128):
    f = FE / f"{tag}.npy"
    if f.exists() and len(np.load(f, mmap_mode="r")) == len(items): print(f"[v6] feats {tag} cached", flush=True); return np.load(f)
    out, t0 = [], time.time()
    for i in range(0, len(items), bs):
        x = torch.from_numpy(np.stack(list(ex.map(lambda it: prep_im(load(it)), items[i:i + bs])))).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): e = model(pixel_values=((x - MEAN) / STD).half()).image_embeds.float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        if (i // bs) % 40 == 0: print(f"[v6] feats {tag} {min(i + bs, len(items))}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(out); np.save(f, X); return X
def fit_linear(X, Y, multilabel, steps=800, lr=1e-3):
    Xt = torch.tensor(X, device=dev); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt = (Xt - mu) / sd
    if multilabel:
        Yt = torch.tensor(Y, device=dev); pos = Yt.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos); nout = Y.shape[1]
    else:
        Yt = torch.tensor(Y, device=dev).long(); lf = torch.nn.CrossEntropyLoss(); nout = int(Y.max()) + 1
    lin = torch.nn.Linear(X.shape[1], nout).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps): opt.zero_grad(); lf(lin(Xt), Yt).backward(); opt.step()
    def pred(Z):
        with torch.no_grad(): o = lin((torch.tensor(Z, device=dev) - mu) / sd)
        return (torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).cpu().numpy()
    return pred
def bal_acc(P, Y): return ((P & Y).sum(0) / np.maximum(Y.sum(0), 1) + (~P & ~Y).sum(0) / np.maximum((~Y).sum(0), 1)) / 2
def fd(a, b):
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False); c, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * c.real))

# ---------- A. race and age-group probes from FairFace
REPO = "HuggingFaceM4/FairFace"
pq = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("1.25/") and f.endswith(".parquet")); print("[v6] FairFace files:", pq, flush=True)
def fairface(split):
    fl = FE / f"fairface125_{split}_labels.npz"
    if fl.exists() and (FE / f"fairface125_{split}.npy").exists(): Lb = np.load(fl); return np.load(FE / f"fairface125_{split}.npy"), Lb["race"], Lb["age"]
    df = pd.concat([pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset")) for f in pq if f.split("/")[1].startswith(split)], ignore_index=True)
    print(f"[v6] FairFace {split}: {len(df)} rows, columns {list(df.columns)}, image cell type {type(df.image.iloc[0]).__name__}", flush=True)
    X = feats([x["bytes"] for x in df.image], f"fairface125_{split}", open_bytes); np.savez(fl, race=df.race.to_numpy(), age=df.age.to_numpy())
    return X, df.race.to_numpy(), df.age.to_numpy()
Xa, ra, aa = fairface("train"); Xv, rv, av = fairface("validation")
RACE = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]; G4 = np.array([0, 1, 2, 3, 3, 3, 0])   # Asian, Indian, Black, White
race_pred = fit_linear(Xa, ra, False, steps=1500, lr=2e-3); age_pred = fit_linear(Xa, aa, False, steps=1500, lr=2e-3)
pv = race_pred(Xv).argmax(1); pa = age_pred(Xv).argmax(1)
print(f"RESULT-V6 race probe on FairFace validation ({len(rv)} images): 7-class accuracy {(pv == rv).mean():.3f}, 4-group accuracy {(G4[pv] == G4[rv]).mean():.3f} | "
      f"age group: 9-class {(pa == av).mean():.3f}, under-20 vs adult {((pa <= 2) == (av <= 2)).mean():.3f}", flush=True)

# ---------- B. CelebA sets, attribute probe, pool
L = open(C / "list_attr_celeba.txt").read().splitlines(); names = L[1].split(); jM = names.index("Male")
A = {p[0]: np.array([int(v) > 0 for v in p[1:]], np.float32) for p in (l.split() for l in L[2:]) if len(p) == len(names) + 1}
idx = pd.read_csv(W / "index" / "celeba.csv"); key = os.path.basename; test = idx[idx.split == "test"].reset_index(drop=True)
ids = dict(l.split() for l in open(C / "identity_CelebA.txt") if l.strip()); used = {ids[key(p)] for p in idx.path if key(p) in ids}
cand = sorted(f for f, i in ids.items() if i not in used and f in A)
real = list(np.random.default_rng(0).choice(cand, 5000, replace=False)); rs = set(real)
trn = list(np.random.default_rng(1).choice([f for f in cand if f not in rs], 20000, replace=False)); P = lambda fs: [str(C / "img_align_celeba" / f) for f in fs]
Xre = feats(P(real), "celeba_u40_real5000", open_path); Yre = np.stack([A[f] for f in real]).astype(bool)
attr_pred = fit_linear(feats(P(trn), "celeba_probe_train20k", open_path), np.stack([A[f] for f in trn]), True)
print(f"[v6] 40-attribute probe on held-out real: mean balanced accuracy {bal_acc(attr_pred(Xre) > 0.5, Yre).mean():.3f}", flush=True)
Fs = feats(test.path.tolist(), "celeba_test_src", open_path); Ys = np.stack([A[key(p)] for p in test.path])
st = pd.read_csv(W / "index" / "celeba_test_state.csv").set_index("image_id").loc[test.image_id]; yaw_s, pit_s = st.yaw.to_numpy(), st.pitch.to_numpy()
gp = pd.read_csv(W / "index" / "genpool_ia.csv"); gp = gp[gp.ia_det == 1].reset_index(drop=True)
Fp = feats(gp.path.tolist(), "pool_all", open_path); agep = age_pred(Fp).argmax(1)
adult = ((gp.ia_age >= 18) & (agep > 2)).to_numpy(); gpa = gp[adult].reset_index(drop=True); Fpa = Fp[adult]
print(f"[v6] pool: {len(gp)} faces, {adult.sum()} kept as adults (dropped {(~adult).sum()}: insightface age <18 or FairFace age group under 20)", flush=True)
Rs, Rp = race_pred(Fs), race_pred(Fpa); Ps, Pp = attr_pred(Fs), attr_pred(Fpa)
print("[v6] source race (7-class argmax):", pd.Series(np.array(RACE)[Rs.argmax(1)]).value_counts().to_dict(), flush=True)
print("[v6] pool   race (7-class argmax):", pd.Series(np.array(RACE)[Rp.argmax(1)]).value_counts().to_dict(), flush=True)
def desc(P40, R7, yaw, pitch):
    Q = P40.copy(); Q[:, jM] *= 3
    return np.concatenate([Q, 2 * R7, (yaw / 15)[:, None], (pitch / 15)[:, None]], 1).astype(np.float32)
Zs, Zp = desc(Ps, Rs, yaw_s, pit_s), desc(Pp, Rp, gpa.ia_yaw.to_numpy(), gpa.ia_pitch.to_numpy()); Zpt = torch.tensor(Zp, device=dev)

# ---------- C. microaggregation with distinct identities
def mdav_distinct(Z, idn, k):
    n = len(Z); rem = np.ones(n, bool); cell = -np.ones(n, int); c = 0
    def take(seed):
        nonlocal c
        ix = np.flatnonzero(rem); order = ix[np.argsort(((Z[ix] - Z[seed]) ** 2).sum(1))]; ch, seen = [], set()
        for j in order:
            if idn[j] not in seen: ch.append(j); seen.add(idn[j])
            if len(ch) == k: break
        if len(ch) < k: return False
        cell[ch] = c; rem[ch] = False; c += 1; return True
    while len(set(idn[rem])) >= 2 * k:
        ix = np.flatnonzero(rem); r = ix[np.argmax(((Z[ix] - Z[ix].mean(0)) ** 2).sum(1))]; take(r)
        ix = np.flatnonzero(rem)
        if len(set(idn[ix])) < k: break
        take(ix[np.argmax(((Z[ix] - Z[r]) ** 2).sum(1))])
    if len(set(idn[rem])) >= k:
        ix = np.flatnonzero(rem); take(ix[np.argmax(((Z[ix] - Z[ix].mean(0)) ** 2).sum(1))])
    cnt = [dict() for _ in range(c)]; size = np.zeros(c, int)
    for j in np.flatnonzero(cell >= 0): cnt[cell[j]][idn[j]] = cnt[cell[j]].get(idn[j], 0) + 1; size[cell[j]] += 1
    cen = np.stack([Z[cell == q].mean(0) for q in range(c)]); dropped = 0
    for j in np.flatnonzero(rem):
        for q in np.argsort(((cen - Z[j]) ** 2).sum(1)):
            if max(max(cnt[q].values()), cnt[q].get(idn[j], 0) + 1) / (size[q] + 1) <= 1 / k + 1e-12:
                cell[j] = q; cnt[q][idn[j]] = cnt[q].get(idn[j], 0) + 1; size[q] += 1; break
        else: dropped += 1
    worst = max(max(d.values()) / s for d, s in zip(cnt, size)); return cell, dropped, worst, size
def nearest_pool(cent, m):
    out = []
    for i in range(0, len(cent), 256): out.append(torch.cdist(torch.tensor(cent[i:i + 256], device=dev), Zpt).topk(m, largest=False).indices.cpu().numpy())
    return np.concatenate(out)

# ---------- D. evaluation on the protocol of the methods table
rng = np.random.default_rng(0); rows = []; idn = test.identity.astype(str).to_numpy()
def evaluate(name, keep, sel, extra=""):
    X, Y = Fpa[sel], Ys[keep]; Xt, Yt, Xe = torch.tensor(X, device=dev), torch.tensor(Y, device=dev), torch.tensor(Xre, device=dev)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xe = (Xt - mu) / sd, (Xe - mu) / sd
    pos = Yt.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    lin = torch.nn.Linear(768, 40).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500): opt.zero_grad(); lf(lin(Xt), Yt).backward(); opt.step()
    with torch.no_grad(): ba = bal_acc((lin(Xe) > 0).cpu().numpy(), Yre)
    g = gpa.iloc[sel]; r_s, r_p = Rs[keep].argmax(1), Rp[sel].argmax(1)
    row = dict(release=name, n=len(sel), mean_ba=ba.mean(), smiling=ba[names.index("Smiling")], eyeglasses=ba[names.index("Eyeglasses")], male=ba[jM],
               gender_match=(g.ia_male.to_numpy() == Y[:, jM]).mean(), young_match=((Pp[sel][:, names.index("Young")] > 0.5) == (Y[:, names.index("Young")] > 0.5)).mean(),
               race7=(r_s == r_p).mean(), race4=(G4[r_s] == G4[r_p]).mean(), yaw_err=np.abs(g.ia_yaw.to_numpy() - yaw_s[keep]).mean(),
               pitch_err=np.abs(g.ia_pitch.to_numpy() - pit_s[keep]).mean(), fd_clip=fd(X, Xre), unique_faces=len(set(sel)) / len(sel))
    rows.append(row)
    print(f"RESULT-V6 {name:28s} n={row['n']} | 40-attr BA {row['mean_ba']:.3f} | Smiling {row['smiling']:.3f} Eyeglasses {row['eyeglasses']:.3f} | gender match {row['gender_match']:.3f} "
          f"young match {row['young_match']:.3f} | race 7-class {row['race7']:.3f} 4-group {row['race4']:.3f} | yaw err {row['yaw_err']:.1f} pitch err {row['pitch_err']:.1f} | "
          f"FD_CLIP {row['fd_clip']:.3f} | distinct faces {row['unique_faces']:.2f} {extra}", flush=True)
allk = np.arange(len(test))
evaluate("random pool face (floor)", allk, rng.integers(0, len(gpa), len(test)))
top = nearest_pool(Zs, 10); evaluate("per-image nearest (no bound)", allk, top[np.arange(len(test)), rng.integers(0, 10, len(test))])
for k in (5, 10, 25):
    t0 = time.time(); cell, dropped, worst, size = mdav_distinct(Zs, idn, k); keep = np.flatnonzero(cell >= 0); nc = cell.max() + 1
    cent = np.stack([Zs[cell == q].mean(0) for q in range(nc)]); m = 2 * k; top = nearest_pool(cent, m)
    sel = top[cell[keep], rng.integers(0, m, len(keep))]
    evaluate(f"microaggregation k={k}", keep, sel, f"| {nc} cells, sizes {size.min()}-{size.max()}, dropped {dropped}, worst identity share {worst:.3f} vs 1/k {1 / k:.3f} ({time.time() - t0:.0f}s)")
    pd.DataFrame(dict(image_id=test.image_id.to_numpy()[keep], cell=cell[keep], pool_path=gpa.path.to_numpy()[sel])).to_csv(W / "index" / f"v6_release_k{k}.csv", index=False)
pd.DataFrame(rows).to_csv(R / "v6_study.csv", index=False); print("[v6] saved", R / "v6_study.csv", flush=True)
