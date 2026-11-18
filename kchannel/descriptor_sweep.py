"""E2: descriptor sweep for k-Channel on a CelebA validation split (identities outside the release, the 20k probe-training
images and the 5,000 held-out test images). No image is rendered: released faces are represented by their cached CLIP features.
Writes work/results/celeba/E2/e2_sweep.csv (resumable: finished configs are skipped)."""
import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, time, itertools, numpy as np, pandas as pd, torch, cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPVisionModelWithProjection
W = Path(ROOT + "/work"); C = Path(ROOT + "/CelebA"); FE = W / "feats"; dev = "cuda"; ex = ThreadPoolExecutor(8)
OUT = W / "results" / "celeba" / "E2"; OUT.mkdir(parents=True, exist_ok=True); CSV = OUT / "e2_sweep.csv"; SEED = 0; K = 5
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2; return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)
def feats(paths, tag, bs=128, cache=True):
    f = FE / f"{tag}.npy"
    if cache and f.exists() and len(np.load(f, mmap_mode="r")) == len(paths): print(f"[e2] feats {tag} cached", flush=True); return np.load(f)
    out, t0 = [], time.time()
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack(list(ex.map(prep, paths[i:i + bs])))).to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): e = model(pixel_values=((x - MEAN) / STD).half()).image_embeds.float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        if (i // bs) % 10 == 0: print(f"[e2] feats {tag} {min(i + bs, len(paths))}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(out)
    if cache: np.save(f, X)
    return X
def fit_linear(X, Y, multilabel, steps=800, lr=1e-3):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, device=dev); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt = (Xt - mu) / sd
    if multilabel: Yt = torch.tensor(Y, device=dev); pos = Yt.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos); nout = Y.shape[1]
    else: Yt = torch.tensor(Y, device=dev).long(); lf = torch.nn.CrossEntropyLoss(); nout = int(Y.max()) + 1
    lin = torch.nn.Linear(X.shape[1], nout).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps): opt.zero_grad(); lf(lin(Xt), Yt).backward(); opt.step()
    def pred(Z):
        with torch.no_grad(): o = lin((torch.tensor(Z, device=dev) - mu) / sd)
        return (torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).cpu().numpy()
    return pred
def probe(Xtr, Ytr, Xte, Yte):  # u40 protocol, seeded
    torch.manual_seed(SEED)
    Xtr, Ytr, Xte = (torch.tensor(v, device=dev) for v in (Xtr, Ytr, Xte))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    pos = Ytr.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    lin = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500): opt.zero_grad(); lf(lin(Xtr), Ytr).backward(); opt.step()
    with torch.no_grad(): P = (lin(Xte) > 0).cpu().numpy()
    Y = Yte.astype(bool)
    return ((P & Y).sum(0) / np.maximum(Y.sum(0), 1) + (~P & ~Y).sum(0) / np.maximum((~Y).sum(0), 1)) / 2
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
    return cell, dropped, max(max(d.values()) / s for d, s in zip(cnt, size))
Lb = np.load(FE / "fairface125_train_labels.npz"); Xa = np.load(FE / "fairface125_train.npy")
race_pred = fit_linear(Xa, Lb["race"], False, steps=1500, lr=2e-3); age_pred = fit_linear(Xa, Lb["age"], False, steps=1500, lr=2e-3)
L = open(C / "list_attr_celeba.txt").read().splitlines(); names = L[1].split(); jM = names.index("Male")
A = {p[0]: np.array([int(v) > 0 for v in p[1:]], np.float32) for p in (l.split() for l in L[2:]) if len(p) == len(names) + 1}
ci = pd.read_csv(W / "index" / "celeba.csv"); key = os.path.basename
ids = dict(l.split() for l in open(C / "identity_CelebA.txt") if l.strip()); used = {ids[key(p)] for p in ci.path if key(p) in ids}
cand = sorted(f for f, i in ids.items() if i not in used and f in A)
real = list(np.random.default_rng(0).choice(cand, 5000, replace=False)); rs = set(real)
trn = list(np.random.default_rng(1).choice([f for f in cand if f not in rs], 20000, replace=False)); ts = set(trn)
attr_pred = fit_linear(np.load(FE / "celeba_probe_train20k.npy"), np.stack([A[f] for f in trn]), True)
print("[e2] classifiers fitted (seed 0)", flush=True)
Xre = np.load(FE / "celeba_u40_real5000.npy"); Yre = np.stack([A[f] for f in real]); assert len(Xre) == 5000
chk = feats([str(C / "img_align_celeba" / f) for f in real[:64]], "chk", cache=False); cs = (chk * Xre[:64]).sum(1)
print(f"[e2] cached real features order check: min cosine {cs.min():.4f}", flush=True); assert cs.min() > 0.98, "celeba_u40_real5000.npy is not in the expected order"
g = defaultdict(list)
for f in cand:
    if f not in rs and f not in ts: g[ids[f]].append(f)
vid = sorted(i for i, fs in g.items() if len(fs) >= 8); vid = list(np.random.default_rng(2).choice(vid, min(500, len(vid)), replace=False))
val = [f for i in vid for f in sorted(g[i])[:8]]; idn = np.array([ids[f] for f in val]); vpaths = [str(C / "img_align_celeba" / f) for f in val]
print(f"[e2] validation: {len(val)} images of {len(vid)} identities (none in the release, the probe training set or the held-out set)", flush=True)
Fv = feats(vpaths, "celeba_val_e2"); Yv = np.stack([A[f] for f in val])
pc = W / "index" / "celeba_val_e2_pose.csv"
if not pc.exists():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=ROOT + "/weights/insightface", allowed_modules=["detection", "landmark_3d_68"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"]); app.prepare(ctx_id=0, det_size=(640, 640)); rows = []
    for n, (f, p) in enumerate(zip(val, vpaths)):
        im = cv2.imread(p); y = pt = np.nan
        for pad in (0, 0.25, 0.5):
            x = cv2.copyMakeBorder(im, *[int(pad * im.shape[0])] * 2, *[int(pad * im.shape[1])] * 2, cv2.BORDER_REPLICATE) if pad else im
            ff = app.get(x)
            if ff: ff = max(ff, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1])); pt, y = float(ff.pose[0]), float(ff.pose[1]); break
        rows.append(dict(file=f, yaw=y, pitch=pt))
        if n % 500 == 0: print(f"[e2] validation pose {n}/{len(val)}", flush=True)
    pd.DataFrame(rows).to_csv(pc, index=False)
st = pd.read_csv(pc).set_index("file").loc[val]; print(f"[e2] pose missing for {st.yaw.isna().mean():.3f} (median used)", flush=True)
yaw, pit = st.yaw.fillna(st.yaw.median()).to_numpy(), st.pitch.fillna(st.pitch.median()).to_numpy()
gp = pd.read_csv(W / "index" / "genpool_ia.csv"); gp = gp[gp.ia_det == 1].reset_index(drop=True); Fp = np.load(FE / "pool_all.npy"); assert len(Fp) == len(gp)
adult = ((gp.ia_age >= 18) & (age_pred(Fp).argmax(1) > 2)).to_numpy(); gpa = gp[adult].reset_index(drop=True); Fpa = Fp[adult]
print(f"[e2] adult pool {len(gpa)}", flush=True)
Ps, Rs, Pp, Rp = attr_pred(Fv), race_pred(Fv), attr_pred(Fpa), race_pred(Fpa)
G4 = None
try:
    src = open(ROOT + "/logs/v6_study.py").read().splitlines(); ln = [l.strip() for l in src if l.strip().startswith("G4 =") or l.strip().startswith("G4=")]
    ns = {"np": np}; exec(ln[0], ns); G4 = np.asarray(ns["G4"]); print("[e2] 4-group map from v6_study:", G4.tolist(), flush=True)
except Exception as e:
    print("[e2] no 4-group map found in v6_study.py, race4 not computed:", repr(e), flush=True)
def desc(P, R, y, p, wg, wr, pdiv):
    Q = P.copy(); Q[:, jM] *= wg; cols = [Q, wr * R]
    if pdiv: cols += [(y / pdiv)[:, None], (p / pdiv)[:, None]]
    return np.concatenate(cols, 1).astype(np.float32)
done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=["config"]); rows = done.to_dict("records")
def run(cfg, sel, keep):
    ba = probe(Fpa[sel], Yv[keep], Xre, Yre); r_s, r_p = Rs[keep].argmax(1), Rp[sel].argmax(1)
    r = dict(config=cfg, n=len(keep), mean_ba=ba.mean(), smiling=ba[names.index("Smiling")], gender=(gpa.ia_male.to_numpy()[sel] == Yv[keep][:, jM]).mean(),
             race7=(r_s == r_p).mean(), race4=(G4[r_s] == G4[r_p]).mean() if G4 is not None else np.nan,
             pitch_err=np.abs(gpa.ia_pitch.to_numpy()[sel] - pit[keep]).mean(), yaw_err=np.abs(gpa.ia_yaw.to_numpy()[sel] - yaw[keep]).mean(), distinct=len(set(sel)) / len(sel))
    rows.append(r); pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"RESULT-E2 {cfg:26s} | BA {r['mean_ba']:.3f} smile {r['smiling']:.3f} | gender {r['gender']:.3f} race4 {r['race4']:.3f} race7 {r['race7']:.3f} | pitch {r['pitch_err']:.1f} yaw {r['yaw_err']:.1f} | distinct {r['distinct']:.2f}", flush=True)
grid = list(itertools.product((1, 3, 5), (0, 1, 2, 4), (15, 30, None)))
for n_, (wg, wr, pdv) in enumerate(grid, 1):
    cfg = f"g{wg}_r{wr}_p{pdv or 'off'}"
    if cfg in set(done.config): print(f"[e2] {n_}/{len(grid)} {cfg} done, skipping", flush=True); continue
    t0 = time.time(); Zs = desc(Ps, Rs, yaw, pit, wg, wr, pdv); Zp = torch.tensor(desc(Pp, Rp, gpa.ia_yaw.to_numpy(), gpa.ia_pitch.to_numpy(), wg, wr, pdv), device=dev)
    cell, dropped, worst = mdav_distinct(Zs, idn, K); keep = np.flatnonzero(cell >= 0); nc = cell.max() + 1
    cent = torch.tensor(np.stack([Zs[cell == q].mean(0) for q in range(nc)]), device=dev)
    top = torch.cdist(cent, Zp).topk(2 * K, largest=False).indices.cpu().numpy()
    sel = top[cell[keep], np.random.default_rng(SEED).integers(0, 2 * K, len(keep))]
    assert worst <= 1 / K + 1e-9, worst
    print(f"[e2] {n_}/{len(grid)} {cfg}: {nc} cells, dropped {dropped}, worst share {worst:.3f}, {time.time() - t0:.0f}s", flush=True); run(cfg, sel, keep)
if "nearest_per_image_nobound" not in set(done.config):
    Zs = desc(Ps, Rs, yaw, pit, 3, 2, 15); Zp = torch.tensor(desc(Pp, Rp, gpa.ia_yaw.to_numpy(), gpa.ia_pitch.to_numpy(), 3, 2, 15), device=dev)
    sel = torch.cat([torch.cdist(torch.tensor(Zs[i:i + 512], device=dev), Zp).argmin(1) for i in range(0, len(Zs), 512)]).cpu().numpy()
    run("nearest_per_image_nobound", sel, np.arange(len(Zs)))
d = pd.DataFrame(rows); print("[e2] TOP 8 by attribute accuracy among configs with gender >= 0.95:", flush=True)
print(d[(d.gender >= 0.95) & d.config.str.startswith("g")].sort_values("mean_ba", ascending=False).head(8).to_string(index=False), flush=True)
print("[e2] DONE", CSV, flush=True)
