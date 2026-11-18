"""E4 (CPU): final k-Channel releases with the combined pool (60k + 100k generated faces), seeded classifiers and the
validated descriptor (gender x3, race group x2, pose / 30 deg). For k = 5, 10, 25 it writes
  1) the evaluation release (manifest format, for the attack suite):  anon/<ds>/genrelease/k-<k>_microagg_v7/...
  2) the publishable export: shuffled, renamed PNGs without metadata + labels:   work/export/v7_<ds>_k<k>/
  3) a private mapping (export file -> source image), kept outside the export: work/index/v7_<ds>_k<k>_private_map.csv
and a summary row per k (utility from cached CLIP features, as in the ablation).  usage: e4_build.py <celeba|mtf>"""
import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, sys, time, shutil, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from anonbench.config import load_config, Paths
from anonbench.data import load_index
from anonbench.manifest import ManifestWriter
ds = sys.argv[1]; W = Path(ROOT + "/work"); C = Path(ROOT + "/CelebA"); FE = W / "feats"; dev = "cpu"
ex = ThreadPoolExecutor(8); torch.set_num_threads(max(4, (os.cpu_count() or 8) // 2)); SEED = 0; WG, WR, PD = 3, 2, 30; TAG = "microagg_v7"
OUTR = W / "results" / ds / "E4"; OUTR.mkdir(parents=True, exist_ok=True); SUM = OUTR / "e4_summary.csv"
print(f"[e4] {ds} | CPU threads {torch.get_num_threads()} | descriptor g{WG}_r{WR}_p{PD} | seed {SEED}", flush=True)
cfg = load_config(None, os.environ["CS"].split() if ds == "celeba" else []); PA = Paths(cfg, ds)
def fit_linear(X, Y, multilabel, steps=800, lr=1e-3):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt = (Xt - mu) / sd
    if multilabel: Yt = torch.tensor(Y); pos = Yt.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos); nout = Y.shape[1]
    else: Yt = torch.tensor(Y).long(); lf = torch.nn.CrossEntropyLoss(); nout = int(Y.max()) + 1
    lin = torch.nn.Linear(X.shape[1], nout); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-3)
    for s in range(steps):
        opt.zero_grad(); lf(lin(Xt), Yt).backward(); opt.step()
    def pred(Z):
        with torch.no_grad(): o = lin((torch.tensor(Z) - mu) / sd)
        return (torch.sigmoid(o) if multilabel else torch.softmax(o, 1)).numpy()
    return pred
def probe(Xtr, Ytr, Xte, Yte):
    torch.manual_seed(SEED)
    Xtr, Ytr, Xte = (torch.tensor(np.asarray(v, np.float32)) for v in (Xtr, Ytr, Xte))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    pos = Ytr.mean(0).clamp(1e-3, 1 - 1e-3); lf = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    lin = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1]); opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500): opt.zero_grad(); lf(lin(Xtr), Ytr).backward(); opt.step()
    with torch.no_grad(): P = (lin(Xte) > 0).numpy()
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
    return cell, dropped, max(max(d.values()) / s for d, s in zip(cnt, size)), size
# ---- classifiers (seeded, CPU)
t0 = time.time(); Lb = np.load(FE / "fairface125_train_labels.npz"); Xa = np.load(FE / "fairface125_train.npy")
race_pred = fit_linear(Xa, Lb["race"], False, steps=1500, lr=2e-3); print(f"[e4] race classifier fitted {time.time() - t0:.0f}s", flush=True)
age_pred = fit_linear(Xa, Lb["age"], False, steps=1500, lr=2e-3); print(f"[e4] age classifier fitted {time.time() - t0:.0f}s", flush=True)
L = open(C / "list_attr_celeba.txt").read().splitlines(); names = L[1].split(); jM = names.index("Male")
A = {p[0]: np.array([int(v) > 0 for v in p[1:]], np.float32) for p in (l.split() for l in L[2:]) if len(p) == len(names) + 1}
def alab(p):
    b = os.path.basename(p); v = A.get(b, A.get(os.path.splitext(b)[0] + ".jpg", A.get(os.path.splitext(b)[0] + ".png"))); return v
ci = pd.read_csv(W / "index" / "celeba.csv")
ids = dict(l.split() for l in open(C / "identity_CelebA.txt") if l.strip()); used = {ids[os.path.basename(p)] for p in ci.path if os.path.basename(p) in ids}
cand = sorted(f for f, i in ids.items() if i not in used and f in A)
real = list(np.random.default_rng(0).choice(cand, 5000, replace=False)); rs = set(real)
trn = list(np.random.default_rng(1).choice([f for f in cand if f not in rs], 20000, replace=False))
attr_pred = fit_linear(np.load(FE / "celeba_probe_train20k.npy"), np.stack([A[f] for f in trn]), True); print(f"[e4] attribute classifier fitted {time.time() - t0:.0f}s", flush=True)
Xre = np.load(FE / "celeba_u40_real5000.npy"); Yre = np.stack([A[f] for f in real])
# ---- source images
idx = load_index(cfg, ds).reset_index(drop=True); test = idx[idx.split == "test"].reset_index(drop=True); idxi = idx.set_index("image_id")
Fs = np.load(FE / f"{ds}_test_src_li.npy"); assert len(Fs) == len(test), (len(Fs), len(test))
if ds == "celeba":
    st = pd.read_csv(W / "index" / "celeba_test_state.csv").set_index("image_id").loc[test.image_id]; yaw_s, pit_s = st.yaw.to_numpy(), st.pitch.to_numpy()
else:
    pc = W / "index" / f"{ds}_test_pose.csv"; assert pc.exists(), f"missing {pc}: run v6_build for {ds} once"
    st = pd.read_csv(pc).set_index("image_id").loc[test.image_id]; yaw_s, pit_s = st.yaw.fillna(st.yaw.median()).to_numpy(), st.pitch.fillna(st.pitch.median()).to_numpy()
Ysrc = np.stack([alab(p) for p in test.path]) if ds == "celeba" else None
if Ysrc is not None: assert not any(v is None for v in Ysrc), "missing CelebA labels for some source images"
gsrc = test.gender.astype(str).str.lower().str.startswith("m").to_numpy() if "gender" in test.columns else None
print(f"[e4] {ds}: {len(test)} source images of {test.identity.nunique()} identities | labels {'yes' if Ysrc is not None else 'no'} | gender column {'yes' if gsrc is not None else 'no'}", flush=True)
# ---- combined pool (old 60k + new 100k), adults only
gp = pd.read_csv(W / "index" / "genpool_ia.csv"); gp = gp[gp.ia_det == 1].reset_index(drop=True); Fp = np.load(FE / "pool_all.npy"); assert len(Fp) == len(gp)
lab = pd.read_csv(W / "index" / "genpool_v7_new_ia.csv"); order = pd.read_csv(W / "index" / "genpool_v7_new_feat_order.csv"); Fn = np.load(FE / "pool_new_v7.npy"); assert len(order) == len(Fn)
gn = lab.set_index("path").loc[order.path].reset_index(); assert (gn.ia_det == 1).all()
G = pd.concat([gp, gn[gp.columns]], ignore_index=True); F = np.concatenate([Fp, Fn]).astype(np.float32)
adult = ((G.ia_age >= 18) & (age_pred(F).argmax(1) > 2)).to_numpy(); gpa = G[adult].reset_index(drop=True); Fpa = F[adult]
print(f"[e4] pool: {len(gp)} old + {len(gn)} new detected faces -> {len(gpa)} adults", flush=True)
def desc(P, R, y, p):
    Q = P.copy(); Q[:, jM] *= WG; return np.concatenate([Q, WR * R, (y / PD)[:, None], (p / PD)[:, None]], 1).astype(np.float32)
Ps, Rs, Pp, Rp = attr_pred(Fs), race_pred(Fs), attr_pred(Fpa), race_pred(Fpa)
Zs = desc(Ps, Rs, yaw_s, pit_s); Zp = torch.tensor(desc(Pp, Rp, gpa.ia_yaw.to_numpy(), gpa.ia_pitch.to_numpy()))
idn = test.identity.astype(str).to_numpy()
load512 = lambda p: np.asarray(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS))
for k in (5, 10, 25):
    t1 = time.time(); cell, dropped, worst, size = mdav_distinct(Zs, idn, k); keep = np.flatnonzero(cell >= 0); nc = cell.max() + 1
    assert worst <= 1 / k + 1e-9, worst
    cent = torch.tensor(np.stack([Zs[cell == q].mean(0) for q in range(nc)]))
    top = torch.cat([torch.cdist(cent[i:i + 64], Zp).topk(2 * k, largest=False).indices for i in range(0, nc, 64)]).numpy()
    rng = np.random.default_rng(SEED); sel = top[cell[keep], rng.integers(0, 2 * k, len(keep))]; paths = gpa.path.to_numpy()[sel]
    print(f"[e4] {ds} k={k}: {len(keep)} images in {nc} cells of {size.min()}-{size.max()} | dropped {dropped} | worst share {worst:.3f} | {time.time() - t1:.0f}s", flush=True)
    out = PA.anon_dir("genrelease", f"k-{k}_{TAG}", "test", 0)
    if os.path.exists(out): shutil.rmtree(out)
    Wm = ManifestWriter(out, meta=dict(method="genrelease", params=dict(k=k, mode=TAG, topm=2 * k, descriptor=f"g{WG}_r{WR}_p{PD}", pool=len(gpa)), seed=SEED))
    exd = W / "export" / f"v7_{ds}_k{k}"
    if exd.exists(): shutil.rmtree(exd)
    (exd / "images").mkdir(parents=True)
    perm = np.random.default_rng(SEED + 1000 + k).permutation(len(keep)); rnd = np.random.default_rng(SEED + 2000 + k)
    fnames = [f"{x:016x}.png" for x in rnd.integers(0, 2 ** 62, len(keep))]; assert len(set(fnames)) == len(fnames)
    pos = np.empty(len(keep), int); pos[perm] = np.arange(len(keep))  # export position of the i-th kept image
    t1 = time.time()
    for i0 in range(0, len(keep), 64):
        for ii, (j, p, arr) in enumerate(zip(keep[i0:i0 + 64], paths[i0:i0 + 64], ex.map(load512, paths[i0:i0 + 64]))):
            iid = test.image_id.iloc[j]; row = dict(idxi.loc[iid]); row["image_id"] = iid
            Wm.save_image(row, arr, "modified", 0, pool_face=p, cell=int(cell[j]))
            Image.fromarray(arr).save(exd / "images" / fnames[pos[i0 + ii]])
        if (i0 // 64) % 8 == 0: print(f"[e4] {ds} k={k} written {min(i0 + 64, len(keep))}/{len(keep)} {time.time() - t1:.0f}s", flush=True)
    Wm.close()
    pd.DataFrame(dict(image_id=test.image_id.to_numpy()[keep], cell=cell[keep], pool_path=paths)).to_csv(W / "index" / f"v7_{ds}_k{k}.csv", index=False)
    pd.DataFrame(dict(file=[fnames[pos[i]] for i in range(len(keep))], image_id=test.image_id.to_numpy()[keep], cell=cell[keep])).to_csv(W / "index" / f"v7_{ds}_k{k}_private_map.csv", index=False)
    r = dict(dataset=ds, k=k, images=len(keep), cells=nc, dropped=dropped, worst_share=worst, distinct=len(set(sel)) / len(sel),
             race7=(Rs[keep].argmax(1) == Rp[sel].argmax(1)).mean(), pitch_err=np.abs(gpa.ia_pitch.to_numpy()[sel] - pit_s[keep]).mean(),
             yaw_err=np.abs(gpa.ia_yaw.to_numpy()[sel] - yaw_s[keep]).mean())
    if gsrc is not None: r["gender"] = (gpa.ia_male.to_numpy()[sel] == gsrc[keep]).mean()
    if Ysrc is not None:
        Yk = Ysrc[keep]; cm = pd.DataFrame(Yk).groupby(cell[keep]).transform("mean").to_numpy().astype(np.float32)
        ex_order = np.argsort(pos)
        pd.DataFrame(Yk[ex_order].astype(int), columns=names).assign(file=[fnames[i] for i in range(len(keep))])[["file"] + names].to_csv(exd / "labels_source.csv", index=False)
        pd.DataFrame(np.round(cm[ex_order], 4), columns=names).assign(file=[fnames[i] for i in range(len(keep))])[["file"] + names].to_csv(exd / "labels_cellmean.csv", index=False)
        ba_s = probe(Fpa[sel], Yk, Xre, Yre); ba_c = probe(Fpa[sel], cm, Xre, Yre)
        r.update(ba_source_labels=ba_s.mean(), ba_cellmean_labels=ba_c.mean(), smiling_source=ba_s[names.index("Smiling")], smiling_cellmean=ba_c[names.index("Smiling")])
    rows = (pd.read_csv(SUM).query("not (dataset == @ds and k == @k)").to_dict("records") if SUM.exists() else []) + [r]
    pd.DataFrame(rows).to_csv(SUM, index=False)
    print("RESULT-E4 " + " | ".join(f"{a} {v:.3f}" if isinstance(v, float) else f"{a} {v}" for a, v in r.items()), flush=True)
    print(f"[e4] {ds} k={k}: export {exd} ({len(keep)} shuffled PNGs, no metadata), private map work/index/v7_{ds}_k{k}_private_map.csv", flush=True)
print(f"[e4] BUILD OK {ds}", flush=True)
