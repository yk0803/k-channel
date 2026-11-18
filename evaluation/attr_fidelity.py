import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import glob, os, sys, time, numpy as np, pandas as pd, cv2
from pathlib import Path
from insightface.app import FaceAnalysis
W = Path(ROOT + "/work")
app = FaceAnalysis(name="buffalo_l", root=ROOT + "/weights/insightface", allowed_modules=["detection", "genderage"],
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
def ga(path):
    im = cv2.imread(str(path))
    if im is None: return None
    for pad in (0, 0.25):
        x = cv2.copyMakeBorder(im, *[int(pad * im.shape[0])] * 2, *[int(pad * im.shape[1])] * 2, cv2.BORDER_REPLICATE) if pad else im
        f = app.get(x)
        if f:
            f = max(f, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
            return int(f.sex == "M"), float(f.age)
    return None
cache = {}
def ga_c(p):
    if p not in cache: cache[p] = ga(p)
    return cache[p]
RUNS = {"celeba": [("EDI-Anon", "edi_anon/blur_kernel-40*"), ("EDIALS", "edials/alpha-0.5*"), ("FAMS", "fams/anonymization_degree-1.25*"),
                   ("NullFace", "nullface/*"), ("CIAGAN", "ciagan/*"), ("FALCO", "falco/m-0.0*"), ("BlendPose", "blendpose/diffusion-True*"),
                   ("k-Channel k5", "genrelease/k-5_pose-3_unlinkable"), ("k-Channel k25", "genrelease/k-25_pose-3_unlinkable")],
        "mtf": [("k-Channel k5", "genrelease/k-5_pose-3_unlinkable"), ("FAMS", "fams/anonymization_degree-1.25*"), ("BlendPose", "blendpose/diffusion-True*")]}
for ds, runs in RUNS.items():
    idx = pd.read_csv(W / "index" / f"{ds}.csv").set_index("image_id"); rows = []
    for name, pat in runs:
        ds_ = [d for d in glob.glob(str(W / "anon" / ds / pat / "seed0" / "test")) if "smoke" not in d and "__n" not in d]
        if len(ds_) != 1: print("SKIP", ds, name, ds_); continue
        d = Path(ds_[0]); m = pd.read_csv(d / "manifest.csv"); t0 = time.time(); res = []
        for k, r in enumerate(m.itertuples()):
            if not isinstance(r.anon_path, str): continue
            a = Path(r.anon_path); a = a if a.is_absolute() else d / a
            if not a.exists(): a = d / "images" / Path(r.anon_path).name
            s, o = ga_c(idx.at[r.image_id, "path"]), ga(a)
            res.append(dict(image_id=r.image_id, identity=idx.at[r.image_id, "identity"], s_det=s is not None, o_det=o is not None,
                            s_m=s[0] if s else np.nan, s_age=s[1] if s else np.nan, o_m=o[0] if o else np.nan, o_age=o[1] if o else np.nan))
            if k % 500 == 0: print(f"[attr] {ds} {name} {k}/{len(m)} {time.time()-t0:.0f}s", flush=True)
        x = pd.DataFrame(res); b = x[x.s_det & x.o_det]
        idm = x.groupby("identity").s_m.mean().round()
        rows.append(dict(dataset=ds, method=name, n=len(x), both_detected=len(b), gender_agree=(b.s_m == b.o_m).mean(),
                         age_mae=(b.s_age - b.o_age).abs().mean(), out_age_lt18=(b.o_age < 18).mean(), src_age_lt18=(b.s_age < 18).mean(),
                         out_age_median=b.o_age.median(), src_age_median=b.s_age.median()))
        print(f"RESULT-ATTR {ds} {name} | gender agree {rows[-1]['gender_agree']:.3f} | age MAE {rows[-1]['age_mae']:.1f} | "
              f"out<18 {rows[-1]['out_age_lt18']:.3f} (src {rows[-1]['src_age_lt18']:.3f}) | n={len(b)}/{len(x)}", flush=True)
    out = W / "results" / ds / "attr_fidelity.csv"; pd.DataFrame(rows).to_csv(out, index=False); print("saved", out)
