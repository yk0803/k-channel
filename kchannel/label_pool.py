import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import os, time, numpy as np, pandas as pd, cv2
from insightface.app import FaceAnalysis
P = ROOT + "/work/index/genpool.csv"; O = ROOT + "/work/index/genpool_ia.csv"
pool = pd.read_csv(P); done = pd.read_csv(O) if os.path.exists(O) else pd.DataFrame(columns=["path"])
todo = [p for p in pool.path if p not in set(done.path)]
print(f"[prel] pool {len(pool)} | done {len(done)} | todo {len(todo)}", flush=True)
app = FaceAnalysis(name="buffalo_l", root=ROOT + "/weights/insightface",
                   allowed_modules=["detection", "genderage", "landmark_3d_68"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
rows, t0 = [], time.time()
def flush():
    global done, rows
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); tmp = O + ".tmp"; done.to_csv(tmp, index=False); os.replace(tmp, O); rows = []
for n, p in enumerate(todo, 1):
    im = cv2.imread(p); rec = dict(path=p, ia_det=0, ia_pad=0, ia_male=np.nan, ia_age=np.nan, ia_pitch=np.nan, ia_yaw=np.nan, ia_roll=np.nan)
    if im is not None:
        for pad in (0, 0.25):
            x = cv2.copyMakeBorder(im, *[int(pad * im.shape[0])] * 2, *[int(pad * im.shape[1])] * 2, cv2.BORDER_REPLICATE) if pad else im
            f = app.get(x)
            if f:
                f = max(f, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
                pp = getattr(f, "pose", None)
                rec.update(ia_det=1, ia_pad=int(pad > 0), ia_male=int(f.sex == "M"), ia_age=float(f.age),
                           ia_pitch=float(pp[0]) if pp is not None else np.nan, ia_yaw=float(pp[1]) if pp is not None else np.nan,
                           ia_roll=float(pp[2]) if pp is not None else np.nan)
                break
    rows.append(rec)
    if n % 1000 == 0:
        r = n / (time.time() - t0); print(f"[prel] {n}/{len(todo)} | {r:.1f} img/s | ETA {(len(todo)-n)/r/60:.1f} min", flush=True)
    if n % 2000 == 0: flush()
flush(); print("[prel] DONE", len(done), flush=True)
