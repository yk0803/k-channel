"""P1.2 - clean-data sanity for every embedder, BEFORE the evaluator freeze (P1.8).

For each embedder on the ORIGINAL MTF images (own landmarks):
  * verification: genuine = same identity, different image; impostor = different identity
    -> AUC, TAR@FAR 1e-3 / 1e-4
  * gallery identification: each test image vs the train+val images (rank-1/5)
Each is reported on ALL images and on the subset whose landmarks were really detected
(`detonly`), so the effect of the fallback template is visible.
For CVLface models both channel orders are measured (B6).

Pass criterion (written before running): AUC >= 0.95 and TAR@1e-3 >= 0.50 on clean MTF (all images).

    python -m anonbench.sanity --embedders arcface_r50_webface600k facenet_casia
      -> work/results/mtf/P1_sanity/clean_sanity.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .align import load_landmarks
from .config import Paths, load_config
from .data import load_index
from .embed import compute, embed_set, items_clean
from .metrics.gallery import identify
from .metrics.verification import pair_scores, verification_report
from .models.embedders import EMBEDDER_INFO, embedder_meta, get_embedder

PASS_AUC, PASS_TAR = 0.95, 0.50


def evaluate(df: pd.DataFrame, E: np.ndarray) -> dict:
    ids, src, split = df.identity.to_numpy(), df.image_id.to_numpy(), df.split.to_numpy()
    gen, imp, _ = pair_scores(E, ids, src, E, ids, src, max_impostors=3_000_000)
    v = verification_report(gen, imp, gen, None)
    te, ga = split == "test", split != "test"
    g = identify(E[te], ids[te], src[te], E[ga], ids[ga], src[ga])
    return dict(auc=v["clean_auc"], tar_1e3=v["clean_TAR@0.001"], tar_1e4=v["clean_TAR@0.0001"],
                rank1=g["rank1"], rank5=g["rank5"], n_images=len(df))


def _save(out, rows):
    """Upsert rows into the results CSV (keyed by embedder + channel order) after EVERY model."""
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~(old.embedder + "|" + old.channel_order.astype(str)).isin(new.embedder + "|" + new.channel_order)]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--embedders", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    idx = load_index(cfg, a.dataset).set_index("image_id", drop=False)
    lm = load_landmarks(cfg, a.dataset)
    det = pd.read_csv(P.landmarks_csv()).set_index("path").detected
    names = a.embedders or cfg["evaluators"]["embedders"]
    out_csv = P.results("P1_sanity") / "clean_sanity.csv"
    jobs = [(n, o) for n in names for o in (["rgb", "bgr"] if EMBEDDER_INFO.get(n, ("fake",))[0] == "cvlface" else [None])]
    for i, (n, order) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {n} {order or ''}: embedding 5246 images + scoring ...", flush=True)
        try:
            if order is None:
                out = embed_set(cfg, a.dataset, n, None, landmarks=lm)
            else:   # B6: both orders, uncached, so neither wins by default
                out = compute(cfg, items_clean(cfg, a.dataset), get_embedder(n, cfg, channel_order=order), "transfer", lm)
            df = idx.loc[list(out["image_id"])]
            ok = df.path.map(det).fillna(0).astype(bool).to_numpy()
            r_all = evaluate(df, out["emb"])
            r_det = evaluate(df[ok], out["emb"][ok])
        except Exception as e:  # keep going: one broken model must not lose the others
            print(f"[{i}/{len(jobs)}] ERROR {n} {order or ''}: {e!r}", flush=True)
            continue
        r = dict(embedder=n, channel_order=order or "-", **embedder_meta(n),
                 **{f"{k}": v for k, v in r_all.items()}, **{f"{k}_detonly": v for k, v in r_det.items()})
        r["pass"] = bool(r["auc"] >= PASS_AUC and r["tar_1e3"] >= PASS_TAR)
        _save(out_csv, [r])
        print(f"[{i}/{len(jobs)}] RESULT {n:28s} {r['channel_order']:4s} AUC {r['auc']:.4f} TAR@1e-3 {r['tar_1e3']:.3f} "
              f"rank1 {r['rank1']:.3f} | detected-only rank1 {r['rank1_detonly']:.3f} | {'PASS' if r['pass'] else 'FAIL'} (saved)",
              flush=True)
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
