"""P1.4 - open-set gallery identification with L2-normalised embeddings.

Gallery = CLEAN embeddings of enrolled identities. Each probe is an anonymized image. The probe's own
clean source is always removed from the gallery (`exclude_same_source`): matching an image to its own
original measures reversibility, not re-identification of the person. Templates are either every
gallery image ('all', max similarity per identity) or one mean template per identity ('mean').
"""
from __future__ import annotations

import numpy as np


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def identify(probe_emb, probe_ids, probe_src, gal_emb, gal_ids, gal_src,
             exclude_same_source: bool = True, template: str = "all", ranks=(1, 5, 10)) -> dict:
    """
    probe_emb [P,D], probe_ids [P] identity, probe_src [P] source image_id of each probe
    gal_emb [G,D], gal_ids [G], gal_src [G] image_id of each gallery image
    Returns rank-k rates + per-probe arrays (true rank, correct@1) for the bootstrap.
    """
    pe, ge = _l2(probe_emb), _l2(gal_emb)
    probe_ids, gal_ids = np.asarray(probe_ids), np.asarray(gal_ids)
    probe_src, gal_src = np.asarray(probe_src), np.asarray(gal_src)
    ids = np.unique(gal_ids)
    col = {k: i for i, k in enumerate(ids)}
    gcol = np.array([col[g] for g in gal_ids])
    S = pe @ ge.T                                                     # [P,G]
    if exclude_same_source:
        S = np.where(probe_src[:, None] == gal_src[None, :], -np.inf, S)
    # identity scores [P,I]
    I = len(ids)
    if template == "all":
        scores = np.full((len(pe), I), -np.inf, dtype=np.float32)
        for j in range(I):
            m = gcol == j
            scores[:, j] = S[:, m].max(1)
    elif template == "mean":
        scores = np.full((len(pe), I), -np.inf, dtype=np.float32)
        for j in range(I):
            m = gcol == j
            for p in range(len(pe)):
                ok = m & np.isfinite(S[p])
                if ok.any():
                    t = _l2(ge[ok].mean(0, keepdims=True))[0]
                    scores[p, j] = float(pe[p] @ t)
    else:
        raise ValueError(template)
    has_gallery = np.array([pid in col and np.isfinite(scores[p, col[pid]]) for p, pid in enumerate(probe_ids)])
    true = np.array([scores[p, col[pid]] if has_gallery[p] else -np.inf for p, pid in enumerate(probe_ids)])
    rank = (scores > true[:, None]).sum(1) + 1                         # ties count against the defender
    rank = np.where(has_gallery, rank, np.iinfo(np.int32).max)
    out = dict(n_probes=int(len(pe)), n_identities=int(I), n_without_gallery=int((~has_gallery).sum()),
               chance_rank1=1.0 / I)
    valid = has_gallery
    for k in ranks:
        out[f"rank{k}"] = float((rank[valid] <= k).mean()) if valid.any() else float("nan")
    out["_per_probe_rank"] = rank
    out["_per_probe_correct1"] = (rank == 1) & valid
    out["_valid"] = valid
    return out
