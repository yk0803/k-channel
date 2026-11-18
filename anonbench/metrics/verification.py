"""P1.5 - 1:1 verification.

Thresholds are fixed on CLEAN genuine/impostor pairs (what an operator would deploy) and then applied
to anonymized-vs-clean pairs. Genuine anonymized pairs exclude the probe's own source.
"""
from __future__ import annotations

import numpy as np


def threshold_at_far(impostor_scores: np.ndarray, far: float) -> float:
    """Smallest threshold t with FAR(t) = P(impostor >= t) <= far."""
    s = np.sort(np.asarray(impostor_scores, dtype=np.float64))[::-1]
    if len(s) == 0:
        raise ValueError("no impostor scores")
    k = int(np.floor(far * len(s)))            # number of impostors allowed at/above t
    if k >= len(s):
        return float(s[-1])
    # threshold strictly above the (k+1)-th highest impostor score
    return float(np.nextafter(s[k], np.inf))


def tar_at(genuine_scores, threshold: float) -> float:
    g = np.asarray(genuine_scores, dtype=np.float64)
    return float((g >= threshold).mean()) if len(g) else float("nan")


def far_at(impostor_scores, threshold: float) -> float:
    i = np.asarray(impostor_scores, dtype=np.float64)
    return float((i >= threshold).mean()) if len(i) else float("nan")


def roc_auc(genuine, impostor) -> float:
    """Mann-Whitney AUC with tie correction."""
    g = np.asarray(genuine, np.float64)
    i = np.asarray(impostor, np.float64)
    allv = np.concatenate([g, i])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    j = 0
    while j < len(sv):
        k = j
        while k + 1 < len(sv) and sv[k + 1] == sv[j]:
            k += 1
        ranks[order[j:k + 1]] = (j + k) / 2 + 1
        j = k + 1
    rg = ranks[: len(g)].sum()
    return float((rg - len(g) * (len(g) + 1) / 2) / (len(g) * len(i)))


def pair_scores(a_emb, a_ids, a_src, b_emb, b_ids, b_src, max_impostors: int | None = 2_000_000, seed: int = 0):
    """All genuine pairs (same identity, different source image) and impostor pairs
    (different identity; subsampled to max_impostors). Embeddings must be L2-normalised.
    Returns genuine scores, impostor scores, and the probe index of each genuine pair."""
    a_emb, b_emb = np.asarray(a_emb, np.float32), np.asarray(b_emb, np.float32)
    a_ids, b_ids, a_src, b_src = map(np.asarray, (a_ids, b_ids, a_src, b_src))
    S = a_emb @ b_emb.T
    same_id = a_ids[:, None] == b_ids[None, :]
    same_src = a_src[:, None] == b_src[None, :]
    gen_mask = same_id & ~same_src
    imp_mask = ~same_id
    gi, gj = np.nonzero(gen_mask)
    genuine = S[gi, gj]
    ii, ij = np.nonzero(imp_mask)
    if max_impostors and len(ii) > max_impostors:
        sel = np.random.default_rng(seed).choice(len(ii), max_impostors, replace=False)
        ii, ij = ii[sel], ij[sel]
    return genuine, S[ii, ij], gi


def verification_report(clean_gen, clean_imp, anon_gen, anon_imp=None, fars=(1e-3, 1e-4)) -> dict:
    out = dict(clean_auc=roc_auc(clean_gen, clean_imp), n_clean_gen=len(clean_gen), n_clean_imp=len(clean_imp),
               n_anon_gen=len(anon_gen))
    for f in fars:
        if f * len(clean_imp) < 1:
            out[f"note_far{f:g}"] = f"only {len(clean_imp)} impostors: FAR {f:g} not resolvable"
        t = threshold_at_far(clean_imp, f)
        out[f"thr@{f:g}"] = t
        out[f"clean_TAR@{f:g}"] = tar_at(clean_gen, t)
        out[f"anon_TAR@{f:g}"] = tar_at(anon_gen, t)
        if anon_imp is not None:
            out[f"anon_FAR@{f:g}"] = far_at(anon_imp, t)
    if anon_imp is not None:
        out["anon_auc"] = roc_auc(anon_gen, anon_imp)
    return out
