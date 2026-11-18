"""P1.6 - identity-clustered bootstrap.

Images of one identity are not independent, so resampling is over IDENTITIES (all images of a drawn
identity come along). Used for 95% CIs and for paired comparisons on the same probes (e.g. diagonal
vs off-diagonal of E2, method A vs method B).
"""
from __future__ import annotations

import numpy as np


def _cluster_index(clusters):
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.nonzero(inv == k)[0] for k in range(len(uniq))]
    return members


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """values: per-image numbers (e.g. 0/1 correct). Returns point estimate + percentile CI."""
    v = np.asarray(values, dtype=np.float64)
    members = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    K = len(members)
    est = float(stat(v))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([members[k] for k in pick])
        boots[b] = stat(v[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(estimate=est, ci_low=float(lo), ci_high=float(hi), n=len(v), n_clusters=K, n_boot=n_boot)


def paired_cluster_bootstrap(a, b, clusters, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference stat(a) - stat(b) on the SAME items (aligned arrays). Two-sided p-value =
    2 * min(P(diff <= 0), P(diff >= 0)) under the bootstrap distribution."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    members = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    K = len(members)
    d0 = float(stat(a) - stat(b))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([members[k] for k in pick])
        boots[i] = stat(a[idx]) - stat(b[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return dict(diff=d0, ci_low=float(lo), ci_high=float(hi), p_value=max(p, 1.0 / n_boot), n_clusters=K)
