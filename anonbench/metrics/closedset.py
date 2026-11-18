"""P1.3 - closed-set identification metrics (classifier predicts one of C enrolled identities).

RIR is reported two ways so the F2 floor is visible rather than hidden:
    rir_raw      = correct / total                       (what the classifier actually does)
    rir_floored  = max(correct / total, 1 / C)           (EDIALS Eq. 1)
`floor_active` says whether the floored value is the floor itself.
"""
from __future__ import annotations

import numpy as np


def topk_correct(logits: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Boolean per row: true label within the k highest logits."""
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return (top == labels[:, None]).any(axis=1)


def closed_set(logits: np.ndarray, labels: np.ndarray, num_classes: int | None = None) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError(f"logits {logits.shape} vs labels {labels.shape}")
    C = int(num_classes or logits.shape[1])
    n = len(labels)
    if n == 0:
        return dict(n=0, num_classes=C)
    c1 = topk_correct(logits, labels, 1)
    c5 = topk_correct(logits, labels, 5)
    raw = float(c1.mean())
    floor = 1.0 / C
    # rank of the true class (1 = top); ties broken pessimistically for the defender
    true_logit = logits[np.arange(n), labels]
    rank = (logits > true_logit[:, None]).sum(1) + 1
    return dict(
        n=n, num_classes=C,
        rank1=raw, rank5=float(c5.mean()),
        rir_raw=raw * 100, rir_floored=max(raw, floor) * 100, chance=floor * 100,
        floor_active=bool(raw < floor),
        mean_true_rank=float(rank.mean()), median_true_rank=float(np.median(rank)),
    )


def per_image_correct(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-image rank-1 correctness (feeds the identity-clustered bootstrap)."""
    return topk_correct(logits, labels, 1)
