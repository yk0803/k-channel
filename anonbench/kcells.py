"""State cells for k-Channel: every cell keeps identity share <= 1/k (Proposition 1 holds per cell)."""
import pandas as pd


def _ok(h, k):
    return h.identity.value_counts(normalize=True).max() <= 1 / k + 1e-12


def assign_levels(m, k, levels):
    """levels: list of column lists, finest first, last one [] (class only).
    Images take the finest level whose cell passes. Leftovers form one class-level cell; if it fails,
    whole cells (coarsest first, largest first) are moved into it until it passes."""
    m = m.copy(); m["level"] = -1; m["cell"] = -1; cid, repaired, last = 0, 0, len(levels) - 1
    for _, g in m.groupby("cls"):
        idx = g.index
        for li, cols in enumerate(levels[:-1]):
            rem = idx[m.loc[idx, "level"].to_numpy() == -1]
            if not len(rem):
                break
            for _, h in m.loc[rem].groupby(cols):
                if _ok(h, k):
                    m.loc[h.index, "level"] = li; m.loc[h.index, "cell"] = cid; cid += 1
        rem = idx[m.loc[idx, "level"].to_numpy() == -1]
        if len(rem):
            L = cid; cid += 1; m.loc[rem, "level"] = last; m.loc[rem, "cell"] = L
            sel = lambda: idx[m.loc[idx, "cell"].to_numpy() == L]
            if not _ok(m.loc[sel()], k):
                repaired += 1
                sub = m.loc[idx]; sub = sub[sub.cell != L]
                order = sub.groupby("cell").agg(level=("level", "first"), n=("level", "size")).sort_values(["level", "n"], ascending=[False, False])
                for cc in order.index:
                    mv = idx[m.loc[idx, "cell"].to_numpy() == cc]; m.loc[mv, "level"] = last; m.loc[mv, "cell"] = L
                    if _ok(m.loc[sel()], k):
                        break
    return m, repaired
