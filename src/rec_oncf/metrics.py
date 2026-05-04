from __future__ import annotations

import numpy as np


def hit_rate_at_k(y_true: np.ndarray, y_proba: np.ndarray, *, k: int) -> float:
    """HitRate@k for multi-class probabilities.

    y_true: shape (n,)
    y_proba: shape (n, n_classes)
    """
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    hits = (topk == y_true.reshape(-1, 1)).any(axis=1)
    return float(np.mean(hits))


def mrr_at_k(y_true: np.ndarray, y_proba: np.ndarray, *, k: int) -> float:
    """Mean Reciprocal Rank at k for multi-class probabilities.

    y_true: shape (n,)
    y_proba: shape (n, n_classes)
    """
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    rr = np.zeros(len(y_true))
    for rank in range(k):
        hit = (topk[:, rank] == y_true) & (rr == 0)
        rr[hit] = 1.0 / (rank + 1)
    return float(np.mean(rr))
