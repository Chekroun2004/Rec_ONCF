from __future__ import annotations

import numpy as np
import pytest
from rec_oncf.metrics import mrr_at_k


def test_mrr_at_k_perfect():
    y_true = np.array([0, 1, 2])
    proba = np.eye(3)
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(1.0)


def test_mrr_at_k_second_place():
    y_true = np.array([0])
    proba = np.array([[0.4, 0.6]])
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.5)


def test_mrr_at_k_not_in_topk():
    y_true = np.array([2])
    proba = np.array([[0.6, 0.3, 0.1]])
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.0)


def test_mrr_at_k_mixed():
    y_true = np.array([0, 2])
    proba = np.array([
        [0.9, 0.05, 0.05],
        [0.5, 0.1, 0.4],
    ])
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(0.75)
