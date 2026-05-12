from __future__ import annotations

import pandas as pd

from rec_oncf.popularity import build_popularity_list, load_popularity, save_popularity


def test_build_popularity_orders_by_descending_frequency():
    df = pd.DataFrame({"LiaisonId": ["A", "A", "A", "B", "B", "C"]})
    assert build_popularity_list(df) == ["A", "B", "C"]


def test_build_popularity_includes_every_liaison_as_str():
    df = pd.DataFrame({"LiaisonId": [10, 20, 30, 20]})
    result = build_popularity_list(df)
    assert set(result) == {"10", "20", "30"}
    assert all(isinstance(x, str) for x in result)


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "sub" / "popularity.joblib"
    save_popularity(["A", "B", "C"], p)
    assert load_popularity(p) == ["A", "B", "C"]
