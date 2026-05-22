from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_make_dataset():
    spec = importlib.util.spec_from_file_location(
        "make_dataset", PROJECT_ROOT / "scripts" / "01_make_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_concat_main_and_extra(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1, 2], "AchteurId": [1, 2], "LiaisonVoyageurSegmentIdSTG": ["L1", "L2"],
    }), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({
        "CodeClient": [3], "AchteurId": [3], "LiaisonVoyageurSegmentIdSTG": ["L3"],
    }), tmp_path / "extra.csv")

    combined = mod.load_and_concat(main, extra)
    assert len(combined) == 3
    assert list(combined.columns) == ["CodeClient", "AchteurId", "LiaisonVoyageurSegmentIdSTG"]


def test_concat_no_extra_returns_main(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1, 2], "AchteurId": [1, 2], "LiaisonVoyageurSegmentIdSTG": ["L1", "L2"],
    }), tmp_path / "main.csv")
    combined = mod.load_and_concat(main, None)
    assert len(combined) == 2


def test_concat_normalizes_aliased_buyer_column(tmp_path):
    """oncf spells it AchteurId, test1 spells it Acheteurid -> must still concat (not a mismatch)."""
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"],
    }), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({
        "CodeClient": [2], "Acheteurid": [2], "LiaisonVoyageurSegmentIdSTG": ["L2"],
    }), tmp_path / "extra.csv")

    combined = mod.load_and_concat(main, extra)
    assert len(combined) == 2
    assert "AchteurId" in combined.columns
    assert "Acheteurid" not in combined.columns
    assert combined["AchteurId"].notna().all()


def test_concat_parses_each_file_dates_with_its_own_convention(tmp_path):
    """oncf is M/D/Y, test1 is D/M/Y. Concatenating must parse each file with its
    OWN convention, not corrupt one with a shared heuristic."""
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"],
        "DatePaiement": ["3/25/2019"],  # M/D/Y -> 25 March
        "DateHeureDepartVoyageSegment": ["3/25/2019 08:00"],
    }), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({
        "CodeClient": [2], "Acheteurid": [2], "LiaisonVoyageurSegmentIdSTG": ["L2"],
        "DatePaiement": ["25/3/2021"],  # D/M/Y -> 25 March
        "DateHeureDepartVoyageSegment": ["25/3/2021 08:00"],
    }), tmp_path / "extra.csv")

    combined = mod.load_and_concat(main, extra)
    dep = combined["DateHeureDepartVoyageSegment"]
    assert pd.api.types.is_datetime64_any_dtype(dep)
    assert dep.iloc[0] == pd.Timestamp("2019-03-25 08:00")  # oncf row intact
    assert dep.iloc[1] == pd.Timestamp("2021-03-25 08:00")  # test1 row intact


def test_concat_real_schema_mismatch_raises(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"],
    }), tmp_path / "main.csv")
    bad = _write(pd.DataFrame({"foo": [1], "bar": [2]}), tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="schema"):
        mod.load_and_concat(main, bad)
