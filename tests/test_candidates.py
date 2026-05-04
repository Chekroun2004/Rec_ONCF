from __future__ import annotations

import pandas as pd
import pytest

from rec_oncf.candidates import generate_candidates


def _make_history(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["DateHeureDepartVoyageSegment"] = pd.to_datetime(df["DateHeureDepartVoyageSegment"])
    return df


def test_returns_list_of_strings():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "100", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "200", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "100", "DateHeureDepartVoyageSegment": "2020-01-03"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)


def test_most_frequent_first():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-03"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-04"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert result[0] == "B"


def test_recency_tiebreaker():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-06-01"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert result[0] == "B"


def test_max_candidates_respected():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": str(i), "DateHeureDepartVoyageSegment": f"2020-01-{i:02d}"}
        for i in range(1, 21)
    ])
    result = generate_candidates(history, user_id="U1", max_candidates=5)
    assert len(result) <= 5


def test_unknown_user_returns_empty():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id="UNKNOWN")
    assert result == []


def test_empty_history_returns_empty():
    history = pd.DataFrame(columns=["CodeClient", "LiaisonId", "DateHeureDepartVoyageSegment"])
    result = generate_candidates(history, user_id="U1")
    assert result == []


def test_no_duplicates():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-03"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert len(result) == len(set(result))


def test_per_user_isolation():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U2", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert "B" not in result
    assert "A" in result


def test_integer_user_id_coerced():
    history = _make_history([
        {"CodeClient": "42", "LiaisonId": "X", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id=42)
    assert result == ["X"]


def test_empty_result_is_falsy():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id="UNKNOWN")
    assert not result


def test_non_empty_result_is_truthy():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert result
