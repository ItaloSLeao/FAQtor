"""Tests for comparative retrieval evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.evaluation import (
    calibrate_confidence_threshold,
    compare_retrievers,
    evaluate_retriever,
    load_test_queries,
)


def test_threshold_calibration_reports_tradeoffs_without_reranking() -> None:
    details = [
        {
            "scored": True,
            "expected_id": "1",
            "expected_fallback": False,
            "hit_at_1": True,
            "top_score": 0.80,
        },
        {
            "scored": True,
            "expected_id": "2",
            "expected_fallback": False,
            "hit_at_1": False,
            "top_score": 0.90,
        },
        {
            "scored": True,
            "expected_id": "",
            "expected_fallback": True,
            "hit_at_1": False,
            "top_score": 0.60,
        },
        {
            "scored": True,
            "expected_id": "",
            "expected_fallback": True,
            "hit_at_1": False,
            "top_score": 0.10,
        },
    ]

    calibration = calibrate_confidence_threshold(
        details,
        selected_threshold=0.50,
        thresholds=[0.50, 0.70, 0.85],
    )

    assert calibration["best_thresholds"] == [0.70]
    assert calibration["best_end_to_end_accuracy"] == pytest.approx(0.75)
    assert calibration["selected"]["false_positives"] == 1
    assert calibration["selected"]["false_negatives"] == 0


class StubRetriever:
    """Return predefined responses and record evaluation calls."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        self.calls.append((query, top_k))
        return deepcopy(self.responses[query])


def response(
    ids: list[str],
    *,
    confident: bool,
    scores: list[float] | None = None,
) -> dict[str, Any]:
    """Build the compatible subset of a retriever response."""
    if scores is None:
        scores = [1.0 - index * 0.1 for index in range(len(ids))]
    results = [
        {"id": faq_id, "score": score}
        for faq_id, score in zip(ids, scores, strict=True)
    ]
    return {
        "results": results,
        "top_result": results[0] if results else None,
        "score": scores[0] if scores else 0.0,
        "is_confident": confident,
        "fallback_message": None if confident else "fallback",
    }


def test_evaluation_counts_negatives_false_positives_and_false_negatives() -> None:
    cases = pd.DataFrame(
        [
            {
                "query": "positive correct",
                "expected_id": "1",
                "expected_fallback": "false",
                "case_type": "positive",
            },
            {
                "query": "negative rejected",
                "expected_id": "",
                "expected_fallback": "true",
                "case_type": "negative",
            },
            {
                "query": "negative accepted",
                "expected_id": "",
                "expected_fallback": "true",
                "case_type": "negative",
            },
            {
                "query": "positive rejected",
                "expected_id": "1",
                "expected_fallback": "false",
                "case_type": "positive",
            },
            {
                "query": "ambiguous review",
                "expected_id": "",
                "expected_fallback": "",
                "case_type": "ambiguous",
            },
        ],
    )
    retriever = StubRetriever(
        {
            "positive correct": response(
                ["1", "2"],
                confident=True,
                scores=[0.90, 0.40],
            ),
            "negative rejected": response(
                ["2", "1"],
                confident=False,
                scores=[0.20, 0.10],
            ),
            "negative accepted": response(
                ["2", "1"],
                confident=True,
                scores=[0.85, 0.30],
            ),
            "positive rejected": response(
                ["2", "1"],
                confident=False,
                scores=[0.45, 0.44],
            ),
            "ambiguous review": response(
                ["1", "2"],
                confident=True,
                scores=[0.60, 0.59],
            ),
        },
    )

    metrics = evaluate_retriever(
        retriever,
        cases,
        top_k=2,
        ambiguity_margin=0.02,
    )

    assert metrics["total_queries"] == 5
    assert metrics["scored_queries"] == 4
    assert metrics["unscored_queries"] == 1
    assert metrics["positive_queries"] == 2
    assert metrics["negative_queries"] == 2
    assert metrics["correct_fallbacks"] == 1
    assert metrics["incorrect_fallbacks"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["end_to_end_correct"] == 2
    assert metrics["end_to_end_accuracy"] == pytest.approx(0.5)

    ambiguous_detail = metrics["details"][-1]
    assert ambiguous_detail["expected_fallback"] is None
    assert ambiguous_detail["scored"] is False
    assert ambiguous_detail["end_to_end_correct"] is None
    assert ambiguous_detail["ambiguous"] is True


def test_compare_retrievers_pairs_the_same_cases_and_groups_outcomes() -> None:
    cases = pd.DataFrame(
        [
            {
                "query": "semantic wins",
                "expected_id": "1",
                "expected_fallback": "false",
            },
            {
                "query": "tfidf wins",
                "expected_id": "1",
                "expected_fallback": "false",
            },
            {
                "query": "both fail",
                "expected_id": "1",
                "expected_fallback": "false",
            },
            {
                "query": "both reject",
                "expected_id": "",
                "expected_fallback": "true",
            },
            {
                "query": "needs review",
                "expected_id": "",
                "expected_fallback": "",
            },
        ],
    )
    tfidf = StubRetriever(
        {
            "semantic wins": response(["2", "1"], confident=True),
            "tfidf wins": response(["1", "2"], confident=True),
            "both fail": response(["2", "3"], confident=True),
            "both reject": response(["2", "3"], confident=False),
            "needs review": response(["2", "1"], confident=True),
        },
    )
    semantic = StubRetriever(
        {
            "semantic wins": response(["1", "2"], confident=True),
            "tfidf wins": response(["2", "1"], confident=True),
            "both fail": response(["3", "2"], confident=True),
            "both reject": response(["3", "2"], confident=False),
            "needs review": response(["1", "2"], confident=True),
        },
    )

    report = compare_retrievers(tfidf, semantic, cases, top_k=2)
    comparison = report["comparison"]

    assert [case["query"] for case in comparison["tfidf_wrong_semantic_correct"]] == [
        "semantic wins",
    ]
    assert [case["query"] for case in comparison["semantic_wrong_tfidf_correct"]] == [
        "tfidf wins",
    ]
    assert [case["query"] for case in comparison["both_wrong"]] == ["both fail"]
    assert [case["query"] for case in comparison["both_correct"]] == [
        "both reject",
    ]
    assert [case["query"] for case in comparison["unscored"]] == ["needs review"]
    assert [
        case["query"] for case in comparison["correct_rejections"]["both"]
    ] == ["both reject"]

    expected_calls = [(query, 3) for query in cases["query"]]
    assert tfidf.calls == expected_calls
    assert semantic.calls == expected_calls


def test_load_test_queries_preserves_empty_and_whitespace_queries(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "queries.csv"
    csv_path.write_text(
        "query,expected_id,categoria,expected_fallback,case_type\n"
        '"",,,,empty\n'
        '"   ",,,,spaces\n',
        encoding="utf-8",
    )

    dataframe = load_test_queries(csv_path)

    assert dataframe["query"].tolist() == ["", "   "]
    assert dataframe["expected_id"].tolist() == ["", ""]
    assert dataframe["query"].isna().sum() == 0
