"""Experimental evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    FAQ_PATH,
    TEST_QUERIES_PATH,
)
from src.retriever import FAQRetriever


def load_faq(path: str | Path = FAQ_PATH) -> pd.DataFrame:
    """Load the FAQ dataset."""
    return pd.read_csv(path)


def load_test_queries(path: str | Path = TEST_QUERIES_PATH) -> pd.DataFrame:
    """Load evaluation queries."""
    return pd.read_csv(path)


def evaluate_retriever(
    retriever: FAQRetriever,
    test_dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Compute ranking metrics for a fitted retriever."""
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    details: list[dict[str, Any]] = []

    for row in test_dataframe.itertuples(index=False):
        query = str(row.query)
        expected_id = str(row.expected_id)
        response = retriever.search(query, top_k=top_k)
        retrieved_ids = [str(result["id"]) for result in response["results"]]

        rank = None
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1

        hit_at_1 = rank == 1
        hit_at_k = rank is not None
        hits_at_1 += int(hit_at_1)
        hits_at_k += int(hit_at_k)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)

        details.append(
            {
                "query": query,
                "expected_id": expected_id,
                "rank": rank,
                "top_id": retrieved_ids[0] if retrieved_ids else None,
                "hit_at_1": hit_at_1,
                f"hit_at_{top_k}": hit_at_k,
            },
        )

    total = len(test_dataframe)
    return {
        "total_queries": total,
        "accuracy_at_1": hits_at_1 / total if total else 0.0,
        f"accuracy_at_{top_k}": hits_at_k / total if total else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / total if total else 0.0
        ),
        "details": details,
    }


def evaluate(
    faq_path: str | Path = FAQ_PATH,
    test_queries_path: str | Path = TEST_QUERIES_PATH,
    top_k: int = DEFAULT_TOP_K,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Load data, fit the retriever and compute metrics."""
    faq_dataframe = load_faq(faq_path)
    test_dataframe = load_test_queries(test_queries_path)
    retriever = FAQRetriever(confidence_threshold=confidence_threshold).fit(
        faq_dataframe,
    )
    return evaluate_retriever(retriever, test_dataframe, top_k=top_k)


def format_report(metrics: dict[str, Any], top_k: int = DEFAULT_TOP_K) -> str:
    """Format evaluation metrics for terminal output."""
    accuracy_at_k_key = f"accuracy_at_{top_k}"
    lines = [
        "FAQtor - Relatório de avaliação",
        f"Consultas avaliadas: {metrics['total_queries']}",
        f"Accuracy@1: {metrics['accuracy_at_1']:.3f}",
        f"Accuracy@{top_k}: {metrics[accuracy_at_k_key]:.3f}",
        f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}",
    ]
    return "\n".join(lines)


def run_evaluation(
    faq_path: str | Path = FAQ_PATH,
    test_queries_path: str | Path = TEST_QUERIES_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run evaluation and print a short report."""
    metrics = evaluate(faq_path, test_queries_path, top_k=top_k)
    print(format_report(metrics, top_k=top_k))
    return metrics
