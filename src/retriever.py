"""Lexical FAQ retrieval utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    AMBIGUITY_MARGIN,
    DEFAULT_TOP_K,
    FALLBACK_MESSAGE,
    TFIDF_SIMILARITY_THRESHOLD,
)
from src.preprocessing import tokenize
from src.search import SearchResult
from src.spelling import suggest_terms as suggest_close_terms
from src.vectorizer import FAQVectorizer, train_vectorizer


REQUIRED_FAQ_COLUMNS = {"id", "pergunta", "resposta", "categoria"}


class TfidfRetriever:
    """Retrieve FAQ answers with TF-IDF and cosine similarity."""

    method = "tfidf"

    def __init__(
        self,
        confidence_threshold: float = TFIDF_SIMILARITY_THRESHOLD,
        ngram_range: tuple[int, int] = (1, 2),
        ambiguity_margin: float = AMBIGUITY_MARGIN,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.ngram_range = ngram_range
        self.ambiguity_margin = ambiguity_margin
        self.faq_dataframe: pd.DataFrame | None = None
        self.vectorizer: FAQVectorizer | None = None
        self.question_matrix: Any | None = None

    def fit(self, faq_dataframe: pd.DataFrame) -> "TfidfRetriever":
        """Fit the retriever with a FAQ dataframe."""
        missing_columns = REQUIRED_FAQ_COLUMNS.difference(faq_dataframe.columns)
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing FAQ columns: {joined}")
        if faq_dataframe.empty:
            raise ValueError("The FAQ dataframe must contain at least one row.")

        self.faq_dataframe = faq_dataframe.copy().reset_index(drop=True)
        questions = self.faq_dataframe["pergunta"].astype(str).tolist()
        self.vectorizer, self.question_matrix = train_vectorizer(
            questions,
            ngram_range=self.ngram_range,
        )
        return self

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return ranked FAQ results with a confidence decision."""
        self._ensure_fitted()
        assert self.faq_dataframe is not None
        assert self.vectorizer is not None
        assert self.question_matrix is not None

        effective_threshold = (
            self.confidence_threshold if threshold is None else float(threshold)
        )
        if query is None or not str(query).strip():
            return self._empty_response(query, effective_threshold)

        query_vector = self.vectorizer.transform(query)
        scores = cosine_similarity(query_vector, self.question_matrix).ravel()
        top_k = min(max(1, top_k), len(scores))
        all_ranked_indices = np.argsort(-scores, kind="stable")
        ranked_indices = all_ranked_indices[:top_k]
        results = [
            self._row_to_result(index, scores[index], rank)
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        top_result = results[0]
        is_confident = top_result["score"] >= effective_threshold
        suggestions = [] if is_confident else self.suggest_terms(query)
        score_gap = self._score_gap(scores, all_ranked_indices)
        is_ambiguous = (
            score_gap is not None and score_gap <= self.ambiguity_margin
        )

        return {
            "query": query,
            "answer": top_result["resposta"] if is_confident else FALLBACK_MESSAGE,
            "is_confident": is_confident,
            "top_result": top_result,
            "results": results,
            "score": top_result["score"],
            "threshold": effective_threshold,
            "fallback_message": None if is_confident else FALLBACK_MESSAGE,
            "suggestions": suggestions,
            "method": self.method,
            "score_gap": score_gap,
            "is_ambiguous": is_ambiguous,
        }

    def vocabulary(self) -> set[str]:
        """Return learned vocabulary terms."""
        self._ensure_fitted()
        assert self.vectorizer is not None
        return self.vectorizer.vocabulary()

    def suggest_terms(self, query: str, max_distance: int = 2) -> list[str]:
        """Suggest vocabulary terms close to query tokens."""
        query_tokens = tokenize(query)
        return suggest_close_terms(
            query_tokens,
            self.vocabulary(),
            max_distance=max_distance,
        )

    def _row_to_result(
        self,
        index: int,
        score: float,
        rank: int,
    ) -> dict[str, Any]:
        assert self.faq_dataframe is not None
        row = self.faq_dataframe.iloc[index]
        return SearchResult(
            id=row["id"],
            pergunta=str(row["pergunta"]),
            resposta=str(row["resposta"]),
            categoria=str(row["categoria"]),
            score=float(score),
            index=int(index),
            rank=rank,
            method=self.method,
        ).to_dict()

    @staticmethod
    def _score_gap(
        scores: np.ndarray,
        ranked_indices: np.ndarray,
    ) -> float | None:
        if len(ranked_indices) < 2:
            return None
        return float(scores[ranked_indices[0]] - scores[ranked_indices[1]])

    def _empty_response(
        self,
        query: str,
        threshold: float,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "is_confident": False,
            "top_result": None,
            "results": [],
            "score": 0.0,
            "threshold": threshold,
            "fallback_message": FALLBACK_MESSAGE,
            "suggestions": [],
            "method": self.method,
            "score_gap": None,
            "is_ambiguous": False,
        }

    def _ensure_fitted(self) -> None:
        if self.faq_dataframe is None or self.vectorizer is None:
            raise RuntimeError("TfidfRetriever must be fitted before search.")


# Preserve the original public name used by the application and external callers.
FAQRetriever = TfidfRetriever
