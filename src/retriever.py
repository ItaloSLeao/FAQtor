"""FAQ retrieval utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_TOP_K
from src.vectorizer import FAQVectorizer, train_vectorizer


REQUIRED_FAQ_COLUMNS = {"id", "pergunta", "resposta", "categoria"}


class FAQRetriever:
    """Retrieve FAQ answers with TF-IDF and cosine similarity."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.ngram_range = ngram_range
        self.faq_dataframe: pd.DataFrame | None = None
        self.vectorizer: FAQVectorizer | None = None
        self.question_matrix: Any | None = None

    def fit(self, faq_dataframe: pd.DataFrame) -> "FAQRetriever":
        """Fit the retriever with a FAQ dataframe."""
        missing_columns = REQUIRED_FAQ_COLUMNS.difference(faq_dataframe.columns)
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing FAQ columns: {joined}")

        self.faq_dataframe = faq_dataframe.copy().reset_index(drop=True)
        questions = self.faq_dataframe["pergunta"].astype(str).tolist()
        self.vectorizer, self.question_matrix = train_vectorizer(
            questions,
            ngram_range=self.ngram_range,
        )
        return self

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return top-k FAQ rows ranked by cosine similarity."""
        self._ensure_fitted()
        assert self.faq_dataframe is not None
        assert self.vectorizer is not None
        assert self.question_matrix is not None

        query_vector = self.vectorizer.transform(query)
        scores = cosine_similarity(query_vector, self.question_matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [self._row_to_result(index, scores[index]) for index in ranked_indices]

    def _row_to_result(self, index: int, score: float) -> dict[str, Any]:
        assert self.faq_dataframe is not None
        row = self.faq_dataframe.iloc[index]
        return {
            "id": row["id"],
            "pergunta": row["pergunta"],
            "resposta": row["resposta"],
            "categoria": row["categoria"],
            "score": float(score),
        }

    def _ensure_fitted(self) -> None:
        if self.faq_dataframe is None or self.vectorizer is None:
            raise RuntimeError("FAQRetriever must be fitted before search.")
