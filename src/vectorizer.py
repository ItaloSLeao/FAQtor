"""TF-IDF vectorization utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_text


class FAQVectorizer:
    """Small wrapper around scikit-learn TF-IDF vectorization."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            lowercase=False,
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit_transform(self, texts: Sequence[str]) -> Any:
        """Fit the vectorizer and return the document matrix."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: str | Sequence[str]) -> Any:
        """Transform one text or a sequence of texts."""
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(texts)

    def vocabulary(self) -> set[str]:
        """Return the learned vocabulary terms."""
        return set(self.vectorizer.get_feature_names_out())


def train_vectorizer(
    questions: Sequence[str],
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[FAQVectorizer, Any]:
    """Fit a FAQVectorizer from base questions."""
    vectorizer = FAQVectorizer(ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix
