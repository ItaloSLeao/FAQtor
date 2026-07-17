"""Dense semantic retrieval for FAQ questions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.config import (
    AMBIGUITY_MARGIN,
    DEFAULT_TOP_K,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    FALLBACK_MESSAGE,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.embedding_cache import EmbeddingCache, compute_embedding_fingerprint
from src.search import SearchResult
from src.semantic_preprocessing import normalize_semantic_text


REQUIRED_FAQ_COLUMNS = {"id", "pergunta", "resposta", "categoria"}
_DEFAULT_CACHE = object()


class EmbeddingModel(Protocol):
    """Describe the subset of SentenceTransformer used by the retriever."""

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode one text or a sequence of texts."""


@lru_cache(maxsize=None)
def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
    revision: str | None = EMBEDDING_MODEL_REVISION,
) -> EmbeddingModel:
    """Load one model, preferring local files before attempting a download."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        message = (
            "Unable to import sentence-transformers. Install the project "
            "dependencies before loading the semantic embedding model."
        )
        raise RuntimeError(message) from error

    model_options: dict[str, Any] = {"device": device}
    if revision is not None:
        model_options["revision"] = revision

    try:
        try:
            return SentenceTransformer(
                model_name,
                local_files_only=True,
                **model_options,
            )
        except OSError:
            return SentenceTransformer(model_name, **model_options)
    except (OSError, RuntimeError, ValueError) as error:
        message = (
            f"Unable to load embedding model '{model_name}'. The first execution "
            "must download it and therefore requires an internet connection. "
            "After that, later executions can reuse the local model cache."
        )
        raise RuntimeError(message) from error


# This descriptive alias keeps the loader easy to discover for callers and tests.
load_sentence_transformer = load_embedding_model


class SemanticRetriever:
    """Retrieve FAQ answers by normalized dense embedding similarity."""

    method = "semantic"

    def __init__(
        self,
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        *,
        confidence_threshold: float | None = None,
        model_name: str = EMBEDDING_MODEL_NAME,
        model_revision: str | None = EMBEDDING_MODEL_REVISION,
        device: str = EMBEDDING_DEVICE,
        cache_dir: str | Path | None | object = _DEFAULT_CACHE,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
        model: EmbeddingModel | None = None,
        model_loader: Callable[[str], EmbeddingModel] | None = None,
    ) -> None:
        if confidence_threshold is not None:
            threshold = confidence_threshold

        self.confidence_threshold = self._validate_threshold(threshold)
        self.threshold = self.confidence_threshold
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        if cache_dir is _DEFAULT_CACHE:
            cache_dir = (
                None
                if model is not None or model_loader is not None
                else EMBEDDING_CACHE_DIR
            )
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.ambiguity_margin = float(ambiguity_margin)
        if not math.isfinite(self.ambiguity_margin) or self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be a finite non-negative value")
        self.model = model
        self.model_loader = model_loader
        self.faq_dataframe: pd.DataFrame | None = None
        self.corpus_embeddings: np.ndarray | None = None
        self.embedding_dimension: int | None = None

    @property
    def question_embeddings(self) -> np.ndarray | None:
        """Expose the in-memory corpus matrix under a domain-specific name."""
        return self.corpus_embeddings

    def fit(self, faq_dataframe: pd.DataFrame) -> "SemanticRetriever":
        """Encode or load all FAQ question embeddings exactly once."""
        self._validate_dataframe(faq_dataframe)
        dataframe = faq_dataframe.copy().reset_index(drop=True)
        questions = [
            normalize_semantic_text(question)
            for question in dataframe["pergunta"].tolist()
        ]
        blank_indices = [
            index for index, question in enumerate(questions) if not question
        ]
        if blank_indices:
            joined = ", ".join(str(index) for index in blank_indices)
            raise ValueError(f"FAQ questions must not be blank (rows: {joined}).")

        model = self._get_model()
        declared_dimension = self._declared_embedding_dimension(model)
        fingerprint = compute_embedding_fingerprint(
            dataframe["id"].tolist(),
            questions,
            self.model_name,
            self.model_revision,
        )

        cache = EmbeddingCache(self.cache_dir) if self.cache_dir is not None else None
        embeddings = None
        if cache is not None:
            embeddings = cache.load(
                fingerprint=fingerprint,
                model_name=self.model_name,
                model_revision=self.model_revision,
                row_count=len(dataframe),
                embedding_dimension=declared_dimension,
            )

        if embeddings is None:
            encoded = model.encode(
                questions,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings = self._prepare_corpus_embeddings(
                encoded,
                row_count=len(dataframe),
                declared_dimension=declared_dimension,
            )
            if cache is not None:
                cache.save(
                    embeddings,
                    fingerprint=fingerprint,
                    model_name=self.model_name,
                    model_revision=self.model_revision,
                )
        else:
            embeddings = self._prepare_corpus_embeddings(
                embeddings,
                row_count=len(dataframe),
                declared_dimension=declared_dimension,
            )

        self.faq_dataframe = dataframe
        self.corpus_embeddings = embeddings
        self.embedding_dimension = int(embeddings.shape[1])
        return self

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return ranked semantic candidates with a confidence decision."""
        self._ensure_fitted()
        assert self.faq_dataframe is not None
        assert self.corpus_embeddings is not None
        assert self.model is not None

        effective_threshold = self._validate_threshold(
            self.confidence_threshold if threshold is None else threshold,
        )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer greater than zero")
        normalized_query = normalize_semantic_text(query)
        if not normalized_query:
            return self._empty_response(query, effective_threshold)

        encoded_query = self.model.encode(
            normalized_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = self._prepare_query_embedding(encoded_query)
        if query_embedding is None:
            return self._empty_response(query, effective_threshold)

        scores = self.corpus_embeddings @ query_embedding
        all_ranked_indices = np.argsort(-scores, kind="stable")
        result_count = min(top_k, len(all_ranked_indices))
        ranked_indices = all_ranked_indices[:result_count]
        results = [
            self._row_to_result(index, scores[index], rank)
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        top_result = results[0]
        is_confident = top_result["score"] >= effective_threshold
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
            "suggestions": [],
            "method": self.method,
            "score_gap": score_gap,
            "is_ambiguous": is_ambiguous,
        }

    def _get_model(self) -> EmbeddingModel:
        if self.model is None:
            if self.model_loader is None:
                self.model = load_embedding_model(
                    self.model_name,
                    self.device,
                    self.model_revision,
                )
            else:
                self.model = self.model_loader(self.model_name)
        return self.model

    @staticmethod
    def _declared_embedding_dimension(model: EmbeddingModel) -> int | None:
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = getattr(
                model,
                "get_sentence_embedding_dimension",
                None,
            )
        if not callable(dimension_getter):
            return None

        dimension = dimension_getter()
        if dimension is None:
            return None
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("The embedding model reported an invalid dimension.")
        return dimension

    @classmethod
    def _prepare_corpus_embeddings(
        cls,
        embeddings: Any,
        *,
        row_count: int,
        declared_dimension: int | None,
    ) -> np.ndarray:
        matrix = np.asarray(embeddings)
        if matrix.ndim == 1 and row_count == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != row_count:
            raise ValueError(
                "The model returned corpus embeddings with an invalid shape.",
            )
        if declared_dimension is not None and matrix.shape[1] != declared_dimension:
            raise ValueError(
                "The model embedding dimension does not match the encoded corpus.",
            )
        return cls._normalize_matrix(matrix)

    def _prepare_query_embedding(self, embedding: Any) -> np.ndarray | None:
        vector = np.asarray(embedding)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(
                "The model returned a query embedding with an invalid shape.",
            )
        if (
            self.embedding_dimension is None
            or vector.shape[0] != self.embedding_dimension
        ):
            raise ValueError(
                "The query embedding dimension does not match the FAQ embeddings.",
            )
        if (
            not np.issubdtype(vector.dtype, np.number)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("The query embedding must contain finite numeric values.")

        vector = vector.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return None
        return vector / norm

    @staticmethod
    def _normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError("Corpus embeddings must contain numeric values.")
        matrix = embeddings.astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("Corpus embeddings must contain only finite values.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Corpus embeddings must not contain zero vectors.")
        return np.ascontiguousarray(matrix / norms)

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

    @staticmethod
    def _validate_dataframe(faq_dataframe: pd.DataFrame) -> None:
        missing_columns = REQUIRED_FAQ_COLUMNS.difference(faq_dataframe.columns)
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing FAQ columns: {joined}")
        if faq_dataframe.empty:
            raise ValueError("The FAQ dataframe must contain at least one row.")

    def _ensure_fitted(self) -> None:
        if self.faq_dataframe is None or self.corpus_embeddings is None:
            raise RuntimeError("SemanticRetriever must be fitted before search.")

    @staticmethod
    def _validate_threshold(value: Any) -> float:
        threshold = float(value)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite value between -1 and 1")
        return threshold
