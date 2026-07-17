"""Persistent cache for FAQ question embeddings."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from src.semantic_preprocessing import normalize_semantic_text


CACHE_FORMAT_VERSION = 2
LOGGER = logging.getLogger(__name__)


def compute_embedding_fingerprint(
    ids: Sequence[Any],
    questions: Sequence[Any],
    model_name: str,
    model_revision: str | None = None,
) -> str:
    """Hash the model weights and ordered, normalized FAQ identity data."""
    if len(ids) != len(questions):
        raise ValueError("FAQ ids and questions must have the same length.")

    payload = {
        "model_name": str(model_name),
        "model_revision": model_revision,
        "rows": [
            [normalize_semantic_text(identifier), normalize_semantic_text(question)]
            for identifier, question in zip(ids, questions, strict=True)
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class EmbeddingCache:
    """Store one validated corpus embedding matrix as NPY and JSON files."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.metadata_path = self.cache_dir / "embeddings.json"

    def load(
        self,
        *,
        fingerprint: str,
        model_name: str,
        model_revision: str | None = None,
        row_count: int,
        embedding_dimension: int | None = None,
    ) -> np.ndarray | None:
        """Return a valid cached matrix or None when it must be regenerated."""
        if not self.embeddings_path.is_file() or not self.metadata_path.is_file():
            return None

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            LOGGER.warning("Ignoring unreadable embedding cache metadata: %s", error)
            return None

        if not self._metadata_matches(
            metadata,
            fingerprint=fingerprint,
            model_name=model_name,
            model_revision=model_revision,
            row_count=row_count,
            embedding_dimension=embedding_dimension,
        ):
            return None

        try:
            embeddings = np.load(self.embeddings_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            LOGGER.warning("Ignoring unreadable embedding cache matrix: %s", error)
            return None

        if not self._array_matches(
            embeddings,
            metadata=metadata,
            row_count=row_count,
            embedding_dimension=embedding_dimension,
        ):
            LOGGER.warning("Ignoring an embedding cache matrix that failed validation.")
            return None

        return np.asarray(embeddings)

    def save(
        self,
        embeddings: np.ndarray,
        *,
        fingerprint: str,
        model_name: str,
        model_revision: str | None = None,
    ) -> None:
        """Atomically persist an embedding matrix and its validation metadata."""
        matrix = np.ascontiguousarray(np.asarray(embeddings))
        if matrix.ndim != 2:
            raise ValueError("Corpus embeddings must be a two-dimensional array.")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("Corpus embeddings must contain numeric values.")
        if not np.isfinite(matrix).all():
            raise ValueError("Corpus embeddings must contain only finite values.")

        metadata = {
            "version": CACHE_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "model_name": str(model_name),
            "model_revision": model_revision,
            "row_count": int(matrix.shape[0]),
            "embedding_dimension": int(matrix.shape[1]),
            "shape": [int(size) for size in matrix.shape],
            "dtype": str(matrix.dtype),
            "array_sha256": self._array_digest(matrix),
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        array_temp = self._temporary_path(".npy")
        metadata_temp = self._temporary_path(".json")
        try:
            np.save(array_temp, matrix, allow_pickle=False)
            metadata_temp.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            array_temp.replace(self.embeddings_path)
            metadata_temp.replace(self.metadata_path)
        finally:
            array_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    def _temporary_path(self, suffix: str) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            dir=self.cache_dir,
            prefix="embedding-cache-",
            suffix=suffix,
        )
        os.close(descriptor)
        return Path(raw_path)

    @staticmethod
    def _metadata_matches(
        metadata: Any,
        *,
        fingerprint: str,
        model_name: str,
        model_revision: str | None,
        row_count: int,
        embedding_dimension: int | None,
    ) -> bool:
        if not isinstance(metadata, dict):
            return False
        if metadata.get("version") != CACHE_FORMAT_VERSION:
            return False
        if metadata.get("fingerprint") != fingerprint:
            return False
        if metadata.get("model_name") != str(model_name):
            return False
        if metadata.get("model_revision") != model_revision:
            return False
        if metadata.get("row_count") != row_count:
            return False
        if (
            embedding_dimension is not None
            and metadata.get("embedding_dimension") != embedding_dimension
        ):
            return False
        return True

    @classmethod
    def _array_matches(
        cls,
        embeddings: np.ndarray,
        *,
        metadata: dict[str, Any],
        row_count: int,
        embedding_dimension: int | None,
    ) -> bool:
        if embeddings.ndim != 2 or embeddings.shape[0] != row_count:
            return False
        if (
            embedding_dimension is not None
            and embeddings.shape[1] != embedding_dimension
        ):
            return False
        if list(embeddings.shape) != metadata.get("shape"):
            return False
        if str(embeddings.dtype) != metadata.get("dtype"):
            return False
        if not np.issubdtype(embeddings.dtype, np.number):
            return False
        if not np.isfinite(embeddings).all():
            return False
        return cls._array_digest(np.ascontiguousarray(embeddings)) == metadata.get(
            "array_sha256",
        )

    @staticmethod
    def _array_digest(embeddings: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(str(embeddings.dtype).encode("ascii"))
        digest.update(json.dumps(list(embeddings.shape)).encode("ascii"))
        digest.update(embeddings.tobytes(order="C"))
        return digest.hexdigest()
