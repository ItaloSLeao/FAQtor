"""Conservative preprocessing for sentence embeddings."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_semantic_text(text: Any) -> str:
    """Normalize Unicode and whitespace while preserving natural language."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFC", str(text))
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    if normalized.isupper():
        normalized = normalized.lower()
    return normalized


def preprocess_semantic_text(text: Any) -> str:
    """Return text prepared for the embedding model."""
    return normalize_semantic_text(text)
