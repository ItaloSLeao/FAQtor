"""Text preprocessing utilities."""

from __future__ import annotations

import re
import unicodedata


PORTUGUESE_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "essa",
    "esse",
    "esta",
    "este",
    "eu",
    "foi",
    "ha",
    "me",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "um",
    "uma",
}


def remove_accents(text: str) -> str:
    """Return text without accent marks."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_text(text: str) -> str:
    """Lowercase text, remove accents and normalize punctuation."""
    text = remove_accents(str(text).lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Split normalized text into simple word tokens."""
    tokens = normalize_text(text).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in PORTUGUESE_STOPWORDS]
    return tokens


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Return normalized text ready for vectorization."""
    return " ".join(tokenize(text, remove_stopwords=remove_stopwords))
