"""Spelling suggestion utilities."""

from __future__ import annotations

from collections.abc import Iterable

from src.preprocessing import normalize_text


def levenshtein_distance(a: str, b: str) -> int:
    """Return the edit distance between two strings."""
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insertion_cost = current_row[j - 1] + 1
            deletion_cost = previous_row[j] + 1
            substitution_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(
                min(insertion_cost, deletion_cost, substitution_cost),
            )
        previous_row = current_row

    return previous_row[-1]


def suggest_terms(
    query_tokens: Iterable[str],
    vocabulary: Iterable[str],
    max_distance: int = 2,
) -> list[str]:
    """Suggest close vocabulary terms for query tokens."""
    vocabulary_terms = sorted(
        {
            normalize_text(term)
            for term in vocabulary
            if normalize_text(term) and " " not in normalize_text(term)
        },
    )
    suggestions: list[str] = []

    for token in query_tokens:
        normalized_token = normalize_text(token)
        if not normalized_token or normalized_token in vocabulary_terms:
            continue

        candidates = [
            (levenshtein_distance(normalized_token, term), term)
            for term in vocabulary_terms
            if abs(len(normalized_token) - len(term)) <= max_distance
        ]
        close_terms = [
            (distance, term)
            for distance, term in candidates
            if distance <= max_distance
        ]
        if not close_terms:
            continue

        _, best_term = min(close_terms, key=lambda item: (item[0], item[1]))
        if best_term not in suggestions:
            suggestions.append(best_term)

    return suggestions
