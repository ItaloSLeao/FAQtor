"""Tests for spelling utilities."""

from src.spelling import levenshtein_distance, suggest_terms


def test_levenshtein_distance_handles_equal_normalized_words() -> None:
    assert levenshtein_distance("matrícula", "matricula") == 0


def test_levenshtein_distance_counts_insertions() -> None:
    assert levenshtein_distance("matriculla", "matricula") == 1


def test_suggest_terms_returns_close_vocabulary_terms() -> None:
    suggestions = suggest_terms(
        ["matriculla", "historicoo"],
        ["matricula", "historico", "biblioteca"],
        max_distance=2,
    )

    assert suggestions == ["matricula", "historico"]
