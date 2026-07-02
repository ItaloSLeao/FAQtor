"""Tests for text preprocessing."""

from src.preprocessing import normalize_text, preprocess_text, tokenize


def test_normalize_text_removes_accents_and_punctuation() -> None:
    assert normalize_text("Matrícula, TCC e Histórico!") == "matricula tcc e historico"


def test_tokenize_removes_stopwords_by_default() -> None:
    tokens = tokenize("Como faço minha matrícula no semestre?")

    assert tokens == ["faco", "matricula", "semestre"]


def test_preprocess_text_returns_joined_tokens() -> None:
    text = preprocess_text("Onde encontro o calendário acadêmico?")

    assert text == "onde encontro calendario academico"
