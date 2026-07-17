"""Tests for FAQ retrieval."""

import pandas as pd

from src.config import FALLBACK_MESSAGE
from src.retriever import FAQRetriever, TfidfRetriever


def make_retriever(confidence_threshold: float = 0.2) -> FAQRetriever:
    faq_dataframe = pd.DataFrame(
        [
            {
                "id": 1,
                "pergunta": "Como fazer matrícula no semestre?",
                "resposta": "A matrícula é feita pelo sistema acadêmico.",
                "categoria": "Matrícula",
            },
            {
                "id": 2,
                "pergunta": "Como trancar uma disciplina?",
                "resposta": "Solicite o trancamento no prazo oficial.",
                "categoria": "Trancamento",
            },
            {
                "id": 3,
                "pergunta": "Como renovar livro da biblioteca?",
                "resposta": "A renovação pode ser feita no sistema da biblioteca.",
                "categoria": "Biblioteca",
            },
        ],
    )
    return FAQRetriever(confidence_threshold=confidence_threshold).fit(faq_dataframe)


def test_search_returns_ranked_results() -> None:
    retriever = make_retriever()
    response = retriever.search("quero trancar disciplina", top_k=2)

    assert response["is_confident"] is True
    assert response["top_result"]["id"] == 2
    assert len(response["results"]) == 2
    assert response["results"][0]["score"] >= response["results"][1]["score"]
    assert response["method"] == "tfidf"
    assert response["top_result"]["method"] == "tfidf"
    assert response["top_result"]["index"] == 1
    assert response["top_result"]["rank"] == 1


def test_original_faq_retriever_name_preserves_tfidf_baseline() -> None:
    assert FAQRetriever is TfidfRetriever


def test_search_returns_fallback_for_distant_query() -> None:
    retriever = make_retriever()
    response = retriever.search("receita de bolo com chocolate", top_k=3)

    assert response["is_confident"] is False
    assert response["answer"] == FALLBACK_MESSAGE
    assert len(response["results"]) == 3
