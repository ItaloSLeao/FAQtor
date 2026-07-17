"""Tests for dense semantic FAQ retrieval without model downloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import FALLBACK_MESSAGE
from src.evaluation import load_faq
from src.search import SearchResult
from src.semantic_preprocessing import normalize_semantic_text
from src.semantic_retriever import SemanticRetriever, load_embedding_model


@dataclass(frozen=True)
class EncodeCall:
    """Record one fake model invocation."""

    texts: tuple[str, ...]
    normalize_embeddings: bool
    convert_to_numpy: bool


class FakeEncoder:
    """Return stable, controllable embeddings and record every encode call."""

    def __init__(
        self,
        vectors: dict[str, Sequence[float]] | None = None,
        dimension: int = 4,
    ) -> None:
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32)
            for text, vector in (vectors or {}).items()
        }
        self.dimension = dimension
        if self.vectors:
            self.dimension = len(next(iter(self.vectors.values())))
        self.calls: list[EncodeCall] = []

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **_: Any,
    ) -> np.ndarray:
        """Mimic the relevant SentenceTransformer.encode behavior."""
        is_single = isinstance(sentences, str)
        texts = [sentences] if is_single else [str(text) for text in sentences]
        self.calls.append(
            EncodeCall(
                texts=tuple(texts),
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
            ),
        )

        matrix = np.vstack([self._vector_for(text) for text in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0.0, 1.0, norms)
        return matrix[0] if is_single else matrix

    def get_sentence_embedding_dimension(self) -> int:
        """Expose the same dimension helper as SentenceTransformer."""
        return self.dimension

    def _vector_for(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return self.vectors[text].copy()

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer(digest[: self.dimension], dtype=np.uint8)
        return values.astype(np.float32) - 127.5


class RecordingLoader:
    """Return a fake encoder while recording model loader calls."""

    def __init__(self, model: FakeEncoder) -> None:
        self.model = model
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeEncoder:
        self.calls.append((args, kwargs))
        return self.model


@pytest.fixture
def faq_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 101,
                "pergunta": "Como redefinir minha senha?",
                "resposta": "Use a recuperacao de senha do portal.",
                "categoria": "Acesso",
            },
            {
                "id": 202,
                "pergunta": "Quais documentos preciso para a matricula?",
                "resposta": "Apresente identidade e comprovante academico.",
                "categoria": "Matricula",
            },
            {
                "id": 303,
                "pergunta": "Como cancelar minha inscricao?",
                "resposta": "Solicite o cancelamento na secretaria.",
                "categoria": "Cancelamento",
            },
        ],
        index=[10, 20, 30],
    )


@pytest.fixture
def semantic_vectors(faq_dataframe: pd.DataFrame) -> dict[str, Sequence[float]]:
    questions = faq_dataframe["pergunta"].tolist()
    return {
        questions[0]: [1.0, 0.0, 0.0, 0.0],
        questions[1]: [0.0, 1.0, 0.0, 0.0],
        questions[2]: [0.0, 0.0, 1.0, 0.0],
        "Esqueci minha credencial. Como crio uma nova?": [1.0, 0.0, 0.0, 0.0],
        "consulta para ordenar": [0.8, 0.6, 0.0, 0.0],
        "receita de bolo de chocolate": [0.0, 0.0, 0.0, 1.0],
        "NAO quero mais participar: qual e o procedimento?": [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    }


def build_retriever(
    faq_dataframe: pd.DataFrame,
    model: FakeEncoder,
    cache_dir: Path,
    *,
    model_name: str = "fake-semantic-model",
    model_revision: str | None = "fake-revision",
    threshold: float = 0.5,
) -> SemanticRetriever:
    return SemanticRetriever(
        model=model,
        model_name=model_name,
        model_revision=model_revision,
        cache_dir=cache_dir,
        threshold=threshold,
    ).fit(faq_dataframe)


def test_search_result_has_explicit_ranked_metadata() -> None:
    result = SearchResult(
        id=202,
        pergunta="Pergunta",
        resposta="Resposta",
        categoria="Categoria",
        score=0.75,
        index=1,
        rank=2,
        method="semantic",
    )

    assert result.to_dict() == {
        "id": 202,
        "pergunta": "Pergunta",
        "resposta": "Resposta",
        "categoria": "Categoria",
        "score": 0.75,
        "index": 1,
        "rank": 2,
        "method": "semantic",
    }


def test_model_loads_once_and_corpus_and_queries_encode_once_each(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    model = FakeEncoder(semantic_vectors)
    loader = RecordingLoader(model)
    retriever = SemanticRetriever(
        model_loader=loader,
        model_name="replaceable-model",
        cache_dir=tmp_path / "cache",
        threshold=0.5,
    ).fit(faq_dataframe)

    assert len(loader.calls) == 1
    assert len(model.calls) == 1
    assert model.calls[0].texts == tuple(faq_dataframe["pergunta"])
    assert model.calls[0].normalize_embeddings is True
    assert model.calls[0].convert_to_numpy is True

    retriever.search("Esqueci minha credencial. Como crio uma nova?")
    retriever.search("consulta para ordenar")

    assert len(loader.calls) == 1
    assert [call.texts for call in model.calls[1:]] == [
        ("Esqueci minha credencial. Como crio uma nova?",),
        ("consulta para ordenar",),
    ]
    assert all(call.normalize_embeddings for call in model.calls)


def test_semantic_reformulation_keeps_question_answer_alignment(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    retriever = build_retriever(
        faq_dataframe,
        FakeEncoder(semantic_vectors),
        tmp_path / "cache",
    )

    response = retriever.search(
        "Esqueci minha credencial. Como crio uma nova?",
        top_k=2,
    )

    assert response["is_confident"] is True
    assert response["answer"] == faq_dataframe.iloc[0]["resposta"]
    assert response["top_result"]["id"] == 101
    assert response["top_result"]["index"] == 0
    assert response["top_result"]["pergunta"] == faq_dataframe.iloc[0]["pergunta"]
    assert response["top_result"]["resposta"] == faq_dataframe.iloc[0]["resposta"]
    assert response["method"] == "semantic"
    assert response["top_result"]["method"] == "semantic"


def test_search_ranks_top_k_and_exposes_result_metadata(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    retriever = build_retriever(
        faq_dataframe,
        FakeEncoder(semantic_vectors),
        tmp_path / "cache",
    )

    response = retriever.search("consulta para ordenar", top_k=2)

    assert len(response["results"]) == 2
    assert [result["id"] for result in response["results"]] == [101, 202]
    assert [result["index"] for result in response["results"]] == [0, 1]
    assert response["results"][0]["score"] > response["results"][1]["score"]
    assert all(result["method"] == "semantic" for result in response["results"])


def test_search_threshold_can_reject_and_accept_same_deterministic_score(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    retriever = build_retriever(
        faq_dataframe,
        FakeEncoder(semantic_vectors),
        tmp_path / "cache",
        threshold=0.81,
    )

    rejected = retriever.search("consulta para ordenar")
    accepted = retriever.search("consulta para ordenar", threshold=0.79)

    assert rejected["score"] < rejected["threshold"]
    assert rejected["is_confident"] is False
    assert rejected["answer"] == FALLBACK_MESSAGE
    assert accepted["score"] >= accepted["threshold"]
    assert accepted["is_confident"] is True
    assert accepted["answer"] == faq_dataframe.iloc[0]["resposta"]


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_query_returns_fallback_without_encoding(
    query: str,
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    model = FakeEncoder(semantic_vectors)
    retriever = build_retriever(faq_dataframe, model, tmp_path / "cache")
    calls_after_fit = len(model.calls)

    response = retriever.search(query)

    assert len(model.calls) == calls_after_fit
    assert response["is_confident"] is False
    assert response["answer"] == FALLBACK_MESSAGE
    assert response["results"] == []
    assert response["top_result"] is None


def test_irrelevant_query_uses_fallback(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    retriever = build_retriever(
        faq_dataframe,
        FakeEncoder(semantic_vectors),
        tmp_path / "cache",
    )

    response = retriever.search("receita de bolo de chocolate")

    assert response["is_confident"] is False
    assert response["score"] < response["threshold"]
    assert response["answer"] == FALLBACK_MESSAGE
    assert response["fallback_message"] == FALLBACK_MESSAGE


def test_semantic_normalization_is_unicode_conservative() -> None:
    raw = "  NA\u0303O\tquero  perder a pontuac\u0327a\u0303o: matri\u0301cula!  "

    normalized = normalize_semantic_text(raw)

    assert normalized == "NÃO quero perder a pontuação: matrícula!"
    assert "NÃO" in normalized
    assert ":" in normalized and "!" in normalized


def test_semantic_normalization_reduces_all_caps_without_losing_accents() -> None:
    normalized = normalize_semantic_text("COMO RENOVAR EMPRÉSTIMO DE LIVROS!!!")

    assert normalized == "como renovar empréstimo de livros!!!"


def test_normalized_unicode_text_is_sent_to_model_unchanged_semantically(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    model = FakeEncoder(semantic_vectors)
    retriever = build_retriever(faq_dataframe, model, tmp_path / "cache")

    retriever.search("  NAO   quero mais participar: qual e o procedimento?  ")

    assert model.calls[-1].texts == (
        "NAO quero mais participar: qual e o procedimento?",
    )


def test_injected_model_replaces_loader_without_changing_retrieval(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    model = FakeEncoder(semantic_vectors)

    def unexpected_loader(*_: Any, **__: Any) -> FakeEncoder:
        raise AssertionError("the loader must not run when a model is injected")

    retriever = SemanticRetriever(
        model=model,
        model_loader=unexpected_loader,
        model_name="a-model-not-installed-locally",
        cache_dir=tmp_path / "cache",
        threshold=0.5,
    ).fit(faq_dataframe)

    assert retriever.search("Esqueci minha credencial. Como crio uma nova?")[
        "top_result"
    ]["id"] == 101


def test_embedding_cache_is_reused_without_reencoding_corpus(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first_model = FakeEncoder(semantic_vectors)
    build_retriever(faq_dataframe, first_model, cache_dir)

    assert len(first_model.calls) == 1
    assert any(cache_dir.iterdir())

    second_model = FakeEncoder(semantic_vectors)
    retriever = build_retriever(faq_dataframe, second_model, cache_dir)

    assert second_model.calls == []
    retriever.search("Esqueci minha credencial. Como crio uma nova?")
    assert len(second_model.calls) == 1
    assert second_model.calls[0].texts == (
        "Esqueci minha credencial. Como crio uma nova?",
    )


@pytest.mark.parametrize(
    "cache_change",
    ["question", "order", "model", "revision"],
)
def test_embedding_cache_invalidates_for_question_order_or_model(
    cache_change: str,
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    build_retriever(
        faq_dataframe,
        FakeEncoder(semantic_vectors),
        cache_dir,
        model_name="model-a",
        model_revision="revision-a",
    )

    changed_dataframe = faq_dataframe.copy()
    changed_model_name = "model-a"
    changed_model_revision = "revision-a"
    if cache_change == "question":
        changed_dataframe.iloc[0, changed_dataframe.columns.get_loc("pergunta")] = (
            "Como criar uma senha nova?"
        )
    elif cache_change == "order":
        changed_dataframe = changed_dataframe.iloc[::-1]
    elif cache_change == "model":
        changed_model_name = "model-b"
    else:
        changed_model_revision = "revision-b"

    second_model = FakeEncoder(semantic_vectors)
    build_retriever(
        changed_dataframe,
        second_model,
        cache_dir,
        model_name=changed_model_name,
        model_revision=changed_model_revision,
    )

    assert len(second_model.calls) == 1
    assert second_model.calls[0].texts == tuple(changed_dataframe["pergunta"])


def test_injected_model_disables_default_persistent_cache(
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
) -> None:
    retriever = SemanticRetriever(
        model=FakeEncoder(semantic_vectors),
        model_name="injected-model",
    ).fit(faq_dataframe)

    assert retriever.cache_dir is None


def test_model_loader_prefers_local_files_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    loaded_model = object()

    def sentence_transformer(model_name: str, **options: Any) -> object:
        calls.append((model_name, options))
        return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer),
    )
    load_embedding_model.cache_clear()

    result = load_embedding_model("model-name", "cpu", "revision-id")

    assert result is loaded_model
    assert calls == [
        (
            "model-name",
            {
                "device": "cpu",
                "revision": "revision-id",
                "local_files_only": True,
            },
        ),
    ]
    load_embedding_model.cache_clear()


def test_model_loader_downloads_only_after_local_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    loaded_model = object()

    def sentence_transformer(_: str, **options: Any) -> object:
        calls.append(options)
        if options.get("local_files_only"):
            raise OSError("local cache miss")
        return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer),
    )
    load_embedding_model.cache_clear()

    result = load_embedding_model("model-name", "cpu", "revision-id")

    assert result is loaded_model
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]
    assert calls[1]["revision"] == "revision-id"
    load_embedding_model.cache_clear()


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_similarity_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        SemanticRetriever(threshold=threshold)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(
    top_k: Any,
    faq_dataframe: pd.DataFrame,
    semantic_vectors: dict[str, Sequence[float]],
) -> None:
    model = FakeEncoder(semantic_vectors)
    retriever = SemanticRetriever(model=model).fit(faq_dataframe)
    calls_after_fit = len(model.calls)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("consulta para ordenar", top_k=top_k)

    assert len(model.calls) == calls_after_fit


def test_default_faq_path_works_outside_project_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    dataframe = load_faq()

    assert not dataframe.empty
    assert {"id", "pergunta", "resposta", "categoria"}.issubset(dataframe.columns)


def test_header_only_faq_is_rejected_without_encoding(
    tmp_path: Path,
) -> None:
    empty_csv = tmp_path / "empty-faq.csv"
    empty_csv.write_text("id,pergunta,resposta,categoria\n", encoding="utf-8")
    dataframe = load_faq(empty_csv)
    model = FakeEncoder()

    with pytest.raises(ValueError, match="(?i)empty|vazi|record|row"):
        SemanticRetriever(
            model=model,
            cache_dir=tmp_path / "cache",
        ).fit(dataframe)

    assert model.calls == []
