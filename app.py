"""Streamlit app entrypoint for FAQtor."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TOP_K, SEMANTIC_SIMILARITY_THRESHOLD
from src.evaluation import load_faq
from src.semantic_retriever import SemanticRetriever


@st.cache_data
def load_faq_data() -> pd.DataFrame:
    """Load FAQ data for the app."""
    return load_faq()


@st.cache_resource
def build_retriever() -> SemanticRetriever:
    """Build and cache the FAQ retriever."""
    faq_dataframe = load_faq_data()
    return SemanticRetriever().fit(faq_dataframe)


def render_sidebar() -> None:
    """Render the NLP pipeline summary."""
    st.sidebar.header("Pipeline")
    st.sidebar.markdown(
        """
1. Pré-processamento semântico
2. Embeddings densos
3. Similaridade cosseno
4. Ranking top-k
5. Fallback por limiar
        """.strip(),
    )
    st.sidebar.divider()
    st.sidebar.caption("FAQtor não usa LLMs nem APIs externas.")


def render_results(response: dict[str, Any]) -> None:
    """Render search results."""
    top_result = response["top_result"]

    if response["is_confident"]:
        st.success(response["answer"])
    else:
        st.warning(response["answer"])

    if response["suggestions"]:
        st.info("Talvez você quis dizer: " + ", ".join(response["suggestions"]))

    left, middle, right = st.columns(3)
    left.metric("Score", f"{response['score']:.3f}")
    middle.metric("Limiar", f"{response['threshold']:.2f}")
    right.metric("Método", str(response["method"]))

    if top_result is None:
        st.caption("Nenhum candidato foi gerado para a consulta.")
        return

    st.subheader("Pergunta mais similar")
    st.write(f"**[{top_result['id']}]** {top_result['pergunta']}")

    st.subheader("Top alternativas")
    alternatives = pd.DataFrame(
        [
            {
                "posição": index,
                "id": result["id"],
                "pergunta": result["pergunta"],
                "categoria": result["categoria"],
                "score": round(result["score"], 3),
            }
            for index, result in enumerate(response["results"], start=1)
        ],
    )
    st.dataframe(alternatives, hide_index=True, width="stretch")


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="FAQtor", layout="wide")
    render_sidebar()

    st.title("FAQtor")
    st.caption("Chatbot FAQ baseado em Recuperação de Informação")

    with st.form("search-form"):
        query = st.text_input(
            "Pergunta",
            placeholder="Ex.: como trancar uma disciplina?",
        )
        col_top_k, col_threshold = st.columns(2)
        top_k = col_top_k.slider(
            "Top-k",
            min_value=1,
            max_value=5,
            value=DEFAULT_TOP_K,
        )
        confidence_threshold = col_threshold.slider(
            "Limiar de confiança",
            min_value=0.0,
            max_value=1.0,
            value=SEMANTIC_SIMILARITY_THRESHOLD,
            step=0.01,
        )
        submitted = st.form_submit_button("Buscar", type="primary")

    if not submitted:
        return

    if not query.strip():
        st.error("Digite uma pergunta para buscar na base de FAQs.")
        return

    retriever = build_retriever()
    response = retriever.search(
        query,
        top_k=top_k,
        threshold=confidence_threshold,
    )
    render_results(response)


if __name__ == "__main__":
    main()
