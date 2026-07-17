"""Project configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FAQ_PATH = DATA_DIR / "faq.csv"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.csv"
EMBEDDING_CACHE_DIR = DATA_DIR / "cache"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Pin the evaluated weights so corpus and query embeddings cannot silently drift.
EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_DEVICE = "cpu"

DEFAULT_TOP_K = 3
TFIDF_SIMILARITY_THRESHOLD = 0.2
SEMANTIC_SIMILARITY_THRESHOLD = 0.54
AMBIGUITY_MARGIN = 0.05

# Keep the original public setting for callers that still use the TF-IDF retriever.
DEFAULT_CONFIDENCE_THRESHOLD = TFIDF_SIMILARITY_THRESHOLD
FALLBACK_MESSAGE = (
    "Não encontrei uma resposta suficientemente confiável para essa pergunta."
)
