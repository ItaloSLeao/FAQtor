"""Project configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FAQ_PATH = DATA_DIR / "faq.csv"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.csv"

DEFAULT_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.2
FALLBACK_MESSAGE = "Não encontrei uma resposta suficientemente confiável para essa pergunta."
