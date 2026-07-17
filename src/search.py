"""Shared search result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Describe one ranked FAQ candidate."""

    id: Any
    pergunta: str
    resposta: str
    categoria: str
    score: float
    index: int
    rank: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Return the result using the legacy dictionary contract."""
        return asdict(self)
