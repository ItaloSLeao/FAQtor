"""Tests for entrypoints executed outside the project directory."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from main import build_parser
from src.config import PROJECT_ROOT


class MetricColumn:
    """Record metric calls made by the Streamlit result renderer."""

    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def metric(self, label: str, value: str) -> None:
        self.calls.append((label, value))


def test_main_help_runs_outside_project_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "search" in result.stdout
    assert "evaluate" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["search", "pergunta", "--top-k", "0"],
        ["search", "pergunta", "--threshold", "nan"],
        ["evaluate", "--threshold", "1.1"],
    ],
)
def test_main_rejects_invalid_search_parameters(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_streamlit_app_renders_outside_project_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

    assert not app.exception
    assert len(app.text_input) == 1


def test_streamlit_result_renderer_handles_missing_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as streamlit_app

    metrics: list[tuple[str, str]] = []
    captions: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "warning", lambda _: None)
    monkeypatch.setattr(
        streamlit_app.st,
        "columns",
        lambda _: [MetricColumn(metrics) for _ in range(3)],
    )
    monkeypatch.setattr(streamlit_app.st, "caption", captions.append)

    streamlit_app.render_results(
        {
            "answer": "fallback",
            "is_confident": False,
            "suggestions": [],
            "score": 0.0,
            "threshold": 0.54,
            "method": "semantic",
            "top_result": None,
            "results": [],
        },
    )

    assert len(metrics) == 3
    assert captions == ["Nenhum candidato foi gerado para a consulta."]
