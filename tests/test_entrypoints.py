"""Tests for entrypoints executed outside the project directory."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from main import build_parser
from src.config import PROJECT_ROOT


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
