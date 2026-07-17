"""Interactive CLI for FAQtor, inspired by modern AI CLIs."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.theme import Theme
except ImportError:
    print("Por favor, instale a biblioteca 'rich' executando:")
    print("pip install rich")
    sys.exit(1)

from src.config import DEFAULT_TOP_K, SEMANTIC_SIMILARITY_THRESHOLD
from src.evaluation import load_faq
from src.semantic_retriever import SemanticRetriever

# Custom highlight colors.
ORANGE = "#FF6600"
LIGHT_GRAY = "#FFFFFF"
WHITE = "#FFFFFF"
DIM = "#A0A0A0"

custom_theme = Theme(
    {
        "info": "white",
        "warning": "yellow",
        "danger": "red",
        "success": "white",
        "user": f"bold {ORANGE}",
        "assistant": f"bold {LIGHT_GRAY}",
        "dim": DIM,
    },
)

console = Console(theme=custom_theme)


def display_welcome() -> None:
    """Display a welcome banner."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = cwd.replace(home, "~")

    welcome_msg = f"""
[bold {ORANGE}]███████████   █████████      ██████    [/][bold {LIGHT_GRAY}] █████                        [/]
[bold {ORANGE}]░░███░░░░░░█  ███░░░░░███   ███░░░░███ [/][bold {LIGHT_GRAY}] ░░███                        [/]
[bold {ORANGE}] ░███   █ ░  ░███    ░███  ███    ░░███[/][bold {LIGHT_GRAY}] ███████    ██████  ████████  [/]  [bold {LIGHT_GRAY}]FAQtor[/] [{LIGHT_GRAY}]CLI 1.0.0[/]
[bold {ORANGE}] ░███████    ░███████████ ░███     ░███[/][bold {LIGHT_GRAY}]░░░███░    ███░░███░░███░░███ [/]  [bold {WHITE}]Assistente:[/] [{WHITE}]Recuperação de Informação[/]
[bold {ORANGE}] ░███░░░█    ░███░░░░░███ ░███   ██░███[/][bold {LIGHT_GRAY}]  ░███    ░███ ░███ ░███ ░░░  [/]  [bold {WHITE}]Pipeline:[/] [{WHITE}]Embeddings + Similaridade de Cosseno[/]
[bold {ORANGE}] ░███  ░     ░███    ░███ ░░███ ░░████ [/][bold {LIGHT_GRAY}]  ░███ ███░███ ░███ ░███      [/]  [bold {WHITE}]Top-K:[/] [{WHITE}]{DEFAULT_TOP_K}[/] | [bold {WHITE}]Limiar:[/] [{WHITE}]{SEMANTIC_SIMILARITY_THRESHOLD}[/]
[bold {ORANGE}] █████       █████   █████ ░░░██████░██[/][bold {LIGHT_GRAY}]  ░░█████ ░░██████  █████     [/]  [{WHITE}]{cwd}[/]
[bold {ORANGE}]░░░░░       ░░░░░   ░░░░░    ░░░░░░ ░░ [/][bold {LIGHT_GRAY}]   ░░░░░   ░░░░░░  ░░░░░      [/]

[{WHITE}]Digite sua [bold {ORANGE}]pergunta[/][{WHITE}] para buscar na base.[/] [{DIM}]Comandos: [/][bold {WHITE}]exit[/][{DIM}] (sair), [/][bold {WHITE}]clear[/][{DIM}] (limpar tela)[/]
"""
    console.print(welcome_msg)


def stream_text(text: str, title: str, border_style: str = DIM) -> None:
    """Render text progressively in the terminal."""
    current_text = ""
    chunk_size = 2

    with Live(
        Panel(
            Markdown(current_text),
            title=title,
            border_style=border_style,
            padding=(1, 2),
        ),
        refresh_per_second=60,
        transient=False,
    ) as live:
        for i in range(0, len(text), chunk_size):
            current_text += text[i : i + chunk_size]
            live.update(
                Panel(
                    Markdown(current_text),
                    title=title,
                    border_style=border_style,
                    padding=(1, 2),
                ),
            )
            time.sleep(0.015)


def print_response(response: dict[str, Any]) -> None:
    """Format and print the retriever response beautifully."""
    if response["is_confident"]:
        border_style = LIGHT_GRAY
        title = f"[bold {LIGHT_GRAY}]FAQtor[/]"
    else:
        border_style = WHITE
        title = f"[bold {LIGHT_GRAY}]FAQtor[/] [{WHITE}](Baixa Confiança)[/]"

    answer_text = response["answer"]
    stream_text(answer_text, title=title, border_style=border_style)

    if response["suggestions"]:
        sug_text = f"[{DIM}]Talvez você quis dizer:[/] " + ", ".join(
            response["suggestions"],
        )
        console.print(sug_text)
        console.print()

    table = Table(
        show_header=True,
        header_style=f"bold {LIGHT_GRAY}",
        border_style=DIM,
        box=None,
    )
    table.add_column("Pos", style=DIM, width=4)
    table.add_column("Score", justify="right", style=DIM, width=8)
    table.add_column("Categoria", style=DIM)
    table.add_column("Pergunta Correspondente")

    for idx, res in enumerate(response["results"], 1):
        score_str = f"{res['score']:.3f}"
        if idx == 1:
            score_str = f"[bold {ORANGE}]{score_str}[/]"
        table.add_row(str(idx), score_str, res["categoria"], res["pergunta"])

    console.print()
    console.print(
        Panel(
            table,
            title=f"[{DIM}]Contexto de Busca[/]",
            border_style=DIM,
            expand=False,
        ),
    )
    console.print()


def main() -> None:
    console.clear()
    with console.status(
        f"[{DIM}]Carregando base de dados...[/]",
        spinner="dots",
        spinner_style=f"bold {LIGHT_GRAY}",
    ):
        faq_df = load_faq()
        retriever = SemanticRetriever(
            confidence_threshold=SEMANTIC_SIMILARITY_THRESHOLD,
        ).fit(faq_df)

    display_welcome()

    while True:
        try:
            console.print()
            console.print(Rule(style=ORANGE))
            query = console.input(f"[bold {ORANGE}]> [/]")
            console.print(Rule(style=ORANGE))
            query = query.strip()

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                console.print(f"[{LIGHT_GRAY}]Encerrando FAQtor...[/]")
                break
            if query.lower() == "clear":
                console.clear()
                display_welcome()
                continue

            with console.status(
                f"[{LIGHT_GRAY}]Consultando base de dados...[/]",
                spinner="bouncingBar",
                spinner_style=f"bold {ORANGE}",
            ):
                response = retriever.search(query, top_k=DEFAULT_TOP_K)

            console.print()
            print_response(response)

        except KeyboardInterrupt:
            console.print(f"\n[{DIM}]Interrompido. Encerrando...[/]")
            break
        except EOFError:
            break
        except (OSError, RuntimeError, ValueError) as error:
            console.print(f"\n[bold red]Erro:[/] {error}")


if __name__ == "__main__":
    main()
