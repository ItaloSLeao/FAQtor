"""Command line entrypoint for FAQtor."""

from __future__ import annotations

import argparse
from typing import Any

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_TOP_K
from src.evaluation import evaluate, format_report, load_faq
from src.retriever import FAQRetriever


def build_parser() -> argparse.ArgumentParser:
    """Build the FAQtor command parser."""
    parser = argparse.ArgumentParser(
        prog="FAQtor",
        description="Chatbot FAQ baseado em Recuperação de Informação.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Buscar uma resposta.")
    search_parser.add_argument("query", help="Pergunta em linguagem natural.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Número de alternativas recuperadas.",
    )
    search_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Limiar mínimo de confiança.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Executar avaliação experimental.",
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Valor de k para accuracy@k.",
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Limiar mínimo de confiança.",
    )

    return parser


def run_search(args: argparse.Namespace) -> None:
    """Run a FAQ search from command line arguments."""
    faq_dataframe = load_faq()
    retriever = FAQRetriever(confidence_threshold=args.threshold).fit(faq_dataframe)
    response = retriever.search(args.query, top_k=args.top_k)
    print_search_response(response)


def print_search_response(response: dict[str, Any]) -> None:
    """Print a ranked search response."""
    top_result = response["top_result"]
    print("FAQtor")
    print(f"Pergunta: {response['query']}")
    print(f"Resposta: {response['answer']}")
    print()
    print("Pergunta mais similar da base:")
    print(f"[{top_result['id']}] {top_result['pergunta']}")
    print(f"Categoria: {top_result['categoria']}")
    print(f"Score: {top_result['score']:.3f}")
    print(f"Confiante: {'sim' if response['is_confident'] else 'não'}")

    if response["suggestions"]:
        print()
        print("Talvez você quis dizer: " + ", ".join(response["suggestions"]))

    print()
    print(f"Top-{len(response['results'])} alternativas:")
    for position, result in enumerate(response["results"], start=1):
        print(
            f"{position}. [{result['id']}] {result['pergunta']} "
            f"(score={result['score']:.3f})",
        )


def run_evaluate(args: argparse.Namespace) -> None:
    """Run experimental evaluation."""
    metrics = evaluate(
        top_k=args.top_k,
        confidence_threshold=args.threshold,
    )
    print(format_report(metrics, top_k=args.top_k))


def main(argv: list[str] | None = None) -> None:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        run_search(args)
        return

    if args.command == "evaluate":
        run_evaluate(args)
        return

    parser.error("Comando inválido.")


if __name__ == "__main__":
    main()
