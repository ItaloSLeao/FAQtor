"""Command line entrypoint for FAQtor."""

from __future__ import annotations

import argparse
import math
from typing import Any

from src.config import DEFAULT_TOP_K, SEMANTIC_SIMILARITY_THRESHOLD
from src.evaluation import evaluate, format_report, load_faq
from src.semantic_retriever import SemanticRetriever


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return parsed


def similarity_threshold(value: str) -> float:
    """Parse a finite cosine-similarity threshold for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "o limiar deve ser finito e estar entre -1 e 1",
        )
    return parsed


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
        type=positive_integer,
        default=DEFAULT_TOP_K,
        help="Número de alternativas recuperadas.",
    )
    search_parser.add_argument(
        "--threshold",
        type=similarity_threshold,
        default=SEMANTIC_SIMILARITY_THRESHOLD,
        help="Limiar mínimo de confiança do embedding semântico.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Executar avaliação experimental.",
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=positive_integer,
        default=DEFAULT_TOP_K,
        help="Valor de k para accuracy@k.",
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=similarity_threshold,
        default=SEMANTIC_SIMILARITY_THRESHOLD,
        help="Limiar mínimo de confiança do embedding semântico.",
    )

    return parser


def run_search(args: argparse.Namespace) -> None:
    """Run a FAQ search from command line arguments."""
    faq_dataframe = load_faq()
    retriever = SemanticRetriever(
        confidence_threshold=args.threshold,
    ).fit(faq_dataframe)
    response = retriever.search(args.query, top_k=args.top_k)
    print_search_response(response)


def print_search_response(response: dict[str, Any]) -> None:
    """Print a ranked search response."""
    top_result = response["top_result"]
    print("FAQtor")
    print(f"Pergunta: {response['query']}")
    print(f"Resposta: {response['answer']}")

    if top_result is None:
        print(f"Score: {response['score']:.3f}")
        print(f"Método: {response['method']}")
        print("Confiante: não")
        return

    print()
    print("Pergunta mais similar da base:")
    print(f"[{top_result['id']}] {top_result['pergunta']}")
    print(f"Categoria: {top_result['categoria']}")
    print(f"Score: {top_result['score']:.3f}")
    print(f"Método: {response['method']}")
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
        semantic_threshold=args.threshold,
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
