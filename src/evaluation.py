"""Comparative retrieval evaluation utilities."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    AMBIGUITY_MARGIN,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    FAQ_PATH,
    FALLBACK_MESSAGE,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TEST_QUERIES_PATH,
    TFIDF_SIMILARITY_THRESHOLD,
)

FAQ_DTYPES = {
    "id": "string",
    "pergunta": "string",
    "resposta": "string",
    "categoria": "string",
}
TEST_QUERY_DTYPES = {
    "query": "string",
    "expected_id": "string",
    "categoria": "string",
    "expected_fallback": "string",
    "case_type": "string",
}
REQUIRED_TEST_COLUMNS = {"query", "expected_id"}
_DEFAULT_SEMANTIC_CACHE = object()
_THRESHOLD_SWEEP = tuple(value / 100 for value in range(40, 81))


def load_faq(path: str | Path = FAQ_PATH) -> pd.DataFrame:
    """Load FAQ data without converting empty strings into missing values."""
    return pd.read_csv(
        path,
        dtype=FAQ_DTYPES,
        keep_default_na=False,
        encoding="utf-8",
    )


def load_test_queries(path: str | Path = TEST_QUERIES_PATH) -> pd.DataFrame:
    """Load evaluation cases while preserving empty and whitespace queries."""
    dataframe = pd.read_csv(
        path,
        dtype=TEST_QUERY_DTYPES,
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = REQUIRED_TEST_COLUMNS.difference(dataframe.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing test query columns: {joined}")
    return dataframe


def _parse_expected_fallback(value: Any) -> bool | None:
    """Parse the explicit fallback label used by the evaluation dataset."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if not normalized:
        return None
    if normalized in {"false", "0", "no", "nao"}:
        return False
    if normalized in {"true", "1", "yes", "sim"}:
        return True
    raise ValueError(f"Invalid expected_fallback value: {value!r}")


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None when a score is unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _result_id(result: Mapping[str, Any]) -> str:
    """Return a normalized identifier from a ranked result."""
    value = result.get("id", result.get("faq_id", ""))
    return "" if value is None else str(value)


def _response_results(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only mapping-like entries from a search response."""
    raw_results = response.get("results", [])
    if not isinstance(raw_results, (list, tuple)):
        return []
    return [result for result in raw_results if isinstance(result, Mapping)]


def _response_is_fallback(
    response: Mapping[str, Any],
    results: list[Mapping[str, Any]],
) -> bool:
    """Read a fallback decision from compatible search response shapes."""
    if "is_fallback" in response:
        return bool(response["is_fallback"])
    if "is_confident" in response:
        return not bool(response["is_confident"])
    if response.get("fallback_message"):
        return True
    if response.get("answer") == FALLBACK_MESSAGE:
        return True
    return not results


def _response_scores(
    response: Mapping[str, Any],
    results: list[Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    """Extract the first two ranking scores from a search response."""
    top_score = _finite_float(response.get("score"))
    if top_score is None and results:
        top_score = _finite_float(results[0].get("score"))
    second_score = None
    if len(results) > 1:
        second_score = _finite_float(results[1].get("score"))
    return top_score, second_score


def _mean(values: list[float]) -> float:
    """Return a stable mean for a possibly empty score collection."""
    return sum(values) / len(values) if values else 0.0


def _summarize_details(
    details: list[dict[str, Any]],
    top_k: int,
    ambiguity_margin: float,
) -> dict[str, Any]:
    """Aggregate ranking, rejection, confidence, and ambiguity metrics."""
    positives = [detail for detail in details if detail["expected_id"]]
    negatives = [
        detail for detail in details if detail["expected_fallback"] is True
    ]
    unscored = [detail for detail in details if not detail["scored"]]
    positive_total = len(positives)
    scored_total = positive_total + len(negatives)
    hits_at_1 = sum(detail["hit_at_1"] for detail in positives)
    hits_at_3 = sum(detail["hit_at_3"] for detail in positives)
    hits_at_k = sum(detail[f"hit_at_{top_k}"] for detail in positives)
    reciprocal_rank_sum = sum(
        0.0 if detail["rank"] is None else 1.0 / detail["rank"]
        for detail in positives
    )
    end_to_end_correct = sum(
        detail["end_to_end_correct"] is True for detail in details
    )
    e2e_correct_scores = [
        detail["top_score"]
        for detail in details
        if detail["end_to_end_correct"] is True
        and detail["top_score"] is not None
    ]
    e2e_incorrect_scores = [
        detail["top_score"]
        for detail in details
        if detail["end_to_end_correct"] is False
        and detail["top_score"] is not None
    ]
    top1_correct_scores = [
        detail["top_score"]
        for detail in positives
        if detail["hit_at_1"] and detail["top_score"] is not None
    ]
    top1_incorrect_scores = [
        detail["top_score"]
        for detail in positives
        if not detail["hit_at_1"] and detail["top_score"] is not None
    ]
    score_gaps = [
        detail["score_gap"]
        for detail in details
        if detail["score_gap"] is not None
    ]
    return {
        "total_queries": len(details),
        "scored_queries": scored_total,
        "unscored_queries": len(unscored),
        "queries_with_similarity_score": sum(
            detail["top_score"] is not None for detail in details
        ),
        "positive_queries": positive_total,
        "negative_queries": len(negatives),
        "ranking_queries": positive_total,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        f"hits_at_{top_k}": hits_at_k,
        "accuracy_at_1": hits_at_1 / positive_total if positive_total else 0.0,
        "accuracy_at_3": hits_at_3 / positive_total if positive_total else 0.0,
        f"accuracy_at_{top_k}": (
            hits_at_k / positive_total if positive_total else 0.0
        ),
        "mean_reciprocal_rank": (
            reciprocal_rank_sum / positive_total if positive_total else 0.0
        ),
        "correct_fallbacks": sum(
            detail["correct_rejection"] for detail in details
        ),
        "incorrect_fallbacks": sum(
            detail["incorrect_fallback"] for detail in details
        ),
        "false_positives": sum(detail["false_positive"] for detail in details),
        "false_negatives": sum(detail["false_negative"] for detail in details),
        "correct_answers": sum(detail["correct_answer"] for detail in details),
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": (
            end_to_end_correct / scored_total if scored_total else 0.0
        ),
        "mean_score_top1_correct": _mean(top1_correct_scores),
        "mean_score_top1_incorrect": _mean(top1_incorrect_scores),
        "mean_score_correct": _mean(top1_correct_scores),
        "mean_score_incorrect": _mean(top1_incorrect_scores),
        "mean_score_end_to_end_correct": _mean(e2e_correct_scores),
        "mean_score_end_to_end_incorrect": _mean(e2e_incorrect_scores),
        "average_score_correct": _mean(top1_correct_scores),
        "average_score_incorrect": _mean(top1_incorrect_scores),
        "ambiguity_margin": ambiguity_margin,
        "ambiguous_queries": sum(detail["ambiguous"] for detail in details),
        "mean_top_gap": _mean(score_gaps),
        "details": details,
    }


def evaluate_retriever(
    retriever: Any,
    test_dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    *,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Evaluate any retriever whose search method returns a response mapping."""
    missing = REQUIRED_TEST_COLUMNS.difference(test_dataframe.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing test query columns: {joined}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin cannot be negative")

    ranking_depth = max(3, top_k)
    details: list[dict[str, Any]] = []

    for case_index, (_, row) in enumerate(test_dataframe.iterrows()):
        query = str(row.get("query", ""))
        expected_id = str(row.get("expected_id", ""))
        expected_fallback = _parse_expected_fallback(
            row.get("expected_fallback", False),
        )
        case_type = str(row.get("case_type", "unspecified"))
        category = str(row.get("categoria", ""))

        raw_response = retriever.search(query, top_k=ranking_depth)
        if not isinstance(raw_response, Mapping):
            raise TypeError("retriever.search() must return a mapping")

        results = _response_results(raw_response)
        retrieved_ids = [_result_id(result) for result in results]
        top_id = retrieved_ids[0] if retrieved_ids else ""
        top_score, second_score = _response_scores(raw_response, results)
        is_fallback = _response_is_fallback(raw_response, results)

        rank = None
        if expected_id and expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1
        hit_at_1 = rank == 1
        hit_at_3 = rank is not None and rank <= 3
        hit_at_k = rank is not None and rank <= top_k

        is_positive = bool(expected_id)
        is_negative = expected_fallback is True
        is_scored = is_positive or is_negative
        if is_positive and is_negative:
            raise ValueError(
                f"Case {case_index} cannot expect both an id and fallback",
            )
        correct_answer = is_positive and not is_fallback and hit_at_1
        correct_rejection = is_negative and is_fallback
        end_to_end_correct = (
            correct_answer or correct_rejection if is_scored else None
        )
        false_positive = is_negative and not is_fallback
        false_negative = is_positive and is_fallback
        incorrect_fallback = is_positive and is_fallback

        score_gap = None
        if top_score is not None and second_score is not None:
            score_gap = top_score - second_score
        is_ambiguous = (
            score_gap is not None
            and score_gap <= ambiguity_margin
        )

        detail = {
            "case_index": case_index,
            "query": query,
            "expected_id": expected_id,
            "expected_fallback": expected_fallback,
            "scored": is_scored,
            "case_type": case_type,
            "categoria": category,
            "rank": rank,
            "retrieved_ids": retrieved_ids,
            "top_id": top_id or None,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "ambiguous": is_ambiguous,
            "is_fallback": is_fallback,
            "correct_answer": correct_answer,
            "correct_rejection": correct_rejection,
            "incorrect_fallback": incorrect_fallback,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            f"hit_at_{top_k}": hit_at_k,
            "end_to_end_correct": end_to_end_correct,
        }
        details.append(detail)

    metrics = _summarize_details(details, top_k, ambiguity_margin)
    baseline_details = [
        detail for detail in details if detail["case_type"] == "baseline_original"
    ]
    controlled_details = [
        detail for detail in details if detail["case_type"] != "baseline_original"
    ]
    metrics["baseline_queries"] = len(baseline_details)
    metrics["controlled_queries"] = len(controlled_details)
    metrics["case_type_counts"] = dict(
        Counter(detail["case_type"] for detail in details),
    )
    metrics["baseline_original"] = _summarize_details(
        baseline_details,
        top_k,
        ambiguity_margin,
    )
    metrics["controlled_cases"] = _summarize_details(
        controlled_details,
        top_k,
        ambiguity_margin,
    )
    return metrics


def calibrate_confidence_threshold(
    details: Sequence[Mapping[str, Any]],
    selected_threshold: float,
    thresholds: Sequence[float] = _THRESHOLD_SWEEP,
) -> dict[str, Any]:
    """Recompute end-to-end decisions over a fixed semantic ranking."""
    selected_threshold = float(selected_threshold)
    candidates = sorted({float(value) for value in thresholds} | {selected_threshold})
    if any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0
        for value in candidates
    ):
        raise ValueError("threshold sweep values must be finite and between -1 and 1")

    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        scored = [detail for detail in details if detail["scored"]]
        positive = [detail for detail in scored if detail["expected_id"]]
        negative = [
            detail for detail in scored if detail["expected_fallback"] is True
        ]

        def accepted(detail: Mapping[str, Any]) -> bool:
            score = detail["top_score"]
            return score is not None and score >= threshold

        correct_answers = sum(
            detail["hit_at_1"] and accepted(detail) for detail in positive
        )
        correct_rejections = sum(not accepted(detail) for detail in negative)
        false_positives = sum(accepted(detail) for detail in negative)
        false_negatives = sum(not accepted(detail) for detail in positive)
        total = len(scored)
        rows.append(
            {
                "threshold": threshold,
                "end_to_end_correct": correct_answers + correct_rejections,
                "end_to_end_accuracy": (
                    (correct_answers + correct_rejections) / total if total else 0.0
                ),
                "correct_answers": correct_answers,
                "correct_rejections": correct_rejections,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
        )

    best_accuracy = max(
        (row["end_to_end_accuracy"] for row in rows),
        default=0.0,
    )
    selected = next(
        row for row in rows if row["threshold"] == selected_threshold
    )
    return {
        "selected_threshold": selected_threshold,
        "selected": selected,
        "best_end_to_end_accuracy": best_accuracy,
        "best_thresholds": [
            row["threshold"]
            for row in rows
            if math.isclose(
                row["end_to_end_accuracy"],
                best_accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ],
        "sweep": rows,
    }


def _comparison_case(
    tfidf_detail: Mapping[str, Any],
    semantic_detail: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact, method-neutral record for a comparison group."""
    return {
        "case_index": tfidf_detail["case_index"],
        "query": tfidf_detail["query"],
        "expected_id": tfidf_detail["expected_id"],
        "expected_fallback": tfidf_detail["expected_fallback"],
        "case_type": tfidf_detail["case_type"],
        "tfidf_top_id": tfidf_detail["top_id"],
        "tfidf_score": tfidf_detail["top_score"],
        "tfidf_fallback": tfidf_detail["is_fallback"],
        "tfidf_gap": tfidf_detail["score_gap"],
        "tfidf_ambiguous": tfidf_detail["ambiguous"],
        "semantic_top_id": semantic_detail["top_id"],
        "semantic_score": semantic_detail["top_score"],
        "semantic_fallback": semantic_detail["is_fallback"],
        "semantic_gap": semantic_detail["score_gap"],
        "semantic_ambiguous": semantic_detail["ambiguous"],
    }


def compare_retrievers(
    tfidf_retriever: Any,
    semantic_retriever: Any,
    test_dataframe: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    *,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Compare fitted lexical and semantic retrievers on identical cases."""
    tfidf_metrics = evaluate_retriever(
        tfidf_retriever,
        test_dataframe,
        top_k=top_k,
        ambiguity_margin=ambiguity_margin,
    )
    semantic_metrics = evaluate_retriever(
        semantic_retriever,
        test_dataframe,
        top_k=top_k,
        ambiguity_margin=ambiguity_margin,
    )

    tfidf_details = tfidf_metrics["details"]
    semantic_details = semantic_metrics["details"]
    if len(tfidf_details) != len(semantic_details):
        raise RuntimeError("Retriever evaluations produced different case counts")

    tfidf_wrong_semantic_correct: list[dict[str, Any]] = []
    semantic_wrong_tfidf_correct: list[dict[str, Any]] = []
    both_wrong: list[dict[str, Any]] = []
    both_correct: list[dict[str, Any]] = []
    tfidf_correct_rejections: list[dict[str, Any]] = []
    semantic_correct_rejections: list[dict[str, Any]] = []
    both_correct_rejections: list[dict[str, Any]] = []
    unscored_cases: list[dict[str, Any]] = []

    for tfidf_detail, semantic_detail in zip(
        tfidf_details,
        semantic_details,
        strict=True,
    ):
        case = _comparison_case(tfidf_detail, semantic_detail)
        tfidf_correct = tfidf_detail["end_to_end_correct"]
        semantic_correct = semantic_detail["end_to_end_correct"]
        if tfidf_correct is None or semantic_correct is None:
            unscored_cases.append(case)
            continue
        if not tfidf_correct and semantic_correct:
            tfidf_wrong_semantic_correct.append(case)
        elif tfidf_correct and not semantic_correct:
            semantic_wrong_tfidf_correct.append(case)
        elif not tfidf_correct and not semantic_correct:
            both_wrong.append(case)
        else:
            both_correct.append(case)

        tfidf_rejection = bool(tfidf_detail["correct_rejection"])
        semantic_rejection = bool(semantic_detail["correct_rejection"])
        if tfidf_rejection:
            tfidf_correct_rejections.append(case)
        if semantic_rejection:
            semantic_correct_rejections.append(case)
        if tfidf_rejection and semantic_rejection:
            both_correct_rejections.append(case)

    comparison = {
        "tfidf_wrong_semantic_correct": tfidf_wrong_semantic_correct,
        "semantic_wrong_tfidf_correct": semantic_wrong_tfidf_correct,
        "both_wrong": both_wrong,
        "both_correct": both_correct,
        "unscored": unscored_cases,
        "correct_rejections": {
            "tfidf": tfidf_correct_rejections,
            "semantic": semantic_correct_rejections,
            "both": both_correct_rejections,
        },
    }
    report = {
        "total_queries": len(test_dataframe),
        "top_k": top_k,
        "tfidf": tfidf_metrics,
        "semantic": semantic_metrics,
        "comparison": comparison,
        "semantic_threshold_calibration": calibrate_confidence_threshold(
            semantic_metrics["details"],
            getattr(
                semantic_retriever,
                "confidence_threshold",
                SEMANTIC_SIMILARITY_THRESHOLD,
            ),
        ),
    }
    report["baseline_original"] = {
        "total_queries": tfidf_metrics["baseline_queries"],
        "tfidf": tfidf_metrics["baseline_original"],
        "semantic": semantic_metrics["baseline_original"],
    }
    report["controlled_cases"] = {
        "total_queries": tfidf_metrics["controlled_queries"],
        "tfidf": tfidf_metrics["controlled_cases"],
        "semantic": semantic_metrics["controlled_cases"],
    }
    return report


def evaluate(
    faq_path: str | Path = FAQ_PATH,
    test_queries_path: str | Path = TEST_QUERIES_PATH,
    top_k: int = DEFAULT_TOP_K,
    semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    tfidf_threshold: float = TFIDF_SIMILARITY_THRESHOLD,
    *,
    semantic_model: Any | None = None,
    semantic_model_loader: Any | None = None,
    semantic_model_name: str = EMBEDDING_MODEL_NAME,
    semantic_model_revision: str | None = EMBEDDING_MODEL_REVISION,
    semantic_cache_dir: str | Path | None | object = _DEFAULT_SEMANTIC_CACHE,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> dict[str, Any]:
    """Load data, fit both retrievers, and return comparative metrics."""
    from src.retriever import TfidfRetriever
    from src.semantic_retriever import SemanticRetriever

    faq_dataframe = load_faq(faq_path)
    test_dataframe = load_test_queries(test_queries_path)
    tfidf_retriever = TfidfRetriever(
        confidence_threshold=tfidf_threshold,
    ).fit(faq_dataframe)

    semantic_options: dict[str, Any] = {
        "confidence_threshold": semantic_threshold,
        "model_name": semantic_model_name,
        "model_revision": semantic_model_revision,
    }
    if semantic_model is not None:
        semantic_options["model"] = semantic_model
    if semantic_model_loader is not None:
        semantic_options["model_loader"] = semantic_model_loader
    if semantic_cache_dir is not _DEFAULT_SEMANTIC_CACHE:
        semantic_options["cache_dir"] = semantic_cache_dir
    elif semantic_model is not None or semantic_model_loader is not None:
        semantic_options["cache_dir"] = None
    semantic_retriever = SemanticRetriever(**semantic_options).fit(faq_dataframe)

    return compare_retrievers(
        tfidf_retriever,
        semantic_retriever,
        test_dataframe,
        top_k=top_k,
        ambiguity_margin=ambiguity_margin,
    )


def _format_method(name: str, metrics: Mapping[str, Any]) -> list[str]:
    """Format one method block for the terminal report."""
    return [
        name,
        f"  Consultas pontuadas: {metrics['scored_queries']}",
        f"  Consultas com similaridade: "
        f"{metrics['queries_with_similarity_score']}",
        f"  Acertos top-1: {metrics['hits_at_1']} "
        f"({metrics['accuracy_at_1']:.3f})",
        f"  Acertos top-3: {metrics['hits_at_3']} "
        f"({metrics['accuracy_at_3']:.3f})",
        f"  MRR: {metrics['mean_reciprocal_rank']:.3f}",
        f"  Fallbacks corretos: {metrics['correct_fallbacks']}",
        f"  Fallbacks incorretos: {metrics['incorrect_fallbacks']}",
        f"  Falsos positivos: {metrics['false_positives']}",
        f"  Falsos negativos: {metrics['false_negatives']}",
        f"  Acuracia end-to-end: {metrics['end_to_end_accuracy']:.3f}",
        f"  Score medio com top-1 correto: "
        f"{metrics['mean_score_top1_correct']:.3f}",
        f"  Score medio com top-1 incorreto: "
        f"{metrics['mean_score_top1_incorrect']:.3f}",
        f"  Casos ambiguos por gap: {metrics['ambiguous_queries']} "
        f"(gap <= {metrics['ambiguity_margin']:.3f})",
        f"  Gap medio entre top-1 e top-2: {metrics['mean_top_gap']:.3f}",
        f"  Baseline original: top-1="
        f"{metrics['baseline_original']['accuracy_at_1']:.3f}, "
        f"top-3={metrics['baseline_original']['accuracy_at_3']:.3f}, "
        f"MRR={metrics['baseline_original']['mean_reciprocal_rank']:.3f}",
        f"  Casos controlados: acuracia end-to-end="
        f"{metrics['controlled_cases']['end_to_end_accuracy']:.3f}",
    ]


def _case_label(case: Mapping[str, Any]) -> str:
    """Format a compact comparison case without changing its query text."""
    query = str(case["query"])
    display_query = repr(query) if not query.strip() else query
    if case["expected_fallback"] is True:
        expected = "fallback"
    elif case["expected_id"]:
        expected = f"id={case['expected_id']}"
    else:
        expected = "nao pontuado"
    return f"  - [{case['case_type']}] {display_query} (esperado: {expected})"


def _format_case_group(
    title: str,
    cases: list[Mapping[str, Any]],
) -> list[str]:
    """Format all cases in one comparison category."""
    lines = [f"{title}: {len(cases)}"]
    lines.extend(_case_label(case) for case in cases)
    return lines


def format_report(
    metrics: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Format a complete lexical-versus-semantic evaluation report."""
    del top_k
    tfidf = metrics["tfidf"]
    semantic = metrics["semantic"]
    calibration = metrics["semantic_threshold_calibration"]
    comparison = metrics["comparison"]
    correct_rejections = comparison["correct_rejections"]
    lines = [
        "FAQtor - Relatorio comparativo de avaliacao",
        f"Consultas totais: {metrics['total_queries']}",
        f"Baseline original: {tfidf['baseline_queries']}",
        f"Casos controlados: {tfidf['controlled_queries']}",
        f"Consultas pontuadas: {tfidf['scored_queries']}",
        f"Consultas nao pontuadas: {tfidf['unscored_queries']}",
        f"Consultas positivas: {tfidf['positive_queries']}",
        f"Consultas negativas: {tfidf['negative_queries']}",
        "",
    ]
    lines.extend(_format_method("TF-IDF", tfidf))
    lines.append("")
    lines.extend(_format_method("Embedding semantico", semantic))
    selected = calibration["selected"]
    best_thresholds = ", ".join(
        f"{threshold:.2f}" for threshold in calibration["best_thresholds"]
    )
    lines.extend(
        [
            "",
            "Calibracao do limiar semantico (sweep 0.40 a 0.80)",
            f"  Limiar selecionado: {calibration['selected_threshold']:.2f}",
            f"  Melhor acuracia end-to-end: "
            f"{calibration['best_end_to_end_accuracy']:.3f}",
            f"  Limiares empatados no melhor resultado: {best_thresholds}",
            f"  No limiar selecionado: falsos positivos="
            f"{selected['false_positives']}, falsos negativos="
            f"{selected['false_negatives']}",
        ],
    )
    lines.extend(["", "Comparacao caso a caso"])
    lines.extend(
        _format_case_group(
            "TF-IDF errou e semantico acertou",
            comparison["tfidf_wrong_semantic_correct"],
        ),
    )
    lines.extend(
        _format_case_group(
            "Semantico errou e TF-IDF acertou",
            comparison["semantic_wrong_tfidf_correct"],
        ),
    )
    lines.extend(
        _format_case_group("Ambos erraram", comparison["both_wrong"]),
    )
    lines.extend(
        _format_case_group(
            "Casos nao pontuados para inspecao de gap",
            comparison["unscored"],
        ),
    )
    lines.extend(
        [
            f"Rejeicoes corretas do TF-IDF: {len(correct_rejections['tfidf'])}",
            f"Rejeicoes corretas do semantico: "
            f"{len(correct_rejections['semantic'])}",
        ],
    )
    lines.extend(
        _format_case_group(
            "Rejeicoes corretas de ambos",
            correct_rejections["both"],
        ),
    )
    return "\n".join(lines)


def run_evaluation(
    faq_path: str | Path = FAQ_PATH,
    test_queries_path: str | Path = TEST_QUERIES_PATH,
    top_k: int = DEFAULT_TOP_K,
    semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    tfidf_threshold: float = TFIDF_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Run the comparative evaluation and print its terminal report."""
    metrics = evaluate(
        faq_path=faq_path,
        test_queries_path=test_queries_path,
        top_k=top_k,
        semantic_threshold=semantic_threshold,
        tfidf_threshold=tfidf_threshold,
    )
    print(format_report(metrics, top_k=top_k))
    return metrics
