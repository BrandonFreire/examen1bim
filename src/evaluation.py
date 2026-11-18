"""Métricas @k: Precision, Recall (binarias) y NDCG con relevancia graduada."""

from __future__ import annotations

from collections.abc import Mapping
from math import log2
from statistics import mean

from src.retrievers import Retriever

MIN_RELEVANT_SCORE = 1  # score >= 1 cuenta como relevante en las métricas binarias


def _check(ranking: list[str], k: int) -> None:
    if k <= 0:
        raise ValueError(f"k debe ser positivo, se recibió {k}")
    if len(set(ranking)) != len(ranking):
        raise ValueError("el ranking contiene doc_id duplicados")


def _relevant_ids(judgments: Mapping[str, int]) -> set[str]:
    return {doc_id for doc_id, score in judgments.items() if score >= MIN_RELEVANT_SCORE}


def precision_at_k(ranking: list[str], judgments: Mapping[str, int], k: int = 10) -> float:
    _check(ranking, k)
    relevant = _relevant_ids(judgments)
    # Se divide siempre por k aunque haya menos de k resultados
    return sum(1 for d in ranking[:k] if d in relevant) / k


def recall_at_k(ranking: list[str], judgments: Mapping[str, int], k: int = 10) -> float:
    _check(ranking, k)
    relevant = _relevant_ids(judgments)
    if not relevant:
        return 0.0
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def ndcg_at_k(ranking: list[str], judgments: Mapping[str, int], k: int = 10) -> float:
    """NDCG con ganancia lineal: DCG = sum(score / log2(pos + 1))."""
    _check(ranking, k)
    dcg = sum(
        judgments.get(doc_id, 0) / log2(pos + 1)
        for pos, doc_id in enumerate(ranking[:k], start=1)
    )
    ideal = sorted((s for s in judgments.values() if s > 0), reverse=True)[:k]
    idcg = sum(s / log2(pos + 1) for pos, s in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    retriever: Retriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Métricas por consulta para las consultas que tienen qrels relevantes."""
    per_query: dict[str, dict[str, float]] = {}
    for qid in sorted(qrels):
        if qid not in queries or not _relevant_ids(qrels[qid]):
            continue
        ranking = [r.doc_id for r in retriever.search(queries[qid], k=k)]
        per_query[qid] = {
            "P@10": precision_at_k(ranking, qrels[qid], k),
            "R@10": recall_at_k(ranking, qrels[qid], k),
            "NDCG@10": ndcg_at_k(ranking, qrels[qid], k),
        }
    return per_query


def summarize(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    metrics = ("P@10", "R@10", "NDCG@10")
    return {m: mean(q[m] for q in per_query.values()) for m in metrics}
