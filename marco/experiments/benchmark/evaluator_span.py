"""Evaluador de retrieval basado en EvidenceSpan [char_start, char_end).

Fail-closed: si alguna pregunta del benchmark no mapea a chunks con
contención completa usando inicio_car/fin_car, lanza error.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class BenchmarkValidationError(RuntimeError):
    """El benchmark no mapea completamente contra el dataset."""


@dataclass(frozen=True)
class QuestionGroundTruth:
    question_id: str
    ticker: str
    fiscal_year: int
    item: str
    question: str
    char_start: int
    char_end: int
    relevant_chunk_ids: tuple[str, ...]


def _chunks_of_section(meta: pd.DataFrame, ticker: str,
                       fiscal_year: int, item: str) -> pd.DataFrame:
    return meta[
        (meta["ticker"].astype(str).str.upper() == ticker.upper())
        & (meta["fiscal_year"].astype(int) == int(fiscal_year))
        & (meta["item"].astype(str) == str(item))
    ]


def map_span_to_chunks(evidence_start: int, evidence_end: int,
                       chunks_of_section: pd.DataFrame) -> tuple[str, ...]:
    """Devuelve los chunk_id que contienen COMPLETAMENTE el span.

    Contención completa: inicio_car <= evidence_start AND fin_car >= evidence_end.
    """
    hits = []
    for _, row in chunks_of_section.iterrows():
        if int(row["inicio_car"]) <= evidence_start \
                and int(row["fin_car"]) >= evidence_end:
            hits.append(str(row["chunk_id"]))
    return tuple(hits)


def validate_benchmark(benchmark_path, meta: pd.DataFrame) -> list[QuestionGroundTruth]:
    """Carga el benchmark y valida que las 48 preguntas mapean.

    Lanza BenchmarkValidationError si alguna no mapea.
    """
    import json

    with open(benchmark_path, encoding="utf-8") as stream:
        raw = [json.loads(line) for line in stream if line.strip()]

    ground_truth: list[QuestionGroundTruth] = []
    failures: list[str] = []

    for q in raw:
        chunks = _chunks_of_section(
            meta, q["ticker"], q["fiscal_year"], q["item"]
        )
        hits = map_span_to_chunks(q["char_start"], q["char_end"], chunks)
        if not hits:
            failures.append(q["question_id"])
            continue
        ground_truth.append(QuestionGroundTruth(
            question_id=q["question_id"],
            ticker=q["ticker"],
            fiscal_year=int(q["fiscal_year"]),
            item=str(q["item"]),
            question=q["question"],
            char_start=int(q["char_start"]),
            char_end=int(q["char_end"]),
            relevant_chunk_ids=hits,
        ))

    if failures:
        raise BenchmarkValidationError(
            f"Fail-closed: {len(failures)} preguntas no mapean a chunks. "
            f"IDs: {failures}"
        )
    if len(ground_truth) != 48:
        raise BenchmarkValidationError(
            f"Se esperaban 48 preguntas, se cargaron {len(ground_truth)}"
        )
    return ground_truth


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def first_relevant_rank(ranking: list[str], relevant: set[str]) -> int | None:
    for rank, chunk_id in enumerate(ranking, 1):
        if chunk_id in relevant:
            return rank
    return None


def recall_at_k(rankings: list[dict], k: int) -> float:
    if not rankings:
        return 0.0
    hits = 0
    for row in rankings:
        relevant = set(row["relevant_chunk_ids"])
        top_k = set(row["ranking"][:k])
        if relevant & top_k:
            hits += 1
    return hits / len(rankings)


def mrr_at_k(rankings: list[dict], k: int) -> float:
    if not rankings:
        return 0.0
    rr = []
    for row in rankings:
        relevant = set(row["relevant_chunk_ids"])
        rank = first_relevant_rank(row["ranking"][:k], relevant)
        rr.append(0.0 if rank is None else 1.0 / rank)
    return sum(rr) / len(rr)


def summarize_rankings(rankings: list[dict]) -> dict:
    return {
        "n_questions": len(rankings),
        "recall@1": recall_at_k(rankings, 1),
        "recall@3": recall_at_k(rankings, 3),
        "recall@5": recall_at_k(rankings, 5),
        "recall@10": recall_at_k(rankings, 10),
        "mrr@10": mrr_at_k(rankings, 10),
    }


# ---------------------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------------------

VIEWS = {
    "ALL-48": lambda q: True,
    "NON-7A-36": lambda q: q.item != "7A",
    "ITEM-1A-12": lambda q: q.item == "1A",
    "ITEM-7-12": lambda q: q.item == "7",
    "ITEM-7A-12": lambda q: q.item == "7A",
    "ITEM-8-12": lambda q: q.item == "8",
}


def build_views(rankings: list[dict],
                questions: dict[str, QuestionGroundTruth]) -> dict[str, list[dict]]:
    """Devuelve las 6 vistas sobre las que se reportan métricas."""
    views: dict[str, list[dict]] = {}
    for name, predicate in VIEWS.items():
        views[name] = [
            r for r in rankings
            if predicate(questions[r["question_id"]])
        ]
    return views