"""Evaluación determinista del retrieval E0 sobre evidencias textuales."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from common.eval.retrieval_metrics import evaluate_rankings
from dani.experiments.benchmark.benchmark_v2 import (
    BENCHMARK_PATH,
    load_benchmark,
    validate_benchmark,
    verify_frozen_benchmark,
)
from dani.experiments.retriever import DenseFaissRetriever


EVALUATION_VIEWS = (
    "ALL-48",
    "ITEM-1A-12",
    "ITEM-7-12",
    "ITEM-7A-12",
    "ITEM-8-12",
    "NON-7A-36",
)


class EvidenceContainmentError(ValueError):
    """Ningún chunk contiene completamente el EvidenceSpan."""


def evaluate_benchmark_views(
    per_question: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """Agrega las vistas predefinidas dando el mismo peso a cada pregunta."""
    selectors = {
        "ALL-48": lambda row: True,
        "ITEM-1A-12": lambda row: row["item"] == "1A",
        "ITEM-7-12": lambda row: row["item"] == "7",
        "ITEM-7A-12": lambda row: row["item"] == "7A",
        "ITEM-8-12": lambda row: row["item"] == "8",
        "NON-7A-36": lambda row: row["item"] in {"1A", "7", "8"},
    }
    return {
        name: evaluate_rankings([row for row in per_question if selector(row)])
        for name, selector in selectors.items()
    }


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", without_marks)).strip()


def _anchor_parts(anchor: str) -> list[str]:
    return [
        normalized
        for part in re.split(r"\s*(?:\.\.\.|\u2026)+\s*", anchor)
        if (normalized := _normalize(part))
    ]


class RetrievalEvaluator:
    """Evalúa anclas, no IDs como verdad conceptual.

    Los IDs actuales se conservan para reportar E0, pero cambiarán al cambiar
    el chunking. Por eso la relevancia se vuelve a localizar desde evidencia
    literal restringida por documento, ejercicio e item.
    """

    def __init__(self, golden_path: Path, metadata: Any, max_k: int = 10):
        self.golden_path = golden_path
        self.metadata = metadata
        self.max_k = max_k
        self.questions = self._load_questions()

    def evaluate(
        self, retriever: DenseFaissRetriever
    ) -> tuple[dict[str, float | int | None], list[dict[str, Any]]]:
        per_question: list[dict[str, Any]] = []
        for question in self.questions:
            relevant = self.relevant_chunk_ids(question)
            outcome = retriever.search(
                question["pregunta"],
                ticker=question["ticker"],
                fiscal_year=int(question["fiscal_year"]),
                item=question.get("item", question.get("item_esperado")),
                k=self.max_k,
            )
            ranking = [
                {
                    "rank": rank,
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "ticker": result["ticker"],
                    "fiscal_year": result["fiscal_year"],
                    "item": result["item"],
                    "posicion": result["posicion"],
                }
                for rank, result in enumerate(outcome.results, 1)
            ]
            relevant_set = set(relevant)
            first_rank = next(
                (row["rank"] for row in ranking
                 if row["chunk_id"] in relevant_set),
                None,
            )
            per_question.append({
                "question_id": question["id"],
                "query": question["pregunta"],
                "ticker": question["ticker"],
                "fiscal_year": int(question["fiscal_year"]),
                "item": question.get("item", question.get("item_esperado")),
                "relevant_chunk_ids": relevant,
                "ranking": ranking,
                "first_relevant_rank": first_rank,
                "latency_s": {
                    "query_embedding": outcome.query_embedding_s,
                    "index_search": outcome.index_search_s,
                    "postfilter": outcome.postfilter_s,
                    "total": outcome.total_s,
                },
            })
        return evaluate_rankings(per_question), per_question

    def relevant_chunk_ids(self, question: dict[str, Any]) -> list[str]:
        raw_anchors = [
            question.get("ancla_texto"),
            *(question.get("anclas_alternativas") or []),
        ]
        anchors = [_anchor_parts(anchor) for anchor in raw_anchors if anchor]
        item = question.get("item", question.get("item_esperado"))
        subset = self.metadata[
            (self.metadata["ticker"].astype(str) == str(question["ticker"]))
            & (self.metadata["fiscal_year"].astype(int)
               == int(question["fiscal_year"]))
            & (self.metadata["item"].astype(str) == str(item))
        ]
        relevant = []
        for _, row in subset.iterrows():
            text = _normalize(str(row["texto"]))
            if any(all(part in text for part in parts) for parts in anchors):
                relevant.append(str(row["chunk_id"]))
        if not relevant:
            raise ValueError(f"{question['id']}: ninguna ancla aparece en el corpus")
        expected = question.get("chunk_id_esperado")
        if expected and expected not in relevant:
            raise ValueError(
                f"{question['id']}: el chunk esperado no contiene el ancla"
            )
        return relevant

    def _load_questions(self) -> list[dict[str, Any]]:
        with self.golden_path.open(encoding="utf-8") as stream:
            questions = [json.loads(line) for line in stream if line.strip()]
        return [question for question in questions if question.get("ancla_texto")]


class BenchmarkV2Evaluator:
    """Evalúa el benchmark congelado proyectando spans sobre el chunking."""

    def __init__(
        self,
        metadata: Any,
        benchmark_path: Path = BENCHMARK_PATH,
        max_k: int = 10,
    ):
        self.benchmark_path = benchmark_path
        self.metadata = metadata
        self.max_k = max_k
        verify_frozen_benchmark(benchmark_path)
        validate_benchmark(benchmark_path)
        self.questions = load_benchmark(benchmark_path)
        self._validate_all_containment()

    @staticmethod
    def candidate_chunks(question: dict[str, Any], metadata: Any) -> Any:
        """Selecciona el pool ticker/año/item sin alterar su orden."""
        required = {
            "chunk_id", "ticker", "fiscal_year", "item",
            "inicio_car", "fin_car",
        }
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"Metadata sin columnas requeridas: {sorted(missing)}")
        return metadata[
            (metadata["ticker"].astype(str) == str(question["ticker"]))
            & (metadata["fiscal_year"].astype(int)
               == int(question["fiscal_year"]))
            & (metadata["item"].astype(str) == str(question["item"]))
        ]

    @classmethod
    def relevant_chunks(
        cls, question: dict[str, Any], metadata: Any
    ) -> tuple[list[str], int]:
        """Deriva todos los chunks que contienen completamente el span."""
        candidates = cls.candidate_chunks(question, metadata)
        start = int(question["char_start"])
        end = int(question["char_end"])
        relevant = [
            str(row.chunk_id)
            for row in candidates.itertuples(index=False)
            if int(row.inicio_car) <= start and int(row.fin_car) >= end
        ]
        if not relevant:
            raise EvidenceContainmentError(
                f"{question['question_id']}: cero chunks con full containment"
            )
        return relevant, len(candidates)

    def relevant_chunk_ids(self, question: dict[str, Any]) -> list[str]:
        return self.relevant_chunks(question, self.metadata)[0]

    def _validate_all_containment(self) -> None:
        failures: list[str] = []
        for question in self.questions:
            try:
                self.relevant_chunk_ids(question)
            except EvidenceContainmentError:
                failures.append(str(question["question_id"]))
        if failures:
            raise EvidenceContainmentError(
                "Benchmark no evaluable; cero full-containment para: "
                + ", ".join(failures)
            )

    def evaluate(
        self, retriever: DenseFaissRetriever
    ) -> tuple[
        dict[str, dict[str, float | int | None]],
        list[dict[str, Any]],
    ]:
        per_question: list[dict[str, Any]] = []
        for question in self.questions:
            relevant, candidate_pool_size = self.relevant_chunks(
                question, self.metadata
            )
            outcome = retriever.search(
                question["question"],
                ticker=question["ticker"],
                fiscal_year=int(question["fiscal_year"]),
                item=question["item"],
                k=self.max_k,
            )
            ranking = [
                {
                    "rank": rank,
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "ticker": result["ticker"],
                    "fiscal_year": result["fiscal_year"],
                    "item": result["item"],
                    "posicion": result["posicion"],
                }
                for rank, result in enumerate(outcome.results, 1)
            ]
            relevant_set = set(relevant)
            first_rank = next(
                (row["rank"] for row in ranking
                 if row["chunk_id"] in relevant_set),
                None,
            )
            per_question.append({
                "question_id": question["question_id"],
                "query": question["question"],
                "ticker": question["ticker"],
                "fiscal_year": int(question["fiscal_year"]),
                "item": question["item"],
                "evidence_type": question["evidence_type"],
                "evidence_span": {
                    "char_start": int(question["char_start"]),
                    "char_end": int(question["char_end"]),
                    "evidence_text": question["evidence_text"],
                    "matching_method": question["matching_method"],
                },
                "candidate_pool_size": candidate_pool_size,
                "relevant_chunk_ids": relevant,
                "n_relevant_chunks": len(relevant),
                "ranking": ranking,
                "first_relevant_rank": first_rank,
                "latency_s": {
                    "query_embedding": outcome.query_embedding_s,
                    "index_search": outcome.index_search_s,
                    "postfilter": outcome.postfilter_s,
                    "total": outcome.total_s,
                },
            })
        return evaluate_benchmark_views(per_question), per_question

