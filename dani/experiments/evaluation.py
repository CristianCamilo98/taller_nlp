"""Evaluación determinista del retrieval E0 sobre evidencias textuales."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from common.eval.retrieval_metrics import evaluate_rankings
from dani.experiments.retriever import DenseFaissRetriever


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

