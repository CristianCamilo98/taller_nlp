"""Evalúa el retriever denso COMMON sobre el benchmark congelado de 48 preguntas.

El runner no reescribe queries, no usa BM25 y no aplica reranking. Cada query
se ordena contra el índice FAISS completo y después se filtra por metadata en
``common.retrieval.dense_baseline.buscar``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from common.benchmark import (
    BENCHMARK_NAME,
    BENCHMARK_PATH,
    FROZEN_SHA256,
)
from common.eval.retrieval_metrics import evaluate_rankings
from common.retrieval.profiles import PROFILE_ENV_VAR, PROFILES, get_embedding_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_VIEWS = (
    "ALL-48",
    "ITEM-1A-12",
    "ITEM-7-12",
    "ITEM-7A-12",
    "ITEM-8-12",
    "NON-7A-36",
)


class RetrievalBenchmarkError(RuntimeError):
    """El benchmark, la selección o sus artefactos no son evaluables."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_frozen_benchmark(path: Path = BENCHMARK_PATH) -> str:
    digest = _sha256(path)
    if digest != FROZEN_SHA256:
        raise RetrievalBenchmarkError(
            f"SHA-256 de benchmark inválido: {digest}; esperado {FROZEN_SHA256}"
        )
    return digest


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    if len(questions) != 48:
        raise RetrievalBenchmarkError(
            f"Benchmark incompleto: {len(questions)} preguntas; esperadas 48"
        )
    ids = [str(question["question_id"]) for question in questions]
    if len(set(ids)) != len(ids):
        raise RetrievalBenchmarkError("El benchmark contiene question_id duplicados")
    return questions


def select_questions(
    questions: list[dict[str, Any]], question_ids: list[str] | None
) -> list[dict[str, Any]]:
    if not question_ids:
        return questions
    requested = list(dict.fromkeys(question_ids))
    by_id = {str(question["question_id"]): question for question in questions}
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        raise RetrievalBenchmarkError(
            f"question_id no encontrado: {', '.join(missing)}"
        )
    return [by_id[question_id] for question_id in requested]


def candidate_chunks(question: dict[str, Any], metadata: Any) -> Any:
    return metadata[
        (metadata["ticker"].astype(str).str.upper()
         == str(question["ticker"]).upper())
        & (metadata["fiscal_year"].astype(int) == int(question["fiscal_year"]))
        & (metadata["item"].astype(str) == str(question["item"]))
    ]


def relevant_chunk_ids(
    question: dict[str, Any], metadata: Any
) -> tuple[list[str], int]:
    """Obtiene chunks cuyo intervalo contiene por completo el EvidenceSpan."""
    candidates = candidate_chunks(question, metadata)
    start = int(question["char_start"])
    end = int(question["char_end"])
    relevant = [
        str(row.chunk_id)
        for row in candidates.itertuples(index=False)
        if int(row.inicio_car) <= start and int(row.fin_car) >= end
    ]
    if not relevant:
        raise RetrievalBenchmarkError(
            f"{question['question_id']}: cero chunks con full containment"
        )
    return relevant, len(candidates)


def evaluate_benchmark_views(
    per_question: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
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


def _ranking(hits: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": str(hit["chunk_id"]),
            "score": float(hit["puntuacion"]),
            "ticker": str(hit["ticker"]),
            "fiscal_year": int(hit["fiscal_year"]),
            "item": str(hit["item"]),
        }
        for rank, hit in enumerate(hits[:max_k], 1)
    ]


def _first_relevant_rank(
    ranking: list[dict[str, Any]], relevant: list[str]
) -> int | None:
    relevant_set = set(relevant)
    return next(
        (row["rank"] for row in ranking if row["chunk_id"] in relevant_set),
        None,
    )


def _commit_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _artifact_hash(path: Path | None) -> str | None:
    return _sha256(path) if path is not None and path.is_file() else None


def run_benchmark(
    *,
    profile_name: str,
    output_path: Path,
    benchmark_path: Path = BENCHMARK_PATH,
    question_ids: list[str] | None = None,
    max_k: int = 10,
    metadata: Any | None = None,
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    provenance_fn: Callable[[], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Ejecuta una selección o las 48 preguntas mediante COMMON."""
    if max_k < 1 or max_k > 10:
        raise RetrievalBenchmarkError("max_k debe estar entre 1 y 10")
    profile = get_embedding_profile(profile_name)
    digest = verify_frozen_benchmark(benchmark_path)
    questions = select_questions(load_benchmark(benchmark_path), question_ids)

    os.environ[PROFILE_ENV_VAR] = profile.name
    from common.config import clear_dataset_path_cache, get_dataset_paths
    from common.retrieval.dense_baseline import (
        buscar,
        clear_dense_retrieval_cache,
        get_last_query_embedding_metadata,
    )

    clear_dataset_path_cache()
    clear_dense_retrieval_cache()
    paths = get_dataset_paths(profile.name)
    if metadata is None:
        import pandas as pd

        metadata = pd.read_parquet(paths.chunks_meta)
    required_columns = {
        "chunk_id", "ticker", "fiscal_year", "item", "inicio_car", "fin_car"
    }
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise RetrievalBenchmarkError(
            f"chunks_meta sin columnas requeridas: {sorted(missing_columns)}"
        )
    for question in questions:
        relevant_chunk_ids(question, metadata)

    actual_search = search_fn or buscar
    actual_provenance = provenance_fn or get_last_query_embedding_metadata
    print(
        f"COMMON retrieval benchmark · {len(questions)} pregunta(s) · "
        f"profile={profile.name} · max_k={max_k}",
        flush=True,
    )

    per_question: list[dict[str, Any]] = []
    for position, question in enumerate(questions, 1):
        relevant, candidate_pool_size = relevant_chunk_ids(question, metadata)
        started = perf_counter()
        hits = actual_search(
            question["question"],
            ticker=str(question["ticker"]),
            fiscal_year=int(question["fiscal_year"]),
            item=str(question["item"]),
            k=max_k,
        )
        latency_s = perf_counter() - started
        ranking = _ranking(hits, max_k)
        first_rank = _first_relevant_rank(ranking, relevant)
        query_provenance = actual_provenance()
        per_question.append({
            "question_id": question["question_id"],
            "query": question["question"],
            "ticker": question["ticker"],
            "fiscal_year": int(question["fiscal_year"]),
            "item": str(question["item"]),
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
            "query_embedding_provenance": query_provenance,
            "latency_s": {"total": round(latency_s, 6)},
        })
        marker = "HIT" if first_rank is not None and first_rank <= 5 else "MISS"
        print(
            f"[{position}/{len(questions)}] {question['question_id']} "
            f"first_rel={first_rank} R@5={marker}",
            flush=True,
        )

    metrics = evaluate_benchmark_views(per_question)
    payload = {
        "manifest": {
            "runner": "common.scripts.evaluate_retrieval_benchmark",
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_sha256": digest,
            "benchmark_path": str(benchmark_path),
            "selection": [row["question_id"] for row in per_question],
            "n_questions": len(per_question),
            "max_k": max_k,
            "profile": profile.name,
            "embedding_model_id": profile.model_name,
            "embedding_dimension": profile.dimension,
            "faiss_index": str(paths.faiss_index),
            "faiss_index_sha256": _artifact_hash(paths.faiss_index),
            "chunks_meta_sha256": _artifact_hash(paths.chunks_meta),
            "retrieval": {
                "type": "dense_faiss_with_metadata_postfilter",
                "policy": "global rank + ticker/fiscal_year/item postfilter",
                "query_rewriting": False,
                "bm25": False,
                "reranking": False,
            },
            "evaluation_views": list(EVALUATION_VIEWS),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "commit_sha": _commit_sha(),
        },
        "metrics": metrics,
        "per_question": per_question,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Artifact: {output_path.resolve()}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa COMMON dense retrieval sobre el benchmark v2 congelado"
    )
    parser.add_argument(
        "--profile", required=True, choices=sorted(PROFILES),
        help="Perfil de embeddings/índice que se evaluará",
    )
    parser.add_argument(
        "--question-id", action="append", dest="question_ids",
        help="Pregunta concreta; se puede repetir. Sin esta opción ejecuta las 48",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--max-k", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(
        profile_name=args.profile,
        output_path=args.output.resolve(),
        benchmark_path=args.benchmark.resolve(),
        question_ids=args.question_ids,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()
