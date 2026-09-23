"""Runner canónico de la ablación de reranking.

Guarda per-question y genera las 6 vistas con todas las métricas.
Fail-closed si el benchmark no mapea.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import os

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from common.config import get_dataset_paths
from common.retrieval.dense_baseline import buscar as buscar_denso

from evaluator_span import (
    BenchmarkValidationError,
    QuestionGroundTruth,
    build_views,
    first_relevant_rank,
    summarize_rankings,
    validate_benchmark,
)


BENCHMARK_PATH = Path(__file__).resolve().parent / "data" / "retrieval_benchmark_v2.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results" / "final" / "frankestein_Qwen3_BGE.json"


def _top20_denso(question: str, gt: QuestionGroundTruth) -> tuple[list[dict], float]:
    """Devuelve top-20 denso y latencia en ms."""
    t0 = time.perf_counter()
    results = buscar_denso(
        question, ticker=gt.ticker, fiscal_year=gt.fiscal_year,
        item=gt.item, k=20,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms


def _top20_reranked(question: str, gt: QuestionGroundTruth) -> tuple[list[dict], float]:
    """Devuelve top-20 rerankeado y latencia del reranker (ms)."""
    from common.retrieval.reranked_baseline import buscar as buscar_reranked
    t0 = time.perf_counter()
    results = buscar_reranked(
        question, ticker=gt.ticker, fiscal_year=gt.fiscal_year,
        item=gt.item, k=20,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms


def run() -> dict:
    # Cargar meta y validar benchmark fail-closed
    paths = get_dataset_paths()
    meta = pd.read_parquet(paths.chunks_meta)

    try:
        ground_truth = validate_benchmark(BENCHMARK_PATH, meta)
    except BenchmarkValidationError as e:
        print(f"FAIL-CLOSED: {e}")
        raise

    questions = {gt.question_id: gt for gt in ground_truth}
    print(f"Benchmark validado: {len(ground_truth)} preguntas mapeadas")

    per_question = []
    latencies_dense = []
    latencies_rerank = []

    for i, gt in enumerate(ground_truth, 1):
        if i % 5 == 0 or i == len(ground_truth):
            print(f"  [{i}/{len(ground_truth)}] {gt.question_id}")

        relevant = set(gt.relevant_chunk_ids)

        # Top-20 denso
        denso, lat_d = _top20_denso(gt.question, gt)
        latencies_dense.append(lat_d)
        dense_ids = [c["chunk_id"] for c in denso]
        dense_rank = first_relevant_rank(dense_ids, relevant)

        # Top-20 rerankeado
        reranked, lat_r = _top20_reranked(gt.question, gt)
        latencies_rerank.append(lat_r)
        reranked_ids = [c["chunk_id"] for c in reranked]
        reranked_rank = first_relevant_rank(reranked_ids, relevant)

        per_question.append({
            "question_id": gt.question_id,
            "ticker": gt.ticker,
            "fiscal_year": gt.fiscal_year,
            "item": gt.item,
            "char_start": gt.char_start,
            "char_end": gt.char_end,
            "relevant_chunk_ids": list(gt.relevant_chunk_ids),
            "dense_top20": [
                {"rank": j, "chunk_id": c["chunk_id"],
                 "score": float(c.get("puntuacion", 0.0))}
                for j, c in enumerate(denso, 1)
            ],
            "reranked_top20": [
                {"rank": j, "chunk_id": c["chunk_id"],
                 "score": float(c.get("puntuacion", 0.0))}
                for j, c in enumerate(reranked, 1)
            ],
            "dense_first_relevant_rank": dense_rank,
            "reranked_first_relevant_rank": reranked_rank,
            "dense_latency_ms": round(lat_d, 2),
            "reranked_latency_ms": round(lat_r, 2),
        })

    # Vistas
    views_dense = build_views(
        [{"question_id": r["question_id"],
          "relevant_chunk_ids": r["relevant_chunk_ids"],
          "ranking": [c["chunk_id"] for c in r["dense_top20"]]}
         for r in per_question],
        questions,
    )
    views_reranked = build_views(
        [{"question_id": r["question_id"],
          "relevant_chunk_ids": r["relevant_chunk_ids"],
          "ranking": [c["chunk_id"] for c in r["reranked_top20"]]}
         for r in per_question],
        questions,
    )

    # Métricas por vista
    metrics = {"dense": {}, "reranked": {}}
    for view_name in views_dense:
        metrics["dense"][view_name] = summarize_rankings(views_dense[view_name])
        metrics["reranked"][view_name] = summarize_rankings(views_reranked[view_name])

    # Candidate Recall@20
    candidate_recall_20 = {}
    for view_name in views_dense:
        rows = views_dense[view_name]
        if not rows:
            candidate_recall_20[view_name] = None
            continue
        hits = sum(
            1 for r in rows
            if set(r["relevant_chunk_ids"]) & set(r["ranking"][:20])
        )
        candidate_recall_20[view_name] = hits / len(rows)

    result = {
        "config": {
            "benchmark_path": str(BENCHMARK_PATH.relative_to(RAIZ)),
            "benchmark_sha256": "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C",
            "embedding_model": os.getenv("MIAX_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            "embedding_query_prefix": os.getenv("MIAX_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "),
            "index_dirname": os.getenv("MIAX_INDEX_DIRNAME", "indice_faiss"),
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "n_candidates": 20,
            "top_k_saved": 20,
        },
        "metrics": metrics,
        "candidate_recall@20": candidate_recall_20,
        "latency_ms": {
            "dense_mean": sum(latencies_dense) / len(latencies_dense),
            "reranker_mean": sum(latencies_rerank) / len(latencies_rerank),
        },
        "per_question": per_question,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGuardado en {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    run()