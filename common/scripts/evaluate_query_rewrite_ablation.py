"""Evalua queries originales y reescritas sobre el retriever denso COMMON.

La cache JSONL es de solo lectura: este runner nunca genera ni modifica
rewrites. El modo ``rewritten`` hace exactamente una busqueda por pregunta y
por tanto permite ejecutar G1 sin volver a embeber las queries originales.
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

from common.benchmark import BENCHMARK_NAME, BENCHMARK_PATH, FROZEN_SHA256
from common.retrieval.profiles import PROFILE_ENV_VAR, PROFILES, get_embedding_profile
from common.scripts.evaluate_retrieval_benchmark import (
    EVALUATION_VIEWS,
    RetrievalBenchmarkError,
    _first_relevant_rank,
    _ranking,
    evaluate_benchmark_views,
    relevant_chunk_ids,
)
from common.scripts.generate_query_rewrites import (
    DEFAULT_BENCHMARK_PATH,
    PROMPT_SHA256,
    load_cache,
    load_frozen_benchmark,
    validate_complete_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = (
    REPO_ROOT / "common" / "results" / "query_rewriting"
    / "query_rewrites_48.jsonl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _artifact_hash(path: Path | None) -> str | None:
    return _sha256(path) if path is not None and path.is_file() else None


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


def _metric_rows(
    per_question: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    return [
        {
            "question_id": row["question_id"],
            "item": row["item"],
            "relevant_chunk_ids": row["relevant_chunk_ids"],
            "ranking": row[variant]["ranking"],
        }
        for row in per_question
    ]


def _in_top(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def paired_analysis(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """Compara first-relevant-rank y cruces de top-k de forma pareada."""
    rows: list[dict[str, Any]] = []
    counts = {
        "rank_improves": 0,
        "rank_worsens": 0,
        "rank_equal": 0,
        "enters_top5": 0,
        "leaves_top5": 0,
        "enters_top10": 0,
        "leaves_top10": 0,
    }
    for row in per_question:
        original = row["original"]["first_relevant_rank"]
        rewritten = row["rewritten"]["first_relevant_rank"]
        original_order = original if original is not None else float("inf")
        rewritten_order = rewritten if rewritten is not None else float("inf")
        if rewritten_order < original_order:
            relation = "improves"
            counts["rank_improves"] += 1
        elif rewritten_order > original_order:
            relation = "worsens"
            counts["rank_worsens"] += 1
        else:
            relation = "equal"
            counts["rank_equal"] += 1
        enters_top5 = not _in_top(original, 5) and _in_top(rewritten, 5)
        leaves_top5 = _in_top(original, 5) and not _in_top(rewritten, 5)
        enters_top10 = not _in_top(original, 10) and _in_top(rewritten, 10)
        leaves_top10 = _in_top(original, 10) and not _in_top(rewritten, 10)
        counts["enters_top5"] += int(enters_top5)
        counts["leaves_top5"] += int(leaves_top5)
        counts["enters_top10"] += int(enters_top10)
        counts["leaves_top10"] += int(leaves_top10)
        rows.append({
            "question_id": row["question_id"],
            "original_first_relevant_rank": original,
            "rewritten_first_relevant_rank": rewritten,
            "rank_change": relation,
            "enters_top5": enters_top5,
            "leaves_top5": leaves_top5,
            "enters_top10": enters_top10,
            "leaves_top10": leaves_top10,
        })
    return {"counts": counts, "per_question": rows}


def run_evaluation(
    *,
    profile_name: str,
    mode: str,
    cache_path: Path,
    output_path: Path,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    max_k: int = 10,
    metadata: Any | None = None,
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    provenance_fn: Callable[[], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    if mode not in {"ablation", "rewritten"}:
        raise RetrievalBenchmarkError(f"Modo no soportado: {mode}")
    if max_k != 10:
        raise RetrievalBenchmarkError("Esta evaluacion congelada exige max_k=10")
    if benchmark_path.resolve() != BENCHMARK_PATH.resolve():
        raise RetrievalBenchmarkError("Debe usarse el benchmark v2 congelado")

    profile = get_embedding_profile(profile_name)
    questions = load_frozen_benchmark(benchmark_path)
    cache = load_cache(cache_path, questions)
    validate_complete_cache(cache, questions)

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

    actual_search = search_fn or buscar
    actual_provenance = provenance_fn or get_last_query_embedding_metadata
    variants = ("original", "rewritten") if mode == "ablation" else ("rewritten",)
    print(
        f"COMMON query rewrite evaluation - {len(questions)} preguntas - "
        f"profile={profile.name} - mode={mode}",
        flush=True,
    )

    per_question: list[dict[str, Any]] = []
    for position, question in enumerate(questions, 1):
        question_id = str(question["question_id"])
        cached = cache[question_id]
        relevant, candidate_pool_size = relevant_chunk_ids(question, metadata)
        row: dict[str, Any] = {
            "question_id": question_id,
            "ticker": str(question["ticker"]),
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
            "rewrite_cache_status": cached["status"],
            "fallback_used": cached["fallback_used"],
        }
        for variant in variants:
            query = (
                str(question["question"])
                if variant == "original"
                else str(cached["query_rewritten"])
            )
            started = perf_counter()
            hits = actual_search(
                query,
                ticker=str(question["ticker"]),
                fiscal_year=int(question["fiscal_year"]),
                item=str(question["item"]),
                k=max_k,
            )
            latency_s = perf_counter() - started
            ranking = _ranking(hits, max_k)
            row[variant] = {
                "query": query,
                "ranking": ranking,
                "first_relevant_rank": _first_relevant_rank(ranking, relevant),
                "query_embedding_provenance": actual_provenance(),
                "latency_s": {"total": round(latency_s, 6)},
            }
        per_question.append(row)
        marker = ", ".join(
            f"{variant}={row[variant]['first_relevant_rank']}"
            for variant in variants
        )
        print(f"[{position}/48] {question_id} {marker}", flush=True)

    metrics = {
        variant: evaluate_benchmark_views(_metric_rows(per_question, variant))
        for variant in variants
    }
    payload: dict[str, Any] = {
        "manifest": {
            "runner": "common.scripts.evaluate_query_rewrite_ablation",
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_sha256": FROZEN_SHA256,
            "benchmark_path": str(benchmark_path),
            "n_questions": 48,
            "selection": [row["question_id"] for row in per_question],
            "mode": mode,
            "variants": list(variants),
            "profile": profile.name,
            "embedding_model_id": profile.model_name,
            "embedding_dimension": profile.dimension,
            "max_k": max_k,
            "faiss_index": str(paths.faiss_index),
            "faiss_index_sha256": _artifact_hash(paths.faiss_index),
            "chunks_sha256": _artifact_hash(paths.chunks),
            "chunks_meta_sha256": _artifact_hash(paths.chunks_meta),
            "rewrite_cache_path": str(cache_path),
            "rewrite_cache_sha256": _sha256(cache_path),
            "rewrite_prompt_sha256": PROMPT_SHA256,
            "rewrite_generation": False,
            "retrieval": {
                "type": "dense_faiss_with_metadata_postfilter",
                "policy": "global rank + ticker/fiscal_year/item postfilter",
                "only_variable": "query",
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
    if mode == "ablation":
        payload["paired_analysis"] = paired_analysis(per_question)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Artifact: {output_path.resolve()}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua queries cacheadas sin regenerar rewrites"
    )
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--mode", required=True, choices=("ablation", "rewritten"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    args = parser.parse_args()
    run_evaluation(
        profile_name=args.profile,
        mode=args.mode,
        cache_path=args.cache.resolve(),
        output_path=args.output.resolve(),
        benchmark_path=args.benchmark.resolve(),
    )


if __name__ == "__main__":
    main()
