"""Evalúa ``miax_s1.buscar`` contra el benchmark de retrieval v2 de Dani.

No usa el golden del agente ni llama al LLM. Mide Recall@k / MRR@10 con la
misma política que Dani (full containment del EvidenceSpan; vistas ALL-48,
ITEM-* y NON-7A-36).

Con ``--rerank``: pool denso (Gemini u otro) + ``BAAI/bge-reranker-v2-m3``.
El JSON sale en formato dual ``metrics.dense`` / ``metrics.reranked`` (como
Marco) para que ``retrieval_viewer`` muestre ambas variantes.

Uso (desde la raíz del repo)::

    python -m cristian.experiments.eval_retrieval_benchmark_v2

    python -m cristian.experiments.eval_retrieval_benchmark_v2 --rerank \\
        --copy-to-comparisons

    python -m cristian.experiments.eval_retrieval_benchmark_v2 --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.eval.retrieval_metrics import evaluate_rankings

from cristian.experiments.benchmark import (
    BENCHMARK_NAME,
    BENCHMARK_PATH,
    FROZEN_SHA256,
)
from cristian.experiments.config import (
    SETTINGS,
    embedding_index_path,
    embedding_model_id,
    get_dataset_paths,
)
from cristian.experiments.reranker import DEFAULT_RERANKER_MODEL

EVALUATION_VIEWS = (
    "ALL-48",
    "ITEM-1A-12",
    "ITEM-7-12",
    "ITEM-7A-12",
    "ITEM-8-12",
    "NON-7A-36",
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPARISONS_CRISTIAN = RESULTS_DIR / "comparisons" / "cristian"


class BenchmarkError(RuntimeError):
    """Benchmark inválido o no evaluable con el chunking actual."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_frozen_benchmark(path: Path = BENCHMARK_PATH) -> str:
    digest = _sha256_file(path)
    if digest != FROZEN_SHA256:
        raise BenchmarkError(
            f"SHA-256 de benchmark inválido: {digest}; esperado {FROZEN_SHA256}"
        )
    return digest


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _commit_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_metadata():
    import pandas as pd

    paths = get_dataset_paths()
    meta = pd.read_parquet(paths.chunks_meta)
    required = {
        "chunk_id",
        "ticker",
        "fiscal_year",
        "item",
        "inicio_car",
        "fin_car",
    }
    missing = required - set(meta.columns)
    if missing:
        raise BenchmarkError(
            f"chunks_meta sin columnas requeridas: {sorted(missing)}"
        )
    return meta


def candidate_chunks(question: dict[str, Any], metadata) -> Any:
    return metadata[
        (metadata["ticker"].astype(str) == str(question["ticker"]))
        & (metadata["fiscal_year"].astype(int) == int(question["fiscal_year"]))
        & (metadata["item"].astype(str) == str(question["item"]))
    ]


def relevant_chunk_ids(question: dict[str, Any], metadata) -> tuple[list[str], int]:
    """Chunks del pool ticker/año/item que contienen por completo el span."""
    candidates = candidate_chunks(question, metadata)
    start = int(question["char_start"])
    end = int(question["char_end"])
    relevant = [
        str(row.chunk_id)
        for row in candidates.itertuples(index=False)
        if int(row.inicio_car) <= start and int(row.fin_car) >= end
    ]
    if not relevant:
        raise BenchmarkError(
            f"{question['question_id']}: cero chunks con full containment"
        )
    return relevant, len(candidates)


def evaluate_benchmark_views(
    per_question: list[dict[str, Any]],
    *,
    ranking_key: str = "ranking",
) -> dict[str, dict[str, float | int | None]]:
    selectors = {
        "ALL-48": lambda row: True,
        "ITEM-1A-12": lambda row: row["item"] == "1A",
        "ITEM-7-12": lambda row: row["item"] == "7",
        "ITEM-7A-12": lambda row: row["item"] == "7A",
        "ITEM-8-12": lambda row: row["item"] == "8",
        "NON-7A-36": lambda row: row["item"] in {"1A", "7", "8"},
    }
    rows_for_eval: list[dict[str, Any]] = []
    for row in per_question:
        rows_for_eval.append({
            **row,
            "ranking": row.get(ranking_key) or row.get("ranking") or [],
        })
    return {
        name: evaluate_rankings([row for row in rows_for_eval if selector(row)])
        for name, selector in selectors.items()
    }


def _ranking_from_hits(hits: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": str(row["chunk_id"]),
            "score": float(row["puntuacion"]),
            "ticker": str(row["ticker"]),
            "fiscal_year": int(row["fiscal_year"]),
            "item": str(row["item"]),
        }
        for rank, row in enumerate(hits[:max_k], 1)
    ]


def _compact_top(
    hits: list[dict[str, Any]],
    *,
    score_key: str,
) -> list[dict[str, Any]]:
    out = []
    for row in hits:
        score = row.get(score_key, row.get("puntuacion"))
        out.append({
            "chunk_id": str(row["chunk_id"]),
            "score": float(score) if score is not None else None,
        })
    return out


def _first_relevant_rank(
    ranking: list[dict[str, Any]],
    relevant: list[str],
) -> int | None:
    relevant_set = set(relevant)
    return next(
        (row["rank"] for row in ranking if row["chunk_id"] in relevant_set),
        None,
    )


def _candidate_recall_at_pool(
    per_question: list[dict[str, Any]],
    pool_key: str,
) -> dict[str, float]:
    """Fracción de preguntas cuyo relevant está en el pool (techo del rerank)."""
    selectors = {
        "ALL-48": lambda row: True,
        "NON-7A-36": lambda row: row["item"] in {"1A", "7", "8"},
        "ITEM-1A-12": lambda row: row["item"] == "1A",
        "ITEM-7-12": lambda row: row["item"] == "7",
        "ITEM-7A-12": lambda row: row["item"] == "7A",
        "ITEM-8-12": lambda row: row["item"] == "8",
    }
    out: dict[str, float] = {}
    for name, selector in selectors.items():
        subset = [row for row in per_question if selector(row)]
        if not subset:
            continue
        hits = 0
        for row in subset:
            pool_ids = {c["chunk_id"] for c in row.get(pool_key) or []}
            if pool_ids & set(row["relevant_chunk_ids"]):
                hits += 1
        out[name] = hits / len(subset)
    return out


def _default_output_path(*, rerank: bool, reranker_model: str) -> Path:
    slug = embedding_model_id().replace("/", "_").replace(":", "_")
    if not rerank:
        return RESULTS_DIR / f"cristian_{slug}_benchmark_v2.json"
    rslug = reranker_model.replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"cristian_{slug}_{rslug}_benchmark_v2.json"


def _print_metrics(
    metrics: dict[str, dict[str, float | int | None]],
    *,
    title: str | None = None,
) -> None:
    if title:
        print(f"\n=== {title} ===")
    else:
        print("\n=== Métricas ===")
    for view in EVALUATION_VIEWS:
        block = metrics.get(view) or {}
        if not block.get("n_questions"):
            print(f"{view:12} (sin preguntas en este subset)")
            continue
        print(
            f"{view:12} R@1={block['recall@1']:.4f} "
            f"R@3={block['recall@3']:.4f} R@5={block['recall@5']:.4f} "
            f"R@10={block['recall@10']:.4f} MRR@10={block['mrr@10']:.4f} "
            f"n={block['n_questions']}"
        )


def run_benchmark(
    *,
    benchmark_path: Path = BENCHMARK_PATH,
    output_path: Path | None = None,
    max_k: int = 10,
    limit: int | None = None,
    embedding_model: str | None = None,
    rerank: bool = False,
    pool_k: int | None = None,
    reranker_model: str | None = None,
    copy_to_comparisons: bool = False,
) -> dict[str, Any]:
    """Ejecuta el benchmark v2 con ``cristian.experiments.miax_s1.buscar``."""
    import dataclasses

    import cristian.experiments.config as config_mod
    from cristian.experiments.miax_s1 import (
        buscar,
        clear_retrieval_cache,
        precargar_retrieval,
        precargar_reranker,
    )
    from cristian.experiments.reranker import rerank_hits

    if embedding_model is not None:
        config_mod.SETTINGS = dataclasses.replace(
            config_mod.SETTINGS, embedding_model=embedding_model
        )
        clear_retrieval_cache()

    reranker = reranker_model or config_mod.SETTINGS.reranker_model
    pool_size = pool_k if pool_k is not None else config_mod.SETTINGS.rerank_pool_k
    if rerank and pool_size < max_k:
        raise BenchmarkError(
            f"pool_k ({pool_size}) debe ser >= max_k ({max_k})"
        )

    digest = verify_frozen_benchmark(benchmark_path)
    questions = load_benchmark(benchmark_path)
    if limit is not None:
        questions = questions[: max(0, int(limit))]

    metadata = _load_metadata()
    for question in questions:
        relevant_chunk_ids(question, metadata)

    mode = (
        f"dense={config_mod.SETTINGS.embedding_model} + rerank={reranker} "
        f"pool_k={pool_size}"
        if rerank
        else f"embedding={config_mod.SETTINGS.embedding_model}"
    )
    print(f"Benchmark v2 · {len(questions)} preguntas · {mode} · max_k={max_k}")
    print("Precargando índice / encoder…")
    precargar_retrieval()
    if rerank:
        print(f"Precargando reranker {reranker}…")
        precargar_reranker(reranker)

    per_question: list[dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        relevant, candidate_pool_size = relevant_chunk_ids(question, metadata)
        query = question["question"]
        filters = {
            "ticker": str(question["ticker"]),
            "fiscal_year": int(question["fiscal_year"]),
            "item": str(question["item"]),
        }

        if rerank:
            t0 = perf_counter()
            pool = buscar(query, **filters, k=pool_size)
            dense_ms = (perf_counter() - t0) * 1000.0
            dense_ranking = _ranking_from_hits(pool, max_k)

            t1 = perf_counter()
            reranked_hits = rerank_hits(
                query,
                pool,
                top_k=pool_size,
                model_name=reranker,
                max_length=config_mod.SETTINGS.rerank_max_length,
            )
            rerank_ms = (perf_counter() - t1) * 1000.0
            reranked_ranking = _ranking_from_hits(reranked_hits, max_k)

            dense_first = _first_relevant_rank(dense_ranking, relevant)
            rerank_first = _first_relevant_rank(reranked_ranking, relevant)
            per_question.append({
                "question_id": question["question_id"],
                "query": query,
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
                "candidate_pool_requested": pool_size,
                "relevant_chunk_ids": relevant,
                "n_relevant_chunks": len(relevant),
                "dense_top20": _compact_top(pool, score_key="puntuacion"),
                "reranked_top20": _compact_top(
                    reranked_hits, score_key="puntuacion"
                ),
                "ranking": reranked_ranking,
                "dense_ranking": dense_ranking,
                "reranked_ranking": reranked_ranking,
                "dense_first_relevant_rank": dense_first,
                "reranked_first_relevant_rank": rerank_first,
                "first_relevant_rank": rerank_first,
                "dense_latency_ms": round(dense_ms, 3),
                "rerank_latency_ms": round(rerank_ms, 3),
                "latency_s": {
                    "dense": round(dense_ms / 1000.0, 6),
                    "rerank": round(rerank_ms / 1000.0, 6),
                    "total": round((dense_ms + rerank_ms) / 1000.0, 6),
                },
            })
            mark = "✓" if rerank_first is not None and rerank_first <= 5 else "·"
            print(
                f"[{index}/{len(questions)}] {question['question_id']} "
                f"item={question['item']} dense={dense_first} "
                f"rerank={rerank_first} {mark}"
            )
        else:
            started = perf_counter()
            hits = buscar(query, **filters, k=max_k)
            latency_s = perf_counter() - started
            ranking = _ranking_from_hits(hits, max_k)
            first_rank = _first_relevant_rank(ranking, relevant)
            per_question.append({
                "question_id": question["question_id"],
                "query": query,
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
                "latency_s": {"total": round(latency_s, 6)},
            })
            mark = "✓" if first_rank is not None and first_rank <= 5 else "·"
            print(
                f"[{index}/{len(questions)}] {question['question_id']} "
                f"item={question['item']} first_rel={first_rank} {mark}"
            )

    index_path = embedding_index_path()
    destination = output_path or _default_output_path(
        rerank=rerank, reranker_model=reranker
    )

    if rerank:
        dense_metrics = evaluate_benchmark_views(
            per_question, ranking_key="dense_ranking"
        )
        reranked_metrics = evaluate_benchmark_views(
            per_question, ranking_key="reranked_ranking"
        )
        # Emparejamiento simple dense vs rerank (como Marco).
        improvements = worsenings = equals = 0
        top5_in = top5_out = 0
        for row in per_question:
            d = row["dense_first_relevant_rank"]
            r = row["reranked_first_relevant_rank"]
            if d is None and r is None:
                equals += 1
            elif d is None and r is not None:
                improvements += 1
                if r <= 5:
                    top5_in += 1
            elif d is not None and r is None:
                worsenings += 1
                if d <= 5:
                    top5_out += 1
            elif r < d:
                improvements += 1
                if d > 5 >= r:
                    top5_in += 1
            elif r > d:
                worsenings += 1
                if r > 5 >= d:
                    top5_out += 1
            else:
                equals += 1

        experiment_id = (
            f"cristian_{embedding_model_id().replace('/', '_')}"
            f"_{reranker.replace('/', '_')}"
        )
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "dense_model": config_mod.SETTINGS.embedding_model,
            "dense_model_id": embedding_model_id(),
            "reranker_model": reranker,
            "candidate_pool_requested": pool_size,
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_sha256": digest,
            "benchmark_path": str(benchmark_path.relative_to(_REPO_ROOT))
            if benchmark_path.is_relative_to(_REPO_ROOT)
            else str(benchmark_path),
            "n_questions": len(questions),
            "max_k": max_k,
            "faiss_index": str(index_path),
            "faiss_index_sha256": _sha256_file(index_path)
            if index_path.is_file()
            else None,
            "config": {
                "reranking": True,
                "reranker_model": reranker,
                "n_candidatos": pool_size,
                "embedding_model": config_mod.SETTINGS.embedding_model,
                "rerank_max_length": config_mod.SETTINGS.rerank_max_length,
            },
            "manifest": {
                "experiment": "cristian",
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_sha256": digest,
                "n_questions": len(questions),
                "max_k": max_k,
                "embedding_model": config_mod.SETTINGS.embedding_model,
                "embedding_model_id": embedding_model_id(),
                "reranker_model": reranker,
                "retrieval": {
                    "tipo": "dense_faiss_con_postfiltrado_metadata + cross_encoder",
                    "policy": "ranking global + filtro + rerank pool",
                    "query_rewriting": False,
                    "bm25": False,
                    "reranking": True,
                    "pool_k": pool_size,
                },
                "evaluation_views": list(EVALUATION_VIEWS),
                "primary_metric": "NON-7A-36 Recall@5",
                "secondary_metric": "NON-7A-36 MRR@10",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "commit_sha": _commit_sha(),
                "comparable_to": (
                    "marco_reranking_v2/*bge_reranker_v2_m3.json "
                    "(mismo reranker; dense distinto)"
                ),
            },
            "metrics": {
                "dense": dense_metrics,
                "reranked": reranked_metrics,
            },
            "candidate_recall@20": _candidate_recall_at_pool(
                per_question, "dense_top20"
            ),
            "paired_comparison": {
                "improvements": improvements,
                "worsenings": worsenings,
                "equals": equals,
                "top5_in": top5_in,
                "top5_out": top5_out,
            },
            "per_question": per_question,
        }
        _print_metrics(dense_metrics, title="Dense (Gemini pool)")
        _print_metrics(reranked_metrics, title="Reranked (BGE cross-encoder)")
        print(
            "\nPaired: "
            f"+{improvements} / -{worsenings} / ={equals} · "
            f"top5_in={top5_in} top5_out={top5_out}"
        )
    else:
        metrics = evaluate_benchmark_views(per_question)
        payload = {
            "manifest": {
                "experiment": "cristian",
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_sha256": digest,
                "benchmark_path": str(benchmark_path.relative_to(_REPO_ROOT))
                if benchmark_path.is_relative_to(_REPO_ROOT)
                else str(benchmark_path),
                "n_questions": len(questions),
                "max_k": max_k,
                "embedding_model": config_mod.SETTINGS.embedding_model,
                "embedding_model_id": embedding_model_id(),
                "faiss_index": str(index_path),
                "faiss_index_sha256": _sha256_file(index_path)
                if index_path.is_file()
                else None,
                "retrieval": {
                    "tipo": "dense_faiss_con_postfiltrado_metadata",
                    "policy": "ranking global + filtro ticker/año/item",
                    "query_rewriting": False,
                    "bm25": False,
                    "reranking": False,
                },
                "evaluation_views": list(EVALUATION_VIEWS),
                "primary_metric": "NON-7A-36 Recall@5",
                "secondary_metric": "NON-7A-36 MRR@10",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "commit_sha": _commit_sha(),
                "comparable_to": (
                    "dani/experiments/results/*_benchmark_v2*.json "
                    "(misma rúbrica de retrieval)"
                ),
            },
            "metrics": metrics,
            "per_question": per_question,
        }
        _print_metrics(metrics)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nGuardado: {destination}")

    if copy_to_comparisons:
        COMPARISONS_CRISTIAN.mkdir(parents=True, exist_ok=True)
        target = COMPARISONS_CRISTIAN / destination.name
        shutil.copy2(destination, target)
        print(f"Copia para viewer: {target}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa el retrieval de Cristian sobre el benchmark v2 de Dani"
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_PATH,
        help="Ruta al JSONL congelado",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON de salida (default según embedding / rerank)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Profundidad del ranking evaluado (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Solo las primeras N preguntas (smoke test)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help=(
            "Override de SETTINGS.embedding_model "
            "(p. ej. BAAI/bge-small-en-v1.5 para comparar con E0 de Dani)"
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Pool denso + cross-encoder "
            f"(default {DEFAULT_RERANKER_MODEL})"
        ),
    )
    parser.add_argument(
        "--pool-k",
        type=int,
        default=None,
        help=f"Tamaño del pool dense antes del rerank (default: {SETTINGS.rerank_pool_k})",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        help=f"Modelo cross-encoder (default: {SETTINGS.reranker_model})",
    )
    parser.add_argument(
        "--copy-to-comparisons",
        action="store_true",
        help="Copia el JSON a results/comparisons/cristian/ para el viewer",
    )
    args = parser.parse_args()
    run_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        max_k=args.max_k,
        limit=args.limit,
        embedding_model=args.embedding,
        rerank=args.rerank,
        pool_k=args.pool_k,
        reranker_model=args.reranker,
        copy_to_comparisons=args.copy_to_comparisons,
    )


if __name__ == "__main__":
    main()
