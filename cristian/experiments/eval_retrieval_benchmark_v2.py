"""Evalúa ``miax_s1.buscar`` contra el benchmark de retrieval v2 de Dani.

No usa el golden del agente ni llama al LLM. Mide Recall@k / MRR@10 con la
misma política que Dani (full containment del EvidenceSpan; vistas ALL-48,
ITEM-* y NON-7A-36).

Uso (desde la raíz del repo)::

    python -m cristian.experiments.eval_retrieval_benchmark_v2

    python -m cristian.experiments.eval_retrieval_benchmark_v2 \\
        --output cristian/experiments/results/cristian_retrieval_benchmark_v2.json

    python -m cristian.experiments.eval_retrieval_benchmark_v2 --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

EVALUATION_VIEWS = (
    "ALL-48",
    "ITEM-1A-12",
    "ITEM-7-12",
    "ITEM-7A-12",
    "ITEM-8-12",
    "NON-7A-36",
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


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


def _default_output_path() -> Path:
    slug = embedding_model_id().replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"cristian_{slug}_benchmark_v2.json"


def run_benchmark(
    *,
    benchmark_path: Path = BENCHMARK_PATH,
    output_path: Path | None = None,
    max_k: int = 10,
    limit: int | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el benchmark v2 con ``cristian.experiments.miax_s1.buscar``."""
    import dataclasses

    import cristian.experiments.config as config_mod
    from cristian.experiments.miax_s1 import (
        buscar,
        clear_retrieval_cache,
        precargar_retrieval,
    )

    if embedding_model is not None:
        config_mod.SETTINGS = dataclasses.replace(
            config_mod.SETTINGS, embedding_model=embedding_model
        )
        clear_retrieval_cache()

    digest = verify_frozen_benchmark(benchmark_path)
    questions = load_benchmark(benchmark_path)
    if limit is not None:
        questions = questions[: max(0, int(limit))]

    metadata = _load_metadata()
    # Valida containment antes de gastar embeddings.
    for question in questions:
        relevant_chunk_ids(question, metadata)

    print(
        f"Benchmark v2 · {len(questions)} preguntas · "
        f"embedding={config_mod.SETTINGS.embedding_model} · max_k={max_k}"
    )
    print("Precargando índice / encoder…")
    precargar_retrieval()

    per_question: list[dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        relevant, candidate_pool_size = relevant_chunk_ids(question, metadata)
        started = perf_counter()
        hits = buscar(
            question["question"],
            ticker=str(question["ticker"]),
            fiscal_year=int(question["fiscal_year"]),
            item=str(question["item"]),
            k=max_k,
        )
        latency_s = perf_counter() - started
        ranking = [
            {
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "score": float(row["puntuacion"]),
                "ticker": str(row["ticker"]),
                "fiscal_year": int(row["fiscal_year"]),
                "item": str(row["item"]),
            }
            for rank, row in enumerate(hits, 1)
        ]
        relevant_set = set(relevant)
        first_rank = next(
            (row["rank"] for row in ranking if row["chunk_id"] in relevant_set),
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
            "latency_s": {"total": round(latency_s, 6)},
        })
        mark = "✓" if first_rank is not None and first_rank <= 5 else "·"
        print(
            f"[{index}/{len(questions)}] {question['question_id']} "
            f"item={question['item']} first_rel={first_rank} {mark}"
        )

    metrics = evaluate_benchmark_views(per_question)
    index_path = embedding_index_path()
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

    destination = output_path or _default_output_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== Métricas ===")
    for view in EVALUATION_VIEWS:
        block = metrics[view]
        if not block.get("n_questions"):
            print(f"{view:12} (sin preguntas en este subset)")
            continue
        print(
            f"{view:12} R@1={block['recall@1']:.4f} "
            f"R@3={block['recall@3']:.4f} R@5={block['recall@5']:.4f} "
            f"R@10={block['recall@10']:.4f} MRR@10={block['mrr@10']:.4f} "
            f"n={block['n_questions']}"
        )
    print(f"\nGuardado: {destination}")
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
        help="JSON de salida (default: results/cristian_<modelo>_benchmark_v2.json)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Profundidad del ranking (default: 10)",
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
    args = parser.parse_args()
    run_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        max_k=args.max_k,
        limit=args.limit,
        embedding_model=args.embedding,
    )


if __name__ == "__main__":
    main()
