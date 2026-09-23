"""Runner canónico de reranking: R0 (BGE-small + BGE-reranker) y
R1 (Qwen3 + BGE-reranker).

Retrieval puro. Sin LLM. Fail-closed.
Uso:
    python marco/experiments/reranking/runner_reranking.py R0
    python marco/experiments/reranking/runner_reranking.py R1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from common.config import get_dataset_paths

from marco.experiments.benchmark.evaluator_span import (
    BenchmarkValidationError,
    build_views,
    first_relevant_rank,
    summarize_rankings,
    validate_benchmark,
)
from marco.experiments.reranking.dense_backends import (
    backend_metadata,
    dense_search,
)


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

BENCHMARK_PATH = (
    RAIZ / "dani" / "experiments" / "benchmark"
    / "retrieval_benchmark_v2.jsonl"
)
BENCHMARK_SHA = "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C"

DEFAULT_K_DENSE = 20
DEFAULT_K_SAVED = 20

CONFIGS = {
    "R0": {
        "experiment_id": "R0_bge_small_bge_reranker_v2_m3",
        "dense_backend": "bge-small",
        "output": RAIZ / "marco" / "results" / "reranking"
        / "bge_small_bge_reranker_v2_m3.json",
    },
    "R1": {
        "experiment_id": "R1_qwen3_bge_reranker_v2_m3",
        "dense_backend": "qwen3",
        "output": RAIZ / "marco" / "results" / "reranking"
        / "qwen3_bge_reranker_v2_m3.json",
    },
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args, text=True, cwd=RAIZ
        ).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION,
                        trust_remote_code=True)


def rerank(query: str, candidates: list[dict], k: int = DEFAULT_K_SAVED
           ) -> list[dict]:
    """Rerankea candidatos con BGE-reranker-v2-m3.

    Devuelve top-k con score del reranker y rank original del dense.
    """
    if not candidates:
        return []
    model = _reranker()
    pares = [(query, c["texto"]) for c in candidates]
    scores = model.predict(pares)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: (-float(x[1]), x[0]["chunk_id"]),
    )
    return [
        {
            "chunk_id": c["chunk_id"],
            "reranker_score": round(float(s), 6),
            "original_dense_rank": i + 1,
        }
        for i, (c, s) in enumerate(ranked[:k])
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(config_name: str) -> dict:
    cfg = CONFIGS[config_name]
    backend = cfg["dense_backend"]
    out_path = cfg["output"]

    paths = get_dataset_paths()
    meta = pd.read_parquet(paths.chunks_meta)

    try:
        ground_truth = validate_benchmark(BENCHMARK_PATH, meta)
    except BenchmarkValidationError as e:
        print(f"FAIL-CLOSED: {e}")
        raise

    print(f"[{config_name}] Benchmark validado: {len(ground_truth)}")
    questions = {gt.question_id: gt for gt in ground_truth}

    per_question = []
    lat_dense_all, lat_rerank_all = [], []

    for i, gt in enumerate(ground_truth, 1):
        if i % 5 == 0 or i == len(ground_truth):
            print(f"  [{i}/{len(ground_truth)}] {gt.question_id}")

        relevant = set(gt.relevant_chunk_ids)

        # Dense
        t0 = time.perf_counter()
        dense = dense_search(
            backend, gt.question, ticker=gt.ticker,
            fiscal_year=gt.fiscal_year, item=gt.item, k=DEFAULT_K_DENSE,
        )
        lat_d = (time.perf_counter() - t0) * 1000.0
        lat_dense_all.append(lat_d)
        dense_ids = [c["chunk_id"] for c in dense]
        dense_rank = first_relevant_rank(dense_ids, relevant)

        # Reranker
        t0 = time.perf_counter()
        reranked = rerank(gt.question, dense, k=DEFAULT_K_SAVED)
        lat_r = (time.perf_counter() - t0) * 1000.0
        lat_rerank_all.append(lat_r)
        reranked_ids = [c["chunk_id"] for c in reranked]
        reranked_rank = first_relevant_rank(reranked_ids, relevant)

        per_question.append({
            "question_id": gt.question_id,
            "ticker": gt.ticker,
            "fiscal_year": gt.fiscal_year,
            "item": gt.item,
            "relevant_chunk_ids": list(gt.relevant_chunk_ids),
            "candidate_pool_size": len(dense),
            "dense_top20": [
                {"chunk_id": c["chunk_id"],
                 "dense_score": float(c["puntuacion"])}
                for c in dense
            ],
            "reranked_top20": reranked,
            "dense_first_relevant_rank": dense_rank,
            "reranked_first_relevant_rank": reranked_rank,
            "dense_latency_ms": round(lat_d, 3),
            "rerank_latency_ms": round(lat_r, 3),
        })

    # Métricas
    dense_rankings = [
        {"question_id": r["question_id"],
         "relevant_chunk_ids": r["relevant_chunk_ids"],
         "ranking": [c["chunk_id"] for c in r["dense_top20"]]}
        for r in per_question
    ]
    reranked_rankings = [
        {"question_id": r["question_id"],
         "relevant_chunk_ids": r["relevant_chunk_ids"],
         "ranking": [c["chunk_id"] for c in r["reranked_top20"]]}
        for r in per_question
    ]

    views_dense = build_views(dense_rankings, questions)
    views_reranked = build_views(reranked_rankings, questions)

    metrics = {"dense": {}, "reranked": {}}
    for v in views_dense:
        metrics["dense"][v] = summarize_rankings(views_dense[v])
        metrics["reranked"][v] = summarize_rankings(views_reranked[v])

    # Candidate Recall@20
    candidate_recall_20 = {}
    for v in views_dense:
        rows = views_dense[v]
        if not rows:
            candidate_recall_20[v] = None
            continue
        hits = sum(
            1 for r in rows
            if set(r["relevant_chunk_ids"]) & set(r["ranking"][:20])
        )
        candidate_recall_20[v] = hits / len(rows)

    # Paired
    improvements = sum(
        1 for r in per_question
        if r["reranked_first_relevant_rank"] is not None
        and (r["dense_first_relevant_rank"] is None
             or r["reranked_first_relevant_rank"]
             < r["dense_first_relevant_rank"])
    )
    worsenings = sum(
        1 for r in per_question
        if r["reranked_first_relevant_rank"] is None
        or (r["dense_first_relevant_rank"] is not None
            and r["reranked_first_relevant_rank"]
            > r["dense_first_relevant_rank"])
    )
    equals = len(per_question) - improvements - worsenings
    top5_in = sum(
        1 for r in per_question
        if (r["reranked_first_relevant_rank"] is not None
            and r["reranked_first_relevant_rank"] <= 5
            and (r["dense_first_relevant_rank"] is None
                 or r["dense_first_relevant_rank"] > 5))
    )
    top5_out = sum(
        1 for r in per_question
        if (r["dense_first_relevant_rank"] is not None
            and r["dense_first_relevant_rank"] <= 5
            and (r["reranked_first_relevant_rank"] is None
                 or r["reranked_first_relevant_rank"] > 5))
    )

    # Provenance
    bm = backend_metadata(backend)
    result = {
        "experiment_id": cfg["experiment_id"],
        "dense_model": bm["dense_model"],
        "dense_model_revision": bm["dense_model_revision"],
        "dense_query_prefix": bm["dense_query_prefix"],
        "dense_index_dir": bm["dense_index_dir"],
        "dense_dimension": bm["dense_dimension"],
        "reranker_model": RERANKER_MODEL,
        "reranker_revision": RERANKER_REVISION,
        "benchmark_path": str(BENCHMARK_PATH.relative_to(RAIZ)),
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "chunks_sha256": sha256_file(
            paths.corpus_dir / "chunks.jsonl"
        ),
        "chunks_meta_sha256": sha256_file(paths.chunks_meta),
        "faiss_sha256": sha256_file(paths.faiss_index),
        "index_type": "IndexFlatIP",
        "candidate_pool_requested": DEFAULT_K_DENSE,
        "candidate_pool_saved": DEFAULT_K_SAVED,
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_worktree": bool(_git(["status", "--porcelain"])),
        "python_version": platform.python_version(),
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "sentence_transformers_version": _pkg_version(
            "sentence-transformers"
        ),
        "faiss_version": _pkg_version("faiss-cpu"),
        "metrics": metrics,
        "candidate_recall@20": candidate_recall_20,
        "paired_comparison": {
            "improvements": improvements,
            "worsenings": worsenings,
            "equals": equals,
            "top5_in": top5_in,
            "top5_out": top5_out,
        },
        "latency_ms": {
            "dense_mean": sum(lat_dense_all) / len(lat_dense_all),
            "reranker_mean": sum(lat_rerank_all) / len(lat_rerank_all),
        },
        "per_question": per_question,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[{config_name}] Guardado en {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=["R0", "R1"])
    args = parser.parse_args()
    run(args.config)