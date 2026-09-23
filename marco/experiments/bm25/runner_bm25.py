"""Runner canónico B0 (dense) vs B1 (dense + BM25 + RRF).

Retrieval puro. Sin LLM. Fail-closed.
"""
from __future__ import annotations

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
from common.eval.retrieval_metrics import evaluate_rankings
from common.retrieval.dense_baseline import buscar as dense_search

from marco.experiments.benchmark.evaluator_span import (
    BenchmarkValidationError,
    build_views,
    first_relevant_rank,
    summarize_rankings,
    validate_benchmark,
)
from marco.experiments.bm25.bm25_hybrid import (
    BM25_B,
    BM25_K1,
    DEFAULT_K_CANDIDATES,
    RRF_K,
    TOKEN_REGEX,
    MIN_TOKEN_LEN,
    bm25_search,
    rrf_fuse,
)


BENCHMARK_PATH = (
    RAIZ / "dani" / "experiments" / "benchmark"
    / "retrieval_benchmark_v2.jsonl"
)
OUTPUT_PATH = RAIZ / "marco" / "results" / "bm25" / "bm25_dense_ablation_v1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=RAIZ
        ).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, cwd=RAIZ,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=RAIZ,
        )
        return bool(out.strip())
    except Exception:
        return True


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def run() -> dict:
    paths = get_dataset_paths()
    meta = pd.read_parquet(paths.chunks_meta)

    # 1. Validar benchmark fail-closed
    try:
        ground_truth = validate_benchmark(BENCHMARK_PATH, meta)
    except BenchmarkValidationError as e:
        print(f"FAIL-CLOSED: {e}")
        raise
    print(f"Benchmark validado: {len(ground_truth)} preguntas")

    questions = {gt.question_id: gt for gt in ground_truth}

    per_question = []
    lat_dense_all, lat_bm25_all, lat_fuse_all = [], [], []

    for i, gt in enumerate(ground_truth, 1):
        if i % 5 == 0 or i == len(ground_truth):
            print(f"  [{i}/{len(ground_truth)}] {gt.question_id}")

        relevant = set(gt.relevant_chunk_ids)

        # B0: dense
        t0 = time.perf_counter()
        dense = dense_search(
            gt.question, ticker=gt.ticker,
            fiscal_year=gt.fiscal_year, item=gt.item,
            k=DEFAULT_K_CANDIDATES,
        )
        lat_dense = (time.perf_counter() - t0) * 1000.0
        lat_dense_all.append(lat_dense)
        dense_ids = [c["chunk_id"] for c in dense]
        dense_rank = first_relevant_rank(dense_ids, relevant)

        # BM25
        t0 = time.perf_counter()
        bm25 = bm25_search(
            gt.question, ticker=gt.ticker,
            fiscal_year=gt.fiscal_year, item=gt.item,
            k=DEFAULT_K_CANDIDATES,
        )
        lat_bm25 = (time.perf_counter() - t0) * 1000.0
        lat_bm25_all.append(lat_bm25)

        # B1: RRF fusion
        t0 = time.perf_counter()
        hybrid = rrf_fuse(dense, bm25, k_rrf=RRF_K, k=DEFAULT_K_CANDIDATES)
        lat_fuse = (time.perf_counter() - t0) * 1000.0
        lat_fuse_all.append(lat_fuse)
        hybrid_ids = [c["chunk_id"] for c in hybrid]
        hybrid_rank = first_relevant_rank(hybrid_ids, relevant)

        per_question.append({
            "question_id": gt.question_id,
            "ticker": gt.ticker,
            "fiscal_year": gt.fiscal_year,
            "item": gt.item,
            "char_start": gt.char_start,
            "char_end": gt.char_end,
            "relevant_chunk_ids": list(gt.relevant_chunk_ids),
            "dense_top20": [
                {"chunk_id": c["chunk_id"],
                 "score": float(c.get("puntuacion", 0.0))}
                for c in dense
            ],
            "bm25_top20": [
                {"chunk_id": c["chunk_id"], "score": c["bm25_score"]}
                for c in bm25
            ],
            "hybrid_top20": [
                {"chunk_id": c["chunk_id"],
                 "rrf_score": c["rrf_score"],
                 "dense_rank": c["dense_rank"],
                 "bm25_rank": c["bm25_rank"]}
                for c in hybrid
            ],
            "dense_first_relevant_rank": dense_rank,
            "hybrid_first_relevant_rank": hybrid_rank,
            "dense_latency_ms": round(lat_dense, 3),
            "bm25_latency_ms": round(lat_bm25, 3),
            "fusion_latency_ms": round(lat_fuse, 4),
        })

    # 2. Métricas por vista
    dense_rankings = [
        {"question_id": r["question_id"],
         "relevant_chunk_ids": r["relevant_chunk_ids"],
         "ranking": [c["chunk_id"] for c in r["dense_top20"]]}
        for r in per_question
    ]
    hybrid_rankings = [
        {"question_id": r["question_id"],
         "relevant_chunk_ids": r["relevant_chunk_ids"],
         "ranking": [c["chunk_id"] for c in r["hybrid_top20"]]}
        for r in per_question
    ]

    views_dense = build_views(dense_rankings, questions)
    views_hybrid = build_views(hybrid_rankings, questions)

    metrics = {"B0_dense": {}, "B1_hybrid": {}}
    for v in views_dense:
        metrics["B0_dense"][v] = summarize_rankings(views_dense[v])
        metrics["B1_hybrid"][v] = summarize_rankings(views_hybrid[v])

    # 3. Comparación pareada
    improvements = sum(
        1 for r in per_question
        if r["hybrid_first_relevant_rank"] is not None
        and (r["dense_first_relevant_rank"] is None
             or r["hybrid_first_relevant_rank"] < r["dense_first_relevant_rank"])
    )
    worsenings = sum(
        1 for r in per_question
        if r["hybrid_first_relevant_rank"] is None
        or (r["dense_first_relevant_rank"] is not None
            and r["hybrid_first_relevant_rank"] > r["dense_first_relevant_rank"])
    )
    equals = len(per_question) - improvements - worsenings

    top5_in = sum(
        1 for r in per_question
        if (r["hybrid_first_relevant_rank"] is not None
            and r["hybrid_first_relevant_rank"] <= 5
            and (r["dense_first_relevant_rank"] is None
                 or r["dense_first_relevant_rank"] > 5))
    )
    top5_out = sum(
        1 for r in per_question
        if (r["dense_first_relevant_rank"] is not None
            and r["dense_first_relevant_rank"] <= 5
            and (r["hybrid_first_relevant_rank"] is None
                 or r["hybrid_first_relevant_rank"] > 5))
    )

    item8_diff = []
    for r in per_question:
        if r["item"] == "8":
            item8_diff.append({
                "question_id": r["question_id"],
                "dense_rank": r["dense_first_relevant_rank"],
                "hybrid_rank": r["hybrid_first_relevant_rank"],
            })

    # 4. Provenance
    result = {
        "config": {
            "benchmark_path": str(BENCHMARK_PATH.relative_to(RAIZ)),
            "benchmark_sha256": sha256_file(BENCHMARK_PATH),
            "chunks_sha256": sha256_file(
                paths.corpus_dir / "chunks.jsonl"
            ),
            "chunks_meta_sha256": sha256_file(paths.chunks_meta),
            "faiss_sha256": sha256_file(paths.faiss_index),
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_query_prefix": (
                "Represent this sentence for searching relevant passages: "
            ),
            "faiss_index_type": "IndexFlatIP",
            "metadata_policy": "global_ranking_then_postfilter",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "bm25_tokenization_regex": TOKEN_REGEX.pattern,
            "bm25_tokenization_min_len": MIN_TOKEN_LEN,
            "bm25_tokenization_note": (
                "lowercase + unicode word regex + min len 2 + no stopwords"
            ),
            "rrf_k": RRF_K,
            "rrf_rank_base": 1,
            "candidate_pool_per_retriever": DEFAULT_K_CANDIDATES,
            "commit": _git_commit(),
            "branch": _git_branch(),
            "dirty_worktree": _git_dirty(),
            "python_version": platform.python_version(),
            "faiss_version": _pkg_version("faiss-cpu"),
            "rank_bm25_version": _pkg_version("rank-bm25"),
            "sentence_transformers_version": _pkg_version(
                "sentence-transformers"
            ),
        },
        "metrics": metrics,
        "paired_comparison": {
            "improvements": improvements,
            "worsenings": worsenings,
            "equals": equals,
            "top5_in": top5_in,
            "top5_out": top5_out,
            "item8_detail": item8_diff,
        },
        "latency_ms": {
            "dense_mean": sum(lat_dense_all) / len(lat_dense_all),
            "bm25_mean": sum(lat_bm25_all) / len(lat_bm25_all),
            "fusion_mean": sum(lat_fuse_all) / len(lat_fuse_all),
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