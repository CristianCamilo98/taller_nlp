"""Tests mínimos del experimento BM25 + RRF."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RAIZ))

import pytest

from marco.experiments.bm25.bm25_hybrid import (
    RRF_K,
    bm25_search,
    rrf_fuse,
    tokenize,
)
from marco.experiments.benchmark.evaluator_span import (
    validate_benchmark,
)

BENCHMARK = RAIZ / "dani" / "experiments" / "benchmark" / "retrieval_benchmark_v2.jsonl"
BENCHMARK_SHA = "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C"


def _load_meta():
    import pandas as pd
    from common.config import get_dataset_paths
    return pd.read_parquet(get_dataset_paths().chunks_meta)


def test_benchmark_sha_correct():
    import hashlib
    h = hashlib.sha256()
    with open(BENCHMARK, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest().upper() == BENCHMARK_SHA


def test_48_questions_validate():
    meta = _load_meta()
    gt = validate_benchmark(BENCHMARK, meta)
    assert len(gt) == 48


def test_no_duplicate_question_ids():
    meta = _load_meta()
    gt = validate_benchmark(BENCHMARK, meta)
    ids = [g.question_id for g in gt]
    assert len(ids) == len(set(ids))


def test_evidence_span_offsets_positive():
    meta = _load_meta()
    gt = validate_benchmark(BENCHMARK, meta)
    for g in gt:
        assert g.char_start >= 0
        assert g.char_end > g.char_start


def test_tokenize_deterministic():
    t1 = tokenize("Revenue grew 15% in FY2025.")
    t2 = tokenize("Revenue grew 15% in FY2025.")
    assert t1 == t2
    assert "revenue" in t1
    assert "15" in t1
    assert "fy2025" in t1


def test_tokenize_min_len():
    tokens = tokenize("a an I of to revenue")
    assert "a" not in tokens
    assert "i" not in tokens  # len 1 filtered
    assert "revenue" in tokens


def test_bm25_search_deterministic():
    r1 = bm25_search("revenue Apple", ticker="AAPL",
                     fiscal_year=2024, item="1A", k=5)
    r2 = bm25_search("revenue Apple", ticker="AAPL",
                     fiscal_year=2024, item="1A", k=5)
    assert [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]


def test_rrf_deterministic():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "d"}, {"chunk_id": "a"}]
    r1 = rrf_fuse(dense, bm25)
    r2 = rrf_fuse(dense, bm25)
    assert [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]


def test_rrf_ranks_1based():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    fused = rrf_fuse(dense, bm25)
    for c in fused:
        if c["dense_rank"] is not None:
            assert c["dense_rank"] >= 1
        if c["bm25_rank"] is not None:
            assert c["bm25_rank"] >= 1


def test_rrf_math():
    """Verifica el cálculo exacto de RRF."""
    dense = [{"chunk_id": "x"}]
    bm25 = [{"chunk_id": "x"}]
    fused = rrf_fuse(dense, bm25, k_rrf=RRF_K)
    expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
    assert abs(fused[0]["rrf_score"] - expected) < 1e-9


def test_rrf_tiebreak_by_dense_rank():
    """Empate en RRF → gana menor dense_rank."""
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    fused = rrf_fuse(dense, bm25, k_rrf=RRF_K)
    # a: dense 1, bm25 2. b: dense 2, bm25 1. Empate RRF.
    # Gana a por mejor dense_rank.
    assert fused[0]["chunk_id"] == "a"


def test_no_reranker_import():
    src = Path(__file__).parent.parent / "bm25_hybrid.py"
    content = src.read_text(encoding="utf-8")
    assert "reranked_baseline" not in content
    assert "CrossEncoder" not in content


def test_no_qwen_import():
    src = Path(__file__).parent.parent / "bm25_hybrid.py"
    content = src.read_text(encoding="utf-8")
    assert "Qwen" not in content


def test_no_llm_api_import():
    src = Path(__file__).parent.parent / "bm25_hybrid.py"
    content = src.read_text(encoding="utf-8")
    for bad in ["openai", "openrouter", "anthropic", "google.generativeai"]:
        assert bad not in content.lower()


def test_common_unchanged():
    """common/ idéntico a baseline-comun-v2."""
    out = subprocess.check_output(
        ["git", "diff", "baseline-comun-v2", "HEAD", "--", "common/"],
        text=True, cwd=RAIZ,
    )
    assert out.strip() == "", f"common/ modificado:\n{out}"