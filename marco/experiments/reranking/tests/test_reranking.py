"""Tests mínimos de la ablación de reranking."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RAIZ))

import pytest

from marco.experiments.benchmark.evaluator_span import validate_benchmark
from marco.experiments.reranking.dense_backends import (
    QWEN3_MODEL, QWEN3_REVISION, backend_metadata,
)
from marco.experiments.reranking.runner_reranking import (
    BENCHMARK_PATH, BENCHMARK_SHA, RERANKER_MODEL, RERANKER_REVISION,
)


def _load_meta():
    import pandas as pd
    from common.config import get_dataset_paths
    return pd.read_parquet(get_dataset_paths().chunks_meta)


def test_benchmark_sha_correct():
    h = hashlib.sha256()
    with open(BENCHMARK_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest().upper() == BENCHMARK_SHA


def test_48_questions():
    meta = _load_meta()
    gt = validate_benchmark(BENCHMARK_PATH, meta)
    assert len(gt) == 48


def test_no_duplicate_question_ids():
    meta = _load_meta()
    gt = validate_benchmark(BENCHMARK_PATH, meta)
    ids = [g.question_id for g in gt]
    assert len(ids) == len(set(ids))


def test_bge_small_metadata():
    bm = backend_metadata("bge-small")
    assert bm["dense_model"] == "BAAI/bge-small-en-v1.5"
    assert bm["dense_dimension"] == 384


def test_qwen3_metadata():
    bm = backend_metadata("qwen3")
    assert bm["dense_model"] == QWEN3_MODEL
    assert bm["dense_model_revision"] == QWEN3_REVISION
    assert bm["dense_dimension"] == 1024


def test_reranker_model_frozen():
    assert RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert RERANKER_REVISION == (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )


def test_no_bm25_import():
    src = Path(__file__).parent.parent / "runner_reranking.py"
    content = src.read_text(encoding="utf-8")
    assert "bm25" not in content.lower()


def test_no_rewriting_import():
    src = Path(__file__).parent.parent / "runner_reranking.py"
    content = src.read_text(encoding="utf-8")
    assert "query_rewriting" not in content.lower()


def test_no_llm_api_import():
    src = Path(__file__).parent.parent / "runner_reranking.py"
    content = src.read_text(encoding="utf-8").lower()
    for bad in ["openai", "openrouter", "anthropic"]:
        assert bad not in content


def test_common_unchanged():
    out = subprocess.check_output(
        ["git", "diff", "baseline-comun-v2", "HEAD", "--", "common/"],
        text=True, cwd=RAIZ,
    )
    assert out.strip() == "", f"common/ modificado:\n{out}"


def test_dense_backends_no_reranker():
    src = Path(__file__).parent.parent / "dense_backends.py"
    content = src.read_text(encoding="utf-8")
    # No debe importar ni instanciar el reranker
    assert "CrossEncoder" not in content
    assert "bge-reranker" not in content
    assert "sentence_transformers.CrossEncoder" not in content