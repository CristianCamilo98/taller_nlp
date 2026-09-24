"""Benchmark de retrieval v2 (congelado por Dani) — copia local.

Fuente: ``experiment/dani-embeddings-chunking``
``dani/experiments/benchmark/retrieval_benchmark_v2.jsonl``

SHA-256 esperado (mayúsculas):
``6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C``
"""

from __future__ import annotations

from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = BENCHMARK_DIR / "retrieval_benchmark_v2.jsonl"
BENCHMARK_NAME = "retrieval_benchmark_v2"
FROZEN_SHA256 = (
    "6BD044EF3135686942B42CC9A654019D43675C1ACB2629ACE4B11274FFD80B8C"
)
