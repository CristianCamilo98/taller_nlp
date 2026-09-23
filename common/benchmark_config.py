"""Configuración congelable del benchmark común, sin efectos laterales."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import os

@dataclass(frozen=True)
class BenchmarkConfig:
    provider: str = "openrouter"
    model: str = "openrouter:google/gemini-3.8-flash"
    requested_model_id: str = "google/gemini-3.8-flash"
    temperature: float = 0.0
    tool_call_run_limit: int = 10
    guardrail_retries: int = 1
    rate_limit_attempts: int = 3
    rate_limit_initial_backoff_s: float = 30.0
    pause_between_questions_s: float = 5.0
    retrieval_k: int = 5
    prompt_version: str = "common-v2-2026-09-19"
    embedding_model: str = os.getenv(
        "MIAX_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    embedding_query_prefix: str = os.getenv(
        "MIAX_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    embedding_normalize: bool = True
    faiss_index_type: str = "IndexFlatIP"
    metadata_filtering: str = "post-filter sobre ranking global"

    def as_dict(self) -> dict:
        return asdict(self)


BENCHMARK = BenchmarkConfig()
