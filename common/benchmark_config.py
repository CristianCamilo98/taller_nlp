"""Configuración congelable del benchmark común, sin efectos laterales."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from common.retrieval.profiles import get_embedding_profile


FINAL_RETRIEVAL_PROFILE = "gemini-embedding-2"
_EMBEDDING_PROFILE = get_embedding_profile(FINAL_RETRIEVAL_PROFILE)


@dataclass(frozen=True)
class BenchmarkConfig:
    provider: str = "openrouter"
    model: str = "openrouter:deepseek/deepseek-v4-flash"
    requested_model_id: str = "deepseek/deepseek-v4-flash"
    temperature: float = 0.0
    tool_call_run_limit: int = 10
    guardrail_retries: int = 1
    rate_limit_attempts: int = 3
    rate_limit_initial_backoff_s: float = 30.0
    pause_between_questions_s: float = 5.0
    retrieval_k: int = 5
    prompt_version: str = "common-v2-2026-09-19"
    retrieval_profile: str = FINAL_RETRIEVAL_PROFILE
    embedding_model: str = _EMBEDDING_PROFILE.model_name
    embedding_revision: str | None = _EMBEDDING_PROFILE.model_revision
    embedding_query_prefix: str = _EMBEDDING_PROFILE.query_prefix
    embedding_dimension: int = _EMBEDDING_PROFILE.dimension
    embedding_pooling: str = _EMBEDDING_PROFILE.pooling
    embedding_normalize: bool = _EMBEDDING_PROFILE.normalize_embeddings
    faiss_index_type: str = _EMBEDDING_PROFILE.index_type
    metadata_filtering: str = "post-filter sobre ranking global"

    def as_dict(self) -> dict:
        return asdict(self)


BENCHMARK = BenchmarkConfig()
