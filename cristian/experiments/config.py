"""Resolución del dataset y ajustes del experimento (sin depender de common)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATASET_ENV_VAR = "MIAX_DATASET_DIR"
# cristian/experiments/config.py → parents[2] = raíz del repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    corpus_dir: Path
    index_dir: Path
    sections: Path
    chunks: Path
    xbrl_facts: Path
    faiss_index: Path
    chunks_meta: Path

    def required_files(self) -> dict[str, Path]:
        return {
            "secciones.jsonl": self.sections,
            "chunks.jsonl": self.chunks,
            "xbrl_facts.parquet": self.xbrl_facts,
            "corpus.faiss": self.faiss_index,
            "chunks_meta.parquet": self.chunks_meta,
        }


@dataclass(frozen=True)
class AgentSettings:
    """Parámetros del LLM y del runner de evaluación."""

    model: str = "openrouter:deepseek/deepseek-v4-flash"
    # Openai fast model openrouter:openai/gpt-3.5-turbo-0613
    # Gemini 3.8 openrouter:google/gemini-3.8-flash
    # Nex AGI FREE  openrouter:nex-agi/nex-n2.5-mini:free
    # MISTRAL (CHEAP) openrouter:mistralai/mistral-nemo
    # QWEN (CHEAP) openrouter:qwen/qwen3.7-flash (siempre falla por que devuelve los datos no estructurados)
    # META LLAMA (CHEAP) openrouter:meta-llama/llama-3.1-8b-instruct
    temperature: float = 0.0
    tool_call_run_limit: int = 10
    llm_call_run_limit: int = 10  # tope de requests OpenRouter por pregunta
    guardrail_retries: int = 1
    rate_limit_attempts: int = 3
    rate_limit_initial_backoff_s: float = 5.0
    pause_between_questions_s: float = 1.0
    eval_max_workers: int = 5  # preguntas en paralelo en evaluar()
    retrieval_k: int = 5
    prompt_version: str = "cristian-exp-v0"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    provider: str = "openrouter"


SETTINGS = AgentSettings()


def dataset_candidates() -> tuple[Path, ...]:
    configured = os.getenv(DATASET_ENV_VAR)
    if configured:
        return (Path(configured).expanduser(),)
    return (
        _REPO_ROOT / "cristian" / "dataset",
        _REPO_ROOT / "dataset",
        _REPO_ROOT.parent / "dataset",
    )


def _build_paths(dataset_dir: Path) -> DatasetPaths:
    root = dataset_dir.resolve()
    corpus = root / "corpus_miax_2026"
    index = root / "indice_faiss"
    return DatasetPaths(
        dataset_dir=root,
        corpus_dir=corpus,
        index_dir=index,
        sections=corpus / "secciones.jsonl",
        chunks=corpus / "chunks.jsonl",
        xbrl_facts=corpus / "xbrl_facts.parquet",
        faiss_index=index / "corpus.faiss",
        chunks_meta=index / "chunks_meta.parquet",
    )


@lru_cache(maxsize=1)
def get_dataset_paths() -> DatasetPaths:
    diagnostics: list[str] = []
    for candidate in dataset_candidates():
        paths = _build_paths(candidate)
        missing = [
            str(path)
            for path in (
                paths.sections,
                paths.chunks,
                paths.xbrl_facts,
                paths.faiss_index,
                paths.chunks_meta,
            )
            if not path.is_file()
        ]
        if not missing:
            return paths
        diagnostics.append(f"{paths.dataset_dir}: faltan {missing}")
    source = (
        f"{DATASET_ENV_VAR} definida"
        if os.getenv(DATASET_ENV_VAR)
        else "autodetección"
    )
    raise RuntimeError(
        f"Dataset no encontrado o incompleto ({source}). "
        + " | ".join(diagnostics)
    )


def env_path() -> Path:
    return _REPO_ROOT / ".env"


def clear_dataset_path_cache() -> None:
    get_dataset_paths.cache_clear()
