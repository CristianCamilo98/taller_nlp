"""Configuración inmutable de las ablaciones de embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Describe las variables que deben permanecer congeladas en E0."""

    experiment_id: str = "e0_bge_small_original"
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_revision: str | None = None
    query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    document_prefix: str = ""
    document_format: str = "raw_chunk_text_without_prefix"
    normalize_embeddings: bool = True
    expected_dimension: int = 384
    expected_max_sequence_length: int = 512
    index_type: str = "IndexFlatIP"
    metadata_policy: str = "global_dense_ranking_then_metadata_postfilter"
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    chunk_source: str = "dataset://corpus_miax_2026/chunks.jsonl"
    golden_path: Path = (
        REPO_ROOT / "common" / "golden_set" / "golden_set_grupo3.jsonl"
    )
    evidence_path: Path = (
        REPO_ROOT / "dani" / "experiments" / "results"
        / "evidence_ground_truth_v1.json"
    )
    requested_device: str | None = "cpu"
    embedding_batch_size: int = 32

    @property
    def max_k(self) -> int:
        return max(self.k_values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["golden_path"] = str(self.golden_path)
        result["evidence_path"] = str(self.evidence_path)
        result["k_values"] = list(self.k_values)
        return result


def e1_bge_large_config() -> ExperimentConfig:
    """E1 cambia exclusivamente identidad, modelo y dimensión del embedding."""
    return replace(
        ExperimentConfig(),
        experiment_id="e1_bge_large_original",
        model_name="BAAI/bge-large-en-v1.5",
        expected_dimension=1024,
    )


def e2_e5_large_v2_config() -> ExperimentConfig:
    """E2 cambia a E5-large-v2 y aplica su formato canónico asimétrico."""
    return replace(
        ExperimentConfig(),
        experiment_id="e2_e5_large_v2_original",
        model_name="intfloat/e5-large-v2",
        query_prefix="query: ",
        document_prefix="passage: ",
        document_format="passage_prefix_plus_raw_chunk_text",
        expected_dimension=1024,
    )


QWEN3_TASK_DESCRIPTION = (
    "Given a financial question, retrieve relevant passages from SEC 10-K "
    "filings that answer the question."
)


def e3_qwen3_embedding_06b_config() -> ExperimentConfig:
    """E3 usa Qwen3 0.6B en dimensión nativa y con instrucción congelada."""
    return replace(
        ExperimentConfig(),
        experiment_id="e3_qwen3_embedding_06b_original",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        query_prefix=f"Instruct: {QWEN3_TASK_DESCRIPTION}\nQuery:",
        document_format="raw_chunk_text_without_prefix",
        expected_dimension=1024,
        expected_max_sequence_length=32768,
        requested_device="cuda",
    )


EXPECTED_E0_METRICS: dict[str, float | int] = {
    "recall@1": 0.16666666666666666,
    "recall@3": 0.3333333333333333,
    "recall@5": 0.6666666666666666,
    "recall@10": 0.8333333333333334,
    "mrr@10": 0.3611111111111111,
    "n_questions": 6,
}
