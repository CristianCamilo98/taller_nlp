"""Configuración inmutable del experimento E0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Describe las variables que deben permanecer congeladas en E0."""

    experiment_id: str = "e0_bge_small_original"
    model_name: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    document_format: str = "raw_chunk_text_without_prefix"
    normalize_embeddings: bool = True
    expected_dimension: int = 384
    index_type: str = "IndexFlatIP"
    metadata_policy: str = "global_dense_ranking_then_metadata_postfilter"
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    chunk_source: str = "dataset://corpus_miax_2026/chunks.jsonl"
    golden_path: Path = (
        REPO_ROOT / "common" / "golden_set" / "golden_set_grupo3.jsonl"
    )
    requested_device: str | None = "cpu"
    embedding_batch_size: int = 32

    @property
    def max_k(self) -> int:
        return max(self.k_values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["golden_path"] = str(self.golden_path)
        result["k_values"] = list(self.k_values)
        return result


EXPECTED_E0_METRICS: dict[str, float | int] = {
    "recall@1": 0.16666666666666666,
    "recall@3": 0.3333333333333333,
    "recall@5": 0.6666666666666666,
    "recall@10": 0.8333333333333334,
    "mrr@10": 0.3611111111111111,
    "n_questions": 6,
}

