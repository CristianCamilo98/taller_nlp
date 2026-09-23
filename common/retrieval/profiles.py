"""Perfiles congelados para el retrieval denso común."""

from __future__ import annotations

import os
from dataclasses import dataclass


PROFILE_ENV_VAR = "MIAX_RETRIEVAL_PROFILE"
DEFAULT_PROFILE = "bge-small-baseline"


class RetrievalProfileError(ValueError):
    """El perfil solicitado no existe o no es válido."""


@dataclass(frozen=True)
class DenseRetrievalProfile:
    name: str
    model_name: str
    model_revision: str | None
    dimension: int
    pooling: str
    normalize_embeddings: bool
    index_type: str
    index_dirname: str
    query_prefix: str
    query_template: str
    document_format: str = "raw"
    manifest_required: bool = False

    def format_query(self, query: str) -> str:
        """Aplica el formato congelado de cada uno de los dos perfiles."""
        if self.name == "bge-small-baseline":
            return f"{self.query_prefix}{query}"
        if self.name == "qwen3-06b":
            return query if query.startswith(self.query_prefix) else (
                f"{self.query_prefix}{query}"
            )
        raise RetrievalProfileError(f"Perfil no soportado: {self.name!r}")


PROFILES = {
    "bge-small-baseline": DenseRetrievalProfile(
        name="bge-small-baseline",
        model_name="BAAI/bge-small-en-v1.5",
        model_revision=None,
        dimension=384,
        pooling="model-default",
        normalize_embeddings=True,
        index_type="IndexFlatIP",
        index_dirname="indice_faiss",
        query_prefix=(
            "Represent this sentence for searching relevant passages: "
        ),
        query_template=(
            "Represent this sentence for searching relevant passages: {query}"
        ),
    ),
    "qwen3-06b": DenseRetrievalProfile(
        name="qwen3-06b",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_revision=(
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        ),
        dimension=1024,
        pooling="last-token",
        normalize_embeddings=True,
        index_type="IndexFlatIP",
        index_dirname="indice_faiss_qwen3_06b",
        query_prefix=(
            "Instruct: Given a financial question, retrieve relevant passages "
            "from SEC 10-K filings that answer the question.\nQuery:"
        ),
        query_template=(
            "Instruct: Given a financial question, retrieve relevant passages "
            "from SEC 10-K filings that answer the question.\nQuery:{query}"
        ),
        manifest_required=True,
    ),
}


def get_embedding_profile(name: str | None = None) -> DenseRetrievalProfile:
    """Resuelve el perfil explícito o el configurado en el entorno."""
    selected = name or os.getenv(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    try:
        return PROFILES[selected]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILES))
        raise RetrievalProfileError(
            f"Perfil de retrieval desconocido {selected!r}; válidos: {valid}"
        ) from exc
