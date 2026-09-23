"""Retriever FAISS exacto que conserva la política de postfiltrado E0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from dani.experiments.config import ExperimentConfig
from dani.experiments.embeddings import EmbeddingAdapter


@dataclass(frozen=True)
class SearchOutcome:
    results: list[dict[str, Any]]
    query_embedding_s: float
    index_search_s: float
    postfilter_s: float
    total_s: float


class DenseFaissRetriever:
    """Ranking global IndexFlatIP seguido de ticker/año/item, como common."""

    def __init__(
        self,
        index: Any,
        metadata: Any,
        adapter: EmbeddingAdapter,
        config: ExperimentConfig,
    ):
        self.index = index
        self.metadata = metadata.reset_index(drop=True)
        self.adapter = adapter
        self.config = config
        self._validate_index()

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        metadata: Any,
        adapter: EmbeddingAdapter,
        config: ExperimentConfig,
    ) -> tuple["DenseFaissRetriever", dict[str, float]]:
        import faiss

        embedding_start = perf_counter()
        vectors = adapter.encode_documents(texts)
        embedding_s = perf_counter() - embedding_start
        if vectors.shape != (len(metadata), config.expected_dimension):
            raise ValueError("Embeddings y metadata no tienen el mismo número de filas")
        if not np.isfinite(vectors).all():
            raise ValueError("Los embeddings contienen NaN o infinito")

        index_start = perf_counter()
        index = faiss.IndexFlatIP(config.expected_dimension)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        index_build_s = perf_counter() - index_start
        retriever = cls(index, metadata, adapter, config)
        return retriever, {
            "document_embedding_s": embedding_s,
            "index_build_s": index_build_s,
        }

    @classmethod
    def load(
        cls,
        index_path: Path,
        metadata: Any,
        adapter: EmbeddingAdapter,
        config: ExperimentConfig,
    ) -> "DenseFaissRetriever":
        import faiss

        return cls(faiss.read_index(str(index_path)), metadata, adapter, config)

    def save(self, index_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    def search(
        self,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> SearchOutcome:
        if k <= 0:
            raise ValueError("k debe ser positivo")
        total_start = perf_counter()
        query_start = perf_counter()
        vector = self.adapter.encode_query(query)
        query_s = perf_counter() - query_start

        search_start = perf_counter()
        scores, positions = self.index.search(vector, self.index.ntotal)
        index_search_s = perf_counter() - search_start

        filter_start = perf_counter()
        results: list[dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0]):
            row = self.metadata.iloc[int(position)]
            if ticker and row["ticker"] != ticker:
                continue
            if fiscal_year and int(row["fiscal_year"]) != int(fiscal_year):
                continue
            if item and row["item"] != item:
                continue
            results.append({
                "chunk_id": str(row["chunk_id"]),
                "score": float(score),
                "ticker": str(row["ticker"]),
                "fiscal_year": int(row["fiscal_year"]),
                "item": str(row["item"]),
                "posicion": int(row["posicion"]),
                "texto": str(row["texto"]),
            })
            if len(results) >= k:
                break
        postfilter_s = perf_counter() - filter_start
        return SearchOutcome(
            results=results,
            query_embedding_s=query_s,
            index_search_s=index_search_s,
            postfilter_s=postfilter_s,
            total_s=perf_counter() - total_start,
        )

    def _validate_index(self) -> None:
        actual_type = type(self.index).__name__
        if actual_type != self.config.index_type:
            raise ValueError(
                f"Tipo de índice {actual_type}; esperado {self.config.index_type}"
            )
        if int(self.index.d) != self.config.expected_dimension:
            raise ValueError(
                f"Dimensión del índice {self.index.d}; esperada "
                f"{self.config.expected_dimension}"
            )
        if int(self.index.ntotal) != len(self.metadata):
            raise ValueError(
                f"Índice con {self.index.ntotal} vectores y metadata con "
                f"{len(self.metadata)} filas"
            )

