"""Dense search backends para la ablación de reranking.

Soporta dos backends causales:
- "bge-small": BAAI/bge-small-en-v1.5 sobre dataset/indice_faiss (baseline)
- "qwen3":     Qwen/Qwen3-Embedding-0.6B sobre dataset/indice_faiss_qwen3

Política común: ranking global → postfilter metadata.
Sin reranker, sin BM25, sin rewriting.
"""
from __future__ import annotations

import functools
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]

QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a financial question, retrieve relevant passages "
    "from SEC 10-K filings that answer the question.\nQuery:"
)
QWEN3_INDEX_DIR = RAIZ / "dataset" / "indice_faiss_qwen3"

BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


@functools.lru_cache(maxsize=1)
def _qwen3_state():
    """Carga Qwen3 + índice + metadatos una sola vez."""
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    idx = faiss.read_index(str(QWEN3_INDEX_DIR / "corpus.faiss"))
    meta = pd.read_parquet(QWEN3_INDEX_DIR / "chunks_meta.parquet")
    if idx.ntotal != len(meta):
        raise RuntimeError(
            f"Qwen3: índice {idx.ntotal} vs meta {len(meta)}"
        )
    model = SentenceTransformer(
        QWEN3_MODEL, revision=QWEN3_REVISION,
    )
    return idx, meta, model


def _qwen3_search(query: str, ticker=None, fiscal_year=None, item=None,
                  k: int = 20) -> list[dict]:
    idx, meta, model = _qwen3_state()
    vector = model.encode(
        [QWEN3_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, positions = idx.search(vector, idx.ntotal)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        row = meta.iloc[int(pos)]
        if ticker and row["ticker"] != ticker:
            continue
        if fiscal_year and int(row["fiscal_year"]) != int(fiscal_year):
            continue
        if item and str(row["item"]) != str(item):
            continue
        results.append({
            "chunk_id": row["chunk_id"],
            "ticker": row["ticker"],
            "fiscal_year": int(row["fiscal_year"]),
            "item": row["item"],
            "texto": row["texto"],
            "puntuacion": round(float(score), 6),
        })
        if len(results) >= k:
            break
    return results


def dense_search(backend: str, query: str, ticker=None, fiscal_year=None,
                 item=None, k: int = 20) -> list[dict]:
    """Buscador dense unificado para la ablación."""
    if backend == "bge-small":
        from common.retrieval.dense_baseline import buscar
        return buscar(query, ticker=ticker, fiscal_year=fiscal_year,
                      item=item, k=k)
    if backend == "qwen3":
        return _qwen3_search(query, ticker=ticker, fiscal_year=fiscal_year,
                             item=item, k=k)
    raise ValueError(f"Backend no soportado: {backend}")


def backend_metadata(backend: str) -> dict:
    """Metadatos de provenance del backend."""
    import os
    if backend == "bge-small":
        return {
            "dense_model": "BAAI/bge-small-en-v1.5",
            "dense_model_revision": None,
            "dense_query_prefix": BGE_QUERY_PREFIX,
            "dense_index_dir": str(
                (RAIZ / "dataset" / "indice_faiss").relative_to(RAIZ)
            ),
            "dense_dimension": 384,
        }
    if backend == "qwen3":
        return {
            "dense_model": QWEN3_MODEL,
            "dense_model_revision": QWEN3_REVISION,
            "dense_query_prefix": QWEN3_QUERY_PREFIX,
            "dense_index_dir": str(
                QWEN3_INDEX_DIR.relative_to(RAIZ)
            ),
            "dense_dimension": 1024,
        }
    raise ValueError(f"Backend no soportado: {backend}")